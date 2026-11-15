"""A backend that runs the "remote" runtime on this machine.

Used for tests and for trying the tool without a cloud account. Each runtime
is a directory under `base_dir` (default: default_state_dir() / "local",
so separate CLI invocations see the same runtimes) with
`remote_root` = <runtime dir>/project. Commands run as local subprocesses
with the given cwd and env. The environment is not rebuilt from the spec:
commands run with `python` resolving to the interpreter in `python`
(default sys.executable), whose environment must already have marimo.

Ports: `create` picks a free local port for each requested port and records
it in handle.port_map; `expose` returns "http://127.0.0.1:<that port>".

Runtimes are tracked in a JSON registry inside `base_dir`, so `get` and
`find` work across LocalBackend instances that share a base_dir.
Spawned processes start in their own session (start_new_session) so they
outlive the CLI process that started them (`nbgpu start --detach`).
`terminate` kills every process spawned in that runtime (each one's process
group) and deletes the runtime directory.

Capabilities: none of the optional ones (no GPU, volumes, idle or max
runtime enforcement). A spec with a GPU raises UnsupportedError.

Implementation notes (decisions beyond the contract):

- Layout: <base_dir>/registry.json, <base_dir>/registry.lock and
  <base_dir>/runtimes/<runtime_id>/{project,logs}.
- Registry concurrency: every read-modify-write of registry.json holds an
  exclusive `fcntl.flock` on registry.lock and replaces the file atomically
  (temp file + os.replace), so readers never take the lock and never see a
  partial file, and concurrent writers in other processes serialize.
- `python` resolution: an argv[0] of exactly "python" or "python3" is
  replaced by the configured interpreter, and that interpreter's directory
  is put first on PATH (unless `env` sets PATH), so `python`, `marimo` and
  friends also resolve inside shell commands such as `sh -c "python ..."`.
- Spawned output (merged stdout/stderr) goes to a log file in the runtime's
  `logs/` directory rather than a pipe, so a detached process never dies of
  SIGPIPE once the CLI that spawned it exits; `lines()` tails that file.
- Each spawned pid (== its process group id, thanks to start_new_session) is
  recorded in the registry. `terminate` sends SIGTERM to every group, waits
  briefly, then SIGKILLs what is left. Pids can in principle be reused by the
  OS after a process dies; a stale pid is only signalled while its runtime is
  being terminated.
- A spec with volumes is also rejected with UnsupportedError (no
  approximation of a missing capability).
- Permissions: base_dir and any directory created on the way to it are made
  with mode 0700 (chmod'ed exactly, so the umask does not matter); the
  registry holds the runtimes' env. Directories that already exist keep
  their permissions.
- write_file's compare-and-swap writes a temp file (TEMP_PREFIX) next to the
  target, then checks `expect` and replaces in one function call
  (`_cas_replace`). ABSENT uses os.link, which fails if the target exists,
  so it is atomic against other writers; the SHA-256 check is not (the
  runtime is a plain directory with no locking), but nothing else runs
  between the check and the os.replace in this process.
- hash_files runs in-process with a stat cache at
  <runtime dir>/nbgpu-hash-cache.json (next to `project`, so never synced);
  see HASH_CACHE_NAME. Its helpers run in-process, so there is no internal
  command to mark with NBGPU_INTERNAL; that variable is removed from the
  inherited environment of user commands (an explicit `env` keeps it).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from nbgpu.backends.base import (
    ABSENT,
    TEMP_PREFIX,
    BackendError,
    ExecResult,
    RuntimeHandle,
    RuntimeRequest,
    UnsupportedError,
)

EXCLUDED_PARTS: frozenset[str] = frozenset({".git", ".venv", "__pycache__", "node_modules"})
PYTHON_NAMES: frozenset[str] = frozenset({"python", "python3"})

# Marks the backend's own helper commands so the idle watchdog can ignore
# their CPU use. This backend runs its helpers in-process (nothing to mark),
# and never lets the variable leak from the CLI's environment into user
# commands; callers that pass it explicitly in `env` keep it.
INTERNAL_ENV_VAR = "NBGPU_INTERNAL"

# hash_files' stat cache, kept next to the project directory (outside what
# is synced): {"root": <hashed root>, "files": {relpath: [size, mtime_ns,
# ctime_ns, inode, sha256]}}. A file is re-hashed unless all four stat
# fields match and it has not changed within HASH_RECENT_NS (timestamp
# granularity). A missing or corrupt cache only means re-hashing.
HASH_CACHE_NAME = "nbgpu-hash-cache.json"
HASH_RECENT_NS = 2_000_000_000

_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_DIR_MODE = 0o700
_TERM_GRACE_S = 3.0
_KILL_GRACE_S = 2.0

# Spawned children of this process, kept until reaped: `terminate` reaps them
# (a zombie leader keeps its process group visible to killpg(pgid, 0)), and
# holding them avoids Popen.__del__'s "still running" ResourceWarning for
# processes that are meant to outlive their LocalProcess object.
_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}


def _track_child(popen: subprocess.Popen[bytes]) -> None:
    for pid, child in list(_CHILDREN.items()):
        if child.poll() is not None:
            del _CHILDREN[pid]
    _CHILDREN[popen.pid] = popen

Record = dict[str, Any]


def _mkdir_private(path: Path) -> None:
    """Create `path` and any missing parents with mode exactly 0700. Existing
    directories are left as they are."""
    missing: list[Path] = []
    p = path
    while not p.exists():
        missing.append(p)
        if p.parent == p:
            break
        p = p.parent
    for d in reversed(missing):
        try:
            os.mkdir(d, _PRIVATE_DIR_MODE)
        except FileExistsError:
            continue  # made concurrently by someone else: not ours to chmod
        os.chmod(d, _PRIVATE_DIR_MODE)
    if not path.is_dir():
        raise BackendError(f"{path} is not a directory")


def _check_expect(expect: str | None) -> str | None:
    if expect is None or expect == ABSENT:
        return expect
    lowered = expect.lower()
    if not _SHA256_RE.match(lowered):
        raise BackendError(f"write_file: expect must be None, ABSENT or a SHA-256 hex digest, got {expect!r}")
    return lowered


def _sha256_of(path: Path) -> str | None:
    """SHA-256 hex of the file, or None if it does not exist."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return None
    return digest.hexdigest()


def _hash_file(path: str) -> str:
    """SHA-256 hex of the file (raises OSError)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hash_cache(cache_path: str, root: str) -> dict[str, list[Any]]:
    try:
        with open(cache_path, "rb") as f:
            data = json.load(f)
        files = data["files"] if data.get("root") == root else {}
        return files if isinstance(files, dict) else {}
    except Exception:  # missing, unreadable or corrupt: re-hash everything
        return {}


def _hash_tree(root: str, cache_path: str) -> dict[str, str]:
    """{relpath: sha256} for every regular file under `root` (see
    hash_files), reusing hashes from the stat cache at `cache_path`."""
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    old = _load_hash_cache(cache_path, root)
    new: dict[str, list[Any]] = {}
    recent = time.time_ns() - HASH_RECENT_NS
    cache_abs = os.path.abspath(cache_path)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_PARTS]
        for filename in filenames:
            if filename in EXCLUDED_PARTS or filename.startswith(TEMP_PREFIX):
                continue
            full = os.path.join(dirpath, filename)
            if os.path.abspath(full) == cache_abs:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel.startswith("/") or ".." in rel.split("/"):
                continue  # cannot happen from os.walk; never hand one out
            try:
                st = os.lstat(full)
                if not stat.S_ISREG(st.st_mode):
                    continue
                key = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]
                entry = old.get(rel)
                if (
                    isinstance(entry, list)
                    and len(entry) == 5
                    and entry[:4] == key
                    and isinstance(entry[4], str)
                    and max(st.st_mtime_ns, st.st_ctime_ns) < recent
                ):
                    digest = entry[4]
                else:
                    digest = _hash_file(full)
            except OSError:
                continue  # removed or unreadable while walking
            out[rel] = digest
            new[rel] = [*key, digest]
    if new != old:
        _write_hash_cache(cache_path, root, new)
    return out


def _write_hash_cache(cache_path: str, root: str, files: dict[str, list[Any]]) -> None:
    """Best effort: a failed write only costs re-hashing next time."""
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=TEMP_PREFIX, dir=os.path.dirname(cache_path))
        with os.fdopen(fd, "w") as f:
            json.dump({"root": root, "files": files}, f)
        os.replace(tmp, cache_path)
        tmp = None
    except OSError:
        pass
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _cas_replace(tmp: str, path: Path, expect: str | None) -> bool:
    """Check `expect` against `path` and move `tmp` onto it, in one call.
    Returns whether it was written; `tmp` never survives either way."""
    written = False
    try:
        if expect is None:
            os.replace(tmp, path)
            written = True
        elif expect == ABSENT:
            try:
                os.link(tmp, path)  # fails if `path` exists, atomically
                written = True
            except FileExistsError:
                pass
            except OSError:  # no hard links on this filesystem
                if not os.path.lexists(path):
                    os.replace(tmp, path)
                    written = True
        elif _sha256_of(path) == expect:
            os.replace(tmp, path)
            written = True
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)  # after os.link, or when not written
    return written


def _atomic_write(path: Path, data: bytes, expect: str | None = None) -> bool:
    """Write `data` to `path` via a TEMP_PREFIX temp file in the same
    directory, then a conditional replace (see _cas_replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=TEMP_PREFIX, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return _cas_replace(tmp, path, expect)


def _group_alive(pgid: int) -> bool:
    child = _CHILDREN.get(pgid)
    if child is not None:
        child.poll()  # reap our own zombie leader
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False  # not ours (pid reused by another user): leave it alone
    return True


def _signal_group(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _kill_groups(pgids: Sequence[int]) -> None:
    """SIGTERM each group, wait, then SIGKILL the survivors."""
    alive = [p for p in pgids if _group_alive(p)]
    for pgid in alive:
        _signal_group(pgid, signal.SIGTERM)
    for sig, grace in ((signal.SIGKILL, _TERM_GRACE_S), (None, _KILL_GRACE_S)):
        deadline = time.monotonic() + grace
        while alive and time.monotonic() < deadline:
            time.sleep(0.05)
            alive = [p for p in alive if _group_alive(p)]
        if sig is None:
            break
        for pgid in alive:
            _signal_group(pgid, sig)


class LocalProcess:
    """A process started by LocalBackend.spawn (implements RemoteProcess)."""

    def __init__(self, popen: subprocess.Popen[bytes], log_path: Path) -> None:
        self._popen = popen
        self._log_path = log_path

    @property
    def pid(self) -> int:
        return self._popen.pid

    def lines(self) -> Iterator[str]:
        buf = b""
        with open(self._log_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    if self._popen.poll() is None:
                        time.sleep(0.05)
                        continue
                    chunk = f.read()  # anything written just before exit
                    if not chunk:
                        break
                buf += chunk
                *complete, buf = buf.split(b"\n")
                for line in complete:
                    yield _decode_line(line)
        if buf:
            yield _decode_line(buf)

    def poll(self) -> int | None:
        return self._popen.poll()

    def terminate(self) -> None:
        if self._popen.poll() is not None:
            return
        _kill_groups([self._popen.pid])
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._popen.wait(timeout=1)


def _decode_line(line: bytes) -> str:
    if line.endswith(b"\r"):
        line = line[:-1]
    return line.decode("utf-8", errors="replace")


class LocalBackend:
    name = "local"
    capabilities: frozenset[str] = frozenset()
    bind_host = "127.0.0.1"

    def __init__(self, base_dir: Path | None = None, python: str = sys.executable) -> None:
        """`base_dir` None -> default_state_dir() / "local", read at call time."""
        if base_dir is None:
            from nbgpu.session import default_state_dir

            base_dir = default_state_dir() / "local"
        self.base_dir = Path(base_dir).expanduser().absolute()
        self.python = python

    # ------------------------------------------------------------------
    # registry

    @property
    def _registry_path(self) -> Path:
        return self.base_dir / "registry.json"

    def _runtime_dir(self, runtime_id: str) -> Path:
        return self.base_dir / "runtimes" / runtime_id

    def _load(self) -> dict[str, Record]:
        try:
            raw = self._registry_path.read_bytes()
        except FileNotFoundError:
            return {}
        try:
            data = json.loads(raw)
            runtimes = data["runtimes"]
            if not isinstance(runtimes, dict):
                raise TypeError("runtimes is not an object")
        except (ValueError, KeyError, TypeError) as exc:
            raise BackendError(f"local runtime registry {self._registry_path} is corrupt: {exc}") from exc
        return runtimes

    @contextlib.contextmanager
    def _locked(self) -> Iterator[dict[str, Record]]:
        """Exclusive read-modify-write of the registry. Mutate the yielded dict;
        it is written back atomically on normal exit."""
        _mkdir_private(self.base_dir)
        with open(self.base_dir / "registry.lock", "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                runtimes = self._load()
                yield runtimes
                payload = json.dumps({"version": 1, "runtimes": runtimes}, indent=2, sort_keys=True)
                _atomic_write(self._registry_path, payload.encode())
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _record(self, runtime_id: str) -> Record | None:
        if not _RUNTIME_ID_RE.match(runtime_id):
            return None
        return self._load().get(runtime_id)

    def _handle(self, record: Record) -> RuntimeHandle:
        return RuntimeHandle(
            backend=self.name,
            runtime_id=record["runtime_id"],
            remote_root=record["remote_root"],
            created_at=float(record["created_at"]),
            tags=dict(record["tags"]),
            port_map={int(k): int(v) for k, v in record["port_map"].items()},
        )

    def _require(self, handle: RuntimeHandle) -> Record:
        record = self._record(handle.runtime_id)
        if record is None:
            raise BackendError(f"local runtime {handle.runtime_id} is not running")
        return record

    # ------------------------------------------------------------------
    # lifecycle

    def create(self, request: RuntimeRequest) -> RuntimeHandle:
        compute = request.spec.compute
        if compute.gpu is not None:
            raise UnsupportedError(
                f"the local backend has no GPUs (spec asks for gpu={compute.gpu!r}); "
                "remove `gpu` or use a GPU provider such as modal"
            )
        if compute.volumes:
            raise UnsupportedError("the local backend does not support volumes; remove `volumes` or use modal")

        runtime_id = f"local-{uuid.uuid4().hex[:12]}"
        runtime_dir = self._runtime_dir(runtime_id)
        remote_root = runtime_dir / "project"
        with self._locked() as runtimes:
            taken = {int(p) for r in runtimes.values() for p in r.get("port_map", {}).values()}
            port_map = self._pick_ports(request.ports, taken)
            remote_root.mkdir(parents=True)
            (runtime_dir / "logs").mkdir()
            record: Record = {
                "runtime_id": runtime_id,
                "dir": str(runtime_dir),
                "remote_root": str(remote_root),
                "created_at": time.time(),
                "tags": dict(request.tags),
                "port_map": {str(k): v for k, v in port_map.items()},
                "env": dict(request.env),
                "pids": [],
            }
            runtimes[runtime_id] = record
        return self._handle(record)

    @staticmethod
    def _pick_ports(requested: Sequence[int], taken: set[int]) -> dict[int, int]:
        """A distinct free local port per requested port, avoiding ports
        already assigned to other registered runtimes."""
        port_map: dict[int, int] = {}
        held: list[socket.socket] = []
        try:
            for port in dict.fromkeys(requested):
                for _ in range(100):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(("127.0.0.1", 0))
                    held.append(sock)  # hold it so the next pick differs
                    chosen = sock.getsockname()[1]
                    if chosen not in taken:
                        break
                else:
                    raise BackendError("could not find a free local port")
                port_map[port] = chosen
                taken.add(chosen)
        finally:
            for sock in held:
                sock.close()
        return port_map

    def get(self, runtime_id: str) -> RuntimeHandle | None:
        record = self._record(runtime_id)
        if record is None or not Path(record["remote_root"]).is_dir():
            return None
        return self._handle(record)

    def find(self, tags: Mapping[str, str]) -> list[RuntimeHandle]:
        wanted = dict(tags).items()
        return [
            self._handle(record)
            for record in sorted(self._load().values(), key=lambda r: r["created_at"])
            if dict(record["tags"]).items() >= wanted and Path(record["remote_root"]).is_dir()
        ]

    def terminate(self, handle: RuntimeHandle) -> None:
        runtime_id = handle.runtime_id
        if not _RUNTIME_ID_RE.match(runtime_id):
            return
        record = self._record(runtime_id)
        if record is not None:
            _kill_groups([int(p) for p in record.get("pids", [])])
            with self._locked() as runtimes:
                runtimes.pop(runtime_id, None)
        shutil.rmtree(self._runtime_dir(runtime_id), ignore_errors=True)

    # ------------------------------------------------------------------
    # commands

    def _resolve(self, handle: RuntimeHandle, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else Path(handle.remote_root) / p

    def _command(
        self,
        handle: RuntimeHandle,
        record: Record,
        argv: Sequence[str],
        cwd: str | None,
        env: Mapping[str, str] | None,
    ) -> tuple[list[str], Path, dict[str, str]]:
        if not argv:
            raise BackendError("empty command")
        args = list(argv)
        if args[0] in PYTHON_NAMES:
            args[0] = self.python
        full_env = dict(os.environ)
        full_env.pop(INTERNAL_ENV_VAR, None)  # only ever set explicitly
        python_dir = os.path.dirname(os.path.abspath(self.python))
        full_env["PATH"] = os.pathsep.join(p for p in (python_dir, full_env.get("PATH", "")) if p)
        full_env.update(record.get("env", {}))
        full_env.update(env or {})
        workdir = self._resolve(handle, cwd) if cwd is not None else Path(handle.remote_root)
        return args, workdir, full_env

    def exec(
        self,
        handle: RuntimeHandle,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        record = self._require(handle)
        args, workdir, full_env = self._command(handle, record, argv, cwd, env)
        try:
            popen = subprocess.Popen(
                args,
                cwd=workdir,
                env=full_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # so a timeout can kill the whole group
            )
        except OSError as exc:
            raise BackendError(f"could not run {args[0]!r}: {exc}") from exc
        try:
            out, err = popen.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _signal_group(popen.pid, signal.SIGKILL)
            popen.communicate()
            raise BackendError(f"command timed out after {timeout_s}s: {' '.join(argv)}") from None
        except BaseException:
            _signal_group(popen.pid, signal.SIGKILL)
            popen.wait()
            raise
        return ExecResult(
            exit_code=popen.returncode,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
        )

    def spawn(
        self,
        handle: RuntimeHandle,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> LocalProcess:
        record = self._require(handle)
        args, workdir, full_env = self._command(handle, record, argv, cwd, env)
        logs = Path(record["dir"]) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{int(time.time())}-{uuid.uuid4().hex[:8]}.log"
        with open(log_path, "wb") as log:
            try:
                popen = subprocess.Popen(
                    args,
                    cwd=workdir,
                    env=full_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # outlives the CLI; killable as a group
                )
            except OSError as exc:
                raise BackendError(f"could not start {args[0]!r}: {exc}") from exc
        _track_child(popen)
        with self._locked() as runtimes:
            current = runtimes.get(handle.runtime_id)
            if current is not None:
                current.setdefault("pids", []).append(popen.pid)
        if current is None:  # terminated while we were starting it
            _kill_groups([popen.pid])
            raise BackendError(f"local runtime {handle.runtime_id} was terminated")
        return LocalProcess(popen, log_path)

    def expose(self, handle: RuntimeHandle, port: int) -> str:
        port_map = handle.port_map
        if port not in port_map:
            record = self._require(handle)
            port_map = {int(k): int(v) for k, v in record["port_map"].items()}
        if port not in port_map:
            raise BackendError(f"port {port} was not requested when runtime {handle.runtime_id} was created")
        return f"http://127.0.0.1:{port_map[port]}"

    # ------------------------------------------------------------------
    # files

    def read_file(self, handle: RuntimeHandle, path: str) -> bytes | None:
        try:
            return self._resolve(handle, path).read_bytes()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return None

    def write_file(
        self,
        handle: RuntimeHandle,
        path: str,
        data: bytes,
        *,
        expect: str | None = None,
    ) -> bool:
        expect = _check_expect(expect)
        try:
            return _atomic_write(self._resolve(handle, path), data, expect)
        except OSError as exc:
            raise BackendError(f"could not write {path}: {exc}") from exc

    def hash_files(self, handle: RuntimeHandle, root: str = ".") -> dict[str, str]:
        base = self._resolve(handle, root)
        return _hash_tree(str(base), self.hash_cache_path(handle))

    @staticmethod
    def hash_cache_path(handle: RuntimeHandle) -> str:
        """Where hash_files keeps its stat cache: next to remote_root."""
        return os.path.join(os.path.dirname(os.path.normpath(handle.remote_root)), HASH_CACHE_NAME)
