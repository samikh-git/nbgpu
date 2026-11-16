"""Keep a running session's Python environment in step with the notebook.

The runtime's image is built from the notebook's dependencies when the
session starts. While it runs, they can change in two ways, both handled
here so the header stays the single source of truth:

- In marimo's UI (installing a missing package, or the packages panel):
  the laptop's proxy (nbgpu.proxy) sees the request and the sync loop adds
  exactly that package to the notebook's header on the laptop, which syncs
  up like any edit; the next start bakes it into the image. marimo's own
  header management (MARIMO_MANAGE_SCRIPT_METADATA) stays off: it also pins
  the version of every package a cell imports, rewriting the header as cells
  run. Installs that bypass the proxy (the remote link, `marimo pair`) are
  listed when the session stops (`unrecorded_installs`).
- On the laptop (the user or an agent edits `dependencies`, or a file named
  by [tool.nbgpu] requirements): `EnvWatcher.tick()`, called from the sync
  loop, notices, and on backends with CAP_LIVE_INSTALL runs
  `uv pip install` with the whole new list inside the runtime, in the
  background (syncing continues). New packages can be imported right away;
  upgraded ones need a kernel restart, which the message says.

What cannot change in place is reported instead: removed packages stay
installed until the next start, and compute (gpu, cpu, memory, volumes,
python) is fixed for the runtime's life.

The applied state is kept in the session record (extra["env"]), so `nbgpu
open` resumes from it.

Without a sync loop (`start --detach`, the way agents work), `nbgpu sync`
and `nbgpu run` call `EnvWatcher.apply()`, which does the same comparison
and installs synchronously, so a package added to the header is importable
by the run that follows. A per-session lock file (`<id>.env.lock` in the
state directory) keeps a loop's background install and a `run`/`sync` from
installing at the same time; both re-read the applied state from the
session record after taking it, so whichever comes second sees the first
one's work and does nothing.
"""

from __future__ import annotations

import collections
import fcntl
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from nbgpu.backends.base import CAP_LIVE_INSTALL, Backend, RemoteProcess, RuntimeHandle
from nbgpu.session import Session, SessionStore
from nbgpu.spec import NotebookSpec, SpecError, load_spec, requirement_name

EXTRA_ENV = "env"
# --system: the runtime's Python is not a virtualenv (it is the image's
# interpreter, the one marimo and its kernels run on).
INSTALL_ARGV = ("uv", "pip", "install", "--system")
_TAIL_LINES = 8
# How long `apply()` waits for another install to finish, and for its own.
INSTALL_TIMEOUT_S = 30 * 60.0
_LOCK_POLL_S = 0.2


class _EnvLock:
    """An exclusive flock on `<state dir>/<session id>.env.lock`."""

    def __init__(self, store: SessionStore, session_id: str) -> None:
        self._path = store.directory / f"{session_id}.env.lock"
        self._fd: int | None = None

    def acquire(self, timeout_s: float = 0.0) -> bool:
        """True once held; False if another process still holds it after
        `timeout_s` (0: do not wait)."""
        if self._fd is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(_LOCK_POLL_S)
            else:
                self._fd = fd
                return True

    def release(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


@dataclass(frozen=True)
class Applied:
    """What `EnvWatcher.apply()` did. Each message starts with a role
    character (see EnvWatcher.tick). `failed`: an install was attempted and
    did not succeed, so the runtime lacks something the header lists."""

    messages: list[str]
    failed: bool = False


PACKAGES_BASELINE_NAME = "nbgpu-packages-at-start.json"

# argv[1]: the baseline file. Without argv[2] ("diff"): write the names of
# the installed distributions to it. With it: print, as JSON, the
# distributions installed since then that no other new one requires (the
# packages someone asked for, not their dependencies).
PACKAGES_CODE = r"""
import json, re, sys
from importlib import metadata
def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()
dists = {norm(d.metadata["Name"]): d for d in metadata.distributions() if d.metadata["Name"]}
if len(sys.argv) < 3:
    with open(sys.argv[1], "w") as f:
        json.dump(sorted(dists), f)
    sys.exit(0)
try:
    with open(sys.argv[1]) as f:
        base = set(json.load(f))
except (OSError, ValueError):
    print("null")
    sys.exit(0)
new = {n: d for n, d in dists.items() if n not in base}
required = set()
for d in new.values():
    for req in d.requires or []:
        if "extra ==" in req:
            continue
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
        if m:
            required.add(norm(m.group(1)))
print(json.dumps(sorted(f"{n}=={d.version}" for n, d in new.items() if n not in required)))
"""


def baseline_argv(path: str) -> list[str]:
    return ["python", "-c", PACKAGES_CODE, path]


def unrecorded_installs(backend: Backend, handle: RuntimeHandle, baseline: str, spec: NotebookSpec) -> list[str] | None:
    """Packages installed in the runtime since it started (top-level only)
    that the notebook's dependencies do not name; None if unknown."""
    if CAP_LIVE_INSTALL not in backend.capabilities:
        return None
    res = backend.exec(handle, ["python", "-c", PACKAGES_CODE, baseline, "diff"], timeout_s=60)
    if res.exit_code != 0:
        return None
    try:
        found = json.loads(res.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    if not isinstance(found, list):
        return None
    named = {requirement_name(d) for d in spec.dependencies}
    return [p for p in found if isinstance(p, str) and requirement_name(p) not in named]


def snapshot(spec: NotebookSpec) -> dict[str, object]:
    c = spec.compute
    return {
        "deps": list(spec.dependencies),
        "compute": {
            "gpu": c.gpu, "cpu": c.cpu, "memory_mib": c.memory_mib,
            "volumes": dict(c.volumes), "python": spec.requires_python, "provider": c.provider,
            "cache": c.cache_volume, "cell-cache": spec.cell_cache,
        },
    }


def snapshot_json(spec: NotebookSpec) -> str:
    return json.dumps(snapshot(spec), sort_keys=True)


@dataclass
class _Install:
    proc: RemoteProcess
    deps: list[str]
    added: list[str]
    tail: collections.deque[str] = field(default_factory=lambda: collections.deque(maxlen=_TAIL_LINES))
    reader: threading.Thread | None = None


class EnvWatcher:
    def __init__(self, session: Session, backend: Backend, handle: RuntimeHandle, store: SessionStore) -> None:
        self._session = session
        self._backend = backend
        self._handle = handle
        self._store = store
        # None: the first tick compares the notebook with what was applied, so
        # a loop resumed by `nbgpu open` catches edits made while it was off.
        self._fingerprint: tuple | None = None
        self._install: _Install | None = None
        self._reported: set[str] = set()  # one-off notes already shown
        self._lock = _EnvLock(store, session.session_id)
        applied = None
        raw = session.extra.get(EXTRA_ENV)
        if raw:
            try:
                applied = json.loads(raw)
            except json.JSONDecodeError:
                applied = None
        if not isinstance(applied, dict):
            # An older record: take the notebook as it is now as applied.
            try:
                applied = snapshot(load_spec(session.notebook))
            except SpecError:
                applied = {"deps": [], "compute": {}}
        self._applied: dict = applied

    @property
    def installing(self) -> bool:
        return self._install is not None

    def _files(self) -> list[Path]:
        files = [self._session.notebook]
        try:
            files += list(load_spec(self._session.notebook).requirement_files)
        except SpecError:
            pass
        return files

    def _current_fingerprint(self) -> tuple:
        out = []
        for path in self._files():
            try:
                st = os.stat(path)
                out.append((str(path), st.st_size, st.st_mtime_ns))
            except OSError:
                out.append((str(path), None, None))
        return tuple(out)

    def _note_once(self, key: str, message: str, messages: list[str]) -> None:
        if key not in self._reported:
            self._reported.add(key)
            messages.append(message)

    def _reload_applied(self) -> None:
        """Take the applied state from the stored record: another process
        (`nbgpu run`/`sync`, or a loop) may have installed since."""
        try:
            record = self._store.load(self._session.session_id)
        except (OSError, ValueError):
            return
        raw = record.extra.get(EXTRA_ENV) if record is not None else None
        if not raw:
            return
        try:
            applied = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(applied, dict):
            self._applied = applied
            self._session.extra = {**self._session.extra, EXTRA_ENV: raw}

    def _save_applied(self) -> None:
        self._session.extra = {**self._session.extra, EXTRA_ENV: json.dumps(self._applied, sort_keys=True)}
        try:
            if self._store.load(self._session.session_id) is not None:
                self._store.save(self._session)
        except OSError:
            pass

    def tick(self) -> list[str]:
        """Check for changes (cheap: one stat per file unless something
        changed) and advance a running install. Returns messages to show."""
        messages: list[str] = []
        if self._install is not None:
            self._poll_install(messages)
            if self._install is not None:
                return messages  # one install at a time; changes are seen after it
        fingerprint = self._current_fingerprint()
        if fingerprint == self._fingerprint:
            return messages
        if not self._lock.acquire():
            # `nbgpu run` or `sync` is installing; look again next tick.
            return messages
        try:
            self._fingerprint = fingerprint
            self._reload_applied()
            try:
                spec = load_spec(self._session.notebook)
            except SpecError as exc:
                self._note_once(f"invalid:{exc}",
                                f"!header: {exc}; environment changes are applied once it is valid", messages)
                return messages
            self._compare(spec, messages)
        finally:
            if self._install is None:  # a background install keeps the lock until it ends
                self._lock.release()
        return messages

    def apply(self, timeout_s: float = INSTALL_TIMEOUT_S) -> Applied:
        """Bring the runtime in line with the header now, waiting for the
        install: for `nbgpu sync` and `nbgpu run`, which have no loop. Waits
        up to `timeout_s` for an install another process is running."""
        if not self._lock.acquire(timeout_s):
            return Applied(["!another install into this runtime is still running; "
                            "dependencies were not checked this time"], failed=True)
        try:
            self._reload_applied()
            try:
                spec = load_spec(self._session.notebook)
            except SpecError as exc:
                return Applied([f"!header: {exc}; environment changes are applied once it is valid"])
            messages: list[str] = []
            failed = self._compare(spec, messages, blocking_timeout_s=timeout_s)
            return Applied(messages, failed)
        finally:
            self._lock.release()

    def _compare(self, spec: NotebookSpec, messages: list[str], blocking_timeout_s: float | None = None) -> bool:
        """Report what changed and install what was added: in the background,
        or with `blocking_timeout_s` synchronously. Returns True if a
        synchronous install failed."""
        new = snapshot(spec)
        old_deps: list[str] = list(self._applied.get("deps", []))
        new_deps: list[str] = list(new["deps"])  # type: ignore[arg-type]

        old_compute = self._applied.get("compute", {})
        for key, value in dict(new["compute"]).items():  # type: ignore[arg-type]
            # The header's provider is only a default: --backend or
            # $NBGPU_BACKEND may have chosen another, and the record holds that.
            if key == "provider":
                continue
            if key in old_compute and old_compute[key] != value:
                self._note_once(
                    f"compute:{key}:{value}",
                    f"!{key} changed ({old_compute[key]} → {value}); it applies on the next start "
                    f"(nbgpu stop, then nbgpu start)",
                    messages,
                )

        new_names = {requirement_name(d) for d in new_deps}
        removed = [d for d in old_deps if requirement_name(d) not in new_names]
        if removed:
            self._note_once(
                "removed:" + ",".join(sorted(removed)),
                f"·no longer listed: {', '.join(removed)} (stays installed until the next start)",
                messages,
            )
        added = [d for d in new_deps if d not in old_deps]
        if not added:
            if new_deps != old_deps:
                self._applied["deps"] = new_deps
                self._save_applied()
            return False
        if CAP_LIVE_INSTALL not in self._backend.capabilities:
            self._note_once(
                "noinstall:" + ",".join(added),
                f"!dependencies changed (+{', '.join(added)}); the {self._backend.name} backend cannot install "
                "into its runtime: restart the session to apply them",
                messages,
            )
            return False
        if blocking_timeout_s is None:
            self._start_install(new_deps, added, messages)
            return False
        return not self._install_now(new_deps, added, messages, blocking_timeout_s)

    def _install_now(self, deps: list[str], added: list[str], messages: list[str], timeout_s: float) -> bool:
        try:
            result = self._backend.exec(self._handle, [*INSTALL_ARGV, *deps], cwd=self._handle.remote_root,
                                        timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001 - reported like a failed install
            messages.append(f"✗installing {', '.join(added)} failed: {exc}. The runtime's environment is unchanged")
            return False
        if result.exit_code == 0:
            self._applied["deps"] = deps
            self._save_applied()
            messages.append(f"✓installed {', '.join(added)} in the runtime")
            return True
        lines = (result.stderr + "\n" + result.stdout).splitlines()
        tail = " | ".join(line.strip() for line in lines[-_TAIL_LINES:] if line.strip())
        messages.append(
            f"✗installing {', '.join(added)} failed (exit {result.exit_code}): {tail or 'no output'}. "
            "The runtime's environment is unchanged; fix the dependencies and try again"
        )
        return False

    def _start_install(self, deps: list[str], added: list[str], messages: list[str]) -> None:
        proc = self._backend.spawn(self._handle, [*INSTALL_ARGV, *deps], cwd=self._handle.remote_root)
        install = _Install(proc=proc, deps=deps, added=added)

        def drain() -> None:
            try:
                for line in proc.lines():
                    install.tail.append(line)
            except Exception:  # noqa: BLE001 - only the tail is lost
                pass

        install.reader = threading.Thread(target=drain, daemon=True)
        install.reader.start()
        self._install = install
        messages.append(f"↻installing in the runtime: {', '.join(added)}")

    def _poll_install(self, messages: list[str]) -> None:
        install = self._install
        assert install is not None
        code = install.proc.poll()
        if code is None:
            return
        if install.reader is not None:
            install.reader.join(5)
        self._install = None
        self._lock.release()
        if code == 0:
            self._applied["deps"] = install.deps
            self._save_applied()
            messages.append(
                f"✓installed {', '.join(install.added)}. New imports work now; restart the kernel "
                "(marimo's ⋯ menu → Restart kernel) to load upgraded packages"
            )
        else:
            tail = " | ".join(line.strip() for line in install.tail if line.strip())
            messages.append(
                f"✗installing {', '.join(install.added)} failed (exit {code}): {tail or 'no output'}. "
                "The runtime's environment is unchanged; fix the dependencies and save to retry"
            )

    def adopt(self) -> None:
        """Take the notebook's current dependencies as applied: they were
        installed by someone else (marimo's UI), so nothing to install."""
        try:
            self._applied["deps"] = list(load_spec(self._session.notebook).dependencies)
        except SpecError:
            return
        self._save_applied()


def applied_extra(spec: NotebookSpec) -> Mapping[str, str]:
    """The session-record entry for a fresh start."""
    return {EXTRA_ENV: snapshot_json(spec)}
