"""Start, find, run and stop notebook sessions. The CLI is a thin layer over this.

Start sequence:
1. load_spec(notebook, overrides). The backend passed in is used as-is;
   choosing it (flag, env var, or the spec's provider) is the CLI's job.
2. Project root = `project_root` if given, else the nearest ancestor of the
   notebook containing pyproject.toml or .git, searched below the user's
   home directory only (never home itself or `/`), else the notebook's
   directory. The project must not contain the nbgpu state directory (it
   holds tokens) unless under an excluded directory, and must be under
   PUSH_MAX_FILES files and PUSH_MAX_TOTAL_BYTES in total (excluded
   directories and files sync skips as too large are not counted).
3. A provisional record (state "starting", no runtime id, URL or token,
   with the creating pid) is saved BEFORE backend.create, so a runtime whose
   creation is slow (image build, GPU scheduling) or interrupted is never
   taken for an orphan. Then backend.create(RuntimeRequest(spec, tags,
   ports=(MARIMO_PORT,))) with tags {TAG_TOOL: "1", TAG_SESSION: id,
   TAG_NOTEBOOK: notebook_tag(notebook), TAG_OWNER: owner_id(store)}, and the
   record gets the runtime id. Records and runtimes are matched by the
   session tag, not the runtime id.
4. SyncEngine.initial_push(). Every engine lifecycle builds writes the sync
   manifest (MANIFEST_NAME next to the project directory), which the
   watchdog compares with the project before an idle stop.
5. Write the token to token_path(handle) (outside the project directory, so
   it is never synced) with backend.write_file, then spawn MARIMO_ARGV
   (read from the module global at call time) from the remote project root,
   with the notebook path relative to the project root, the host from
   backend.bind_host, and the port from handle.port_for(MARIMO_PORT).
   marimo is started through a tiny wrapper that writes its pid to
   MARIMO_PIDFILE_NAME (next to the token) and execs it.
6. Poll GET <url>/health (no auth needed; 200 when ready) until it answers,
   up to `ready_timeout_s`. If marimo exits or the time runs out, terminate
   the runtime and raise StartError including the last lines of marimo's
   output.
7. Upload nbgpu/watchdog.py next to the token and spawn it (see that module):
   after the spec's idle timeout without activity (and, while edits are not
   yet synced, up to an extra grace) it writes IDLE_MARKER_NAME next to the
   token and stops marimo. The CLI's sync loop then pulls, terminates the
   runtime and removes the record. The watchdog writes a heartbeat
   (HEARTBEAT_NAME) so the loop can restart it if it dies.
8. Save the complete record and return the Session.

If any step after `create` fails, the runtime is terminated before the
error is raised, so a failed start never leaves something billing. If that
terminate itself fails, the provisional record is kept so the runtime stays
visible. Terminates are shielded from Ctrl-C and bounded by
TERMINATE_TIMEOUT_S (TerminateTimeout, record kept), and SIGTERM/SIGHUP are
turned into a KeyboardInterrupt (Terminated) during start, run and stop so
the same cleanup runs.

Orphans: runtimes tagged with this state directory's owner id whose session
tag no record carries. Runtimes of other owners (another laptop, a
teammate, another NBGPU_STATE_DIR) are reported separately and never
terminated.

The time of the last successful sync of each session is kept next to its
record (`<id>.last-sync`), so `stop`/`status` can say which browser edits
may be lost when the runtime is gone.

One-off runs (`run` without a session) save a record of kind "run" carrying
the owner's pid while they run, so their runtime is not mistaken for an
orphan; such records are not sessions (see session_records).

Decisions beyond the contract:
- The session's sync-state path and its hourly rate (computed at start, with
  the overrides then in force) are recorded in `Session.extra`, so `status`
  and `run(session=...)` do not need to re-read the notebook or the store.
- The rate is None when the price table has no prices at all (nothing is
  known), else cost.hourly_rate's Rate (which may be incomplete).
"""

from __future__ import annotations

import collections
import contextlib
import importlib.resources
import json
import os
import posixpath
import re
import secrets
import signal
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from nbgpu import cost, sync, watchdog
from nbgpu.backends.base import Backend, BackendError, RemoteProcess, RuntimeHandle, RuntimeRequest
from nbgpu.cost import Rate
from nbgpu.session import (
    TAG_NOTEBOOK,
    TAG_OWNER,
    TAG_SESSION,
    TAG_TOOL,
    Session,
    SessionStore,
    default_state_dir,
    new_session_id,
    new_token,
    notebook_tag,
)
from nbgpu.ignore import IgnoreRules
from nbgpu.spec import ComputeSpec, load_spec
from nbgpu.sync import SyncEngine, SyncReport

MARIMO_PORT = 2718
MARIMO_ARGV = (
    "python", "-m", "marimo", "edit", "{notebook}",
    "--headless", "--watch", "--skip-update-check", "--no-sandbox",
    "--host", "{host}", "--port", "{port}",
    "--token-password-file", "{token_path}",
)

# Files next to the remote project directory (never synced).
TOKEN_NAME = "nbgpu-token"
IDLE_MARKER_NAME = "nbgpu-idle"
WATCHDOG_NAME = "nbgpu-watchdog.py"
MARIMO_PIDFILE_NAME = "nbgpu-marimo.pid"
MANIFEST_NAME = "nbgpu-synced.json"
UNSYNCED_MARKER_NAME = "nbgpu-idle-unsynced"
HEARTBEAT_NAME = "nbgpu-watchdog.beat"

# A terminate that has not finished after this long is given up on (the
# record is kept), so a network outage cannot hang the CLI.
TERMINATE_TIMEOUT_S = 90.0
OWNER_ID_NAME = "owner-id"

# Upper bounds on what start/run upload, so a mis-detected project root
# (e.g. a home directory) fails fast instead of uploading everything.
PUSH_MAX_FILES = 2_000
PUSH_MAX_TOTAL_BYTES = 500 * 1024 * 1024

# `stop --orphans` leaves untracked runtimes younger than this alone: a
# concurrent start may have created one and not yet saved its record.
ORPHAN_GRACE_S = 120.0

# Session states reported by status().
STATE_RUNNING = "running"
STATE_STARTING = "starting"
STATE_IDLE_STOPPED = "idle-stopped"
STATE_IDLE_UNSYNCED = "idle-unsynced"  # idle, waiting for unsynced edits to be pulled
STATE_NOT_RUNNING = "not-running"

_PROJECT_MARKERS = ("pyproject.toml", ".git")
_START_ERROR_TAIL_LINES = 20
_HEALTH_POLL_INTERVAL_S = 0.25
_HEALTH_REQUEST_TIMEOUT_S = 2.0
_TAIL_READ_TIMEOUT_S = 5.0
_WATCHDOG_STARTUP_CHECK_S = 0.5
_PLACEHOLDER_RE = re.compile(r"\{(notebook|host|port|token_path)\}")
# Writes the process's pid to argv[1], then execs argv[2:] (same pid), so the
# watchdog knows which process is marimo on any backend.
_PIDFILE_EXEC = (
    "import os, sys\n"
    "with open(sys.argv[1], 'w') as f:\n"
    "    f.write(str(os.getpid()))\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n"
)

# Keys in Session.extra.
_EXTRA_SYNC_STATE = "sync_state"
_EXTRA_RATE = "usd_per_hour"
_EXTRA_RATE_COMPLETE = "rate_complete"
_EXTRA_RATE_NOTE = "rate_note"
_EXTRA_STATE = "state"  # STATE_STARTING while a start is in progress
_EXTRA_KIND = "kind"  # "run" for a one-off run's record
_EXTRA_OWNER_PID = "owner_pid"  # the process creating (or, for a run, using) the runtime
_EXTRA_IDLE_TIMEOUT = "idle_timeout_s"
_EXTRA_UNSYNCED_GRACE = "unsynced_grace_s"
_EXTRA_EXCLUDE = "exclude"  # JSON list of [tool.nbgpu] exclude patterns, fixed at start
_KIND_RUN = "run"


class StartError(RuntimeError):
    pass


class StopRefused(RuntimeError):
    """Stop was refused because of open sync conflicts; the message lists them."""


class SessionNotRunning(RuntimeError):
    """The session's runtime no longer exists (stopped, timed out or terminated)."""


class RunRefused(RuntimeError):
    """A run in a session was refused because the notebook has an open sync
    conflict, so the runtime's copy is not the local one."""


class TerminateTimeout(BackendError):
    """backend.terminate did not finish within TERMINATE_TIMEOUT_S."""


class Terminated(KeyboardInterrupt):
    """Raised by the SIGTERM/SIGHUP handler of signals_as_interrupt(), so
    cleanup written for Ctrl-C also runs when the process is told to stop."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received {signal.Signals(signum).name}")
        self.signum = signum


@dataclass(frozen=True)
class SessionStatus:
    session: Session
    running: bool
    uptime_s: float
    rate: Rate | None
    cost_usd: float | None
    conflicts: tuple[str, ...]
    state: str = STATE_RUNNING
    last_sync_at: float | None = None
    # Not running and not a start: browser edits after last_sync_at are lost.
    warning: str | None = None
    unsynced: tuple[str, ...] = ()  # from the watchdog's markers, if any


@dataclass(frozen=True)
class Orphan:
    """A runtime this tool created that no record tracks: still billing.
    (Also used for runtimes of other owners, which are only listed.)"""

    backend: str
    handle: RuntimeHandle
    uptime_s: float

    @property
    def runtime_id(self) -> str:
        return self.handle.runtime_id

    @property
    def owner(self) -> str | None:
        return self.handle.tags.get(TAG_OWNER)


@dataclass
class OrphanStopResult:
    terminated: list[Orphan] = field(default_factory=list)
    skipped_young: list[Orphan] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # runtime_id -> error


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    output: str  # bounded: at most max_output_lines lines
    truncated: bool
    full_output_path: Path | None  # local file holding the full output when truncated


# ---------------------------------------------------------------------------
# signals


def _in_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


@contextlib.contextmanager
def signals_as_interrupt() -> Iterator[None]:
    """While active, SIGTERM and SIGHUP raise Terminated (a KeyboardInterrupt)
    in the main thread, so `except KeyboardInterrupt` / `except BaseException`
    cleanup runs. Previous handlers are restored on exit. No-op outside the
    main thread."""
    if not _in_main_thread():
        yield
        return

    def handler(signum: int, frame: object) -> None:
        raise Terminated(signum)

    previous: dict[int, object] = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(ValueError, OSError):
            previous[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, old in previous.items():
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(sig, old if old is not None else signal.SIG_DFL)  # type: ignore[arg-type]


def _terminate_with_deadline(backend: Backend, handle: RuntimeHandle) -> None:
    outcome: list[BaseException | None] = []

    def target() -> None:
        try:
            backend.terminate(handle)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome.append(exc)
        else:
            outcome.append(None)

    worker = threading.Thread(target=target, name=f"nbgpu-terminate-{handle.runtime_id}", daemon=True)
    worker.start()
    worker.join(TERMINATE_TIMEOUT_S)  # module global, read now
    if not outcome:
        raise TerminateTimeout(
            f"terminating runtime {handle.runtime_id} on {handle.backend} did not finish within "
            f"{TERMINATE_TIMEOUT_S:g}s (network problem?); it may still be billing. Retry `nbgpu stop`, "
            f"run `nbgpu stop --orphans`, or terminate {handle.runtime_id} in the provider's console"
        )
    if outcome[0] is not None:
        raise outcome[0]


def _terminate(backend: Backend, handle: RuntimeHandle) -> None:
    """backend.terminate, shielded from Ctrl-C (and SIGTERM/SIGHUP): a second
    interrupt during cleanup must not leave the runtime billing. Bounded by
    TERMINATE_TIMEOUT_S: on timeout the signals are restored and
    TerminateTimeout raised (callers keep the record)."""
    if not _in_main_thread():
        _terminate_with_deadline(backend, handle)
        return
    previous: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(ValueError, OSError):
            previous[sig] = signal.signal(sig, signal.SIG_IGN)
    try:
        _terminate_with_deadline(backend, handle)
    finally:
        for sig, old in previous.items():
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(sig, old if old is not None else signal.SIG_DFL)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# helpers


def find_project_root(notebook: Path) -> Path:
    """The nearest ancestor of `notebook` holding a project marker, searched
    only below the user's home directory and never `/`; else the notebook's
    directory. A notebook directly in home (or `/`) is refused: pass an
    explicit project root instead."""
    notebook = Path(notebook).resolve()
    stops = {Path(notebook.anchor)}
    with contextlib.suppress(RuntimeError, KeyError, OSError):
        stops.add(Path.home().resolve())
    if notebook.parent in stops:
        raise StartError(
            f"{notebook} is directly in {notebook.parent}; refusing to upload that whole directory. "
            "Move the notebook into a project directory or pass --project-root"
        )
    for directory in notebook.parents:
        if directory in stops:
            break
        if any((directory / marker).exists() for marker in _PROJECT_MARKERS):
            return directory
    return notebook.parent


def _excluded_parts() -> frozenset[str]:
    return frozenset(getattr(sync, "EXCLUDED_PARTS", frozenset({".git", ".venv", "__pycache__", "node_modules"})))


def check_push_size(root: Path, exclude: Sequence[str] = ()) -> tuple[int, int]:
    """Count the files and bytes an initial push of `root` would upload
    (excluded directories and files over sync's per-file limit left out).
    Raises StartError once either exceeds PUSH_MAX_FILES /
    PUSH_MAX_TOTAL_BYTES; returns (files, bytes) otherwise."""
    excluded = _excluded_parts()
    ignore = IgnoreRules(root, exclude)
    per_file_max = getattr(sync, "PUSH_MAX_BYTES", None)
    max_files, max_bytes = PUSH_MAX_FILES, PUSH_MAX_TOTAL_BYTES
    files = total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dir_rel = Path(dirpath).relative_to(root).as_posix()
        dir_rel = "" if dir_rel == "." else dir_rel
        dirnames[:] = [
            d for d in dirnames if d not in excluded and not ignore.is_ignored(f"{dir_rel}/{d}" if dir_rel else d, True)
        ]
        for name in filenames:
            if ignore.is_ignored(f"{dir_rel}/{name}" if dir_rel else name):
                continue
            try:
                st = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or (per_file_max is not None and st.st_size > per_file_max):
                continue
            files += 1
            total += st.st_size
            if files > max_files or total > max_bytes:
                raise StartError(
                    f"the project directory {root} is too large to upload: at least {files} files / "
                    f"{total / 1e6:.0f} MB (limits: {max_files} files, {max_bytes / 1e6:.0f} MB). "
                    "Pass --project-root to choose a smaller directory, list large paths in .gitignore or "
                    "[tool.nbgpu] exclude, or move data out of it "
                    f"(directories named {', '.join(sorted(excluded))} are not uploaded)"
                )
    return files, total


def _check_notebook_not_ignored(rel_notebook: str, root: Path, exclude: Sequence[str]) -> None:
    if IgnoreRules(root, exclude).is_ignored(rel_notebook):
        raise StartError(
            f"{rel_notebook} is ignored by .gitignore or [tool.nbgpu] exclude, so it would not be uploaded; "
            "un-ignore it (e.g. add `!" + rel_notebook + "` to exclude) or pass --project-root"
        )


def _check_state_dir_outside(root: Path, state_dir: Path) -> None:
    """The state directory holds tokens and copies of files: never upload it."""
    try:
        rel = Path(state_dir).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return
    if any(part in _excluded_parts() for part in rel.parts):
        return  # sync never looks inside it
    raise StartError(
        f"the nbgpu state directory {state_dir} is inside the project directory {root}, so its tokens "
        "would be uploaded; set NBGPU_STATE_DIR to a directory outside the project or pass --project-root"
    )


def runtime_path(handle: RuntimeHandle, name: str) -> str:
    """`name` next to the remote project directory: per runtime, never synced."""
    return posixpath.join(posixpath.dirname(handle.remote_root), name)


def token_path(handle: RuntimeHandle) -> str:
    """`nbgpu-token` next to the remote project directory:
    posixpath.join(posixpath.dirname(handle.remote_root), "nbgpu-token").
    Per runtime, and outside what is synced."""
    return runtime_path(handle, TOKEN_NAME)


def idle_marker_path(handle: RuntimeHandle) -> str:
    return runtime_path(handle, IDLE_MARKER_NAME)


def manifest_path(handle: RuntimeHandle) -> str:
    """Where sync engines record what the laptop has agreed with."""
    return runtime_path(handle, MANIFEST_NAME)


def _read_json_marker(backend: Backend, handle: RuntimeHandle, path: str) -> dict[str, object] | None:
    data = backend.read_file(handle, path)
    if data is None:
        return None
    try:
        info = json.loads(data)
    except ValueError:
        return {}
    return info if isinstance(info, dict) else {}


def read_idle_marker(backend: Backend, handle: RuntimeHandle) -> dict[str, object] | None:
    """The watchdog's marker if it stopped marimo for idleness, else None.
    With unsynced edits at that time, "unsynced" lists their paths."""
    return _read_json_marker(backend, handle, idle_marker_path(handle))


def read_unsynced_marker(backend: Backend, handle: RuntimeHandle) -> dict[str, object] | None:
    """The watchdog's marker while it is idle but waiting for unsynced edits
    ("unsynced": paths, "stop_at": unix time), else None."""
    return _read_json_marker(backend, handle, runtime_path(handle, UNSYNCED_MARKER_NAME))


def read_heartbeat(backend: Backend, handle: RuntimeHandle) -> bytes | None:
    """The watchdog's heartbeat file (rewritten every check; removed when it
    exits), or None."""
    return backend.read_file(handle, runtime_path(handle, HEARTBEAT_NAME))


def marker_paths(info: Mapping[str, object] | None) -> list[str]:
    raw = (info or {}).get("unsynced")
    return [str(p) for p in raw] if isinstance(raw, list) else []


def owner_id(store: SessionStore) -> str:
    """A random id for this state directory, created once (file 0600) and
    put on every runtime as TAG_OWNER."""
    path = store.directory / OWNER_ID_NAME
    with contextlib.suppress(OSError):
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    _ensure_private_dir(store.directory)
    value = secrets.token_hex(8)
    fd, tmp = tempfile.mkstemp(dir=store.directory, prefix=".owner-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(value + "\n")
        try:
            os.link(tmp, path)  # fails if a concurrent process won
        except FileExistsError:
            pass
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
    return path.read_text(encoding="utf-8").strip()


def _last_sync_path(session_id: str, store: SessionStore) -> Path:
    return store.directory / f"{session_id}.last-sync"


def note_synced(session: Session, store: SessionStore, when: float | None = None) -> None:
    """Record that the session synced successfully (now, or `when`). Only
    while its record exists, so a concurrent stop is not undone."""
    if store.load(session.session_id) is None:
        return
    path = _last_sync_path(session.session_id, store)
    with contextlib.suppress(OSError):
        fd, tmp = tempfile.mkstemp(dir=store.directory, prefix=".sync-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(repr(time.time() if when is None else when))
        os.replace(tmp, path)


def last_sync_at(session: Session, store: SessionStore) -> float | None:
    with contextlib.suppress(OSError, ValueError):
        return float(_last_sync_path(session.session_id, store).read_text(encoding="utf-8"))
    return None


def format_time(ts: float | None) -> str:
    if ts is None:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def lost_edits_warning(session: Session, store: SessionStore) -> str:
    when = last_sync_at(session, store)
    since = f"after the last successful sync ({format_time(when)})" if when else "since the session started"
    return (
        f"the runtime of session {session.session_id} ({session.runtime_id or 'never created'}) is gone, so edits "
        f"made in the browser {since} could not be recovered; your local files are as of that sync"
    )


def session_idle_timeout_s(session: Session) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(session.extra[_EXTRA_IDLE_TIMEOUT])
    return None


def session_unsynced_grace_s(session: Session) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(session.extra[_EXTRA_UNSYNCED_GRACE])
    return session_idle_timeout_s(session)


def unsynced_grace_for(idle_timeout_s: float) -> float:
    grace = watchdog.UNSYNCED_GRACE_S  # module global, read now
    return float(idle_timeout_s if grace is None else grace)


def sync_state_path(session_id: str, store: SessionStore) -> Path:
    return store.directory / f"{session_id}.sync.json"


def rate_for(compute: ComputeSpec) -> Rate | None:
    """The hourly rate, or None when the price table knows no prices at all."""
    prices = cost.DEFAULT_PRICES
    if not (prices.gpu_hourly or prices.cpu_core_hourly or prices.memory_gib_hourly):
        return None
    return cost.hourly_rate(compute, prices)


def session_rate(session: Session) -> Rate | None:
    """The rate recorded at start, or None if none was recorded."""
    raw = session.extra.get(_EXTRA_RATE)
    if raw is None:
        return None
    try:
        usd = float(raw)
    except ValueError:
        return None
    return Rate(
        usd_per_hour=usd,
        complete=session.extra.get(_EXTRA_RATE_COMPLETE) == "1",
        note=session.extra.get(_EXTRA_RATE_NOTE, ""),
    )


def is_starting(session: Session) -> bool:
    return session.extra.get(_EXTRA_STATE) == STATE_STARTING


def is_run_record(session: Session) -> bool:
    return session.extra.get(_EXTRA_KIND) == _KIND_RUN


def session_records(store: SessionStore, notebook: Path | None = None) -> list[Session]:
    """Session records (one-off run records left out), oldest first; with
    `notebook`, only that notebook's, newest first."""
    records = store.list() if notebook is None else store.find_by_notebook(notebook)
    return [s for s in records if not is_run_record(s)]


def _rate_extra(rate: Rate | None) -> dict[str, str]:
    if rate is None:
        return {}
    return {
        _EXTRA_RATE: repr(rate.usd_per_hour),
        _EXTRA_RATE_COMPLETE: "1" if rate.complete else "0",
        _EXTRA_RATE_NOTE: rate.note,
    }


def _ensure_private_dir(directory: Path) -> None:
    """Make sure `directory` exists and only the user can use it: created
    0700, and an existing one we own has group/other permissions removed."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        st = directory.stat()
        if st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode) & 0o077:
            directory.chmod(stat.S_IMODE(st.st_mode) & 0o700)
        return
    directory.chmod(0o700)  # mkdir's mode is filtered by the umask


def _private_file(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.chmod(0o600)


def _relative_notebook(notebook: Path, project_root: Path) -> str:
    try:
        return notebook.relative_to(project_root).as_posix()
    except ValueError:
        raise StartError(f"notebook {notebook} is not inside the project directory {project_root}") from None


def _marimo_argv(*, notebook: str, host: str, port: int, token_file: str) -> list[str]:
    # One pass of re.sub per argument: values are never re-scanned, so a
    # notebook path containing "{host}" is left as it is, and other braces
    # (e.g. code passed to `python -c`) are untouched.
    values = {"notebook": notebook, "host": host, "port": str(port), "token_path": token_file}
    return [_PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], arg) for arg in MARIMO_ARGV]  # global, read now


def _with_pidfile(pidfile: str, argv: list[str]) -> list[str]:
    return ["python", "-c", _PIDFILE_EXEC, pidfile, *argv]


def _health_ok(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=_HEALTH_REQUEST_TIMEOUT_S) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _last_lines(proc: RemoteProcess, count: int) -> list[str]:
    """The last `count` output lines of a process that has been told to stop.
    Read in a thread with a deadline, since a stream may not end promptly."""
    tail: collections.deque[str] = collections.deque(maxlen=count)

    def read() -> None:
        with contextlib.suppress(Exception):
            for line in proc.lines():
                tail.append(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(_TAIL_READ_TIMEOUT_S)
    return list(tail)


def _wait_ready(proc: RemoteProcess, base_url: str, timeout_s: float) -> None:
    """Return once GET <base_url>/health answers 200; raise StartError if the
    process exits or the time runs out (the process is stopped first)."""
    deadline = time.monotonic() + timeout_s
    while True:
        exit_code = proc.poll()
        if exit_code is not None:
            problem = f"marimo exited with code {exit_code} before it was ready"
            break
        if _health_ok(base_url):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            problem = f"marimo was not ready after {timeout_s:g}s"
            break
        time.sleep(min(_HEALTH_POLL_INTERVAL_S, remaining))
    proc.terminate()
    lines = _last_lines(proc, _START_ERROR_TAIL_LINES)
    output = "\n".join(lines) if lines else "(no output)"
    raise StartError(f"{problem}; last output:\n{output}")


def _start_watchdog(
    backend: Backend, handle: RuntimeHandle, idle_timeout_s: float, unsynced_grace_s: float
) -> RemoteProcess:
    script = importlib.resources.files("nbgpu").joinpath("watchdog.py").read_bytes()
    path = runtime_path(handle, WATCHDOG_NAME)
    backend.write_file(handle, path, script)
    argv = [
        "python", path,
        "--project", handle.remote_root,
        "--marker", idle_marker_path(handle),
        "--marimo-pidfile", runtime_path(handle, MARIMO_PIDFILE_NAME),
        "--idle-timeout", repr(float(idle_timeout_s)),
        "--unsynced-grace", repr(float(unsynced_grace_s)),
        "--manifest", manifest_path(handle),
        "--unsynced-marker", runtime_path(handle, UNSYNCED_MARKER_NAME),
        "--heartbeat", runtime_path(handle, HEARTBEAT_NAME),
        # Asked from inside the runtime whether a cell is running.
        "--marimo-url", f"http://127.0.0.1:{handle.port_for(MARIMO_PORT)}",
        "--token-file", token_path(handle),
        # Module globals read now, so tests can shrink them.
        "--interval", repr(float(watchdog.CHECK_INTERVAL_S)),
        "--cpu-threshold", repr(float(watchdog.CPU_BUSY_FRACTION)),
        "--gpu-threshold", repr(float(watchdog.GPU_BUSY_PERCENT)),
        "--kill-grace", repr(float(watchdog.KILL_GRACE_S)),
        # The local backend shares the machine with everything else the user
        # runs; only marimo's process tree is the session's there.
        "--scope", "tree" if backend.name == "local" else "all",
    ]
    proc = backend.spawn(handle, argv, cwd=posixpath.dirname(handle.remote_root))
    time.sleep(_WATCHDOG_STARTUP_CHECK_S)
    code = proc.poll()
    if code is not None:
        lines = _last_lines(proc, _START_ERROR_TAIL_LINES)
        raise StartError(f"the idle watchdog exited with code {code}; last output:\n" + ("\n".join(lines) or "(no output)"))
    return proc


def restart_watchdog(session: Session, backend: Backend, handle: RuntimeHandle) -> None:
    """Start the session's idle watchdog again (after it died). Raises
    StartError if it exits right away (e.g. marimo is no longer running)."""
    idle = session_idle_timeout_s(session)
    if idle is None:
        idle = float(load_spec(session.notebook).compute.idle_timeout_s)
    grace = session_unsynced_grace_s(session)
    _start_watchdog(backend, handle, idle, unsynced_grace_for(idle) if grace is None else grace)


def resolve_handle(session: Session, backend: Backend) -> RuntimeHandle | None:
    """The session's runtime: by runtime id, or (for a start that has not
    recorded one yet) by its session tag."""
    if session.runtime_id:
        return backend.get(session.runtime_id)
    found = backend.find({TAG_SESSION: session.session_id})
    return found[0] if found else None


def _require_handle(session: Session, backend: Backend) -> RuntimeHandle:
    handle = backend.get(session.runtime_id) if session.runtime_id else None
    if handle is None:
        raise SessionNotRunning(
            f"the runtime of session {session.session_id} ({session.runtime_id}) is not running"
        )
    return handle


def _state_path(session: Session, store: SessionStore | None) -> Path:
    if store is not None:
        return sync_state_path(session.session_id, store)
    recorded = session.extra.get(_EXTRA_SYNC_STATE)
    return Path(recorded) if recorded else sync_state_path(session.session_id, SessionStore())


def _new_engine(
    root: Path, backend: Backend, handle: RuntimeHandle, state_path: Path, exclude: Sequence[str] = ()
) -> SyncEngine:
    return SyncEngine(root, backend, handle, state_path, manifest_path=manifest_path(handle), exclude=exclude)


def session_exclude(session: Session) -> tuple[str, ...]:
    """The exclude patterns recorded at start (() for older records)."""
    raw = session.extra.get(_EXTRA_EXCLUDE)
    if not raw:
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    return tuple(p for p in value if isinstance(p, str)) if isinstance(value, list) else ()


def _engine(session: Session, backend: Backend, handle: RuntimeHandle, state_path: Path) -> SyncEngine:
    return _new_engine(session.project_root, backend, handle, state_path, session_exclude(session))


def session_engine(session: Session, *, backend: Backend, store: SessionStore) -> SyncEngine:
    """A SyncEngine for the session, to be kept and reused across polls.
    Raises SessionNotRunning if the runtime is gone."""
    handle = _require_handle(session, backend)
    state_path = sync_state_path(session.session_id, store)
    _private_file(state_path)
    return _engine(session, backend, handle, state_path)


# ---------------------------------------------------------------------------
# start


def start(
    notebook: Path,
    *,
    backend: Backend,
    store: SessionStore,
    overrides: Mapping[str, object] | None = None,
    project_root: Path | None = None,
    ready_timeout_s: float = 300.0,
    progress: Callable[[str], None] | None = None,
) -> Session:
    """`progress`, if given, is called with a short description as each
    startup step begins ("creating the runtime", "uploading 44 files", ...)."""
    with signals_as_interrupt():
        return _start(notebook, backend, store, overrides, project_root, ready_timeout_s, progress)


def _start(
    notebook: Path,
    backend: Backend,
    store: SessionStore,
    overrides: Mapping[str, object] | None,
    project_root: Path | None,
    ready_timeout_s: float,
    progress: Callable[[str], None] | None = None,
) -> Session:
    step = progress or (lambda _message: None)
    spec = load_spec(notebook, overrides)
    notebook = spec.path
    root = Path(project_root).resolve() if project_root is not None else find_project_root(notebook)
    rel_notebook = _relative_notebook(notebook, root)
    _ensure_private_dir(store.directory)
    _check_state_dir_outside(root, store.directory)
    _check_notebook_not_ignored(rel_notebook, root, spec.exclude)
    files, size = check_push_size(root, spec.exclude)

    session_id = new_session_id()
    tags = {
        TAG_TOOL: "1",
        TAG_SESSION: session_id,
        TAG_NOTEBOOK: notebook_tag(notebook),
        TAG_OWNER: owner_id(store),
    }
    state_path = sync_state_path(session_id, store)
    idle_timeout_s = float(spec.compute.idle_timeout_s)
    grace_s = unsynced_grace_for(idle_timeout_s)
    extra = {
        _EXTRA_SYNC_STATE: str(state_path),
        _EXTRA_IDLE_TIMEOUT: repr(idle_timeout_s),
        _EXTRA_UNSYNCED_GRACE: repr(grace_s),
        **({_EXTRA_EXCLUDE: json.dumps(list(spec.exclude))} if spec.exclude else {}),
        **_rate_extra(rate_for(spec.compute)),
    }
    session = Session(
        session_id=session_id,
        notebook=notebook,
        project_root=root,
        backend=backend.name,
        runtime_id="",
        remote_root="",
        url="",
        token="",
        created_at=time.time(),
        gpu=spec.compute.gpu,
        extra={**extra, _EXTRA_STATE: STATE_STARTING, _EXTRA_OWNER_PID: str(os.getpid())},
    )
    # Provisional, before create: the runtime is never untracked, however
    # long creating it takes, and a crash leaves a visible record.
    store.save(session)

    step(f"creating the runtime ({_shape(spec.compute)})")
    try:
        handle = backend.create(RuntimeRequest(spec=spec, tags=tags, ports=(MARIMO_PORT,)))
    except BaseException as exc:
        _abandon_create(session, backend, store, exc)
        raise
    try:
        session.runtime_id = handle.runtime_id
        session.remote_root = handle.remote_root
        session.created_at = handle.created_at
        store.save(session)

        step(f"uploading {files} file{'s' if files != 1 else ''} ({_size_text(size)}) from {root.name}/")
        _new_engine(root, backend, handle, state_path, spec.exclude).initial_push()
        _private_file(state_path)
        note_synced(session, store)

        token = new_token()
        token_file = token_path(handle)
        backend.write_file(handle, token_file, token.encode("utf-8"))
        pidfile = runtime_path(handle, MARIMO_PIDFILE_NAME)
        argv = _marimo_argv(
            notebook=rel_notebook,
            host=backend.bind_host,
            port=handle.port_for(MARIMO_PORT),
            token_file=token_file,
        )
        step("starting marimo")
        proc = backend.spawn(handle, _with_pidfile(pidfile, argv), cwd=handle.remote_root)
        url = backend.expose(handle, MARIMO_PORT).rstrip("/")
        _wait_ready(proc, url, ready_timeout_s)
        _start_watchdog(backend, handle, idle_timeout_s, grace_s)

        session.url = url
        session.token = token
        session.extra = extra
        store.save(session)
        return session
    except BaseException as exc:
        try:
            _terminate(backend, handle)
        except Exception as term_exc:  # noqa: BLE001
            # Keep the provisional record so status/stop still see the runtime.
            raise StartError(
                f"{str(exc) or type(exc).__name__}; in addition the runtime {handle.runtime_id} could not be "
                f"terminated ({term_exc}); it may still be billing: see `nbgpu status`, then `nbgpu stop`"
            ) from exc
        _forget(session, store)
        raise


def _shape(compute: ComputeSpec) -> str:
    parts = [f"gpu {compute.gpu}" if compute.gpu else "no gpu"]
    if compute.cpu is not None:
        parts.append(f"{compute.cpu:g} cpu")
    if compute.memory_mib is not None:
        parts.append(f"{compute.memory_mib / 1024:g} GiB")
    return ", ".join(parts)


def _size_text(size: int) -> str:
    for unit, scale in (("MB", 1e6), ("kB", 1e3)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} B"


def _abandon_create(session: Session, backend: Backend, store: SessionStore, exc: BaseException) -> None:
    """backend.create failed or was interrupted: a runtime may exist anyway
    (e.g. Ctrl-C while the provider was creating it). Terminate whatever
    carries the session tag; keep the provisional record if that fails."""
    try:
        with signals_as_interrupt():
            for handle in backend.find({TAG_SESSION: session.session_id}):
                _terminate(backend, handle)
    except BaseException:  # noqa: BLE001 - keep the record: the runtime may still exist
        return
    _forget(session, store)


# ---------------------------------------------------------------------------
# sync, status, stop


def sync_once(session: Session, *, backend: Backend, store: SessionStore) -> SyncReport:
    """One SyncEngine.poll_once for the session, with the engine's state kept
    at store.directory / f"{session_id}.sync.json". For repeated syncing,
    keep one engine from session_engine() instead."""
    report = session_engine(session, backend=backend, store=store).poll_once()
    _private_file(sync_state_path(session.session_id, store))
    note_synced(session, store)
    return report


def _owner_alive(session: Session) -> bool:
    try:
        pid = int(session.extra.get(_EXTRA_OWNER_PID, ""))
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status(*, backends: Mapping[str, Backend], store: SessionStore) -> list[SessionStatus]:
    """One entry per stored session. `running` comes from the backend.
    Records whose runtime is gone are reported as not running, not deleted,
    with a `warning` that browser edits after `last_sync_at` are lost.
    `conflicts` are the open conflicts in the session's saved sync state
    (store.directory / f"{session_id}.sync.json"); no poll is done.
    `state` is STATE_STARTING for a provisional record (also while its
    runtime is still being created), STATE_IDLE_STOPPED when the watchdog
    stopped marimo, STATE_IDLE_UNSYNCED while it waits for unsynced edits to
    be pulled, else running / not-running.

    A runtime that is not running has uptime 0 and an unknown cost (when it
    stopped is not known). Cost is None when the rate is unknown or incomplete."""
    now = time.time()
    entries: list[SessionStatus] = []
    for session in session_records(store):
        backend = backends.get(session.backend)
        if backend is None:
            raise BackendError(f"no backend {session.backend!r} given for session {session.session_id}")
        handle = resolve_handle(session, backend)
        running = handle is not None
        uptime_s = max(0.0, now - session.created_at) if running else 0.0
        rate = session_rate(session)
        cost_usd = cost.cost_so_far(rate, uptime_s) if running and rate is not None and rate.complete else None
        warning: str | None = None
        unsynced: list[str] = []
        if is_starting(session):
            state = STATE_STARTING if running or _owner_alive(session) else STATE_NOT_RUNNING
        elif not running:
            state = STATE_NOT_RUNNING
            warning = lost_edits_warning(session, store)
        else:
            state = STATE_RUNNING
            with contextlib.suppress(Exception):
                if handle is not None:
                    idle = read_idle_marker(backend, handle)
                    if idle is not None:
                        state, unsynced = STATE_IDLE_STOPPED, marker_paths(idle)
                    else:
                        waiting = read_unsynced_marker(backend, handle)
                        if waiting is not None:
                            state, unsynced = STATE_IDLE_UNSYNCED, marker_paths(waiting)
        # Loading the engine only reads its state file; no backend call is made.
        offline = handle or RuntimeHandle(
            backend=session.backend,
            runtime_id=session.runtime_id,
            remote_root=session.remote_root,
            created_at=session.created_at,
        )
        conflicts = _engine(session, backend, offline, sync_state_path(session.session_id, store)).conflicts
        entries.append(
            SessionStatus(
                session=session,
                running=running,
                uptime_s=uptime_s,
                rate=rate,
                cost_usd=cost_usd,
                conflicts=tuple(conflicts),
                state=state,
                last_sync_at=last_sync_at(session, store),
                warning=warning,
                unsynced=tuple(unsynced),
            )
        )
    return entries


def find_orphans(
    *,
    backends: Mapping[str, Backend],
    store: SessionStore,
    errors: dict[str, str] | None = None,
    others: list[Orphan] | None = None,
) -> list[Orphan]:
    """Runtimes tagged TAG_TOOL="1" and this state directory's owner id
    (owner_id) on the given backends whose session tag no record carries (a
    one-off run's record counts only while its owner process is alive).
    Runtimes of other owners (or none) are never orphans: they are appended
    to `others` when given. A backend whose `find` fails is skipped and its
    error put in `errors` (raised if `errors` is None)."""
    me = owner_id(store)
    tracked = {
        (s.backend, s.session_id)
        for s in store.list()
        if not is_run_record(s) or _owner_alive(s)
    }
    now = time.time()
    orphans: list[Orphan] = []
    for name, backend in backends.items():
        try:
            handles = backend.find({TAG_TOOL: "1"})
        except Exception as exc:
            if errors is None:
                raise
            errors[name] = str(exc) or type(exc).__name__
            continue
        for handle in handles:
            entry = Orphan(name, handle, max(0.0, now - handle.created_at))
            if handle.tags.get(TAG_OWNER) != me:
                if others is not None:
                    others.append(entry)
            elif (name, handle.tags.get(TAG_SESSION, "")) not in tracked:
                orphans.append(entry)
    return orphans


def stop_orphans(
    orphans: list[Orphan],
    *,
    backends: Mapping[str, Backend],
    store: SessionStore,
    min_age_s: float | None = None,
) -> OrphanStopResult:
    """Terminate the given orphans, except ones younger than `min_age_s`
    (default ORPHAN_GRACE_S), which a concurrent start may still own. Stale
    one-off run records for terminated runtimes are removed."""
    grace = ORPHAN_GRACE_S if min_age_s is None else min_age_s
    me = owner_id(store)
    result = OrphanStopResult()
    with signals_as_interrupt():
        for orphan in orphans:
            if orphan.owner != me:  # never another owner's runtime
                result.failed[orphan.runtime_id] = "not created from this state directory; left alone"
                continue
            if orphan.uptime_s < grace:
                result.skipped_young.append(orphan)
                continue
            try:
                _terminate(backends[orphan.backend], orphan.handle)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                result.failed[orphan.runtime_id] = str(exc) or type(exc).__name__
                continue
            result.terminated.append(orphan)
            tag = orphan.handle.tags.get(TAG_SESSION)
            for record in store.list():
                if record.backend == orphan.backend and (
                    record.session_id == tag or (record.runtime_id and record.runtime_id == orphan.runtime_id)
                ):
                    store.remove(record.session_id)
    return result


def _forget(session: Session, store: SessionStore) -> None:
    store.remove(session.session_id)
    with contextlib.suppress(OSError):
        sync_state_path(session.session_id, store).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        _last_sync_path(session.session_id, store).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        state = sync_state_path(session.session_id, store)
        state.with_name(state.name + ".lock").unlink(missing_ok=True)


def _with_warnings(report: SyncReport, *warnings: str) -> SyncReport:
    existing = list(getattr(report, "warnings", None) or [])
    report.warnings = existing + [w for w in warnings if w not in existing]  # type: ignore[attr-defined]
    return report


def stop(
    session: Session,
    *,
    backend: Backend,
    store: SessionStore,
    force: bool = False,
    ignore_conflicts: bool = False,
    terminate_on_error: bool = False,
    engine: SyncEngine | None = None,
) -> SyncReport:
    """Final pull, then terminate and remove the record (and its sync state).
    With open conflicts, raise StopRefused unless ignore_conflicts (then:
    still pull what is clean, keep conflict copies, then terminate; the
    returned report lists the open conflicts). If the final pull itself
    fails, the error is raised and the runtime and record are kept, unless
    terminate_on_error (only for an explicit user --force), in which case
    the runtime is terminated anyway and the error raised afterwards.
    force=True means both. `engine` reuses the caller's engine.

    If the runtime is already gone there is nothing to pull: the record is
    removed and the report's `warnings` say that browser edits after the
    last successful sync were lost. A session whose start never finished is
    terminated without a pull (nothing was edited on it)."""
    ignore_conflicts = ignore_conflicts or force
    terminate_on_error = terminate_on_error or force
    with signals_as_interrupt():
        handle = resolve_handle(session, backend)
        if handle is None:
            warning = None if is_starting(session) else lost_edits_warning(session, store)
            _forget(session, store)
            return _with_warnings(SyncReport(), *([warning] if warning else []))
        if is_starting(session):
            _terminate(backend, handle)
            _forget(session, store)
            return _with_warnings(SyncReport())

        if engine is None:
            engine = _engine(session, backend, handle, sync_state_path(session.session_id, store))
        try:
            report = engine.final_pull()
        except Exception as exc:
            if not terminate_on_error:
                raise
            _terminate(backend, handle)
            warning = lost_edits_warning(session, store)
            _forget(session, store)
            raise BackendError(
                f"final pull failed ({exc}); the runtime was terminated anyway (--force): {warning}"
            ) from exc
        note_synced(session, store)

        if report.conflicts and not ignore_conflicts:
            listed = ", ".join(report.conflicts)
            raise StopRefused(
                f"open sync conflicts: {listed}. Resolve them (delete the .remote copies) "
                "or stop with --force to keep the local versions and the conflict copies"
            )
        _terminate(backend, handle)
        _forget(session, store)
        return _with_warnings(report)


# ---------------------------------------------------------------------------
# run


def _bound_output(lines: list[str], max_lines: int, full_output_path: Path) -> str:
    """First and last halves of `lines` with a marker line between, in at
    most `max_lines` lines in total (the marker included)."""
    keep = max_lines - 1
    head_n = (keep + 1) // 2
    tail_n = keep - head_n
    omitted = len(lines) - keep
    marker = f"... {omitted} lines omitted; full output in {full_output_path} ..."
    head = lines[:head_n]
    tail = lines[len(lines) - tail_n:] if tail_n else []
    return "\n".join([*head, marker, *tail])


def _run_in(
    backend: Backend, handle: RuntimeHandle, rel_notebook: str, max_output_lines: int, output_dir: Path
) -> RunResult:
    result = backend.exec(handle, ["python", rel_notebook], cwd=handle.remote_root)
    combined = result.stdout
    if combined and result.stderr and not combined.endswith("\n"):
        combined += "\n"
    combined += result.stderr
    lines = combined.splitlines()
    if len(lines) <= max_output_lines:
        return RunResult(result.exit_code, "\n".join(lines), False, None)

    if not output_dir.exists():
        _ensure_private_dir(output_dir)
    stem = posixpath.splitext(posixpath.basename(rel_notebook))[0]
    fd, name = tempfile.mkstemp(prefix=f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}-", suffix=".log", dir=output_dir)
    full_path = Path(name)
    with open(fd, "w", encoding="utf-8") as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(combined)
    return RunResult(result.exit_code, _bound_output(lines, max_output_lines, full_path), True, full_path)


def run(
    notebook: Path,
    *,
    backend: Backend,
    session: Session | None = None,
    overrides: Mapping[str, object] | None = None,
    max_output_lines: int = 200,
    output_dir: Path | None = None,
    project_root: Path | None = None,
    store: SessionStore | None = None,
) -> RunResult:
    """Run the whole notebook as a script (`python <notebook>`) on the runtime
    and return its exit code and output. With `session`, sync first and run
    in that runtime; without, create a runtime, push, run, and terminate it.
    Output over max_output_lines keeps the first and last halves with a
    marker line between (the marker counts towards the limit), and writes
    the full output to output_dir.

    Output is stdout followed by stderr. output_dir defaults to
    default_state_dir() / "runs". In a session, a run is refused (RunRefused)
    if the notebook has an open sync conflict after syncing. A one-off run
    keeps a "run" record in `store` (default SessionStore()) while it runs."""
    if max_output_lines < 1:
        raise ValueError(f"max_output_lines must be at least 1, got {max_output_lines}")
    out_dir = Path(output_dir) if output_dir is not None else default_state_dir() / "runs"

    with signals_as_interrupt():
        if session is not None:
            notebook = Path(notebook).resolve()
            rel_notebook = _relative_notebook(notebook, session.project_root)
            handle = _require_handle(session, backend)
            report = _engine(session, backend, handle, _state_path(session, store)).poll_once()
            if store is not None:
                note_synced(session, store)
            if rel_notebook in report.conflicts:
                raise RunRefused(
                    f"{rel_notebook} has an open sync conflict, so the runtime's copy differs from yours; "
                    "resolve it (delete the .remote copy) and run again"
                )
            return _run_in(backend, handle, rel_notebook, max_output_lines, out_dir)

        store = store if store is not None else SessionStore()
        spec = load_spec(notebook, overrides)
        root = Path(project_root).resolve() if project_root is not None else find_project_root(spec.path)
        rel_notebook = _relative_notebook(spec.path, root)
        _check_state_dir_outside(root, store.directory)
        _check_notebook_not_ignored(rel_notebook, root, spec.exclude)
        check_push_size(root, spec.exclude)
        run_id = new_session_id()
        tags = {
            TAG_TOOL: "1",
            TAG_SESSION: run_id,
            TAG_NOTEBOOK: notebook_tag(spec.path),
            TAG_OWNER: owner_id(store),
        }
        record = Session(
            session_id=run_id,
            notebook=spec.path,
            project_root=root,
            backend=backend.name,
            runtime_id="",
            remote_root="",
            url="",
            token="",
            created_at=time.time(),
            gpu=spec.compute.gpu,
            extra={_EXTRA_KIND: _KIND_RUN, _EXTRA_OWNER_PID: str(os.getpid())},
        )
        store.save(record)  # before create: the runtime is tracked while we live
        try:
            handle = backend.create(RuntimeRequest(spec=spec, tags=tags))
        except BaseException as exc:
            _abandon_create(record, backend, store, exc)
            raise
        try:
            record.runtime_id, record.remote_root = handle.runtime_id, handle.remote_root
            with contextlib.suppress(Exception):
                store.save(record)
            with tempfile.TemporaryDirectory(prefix="nbgpu-run-") as tmp:
                _new_engine(root, backend, handle, Path(tmp) / "sync.json", spec.exclude).initial_push()
            return _run_in(backend, handle, rel_notebook, max_output_lines, out_dir)
        finally:
            _terminate(backend, handle)
            with contextlib.suppress(Exception):
                store.remove(run_id)
