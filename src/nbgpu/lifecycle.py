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
import datetime
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import posixpath
import re
import secrets
import shlex
import signal
import stat
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

from nbgpu import cost, sync, watchdog
from nbgpu.backends.base import (
    CAP_BILLING,
    CAP_LIVE_INSTALL,
    CAP_SNAPSHOT,
    CAP_VOLUMES,
    CACHE_MOUNT,
    Backend,
    BackendError,
    RemoteProcess,
    RuntimeHandle,
    RuntimeRequest,
    UnsupportedError,
)
from nbgpu.cost import Rate
from nbgpu.session import (
    TAG_NOTEBOOK,
    TAG_OWNER,
    TAG_SESSION,
    TAG_TOOL,
    Session,
    SessionStore,
    default_state_dir,
    ensure_private_dir,
    new_session_id,
    new_token,
    notebook_tag,
)
from nbgpu import envsync
from nbgpu.envsync import applied_extra
from nbgpu.ignore import IgnoreRules
from nbgpu.spec import ComputeSpec, NotebookSpec, SpecError, load_spec
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
# Touched by `exec` code running in the kernel; the watchdog counts its mtime as activity.
ACTIVITY_NAME = "nbgpu-activity"

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
_EXTRA_PRICES_CHECKED = "prices_checked"
_EXTRA_PRICES_SOURCE = "prices_source"
_EXTRA_TIMINGS = "timings"
_EXTRA_FROM_SNAPSHOT = "from_snapshot"
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
    pulled: list[str] = field(default_factory=list)  # files brought back after the run
    warnings: list[str] = field(default_factory=list)  # files not brought back, conflicts, sync errors
    env: list[str] = field(default_factory=list)  # EnvWatcher messages (role character first)


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


def check_push_size(root: Path, exclude: Sequence[str] = ()) -> tuple[int, int]:
    """Count the files and bytes an initial push of `root` would upload
    (excluded directories and files over sync's per-file limit left out).
    Raises StartError once either exceeds PUSH_MAX_FILES /
    PUSH_MAX_TOTAL_BYTES; returns (files, bytes) otherwise."""
    excluded = sync.EXCLUDED_PARTS
    ignore = IgnoreRules(root, exclude)
    per_file_max = sync.PUSH_MAX_BYTES  # module global, read now
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
            if not stat.S_ISREG(st.st_mode) or st.st_size > per_file_max:
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
    if any(part in sync.EXCLUDED_PARTS for part in rel.parts):
        return  # sync never looks inside it
    raise StartError(
        f"the nbgpu state directory {state_dir} is inside the project directory {root}, so its tokens "
        "would be uploaded; set NBGPU_STATE_DIR to a directory outside the project or pass --project-root"
    )


def runtime_path(handle: RuntimeHandle, name: str) -> str:
    """`name` next to the remote project directory: per runtime, never synced."""
    return posixpath.join(posixpath.dirname(handle.remote_root), name)


def token_path(handle: RuntimeHandle) -> str:
    """marimo's access token file: per runtime, outside what is synced."""
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
    ensure_private_dir(store.directory)
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


def rate_for(compute: ComputeSpec, state_dir: Path | None = None) -> Rate | None:
    """The hourly rate, or None when the price table knows no prices at all.
    `state_dir`: where cached live rates are (see cost.current_prices)."""
    prices = cost.current_prices(state_dir)
    if not (prices.gpu_hourly or prices.cpu_core_hourly or prices.memory_gib_hourly):
        return None
    return cost.hourly_rate(compute, prices)


def session_timings(session: Session) -> dict[str, float]:
    """Seconds per start phase (create, upload, ready, total), or {}."""
    try:
        value = json.loads(session.extra.get(_EXTRA_TIMINGS, "") or "{}")
    except json.JSONDecodeError:
        return {}
    return {k: float(v) for k, v in value.items() if isinstance(v, int | float)} if isinstance(value, dict) else {}


def session_snapshot(session: Session) -> str | None:
    """The snapshot image the session booted from, or None."""
    return session.extra.get(_EXTRA_FROM_SNAPSHOT) or None


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
        prices_checked=session.extra.get(_EXTRA_PRICES_CHECKED, ""),
        prices_source=session.extra.get(_EXTRA_PRICES_SOURCE, ""),
    )


def refresh_live_prices(backend: Backend, state_dir: Path, *, force: bool = False) -> str | None:
    """Cache the provider's current prices (cost.save_live_prices) if the
    backend can report them and the cache is older than
    cost.LIVE_REFRESH_AFTER_S (or `force`). Returns an error message, or None
    (refreshed, fresh enough, or not supported). Never raises for provider
    errors: prices fall back to the cache or the shipped table."""
    if CAP_BILLING not in backend.capabilities:
        return None
    age = cost.live_cache_age(state_dir, backend.name)
    if not force and age is not None and 0 <= age < cost.LIVE_REFRESH_AFTER_S:
        return None
    try:
        prices = backend.current_prices()  # type: ignore[attr-defined]
        cost.save_live_prices(state_dir, backend.name, prices)
    except Exception as exc:  # noqa: BLE001 - a price refresh never blocks anything
        return str(exc) or type(exc).__name__
    return None


@dataclass(frozen=True)
class Spend:
    days: list[tuple[str, float]]  # (ISO date, USD), oldest first, days with no cost left out
    by_resource: dict[str, float]  # USD over the period, largest first
    total: float
    start: datetime.datetime  # UTC, inclusive
    end: datetime.datetime  # UTC, exclusive: the end of the last complete hour


def spend(backend: Backend, days: int, now: datetime.datetime | None = None) -> Spend:
    """What this tool's runtimes on `backend` were billed over the last
    `days` complete UTC days plus today so far (hourly rows, so complete
    hours only). Raises UnsupportedError without CAP_BILLING."""
    if CAP_BILLING not in backend.capabilities:
        raise UnsupportedError(f"the {backend.name} backend cannot report what was billed")
    if not 1 <= days <= 31:
        raise ValueError(f"days must be between 1 and 31, got {days}")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - datetime.timedelta(days=days)
    hour = now.replace(minute=0, second=0, microsecond=0)
    rows = list(backend.spend(start, today, "d"))  # type: ignore[attr-defined]
    if hour > today:
        rows += backend.spend(today, hour, "h")  # type: ignore[attr-defined]
    per_day: dict[str, float] = {}
    by_resource: dict[str, float] = {}
    for row in rows:
        day = row.interval_start.astimezone(datetime.timezone.utc).date().isoformat()
        per_day[day] = per_day.get(day, 0.0) + row.cost
        for name, value in row.by_resource.items():
            by_resource[name] = by_resource.get(name, 0.0) + value
    return Spend(
        days=sorted((d, c) for d, c in per_day.items() if c > 0),
        by_resource=dict(sorted(((k, v) for k, v in by_resource.items() if v > 0), key=lambda kv: -kv[1])),
        total=sum(per_day.values()),
        start=start,
        end=hour,
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
        _EXTRA_PRICES_CHECKED: rate.prices_checked,
        _EXTRA_PRICES_SOURCE: rate.prices_source,
    }


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


def _write_watchdog_script(backend: Backend, handle: RuntimeHandle) -> None:
    script = importlib.resources.files("nbgpu").joinpath("watchdog.py").read_bytes()
    backend.write_file(handle, runtime_path(handle, WATCHDOG_NAME), script)


def _start_watchdog(
    backend: Backend, handle: RuntimeHandle, idle_timeout_s: float, unsynced_grace_s: float
) -> RemoteProcess:
    _write_watchdog_script(backend, handle)
    proc = _spawn_watchdog(backend, handle, idle_timeout_s, unsynced_grace_s)
    time.sleep(_WATCHDOG_STARTUP_CHECK_S)
    _check_watchdog(proc)
    return proc


def _check_watchdog(proc: RemoteProcess) -> None:
    code = proc.poll()
    if code is not None:
        lines = _last_lines(proc, _START_ERROR_TAIL_LINES)
        raise StartError(f"the idle watchdog exited with code {code}; last output:\n" + ("\n".join(lines) or "(no output)"))


def _spawn_watchdog(
    backend: Backend, handle: RuntimeHandle, idle_timeout_s: float, unsynced_grace_s: float
) -> RemoteProcess:
    path = runtime_path(handle, WATCHDOG_NAME)
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
        "--activity-file", runtime_path(handle, ACTIVITY_NAME),
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
    return backend.spawn(handle, argv, cwd=posixpath.dirname(handle.remote_root))


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
    root: Path,
    backend: Backend,
    handle: RuntimeHandle,
    state_path: Path,
    exclude: Sequence[str] = (),
    *,
    defer_manifest: bool = False,
) -> SyncEngine:
    return SyncEngine(root, backend, handle, state_path, manifest_path=manifest_path(handle), exclude=exclude,
                      defer_manifest=defer_manifest)


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


def session_engine(
    session: Session, *, backend: Backend, store: SessionStore, defer_manifest: bool = False
) -> SyncEngine:
    """A SyncEngine for the session, to be kept and reused across polls
    (`defer_manifest=True` for such a long-lived engine: see nbgpu.sync).
    Raises SessionNotRunning if the runtime is gone."""
    handle = _require_handle(session, backend)
    state_path = sync_state_path(session.session_id, store)
    _private_file(state_path)
    return _new_engine(session.project_root, backend, handle, state_path, session_exclude(session),
                       defer_manifest=defer_manifest)


# ---------------------------------------------------------------------------
# filesystem snapshots (CAP_SNAPSHOT)
#
# `stop` can save the runtime's filesystem; the next `start` of the same
# notebook boots from it when nothing that shaped that filesystem changed:
# the backend, the dependencies, requires-python, this machine's Python
# minor version (the runtime's Python follows it) and marimo's version
# (pinned to the local one). Compute (gpu, cpu, memory, volumes) may differ.
# Records live in the state directory (SNAPSHOTS_NAME), one per notebook.

SNAPSHOTS_NAME = "snapshots.json"
# A little under Modal's retention (backends.modal.SNAPSHOT_TTL_S), so an
# expired image is not even tried.
SNAPSHOT_MAX_AGE_S = 29 * 24 * 3600


def snapshot_key(backend_name: str, dependencies: Sequence[str], requires_python: str | None) -> dict[str, object]:
    try:
        marimo_version = importlib.metadata.version("marimo")
    except importlib.metadata.PackageNotFoundError:
        marimo_version = ""
    return {
        "backend": backend_name,
        "deps": sorted(dependencies),
        "requires_python": requires_python,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "marimo": marimo_version,
    }


def _read_snapshots(store: SessionStore) -> dict[str, dict]:
    try:
        data = json.loads((store.directory / SNAPSHOTS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _write_snapshots(store: SessionStore, records: Mapping[str, dict]) -> None:
    ensure_private_dir(store.directory)
    path = store.directory / SNAPSHOTS_NAME
    fd, tmp = tempfile.mkstemp(prefix=".snapshots-", dir=store.directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), 0o600)
            json.dump(dict(records), f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def snapshot_for(notebook: Path, store: SessionStore) -> dict | None:
    """The notebook's snapshot record (any key), or None."""
    return _read_snapshots(store).get(str(Path(notebook).resolve()))


def matching_snapshot(spec: NotebookSpec, backend: Backend, store: SessionStore,
                      now: float | None = None) -> dict | None:
    """The notebook's snapshot if `backend` can boot it and it fits `spec`."""
    if CAP_SNAPSHOT not in backend.capabilities:
        return None
    record = snapshot_for(spec.path, store)
    if record is None or not isinstance(record.get("image"), str):
        return None
    age = (time.time() if now is None else now) - float(record.get("created_at", 0))
    if not 0 <= age <= SNAPSHOT_MAX_AGE_S:
        return None
    if record.get("key") != snapshot_key(backend.name, spec.dependencies, spec.requires_python):
        return None
    return record


def forget_snapshot(notebook: Path, store: SessionStore) -> bool:
    """Drop the notebook's snapshot record. True if there was one. (The
    provider deletes the image itself when it expires.)"""
    records = _read_snapshots(store)
    if records.pop(str(Path(notebook).resolve()), None) is None:
        return False
    _write_snapshots(store, records)
    return True


def _save_snapshot(session: Session, backend: Backend, handle: RuntimeHandle, store: SessionStore) -> dict:
    """Snapshot the runtime and record it for the notebook's next start. The
    key uses the dependencies the runtime actually has (live installs
    included) and the header's requires-python."""
    try:
        applied = json.loads(session.extra.get(envsync.EXTRA_ENV, "") or "{}")
    except json.JSONDecodeError:
        applied = {}
    deps = applied.get("deps") if isinstance(applied, dict) else None
    spec = load_spec(session.notebook)
    if not isinstance(deps, list):
        deps = list(spec.dependencies)
    started = time.monotonic()
    image = backend.snapshot(handle)  # type: ignore[attr-defined]
    record = {
        "image": image,
        "created_at": time.time(),
        "seconds": round(time.monotonic() - started, 1),
        "session_id": session.session_id,
        "key": snapshot_key(backend.name, deps, spec.requires_python),
    }
    records = _read_snapshots(store)
    records[str(session.notebook)] = record
    _write_snapshots(store, records)
    return record


def project_cache_id(project_root: Path) -> str:
    """A short, stable id for a local project. Every project lands at the
    same remote path, so per-project caches on a shared volume need it."""
    return hashlib.sha256(str(Path(project_root).resolve()).encode()).hexdigest()[:12]


def cell_cache_paths(spec: NotebookSpec, backend: Backend, handle: RuntimeHandle, project_root: Path,
                     rel_notebook: str) -> tuple[str, str] | None:
    """(link, target) for `cell-cache`, or None when it does not apply.

    marimo keeps mo.persistent_cache entries in `__marimo__/cache` next to
    the notebook. With cell-cache that folder on the runtime is a symlink
    into the cache volume, per project and notebook directory, so the
    entries outlive the runtime. Only that folder moves: pointing
    PYTHONPYCACHEPREFIX at the volume would also move Python's bytecode,
    and reading it from the volume made `import torch` take minutes
    (measured 2026-11-15). `__marimo__` is never synced, so the link stays
    on the runtime."""
    if not (spec.cell_cache and spec.compute.cache_volume is not None and CAP_VOLUMES in backend.capabilities):
        return None
    directory = posixpath.dirname(rel_notebook)
    link = posixpath.join(handle.remote_root, directory, "__marimo__", "cache")
    target = posixpath.join(CACHE_MOUNT, "marimo", project_cache_id(project_root), directory or ".")
    return link, posixpath.normpath(target)


def _link_cell_cache(backend: Backend, handle: RuntimeHandle, paths: tuple[str, str] | None) -> None:
    if paths is None:
        return
    link, target = paths
    q = shlex.quote
    script = (f"mkdir -p {q(target)} {q(posixpath.dirname(link))} && rm -rf {q(link)} "
              f"&& ln -s {q(target)} {q(link)}")
    result = backend.exec(handle, ["sh", "-c", script], cwd="/", timeout_s=120)
    if result.exit_code != 0:
        raise BackendError(f"linking marimo's cache to the cache volume failed: "
                           f"{result.stderr.strip() or result.stdout.strip()}")


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _reset_runtime_dir(backend: Backend, handle: RuntimeHandle) -> None:
    """A runtime booted from a snapshot has the old project, token and
    markers; remove them so the upload and the new token are all there is."""
    parent = posixpath.dirname(handle.remote_root.rstrip("/"))
    script = f"rm -rf {shlex.quote(parent)} && mkdir -p {shlex.quote(handle.remote_root)}"
    result = backend.exec(handle, ["sh", "-c", script], cwd="/", timeout_s=120)
    if result.exit_code != 0:
        raise BackendError(f"clearing the snapshot's old project failed: {result.stderr.strip() or result.stdout.strip()}")


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
    fresh: bool = False,
) -> Session:
    """`progress`, if given, is called with a short description as each
    startup step begins ("creating the runtime", "uploading 44 files", ...).
    The notebook's filesystem snapshot is used when one matches
    (matching_snapshot), unless `fresh`. Phase durations are recorded in the
    session (session_timings)."""
    with signals_as_interrupt():
        return _start(notebook, backend, store, overrides, project_root, ready_timeout_s, progress, fresh)


def _start(
    notebook: Path,
    backend: Backend,
    store: SessionStore,
    overrides: Mapping[str, object] | None,
    project_root: Path | None,
    ready_timeout_s: float,
    progress: Callable[[str], None] | None = None,
    fresh: bool = False,
) -> Session:
    step = progress or (lambda _message: None)
    began = time.monotonic()
    timings: dict[str, float] = {}
    spec = load_spec(notebook, overrides)
    notebook = spec.path
    root = Path(project_root).resolve() if project_root is not None else find_project_root(notebook)
    rel_notebook = _relative_notebook(notebook, root)
    ensure_private_dir(store.directory)
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
        **applied_extra(spec),
        **_rate_extra(rate_for(spec.compute, store.directory)),
    }
    # Fetch the provider's current prices while the runtime is created; the
    # complete record below is priced with them (at most once a day).
    price_refresh = threading.Thread(target=refresh_live_prices, args=(backend, store.directory), daemon=True)
    price_refresh.start()
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

    if spec.cell_cache and CAP_VOLUMES not in backend.capabilities:
        step(f"cell-cache: the {backend.name} backend has no cache volume, so marimo's cache stays on the runtime")
    base = None if fresh else matching_snapshot(spec, backend, store)
    if base is not None:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(base["created_at"])))
        step(f"creating the runtime ({_shape(spec.compute)}) from the snapshot of {when}")
    else:
        step(f"creating the runtime ({_shape(spec.compute)})")
    phase = time.monotonic()
    try:
        try:
            handle = backend.create(RuntimeRequest(spec=spec, tags=tags, ports=(MARIMO_PORT,),
                                                   base_image=base["image"] if base else None))
        except BackendError as exc:
            if base is None:
                raise
            # The image may have expired or been deleted: build instead.
            forget_snapshot(spec.path, store)
            step(f"the snapshot could not be used ({_first_line(str(exc))}); building the environment instead")
            base = None
            handle = backend.create(RuntimeRequest(spec=spec, tags=tags, ports=(MARIMO_PORT,)))
    except BaseException as exc:
        _abandon_create(session, backend, store, exc)
        raise
    timings["create"] = time.monotonic() - phase
    try:
        session.runtime_id = handle.runtime_id
        session.remote_root = handle.remote_root
        session.created_at = handle.created_at
        store.save(session)
        if base is not None:
            _reset_runtime_dir(backend, handle)

        phase = time.monotonic()
        step(f"uploading {files} file{'s' if files != 1 else ''} ({_size_text(size)}) from {root.name}/")
        token = new_token()
        token_file = token_path(handle)

        def side_files() -> None:
            # Independent of the upload, so it runs alongside it.
            backend.write_file(handle, token_file, token.encode("utf-8"))
            _write_watchdog_script(backend, handle)
            if CAP_LIVE_INSTALL in backend.capabilities:
                # What is installed now, so stop can tell what was added later.
                with contextlib.suppress(Exception):
                    backend.exec(handle, envsync.baseline_argv(runtime_path(handle, envsync.PACKAGES_BASELINE_NAME)),
                                 cwd="/", timeout_s=60)

        _in_parallel(
            lambda: _new_engine(root, backend, handle, state_path, spec.exclude).initial_push(),
            side_files,
        )
        _private_file(state_path)
        note_synced(session, store)
        _link_cell_cache(backend, handle, cell_cache_paths(spec, backend, handle, root, rel_notebook))
        timings["upload"] = time.monotonic() - phase
        phase = time.monotonic()

        pidfile = runtime_path(handle, MARIMO_PIDFILE_NAME)
        argv = _marimo_argv(
            notebook=rel_notebook,
            host=backend.bind_host,
            port=handle.port_for(MARIMO_PORT),
            token_file=token_file,
        )
        step("starting marimo")
        proc = backend.spawn(handle, _with_pidfile(pidfile, argv), cwd=handle.remote_root)
        # The watchdog waits for marimo's pid file itself, so it starts now
        # and is checked once marimo answers (no separate startup sleep).
        dog = _spawn_watchdog(backend, handle, idle_timeout_s, grace_s)
        url = backend.expose(handle, MARIMO_PORT).rstrip("/")
        _wait_ready(proc, url, ready_timeout_s)
        _check_watchdog(dog)

        session.url = url
        session.token = token
        timings["ready"] = time.monotonic() - phase
        timings["total"] = time.monotonic() - began
        price_refresh.join(timeout=2.0)
        session.extra = {
            **extra,
            **_rate_extra(rate_for(spec.compute, store.directory)),
            _EXTRA_TIMINGS: json.dumps({k: round(v, 2) for k, v in timings.items()}),
            **({_EXTRA_FROM_SNAPSHOT: base["image"]} if base is not None else {}),
        }
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


def _in_parallel(*calls: Callable[[], object]) -> None:
    """Run the calls on threads and wait for all; re-raise the first error.
    Each call talks to the backend (network-bound), so threads overlap well."""
    errors: list[BaseException] = []

    def run(call: Callable[[], object]) -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(c,), daemon=True) for c in calls[1:]]
    for t in threads:
        t.start()
    run(calls[0])  # the first on this thread, so Ctrl-C still interrupts it
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


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
    _forget_kernel_id(session, store)
    with contextlib.suppress(OSError):
        state = sync_state_path(session.session_id, store)
        state.with_name(state.name + ".lock").unlink(missing_ok=True)


def _with_warnings(report: SyncReport, *warnings: str) -> SyncReport:
    report.warnings += [w for w in warnings if w not in report.warnings]
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
    snapshot: bool | None = None,
) -> SyncReport:
    """Final pull, then terminate and remove the record (and its sync state).
    With open conflicts, raise StopRefused unless ignore_conflicts (then:
    still pull what is clean, keep conflict copies, then terminate; the
    returned report lists the open conflicts). If the final pull itself
    fails, the error is raised and the runtime and record are kept, unless
    terminate_on_error (only for an explicit user --force), in which case
    the runtime is terminated anyway and the error raised afterwards.
    force=True means both. `engine` reuses the caller's engine.

    `snapshot` (None: the header's `snapshot` key) saves the runtime's
    filesystem before terminating, on backends with CAP_SNAPSHOT, so the next
    start resumes from it; the report's `snapshot` holds the record. A failed
    snapshot is a warning, never a reason to keep the runtime.

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
        unrecorded = _unrecorded_installs(session, backend, handle)
        notes = []
        if snapshot is None:
            try:
                snapshot = load_spec(session.notebook).snapshot
            except SpecError:
                snapshot = False
        if snapshot and CAP_SNAPSHOT not in backend.capabilities:
            notes.append(f"the {backend.name} backend cannot snapshot a runtime; nothing was saved")
        elif snapshot:
            try:
                report.snapshot = _save_snapshot(session, backend, handle, store)
            except Exception as exc:  # noqa: BLE001 - never keeps a runtime billing
                notes.append(f"the snapshot failed ({exc}); the next start builds the environment instead")
        _terminate(backend, handle)
        _forget(session, store)
        if unrecorded:
            notes.append(
                f"installed during the session but not in {session.notebook.name}'s dependencies: "
                f"{', '.join(unrecorded)}; add them to its header (or a [tool.nbgpu] requirements file) "
                "to have them next time"
            )
        return _with_warnings(report, *notes)


def _unrecorded_installs(session: Session, backend: Backend, handle: RuntimeHandle) -> list[str]:
    """Best effort (one exec): packages installed in the runtime that the
    notebook does not list, e.g. installed from the remote link or by an agent."""
    try:
        spec = load_spec(session.notebook)
        found = envsync.unrecorded_installs(
            backend, handle, runtime_path(handle, envsync.PACKAGES_BASELINE_NAME), spec)
    except Exception:  # noqa: BLE001 - never blocks a stop
        return []
    return found or []


# ---------------------------------------------------------------------------
# run


def _bound_output(lines: list[str], max_lines: int, full_output_path: Path) -> str:
    """First and last halves of `lines` with a marker line between, in at
    most `max_lines` lines in total (the marker included)."""
    keep = max_lines - 1
    head_n = (keep + 1) // 2
    tail_n = keep - head_n
    marker = _omitted_marker(len(lines) - keep, full_output_path)
    head = lines[:head_n]
    tail = lines[len(lines) - tail_n:] if tail_n else []
    return "\n".join([*head, marker, *tail])


def _omitted_marker(omitted: int, full_output_path: Path) -> str:
    return f"... {omitted} lines omitted; full output in {full_output_path} ..."


# Per-file limit for `run --pull`: larger outputs belong on a volume.
RUN_PULL_MAX_BYTES = 200 * 1024 * 1024
# At most this many files are pulled by one `run --pull`.
RUN_PULL_MAX_FILES = 1000

_SIZES_CODE = "import json,os,sys;print(json.dumps({p: os.path.getsize(p) for p in sys.argv[1:] if os.path.isfile(p)}))"


def pull_matching(
    root: Path, backend: Backend, handle: RuntimeHandle, patterns: Sequence[str], exclude: Sequence[str] = ()
) -> tuple[list[str], list[str]]:
    """Bring back runtime files matching `patterns` (.gitignore syntax,
    relative to the project root) that differ from the local copy, for
    outputs sync would not pull (any type or size, ignored files included).

    A local file that sync manages (exists and is not ignored) is never
    overwritten: sync handles it both ways. A missing local file is created;
    an ignored one (an output from an earlier run) is replaced. Files over
    RUN_PULL_MAX_BYTES, unsafe paths and files changed mid-read are left out
    with a warning. Returns (pulled, warnings)."""
    if not patterns:
        return [], []
    spec = pathspec.GitIgnoreSpec.from_lines(list(patterns))
    rules = IgnoreRules(root, exclude)
    pulled: list[str] = []
    warnings: list[str] = []
    candidates: dict[str, str] = {}
    for rel, digest in sorted(backend.hash_files(handle).items()):
        if not spec.match_file(rel) or sync.is_excluded(rel):
            continue
        target = sync.local_target(root, rel)
        if target is None:
            warnings.append(f"{rel}: not pulled (unsafe local path)")
            continue
        try:
            local = target.read_bytes() if target.exists() else None
        except OSError as exc:
            warnings.append(f"{rel}: not pulled ({exc.strerror or exc})")
            continue
        if local is not None and hashlib.sha256(local).hexdigest() == digest:
            continue  # already identical
        if local is not None and not rules.is_ignored(rel):
            continue  # a synced project file: sync decides, so local edits are never lost
        candidates[rel] = digest
    if len(candidates) > RUN_PULL_MAX_FILES:
        warnings.append(f"{len(candidates)} files match --pull; only the first {RUN_PULL_MAX_FILES} were pulled")
        candidates = dict(list(candidates.items())[:RUN_PULL_MAX_FILES])
    if not candidates:
        return pulled, warnings
    result = backend.exec(handle, ["python", "-c", _SIZES_CODE, *candidates], cwd=handle.remote_root)
    try:
        sizes = json.loads(result.stdout) if result.exit_code == 0 else {}
    except json.JSONDecodeError:
        sizes = {}
    limit_mb = RUN_PULL_MAX_BYTES // (1024 * 1024)
    for rel, digest in candidates.items():
        size = sizes.get(rel)
        if not isinstance(size, int):
            warnings.append(f"{rel}: not pulled (could not read its size)")
            continue
        if size > RUN_PULL_MAX_BYTES:
            warnings.append(f"{rel}: not pulled ({_size_text(size)} is over the {limit_mb} MB --pull limit; "
                            "write large outputs to a volume)")
            continue
        data = backend.read_file(handle, posixpath.join(handle.remote_root, rel))
        if data is None or hashlib.sha256(data).hexdigest() != digest:
            warnings.append(f"{rel}: not pulled (changed or removed while reading)")
            continue
        target = sync.local_target(root, rel)
        if target is None:
            warnings.append(f"{rel}: not pulled (unsafe local path)")
            continue
        sync.write_local_file(target, data)
        pulled.append(rel)
    return pulled, warnings


def _sync_after_run(
    engine: SyncEngine, root: Path, backend: Backend, handle: RuntimeHandle, pull: Sequence[str],
    exclude: Sequence[str], result: RunResult,
) -> None:
    """Bring back what the run wrote: one sync pass (small text files, as
    `nbgpu sync`), then `pull` patterns. Failures become warnings, so the
    run's own result is never lost."""
    try:
        report = engine.poll_once()
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"syncing after the run failed: {exc}")
    else:
        result.pulled.extend(report.pulled + report.merged)
        result.warnings.extend(f"{c}: sync conflict (remote version saved as a .remote copy)" for c in report.conflicts)
    try:
        pulled, warnings = pull_matching(root, backend, handle, pull, exclude)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"--pull failed: {exc}")
    else:
        result.pulled.extend(p for p in pulled if p not in result.pulled)
        result.warnings.extend(warnings)


def apply_env(session: Session, *, backend: Backend, handle: RuntimeHandle, store: SessionStore) -> envsync.Applied:
    """Install header changes into the session's runtime now (for `sync` and
    `run`, which have no loop). See EnvWatcher.apply."""
    return envsync.EnvWatcher(session, backend, handle, store).apply()


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
        ensure_private_dir(output_dir)
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
    pull: Sequence[str] = (),
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
    keeps a "run" record in `store` (default SessionStore()) while it runs.

    In a session, header changes are installed first (apply_env); a failed
    install refuses the run, since the notebook would fail on the import.
    After the run, in both modes, one sync pass brings back what `nbgpu sync`
    would, and `pull` patterns bring back other outputs (pull_matching);
    see RunResult.pulled and .warnings."""
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
            env_messages: list[str] = []
            if store is not None:
                applied = apply_env(session, backend=backend, handle=handle, store=store)
                if applied.failed:
                    detail = "; ".join(m[1:] for m in applied.messages)
                    raise RunRefused(f"the notebook was not run: {detail}")
                env_messages = applied.messages
            result = _run_in(backend, handle, rel_notebook, max_output_lines, out_dir)
            result.env.extend(env_messages)
            _sync_after_run(_engine(session, backend, handle, _state_path(session, store)), session.project_root,
                            backend, handle, pull, session_exclude(session), result)
            if store is not None:
                note_synced(session, store)
            return result

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
                engine = _new_engine(root, backend, handle, Path(tmp) / "sync.json", spec.exclude)
                engine.initial_push()
                _link_cell_cache(backend, handle, cell_cache_paths(spec, backend, handle, root, rel_notebook))
                result = _run_in(backend, handle, rel_notebook, max_output_lines, out_dir)
                _sync_after_run(engine, root, backend, handle, pull, spec.exclude, result)
            return result
        finally:
            _terminate(backend, handle)
            with contextlib.suppress(Exception):
                store.remove(run_id)


# ---------------------------------------------------------------------------
# exec: code in the live kernel


@dataclass
class KernelExecResult:
    ok: bool
    value: str | None  # repr() of the code's last expression, if any
    error: str | None  # one line, when not ok
    output: str  # stdout and stderr in arrival order, bounded like run's
    truncated: bool
    full_output_path: Path | None
    cells: list[dict[str, object]] = field(default_factory=list)  # --list: all; cells: those in error
    kernel_created: bool = False  # no kernel existed; exec started one
    ran_notebook: bool = False  # ... and ran every cell first
    pushed: list[str] = field(default_factory=list)  # files pushed by the sync before
    warnings: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)  # EnvWatcher messages (role character first)


def exec_in_kernel(
    session: Session,
    *,
    backend: Backend,
    store: SessionStore,
    code: str | None = None,
    cells: Sequence[str] | str | None = None,
    list_cells: bool = False,
    sync_first: bool | None = None,
    max_output_lines: int = 200,
    output_dir: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> KernelExecResult:
    """Run `code` in the session's live marimo kernel (exactly one of `code`,
    `cells` (names or ids, or "all"), `list_cells`), see nbgpu.kernel.

    sync_first (default: only when running cells) does one sync pass first,
    refuses if the notebook has a conflict, and installs header changes, as
    `run` does; when it pushed the notebook, cells run only once marimo's
    file watcher has loaded it. Without a kernel, one is started. Unless only
    listing, if no cell has run in the kernel yet, every cell runs first so
    the notebook's variables exist (ran_notebook).
    Output is bounded like run's (full text under output_dir). With
    `on_line`, output lines are also passed to it as they arrive, bounded
    the same way (LiveLines): the same lines result.output holds.

    The kernel's stable session id is remembered next to the session's
    record (`<id>.kernel`), so a call is one request; if marimo no longer
    knows it, it is looked up (or a kernel created) again."""
    from nbgpu import kernel as kernel_mod

    modes = [code is not None, cells is not None, list_cells]
    if sum(modes) != 1:
        raise ValueError("exactly one of code, cells and list_cells is needed")
    out_dir = Path(output_dir) if output_dir is not None else default_state_dir() / "runs"
    rel_notebook = _relative_notebook(session.notebook, session.project_root)
    remote_notebook = posixpath.join(session.remote_root, rel_notebook)
    activity = posixpath.join(posixpath.dirname(session.remote_root), ACTIVITY_NAME)
    result = KernelExecResult(ok=False, value=None, error=None, output="", truncated=False, full_output_path=None)

    do_sync = sync_first if sync_first is not None else cells is not None
    notebook_pushed = False
    if do_sync:
        handle = _require_handle(session, backend)
        report = _engine(session, backend, handle, _state_path(session, store)).poll_once()
        note_synced(session, store)
        if rel_notebook in report.conflicts:
            raise RunRefused(
                f"{rel_notebook} has an open sync conflict, so the runtime's copy differs from yours; "
                "resolve it (delete the .remote copy) and run again"
            )
        result.pushed = list(report.pushed)
        result.warnings.extend(report.warnings)
        notebook_pushed = rel_notebook in report.pushed or rel_notebook in report.merged
        applied = apply_env(session, backend=backend, handle=handle, store=store)
        if applied.failed:
            detail = "; ".join(m[1:] for m in applied.messages)
            raise RunRefused(f"nothing was run: {detail}")
        result.env = applied.messages

    pieces: list[str] = []
    live = LiveLines(on_line, max_output_lines) if on_line is not None else None

    def collect(channel: str, text: str) -> None:
        pieces.append(text)
        if live is not None:
            live.feed(text)

    params: dict[str, object] = {"activity": activity, "prime": True}
    if list_cells:
        params["mode"] = "list"
    elif code is not None:
        params.update(mode="code", code=code)
    else:
        params.update(mode="cells", cells="all" if cells == "all" else list(cells or ()))
        if notebook_pushed:
            params["wait_for"] = remote_notebook  # a kernel created just now matches at once

    client = kernel_mod.KernelClient(session.url, session.token)
    try:
        outcome = None
        cached = _cached_kernel_id(session, store)
        if cached is not None:
            try:
                outcome = kernel_mod.run_in_kernel(client, cached, params, collect)
            except kernel_mod.UnknownKernelSession:
                _forget_kernel_id(session, store)  # marimo restarted it, or it was closed
        if outcome is None:
            kernel_id, result.kernel_created = client.ensure_kernel(rel_notebook)
            _remember_kernel_id(session, store, kernel_id)
            outcome = kernel_mod.run_in_kernel(client, kernel_id, params, collect)
    except kernel_mod.KernelError:
        _require_handle(session, backend)  # a runtime that is gone is the better error
        raise
    result.ok, result.value, result.error, result.cells = outcome.ok, outcome.value, outcome.error, outcome.cells
    result.ran_notebook = outcome.ran_notebook
    if outcome.notebook_error and code is not None:
        result.warnings.append(f"running the notebook first: {outcome.notebook_error}")

    text = kernel_mod.strip_results("".join(pieces))[0].rstrip("\n")
    lines = text.split("\n") if text else []
    if len(lines) <= max_output_lines:
        result.output = text
    else:
        if not out_dir.exists():
            ensure_private_dir(out_dir)
        stem = posixpath.splitext(posixpath.basename(rel_notebook))[0]
        fd, name = tempfile.mkstemp(prefix=f"{stem}-exec-{time.strftime('%Y%m%d-%H%M%S')}-", suffix=".log",
                                    dir=out_dir)
        with open(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(text)
        result.full_output_path = Path(name)
        result.truncated = True
        result.output = _bound_output(lines, max_output_lines, result.full_output_path)
    if live is not None:
        live.finish(len(lines), result.full_output_path)
    return result


class LiveLines:
    """Passes output lines to `emit` as they arrive, bounded like
    _bound_output: the first half of `max_lines` at once, the rest held
    until finish(), which emits them all if the output fit, else the marker
    line and the last half. The runner's result lines are left out."""

    def __init__(self, emit: Callable[[str], None], max_lines: int) -> None:
        self._emit = emit
        keep = max_lines - 1
        self._head_n = (keep + 1) // 2
        self._tail_n = keep - self._head_n
        self._max = max_lines
        self._held: collections.deque[str] = collections.deque(maxlen=max_lines - self._head_n)
        self._partial = ""
        self.count = 0

    def feed(self, text: str) -> None:
        *complete, self._partial = (self._partial + text).split("\n")
        for line in complete:
            self._line(line + "\n")

    def _line(self, raw: str) -> None:
        from nbgpu.kernel import strip_results

        clean = strip_results(raw)[0]
        if not clean and raw.strip():
            return  # only a result line
        line = clean.rstrip("\n")
        self.count += 1
        if self.count <= self._head_n:
            self._emit(line)
        else:
            self._held.append(line)

    def finish(self, total_lines: int, full_output_path: Path | None) -> None:
        """`total_lines` and `full_output_path` as exec_in_kernel counted
        and wrote them, so the marker matches result.output's."""
        if self._partial:
            self._line(self._partial)
            self._partial = ""
        if full_output_path is None:
            for line in self._held:
                self._emit(line)
            return
        self._emit(_omitted_marker(total_lines - (self._max - 1), full_output_path))
        held = list(self._held)
        for line in held[len(held) - self._tail_n:] if self._tail_n else []:
            self._emit(line)


def _kernel_id_path(session_id: str, store: SessionStore) -> Path:
    return store.directory / f"{session_id}.kernel"


def _cached_kernel_id(session: Session, store: SessionStore) -> str | None:
    with contextlib.suppress(OSError):
        value = _kernel_id_path(session.session_id, store).read_text(encoding="utf-8").strip()
        return value or None
    return None


def _remember_kernel_id(session: Session, store: SessionStore, kernel_id: str) -> None:
    """Only while the session's record exists, so a concurrent stop is not undone."""
    if store.load(session.session_id) is None:
        return
    with contextlib.suppress(OSError):
        fd, tmp = tempfile.mkstemp(dir=store.directory, prefix=".kernel-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(kernel_id)
        os.replace(tmp, _kernel_id_path(session.session_id, store))


def _forget_kernel_id(session: Session, store: SessionStore) -> None:
    with contextlib.suppress(OSError):
        _kernel_id_path(session.session_id, store).unlink(missing_ok=True)
