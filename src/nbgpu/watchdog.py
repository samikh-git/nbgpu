"""Idle watchdog: runs inside the runtime and stops marimo when the session is idle.

Self-contained and stdlib-only: lifecycle uploads this file to the runtime
(next to the token, outside the synced project directory) and runs it with
the runtime's python, so it must never import nbgpu.

Providers count any running process (the marimo server, nbgpu's sync execs)
as activity, so their own idle timeouts never fire while marimo runs. This
script decides idleness from what the user actually does. The session is
active while any of these holds:

- a file under the project directory was modified recently (newest mtime);
- processes other than this one use more than CPU_BUSY_FRACTION of a core
  between two checks (read from /proc where available, else `ps`). Only
  processes alive at both samples count, so short-lived commands such as
  the sync loop's hashing execs do not keep a session alive;
- a GPU is busy (`nvidia-smi` reports utilization above GPU_BUSY_PERCENT);
- marimo itself reports a cell running or queued: GET /api/sessions, then
  GET /api/kernel/status (header Marimo-Session-Id) for each session,
  authenticated with `Authorization: Bearer <token>` (`--marimo-url`,
  `--token-file`). This catches cells that wait on an API, poll a job or
  sleep while using no CPU. If marimo cannot be asked, the other signals
  still apply.

Processes whose environment has NBGPU_INTERNAL=1 (the backends' own helper
execs: hashing, compare-and-swap writes) never count, where the environment
is readable (/proc on Linux). An open browser tab alone is not activity.

While running it rewrites `--heartbeat` every check (JSON with its pid and
the time) and removes it on exit, so the laptop's sync loop can restart a
watchdog that died.

After `--idle-timeout` seconds without activity it checks for unsynced
edits: project files whose content differs from the manifest the laptop's
sync engine writes (`--manifest`, {"files": {relpath: sha256}}), or files
the laptop would pull that the manifest does not list. A missing or
unreadable manifest counts every such file as unsynced. If there are any,
it does not stop yet: it writes `--unsynced-marker` (JSON with the paths
and the deadline) and waits up to `--unsynced-grace` seconds more for a
sync (or new activity, which starts the idle clock again and removes that
marker). Billing stays bounded: after the grace it stops anyway.

Stopping writes a JSON marker file (`--marker`, atomically; with an
"unsynced" list of paths when it stopped with unsynced edits), terminates
marimo's process group (marimo's pid is read from `--marimo-pidfile`) and
exits. It also exits, without a marker, when marimo exits on its own or its
pid file never appears.

`--scope all` watches every process (right for an isolated container);
`--scope tree` only marimo and its descendants (for the local backend,
where other processes on the machine are not the session's).

The module constants are defaults; lifecycle passes its current values on
the command line, so tests can shrink them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

CHECK_INTERVAL_S = 30.0
CPU_BUSY_FRACTION = 0.10  # of one core, summed over the watched processes
GPU_BUSY_PERCENT = 5
PIDFILE_WAIT_S = 120.0
KILL_GRACE_S = 5.0
MAX_WALK_FILES = 50_000
# Extra time an idle session with unsynced edits is kept for a sync to pull
# them. None: equal to the idle timeout.
UNSYNCED_GRACE_S: float | None = None
MARIMO_REQUEST_TIMEOUT_S = 5.0
INTERNAL_ENV = b"NBGPU_INTERNAL=1"
# Mirrors nbgpu.sync (tested to match): what takes part in sync and what a
# new remote file must look like to be pulled.
EXCLUDED_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules", ".nbgpu", "__marimo__"})
PULL_SUFFIXES = frozenset({".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini"})
PULL_MAX_BYTES = 1024 * 1024
_TEMP_PREFIX = ".nbgpu-tmp-"


# ---------------------------------------------------------------------------
# processes


def _parse_cputime(text: str) -> float:
    """`ps` cputime: "[dd-][hh:]mm:ss[.cc]" -> seconds."""
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return days * 86400 + seconds


def _proc_table() -> dict[int, tuple[int, float, bool]]:
    """pid -> (ppid, cpu seconds, zombie) for every visible process."""
    table: dict[int, tuple[int, float, bool]] = {}
    if os.path.isdir("/proc/self"):
        ticks = os.sysconf("SC_CLK_TCK") or 100
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as f:
                    raw = f.read().decode("utf-8", "replace")
            except OSError:
                continue
            fields = raw[raw.rfind(")") + 2:].split()
            try:
                table[int(name)] = (int(fields[1]), (int(fields[11]) + int(fields[12])) / ticks, fields[0] == "Z")
            except (IndexError, ValueError):
                continue
        return table
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,stat=,time="], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return table
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            table[int(parts[0])] = (int(parts[1]), _parse_cputime(parts[3]), parts[2].startswith("Z"))
        except ValueError:
            continue
    return table


def _descendants(table: dict[int, tuple[int, float, bool]], root: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, (ppid, _, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    found = {root}
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    entry = _proc_table().get(pid)
    return entry is None or not entry[2]  # a zombie is not alive


_internal_cache: dict[int, bool] = {}


def is_internal(pid: int) -> bool:
    """Whether the process's environment has NBGPU_INTERNAL=1 (Linux /proc
    only; elsewhere False). Cached per pid: an environment does not change."""
    cached = _internal_cache.get(pid)
    if cached is not None:
        return cached
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            internal = INTERNAL_ENV in f.read().split(b"\0")
    except OSError:
        internal = False
    _internal_cache[pid] = internal
    return internal


def cpu_seconds(scope: str, marimo_pid: int) -> dict[int, float]:
    """CPU seconds used so far by each watched process (this one and nbgpu's
    internal helpers excluded)."""
    table = _proc_table()
    for pid in list(_internal_cache):
        if pid not in table:
            del _internal_cache[pid]  # pids are reused
    pids = set(table) if scope == "all" else _descendants(table, marimo_pid)
    pids.discard(os.getpid())
    return {
        pid: table[pid][1]
        for pid in pids
        if pid in table and not table[pid][2] and not is_internal(pid)
    }


def cpu_busy(before: dict[int, float], after: dict[int, float], elapsed_s: float, threshold: float) -> bool:
    """Whether processes alive at both samples used more than `threshold`
    of one core in between."""
    if elapsed_s <= 0:
        return False
    used = sum(max(0.0, after[pid] - before[pid]) for pid in after.keys() & before.keys())
    return used / elapsed_s > threshold


# ---------------------------------------------------------------------------
# GPU and files


def gpu_busy(threshold: float) -> bool:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        try:
            if float(line.strip()) > threshold:
                return True
        except ValueError:
            continue
    return False


def _marimo_get(url: str, token: str, extra_headers: dict[str, str] | None = None) -> object:
    headers = {"Authorization": f"Bearer {token}", **(extra_headers or {})}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, headers=headers), timeout=MARIMO_REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def marimo_busy(base_url: str, token_file: str) -> bool | None:
    """True if marimo reports a cell running or queued in any session, False
    if none, None if marimo could not be asked."""
    try:
        with open(token_file, encoding="utf-8") as f:
            token = f.read().strip()
        sessions = _marimo_get(base_url + "/api/sessions", token)
        if not isinstance(sessions, dict):
            return None
        for session_id in sessions:
            status = _marimo_get(base_url + "/api/kernel/status", token, {"Marimo-Session-Id": str(session_id)})
            if isinstance(status, dict) and status.get("state") == "running":
                return True
        return False
    except (OSError, ValueError, urllib.error.URLError):
        return None


def newest_mtime(root: str) -> float | None:
    newest: float | None = None
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            seen += 1
            if seen > MAX_WALK_FILES:
                return newest
            try:
                mtime = os.lstat(os.path.join(dirpath, name)).st_mtime
            except OSError:
                continue
            if newest is None or mtime > newest:
                newest = mtime
    return newest


# ---------------------------------------------------------------------------
# unsynced edits


def read_manifest(path: str) -> dict[str, str] | None:
    """relpath -> sha256 the laptop agreed with, or None if unknown."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict):
        return None
    return {str(k): str(v) for k, v in files.items()}


class HashCache:
    """sha256 of files, reused while (size, mtime_ns, inode) are unchanged."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[tuple[int, int, int], str]] = {}

    def digest(self, path: str, st: os.stat_result) -> str | None:
        key = (st.st_size, st.st_mtime_ns, st.st_ino)
        hit = self._entries.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            return None
        self._entries[path] = (key, h.hexdigest())
        return h.hexdigest()

    def retain(self, paths: set[str]) -> None:
        for path in list(self._entries):
            if path not in paths:
                del self._entries[path]


def unsynced_paths(root: str, manifest: dict[str, str] | None, cache: HashCache) -> list[str]:
    """Project files the laptop has not got: listed in the manifest with
    other content, or not listed but of a kind sync pulls. Deleted files do
    not count (deletions are not synced)."""
    known = manifest or {}
    found: list[str] = []
    seen_abs: set[str] = set()
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            if name.startswith(_TEMP_PREFIX):
                continue
            seen += 1
            if seen > MAX_WALK_FILES:
                cache.retain(seen_abs)
                return found
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
            try:
                st = os.lstat(abs_path)
            except OSError:
                continue
            if not os.path.isfile(abs_path) or os.path.islink(abs_path):
                continue
            expected = known.get(rel)
            if expected is None:
                if os.path.splitext(name)[1] in PULL_SUFFIXES and st.st_size <= PULL_MAX_BYTES:
                    found.append(rel)
                continue
            seen_abs.add(abs_path)
            if cache.digest(abs_path, st) != expected:
                found.append(rel)
    cache.retain(seen_abs)
    return found


# ---------------------------------------------------------------------------
# stopping


def remove_file(path: str | None) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def write_marker(path: str, info: dict[str, object]) -> None:
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, f"{_TEMP_PREFIX}{os.path.basename(path)}.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stop_marimo(pid: int, grace_s: float) -> None:
    """SIGTERM marimo's process group (just marimo if it shares ours), then
    SIGKILL whatever is left after `grace_s`."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    own = pgid == os.getpgrp()

    def send(sig: int) -> None:
        try:
            if own:
                os.kill(pid, sig)
            else:
                os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.1)
    send(signal.SIGKILL)


def read_pid(pidfile: str, wait_s: float) -> int | None:
    deadline = time.monotonic() + wait_s
    while True:
        try:
            with open(pidfile, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# main


def is_active(args: argparse.Namespace, sample: dict[int, float], current: dict[int, float], elapsed_s: float) -> bool:
    if cpu_busy(sample, current, elapsed_s, args.cpu_threshold):
        return True
    if gpu_busy(args.gpu_threshold):
        return True
    if args.marimo_url and args.token_file:
        return marimo_busy(args.marimo_url.rstrip("/"), args.token_file) is True
    return False


def watch(args: argparse.Namespace) -> int:
    pid = read_pid(args.marimo_pidfile, PIDFILE_WAIT_S)
    if pid is None:
        print(f"nbgpu-watchdog: no marimo pid in {args.marimo_pidfile}; exiting", file=sys.stderr)
        return 1
    grace = args.unsynced_grace if args.unsynced_grace is not None else args.idle_timeout
    cache = HashCache()
    beat(args, "watching")
    try:
        return _watch(args, pid, grace, cache)
    finally:
        remove_file(args.heartbeat)


def beat(args: argparse.Namespace, state: str) -> None:
    """Rewrite the heartbeat file, so the laptop can tell this process is alive."""
    if args.heartbeat:
        try:
            write_marker(args.heartbeat, {
                "pid": os.getpid(), "at": time.time(), "interval": args.interval, "state": state,
            })
        except OSError:
            pass


def _watch(args: argparse.Namespace, pid: int, grace: float, cache: HashCache) -> int:
    last_active = time.time()
    unsynced_since: float | None = None
    shown: list[str] | None = None
    sample_at = time.monotonic()
    sample = cpu_seconds(args.scope, pid)
    while True:
        time.sleep(args.interval)
        beat(args, "watching" if unsynced_since is None else "idle-unsynced")
        if not _alive(pid):
            remove_file(args.unsynced_marker)
            return 0  # marimo exited on its own: nothing left to watch
        now = time.time()
        mono = time.monotonic()
        current = cpu_seconds(args.scope, pid)
        if is_active(args, sample, current, mono - sample_at):
            last_active = now
        sample, sample_at = current, mono
        newest = newest_mtime(args.project)
        if newest is not None:
            last_active = max(last_active, min(newest, now))
        idle_s = now - last_active
        if idle_s < args.idle_timeout:
            if unsynced_since is not None:
                unsynced_since, shown = None, None
                remove_file(args.unsynced_marker)
            continue

        unsynced: list[str] = []
        if args.manifest:
            unsynced = unsynced_paths(args.project, read_manifest(args.manifest), cache)
        if unsynced:
            if unsynced_since is None:
                unsynced_since = now
            if now - unsynced_since < grace:
                if unsynced != shown and args.unsynced_marker:
                    write_marker(args.unsynced_marker, {
                        "reason": "idle-unsynced",
                        "unsynced": unsynced,
                        "idle_timeout_s": args.idle_timeout,
                        "since": unsynced_since,
                        "stop_at": unsynced_since + grace,
                    })
                    shown = unsynced
                continue
        info: dict[str, object] = {
            "reason": "idle",
            "idle_s": round(idle_s, 1),
            "idle_timeout_s": args.idle_timeout,
            "stopped_at": now,
        }
        if unsynced:
            info["unsynced"] = unsynced
            info["unsynced_grace_s"] = grace
        write_marker(args.marker, info)
        remove_file(args.unsynced_marker)
        stop_marimo(pid, args.kill_grace)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nbgpu-watchdog")
    parser.add_argument("--project", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--marimo-pidfile", required=True)
    parser.add_argument("--idle-timeout", type=float, required=True)
    parser.add_argument("--interval", type=float, default=CHECK_INTERVAL_S)
    parser.add_argument("--cpu-threshold", type=float, default=CPU_BUSY_FRACTION)
    parser.add_argument("--gpu-threshold", type=float, default=GPU_BUSY_PERCENT)
    parser.add_argument("--kill-grace", type=float, default=KILL_GRACE_S)
    parser.add_argument("--scope", choices=("all", "tree"), default="all")
    parser.add_argument("--heartbeat", help="rewritten every check while running; removed on exit")
    parser.add_argument("--manifest", help="JSON manifest of what the laptop has synced")
    parser.add_argument("--unsynced-marker", help="written while waiting for unsynced edits to be pulled")
    parser.add_argument("--unsynced-grace", type=float, default=UNSYNCED_GRACE_S)
    parser.add_argument("--marimo-url", help="marimo's base URL as seen from inside the runtime")
    parser.add_argument("--token-file", help="file holding marimo's access token")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    return watch(args)


if __name__ == "__main__":
    sys.exit(main())
