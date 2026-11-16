"""The idle watchdog script (src/nbgpu/watchdog.py), as a unit and as the
separate process it runs as inside a runtime."""

from __future__ import annotations

import ast
import hashlib
import http.server
import json
import threading
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nbgpu import watchdog

SCRIPT = Path(watchdog.__file__)


def test_script_is_stdlib_only() -> None:
    tree = ast.parse(SCRIPT.read_text())
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "nbgpu" not in modules
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    assert modules <= stdlib, modules - stdlib


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("0:00.03", 0.03), ("12:34.50", 754.5), ("01:02:03", 3723.0), ("2-00:00:01", 172801.0)],
)
def test_parse_cputime(text: str, seconds: float) -> None:
    assert watchdog._parse_cputime(text) == pytest.approx(seconds)


def test_cpu_busy_counts_only_processes_seen_twice() -> None:
    before = {1: 10.0, 2: 5.0}
    # pid 3 is new (e.g. a short sync exec): its CPU time does not count.
    after = {1: 10.05, 2: 5.0, 3: 50.0}
    assert not watchdog.cpu_busy(before, after, elapsed_s=1.0, threshold=0.10)
    assert watchdog.cpu_busy(before, {1: 10.5, 2: 5.0}, elapsed_s=1.0, threshold=0.10)
    assert not watchdog.cpu_busy(before, after, elapsed_s=0.0, threshold=0.10)


def test_cpu_seconds_sees_this_process_tree_but_not_itself() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        table = watchdog.cpu_seconds("tree", os.getpid())
        assert os.getpid() not in table  # the watchdog never counts itself
        assert child.pid in table
        assert child.pid in watchdog.cpu_seconds("all", os.getpid())
    finally:
        child.kill()
        child.wait()


def test_newest_mtime_skips_excluded_directories(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    os.utime(tmp_path / "a.py", (1000, 1000))
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("y")  # recent, but excluded
    assert watchdog.newest_mtime(str(tmp_path)) == 1000
    assert watchdog.newest_mtime(str(tmp_path / "missing")) is None


def test_write_marker_is_atomic_json(tmp_path: Path) -> None:
    marker = tmp_path / "nbgpu-idle"
    watchdog.write_marker(str(marker), {"reason": "idle", "idle_s": 3})
    assert json.loads(marker.read_text()) == {"reason": "idle", "idle_s": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["nbgpu-idle"]


# ---------------------------------------------------------------------------
# as a process


class FakeMarimo:
    """A stand-in for the marimo server: a process group whose leader writes
    its pid file, optionally burns CPU for a while, then sleeps."""

    def __init__(self, tmp: Path, busy_s: float = 0.0) -> None:
        self.pidfile = tmp / "marimo.pid"
        code = (
            "import os, sys, time\n"
            "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
            "end = time.monotonic() + float(sys.argv[2])\n"
            "while time.monotonic() < end:\n"
            "    pass\n"
            "time.sleep(120)\n"
        )
        self.proc = subprocess.Popen([sys.executable, "-c", code, str(self.pidfile), str(busy_s)], start_new_session=True)

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait()


def run_watchdog(
    tmp: Path, fake: FakeMarimo, idle_timeout: float, interval: float = 0.2, extra: list[str] | None = None
) -> subprocess.Popen[str]:
    project = tmp / "project"
    project.mkdir(exist_ok=True)
    return subprocess.Popen(
        [
            sys.executable, str(SCRIPT),
            "--project", str(project),
            "--marker", str(tmp / "nbgpu-idle"),
            "--marimo-pidfile", str(fake.pidfile),
            "--idle-timeout", str(idle_timeout),
            "--interval", str(interval),
            "--cpu-threshold", "0.10",
            "--kill-grace", "1",
            "--scope", "tree",
            *(extra or []),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@pytest.fixture
def fake_marimo(tmp_path: Path):
    made: list[FakeMarimo] = []

    def make(busy_s: float = 0.0) -> FakeMarimo:
        fake = FakeMarimo(tmp_path, busy_s)
        made.append(fake)
        return fake

    yield make
    for fake in made:
        fake.stop()


def test_idle_session_is_stopped_and_marked(tmp_path: Path, fake_marimo) -> None:
    fake = fake_marimo()
    started = time.monotonic()
    wd = run_watchdog(tmp_path, fake, idle_timeout=1.0)
    out, _ = wd.communicate(timeout=30)
    assert wd.returncode == 0, out
    assert time.monotonic() - started >= 1.0
    info = json.loads((tmp_path / "nbgpu-idle").read_text())
    assert info["reason"] == "idle" and info["idle_s"] >= 1.0
    assert fake.proc.wait(timeout=10) != 0  # marimo's group was terminated


def test_file_changes_keep_the_session_alive(tmp_path: Path, fake_marimo) -> None:
    fake = fake_marimo()
    wd = run_watchdog(tmp_path, fake, idle_timeout=1.5)
    project = tmp_path / "project"
    deadline = time.monotonic() + 4.0
    try:
        while time.monotonic() < deadline:
            (project / "notebook.py").write_text(str(time.time()))  # the user keeps editing
            time.sleep(0.3)
            assert wd.poll() is None, "stopped although files kept changing"
        assert not (tmp_path / "nbgpu-idle").exists()
        # Edits stop: it is stopped after the timeout.
        wd.wait(timeout=30)
        assert (tmp_path / "nbgpu-idle").exists()
    finally:
        if wd.poll() is None:
            wd.kill()
            wd.wait()


def test_the_activity_file_keeps_the_session_alive(tmp_path: Path, fake_marimo) -> None:
    fake = fake_marimo()
    activity = tmp_path / "nbgpu-activity"
    wd = run_watchdog(tmp_path, fake, idle_timeout=1.5, extra=["--activity-file", str(activity)])
    deadline = time.monotonic() + 4.0
    try:
        while time.monotonic() < deadline:
            activity.touch()  # `nbgpu exec` keeps running code in the kernel
            time.sleep(0.3)
            assert wd.poll() is None, "stopped although exec kept touching the activity file"
        assert not (tmp_path / "nbgpu-idle").exists()
        wd.wait(timeout=30)  # exec stops: idle after the timeout
    finally:
        if wd.poll() is None:
            wd.kill()
            wd.wait()
    assert (tmp_path / "nbgpu-idle").exists()


def test_file_mtime_of_a_missing_file_is_none(tmp_path: Path) -> None:
    assert watchdog.file_mtime(str(tmp_path / "missing")) is None
    assert watchdog.file_mtime(None) is None


def test_cpu_use_keeps_the_session_alive(tmp_path: Path, fake_marimo) -> None:
    fake = fake_marimo(busy_s=3.5)  # "a cell is running" for 3.5s
    started = time.monotonic()
    wd = run_watchdog(tmp_path, fake, idle_timeout=1.0)
    wd.wait(timeout=30)
    elapsed = time.monotonic() - started
    assert (tmp_path / "nbgpu-idle").exists()
    # Not stopped before the busy period plus the idle timeout (with slack
    # for one check interval and the process starting up).
    assert elapsed >= 3.5 + 1.0 - 0.5, elapsed


def test_exits_without_marker_when_marimo_exits(tmp_path: Path, fake_marimo) -> None:
    fake = fake_marimo()
    wd = run_watchdog(tmp_path, fake, idle_timeout=60)
    time.sleep(0.5)
    fake.stop()
    wd.wait(timeout=10)
    assert wd.returncode == 0
    assert not (tmp_path / "nbgpu-idle").exists()


# ---------------------------------------------------------------------------
# unsynced edits, heartbeat, marimo's busy signal


def test_constants_mirror_sync() -> None:
    from nbgpu import sync
    from nbgpu.backends.base import TEMP_PREFIX

    assert watchdog.EXCLUDED_DIRS == sync.EXCLUDED_PARTS
    assert watchdog.PULL_SUFFIXES == sync.PULL_SUFFIXES
    assert watchdog.PULL_MAX_BYTES == sync.PULL_MAX_BYTES
    assert watchdog._TEMP_PREFIX == TEMP_PREFIX


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(path: Path, files: dict[str, bytes]) -> None:
    path.write_text(json.dumps({"version": 1, "written_at": time.time(),
                                "files": {k: _sha(v) for k, v in files.items()}}))


def test_unsynced_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "sub").mkdir(parents=True)
    (root / "nb.py").write_bytes(b"a")
    (root / "sub" / "util.py").write_bytes(b"u")
    manifest = {k: _sha(v) for k, v in {"nb.py": b"a", "sub/util.py": b"u", "deleted.py": b"d"}.items()}
    cache = watchdog.HashCache()
    assert watchdog.unsynced_paths(str(root), manifest, cache) == []  # listed but missing: not an edit

    (root / "nb.py").write_bytes(b"edited in the browser")
    (root / "new.py").write_bytes(b"new cell module")
    (root / "plot.png").write_bytes(b"\x89PNG")  # never pulled: not an edit
    (root / "big.txt").write_bytes(b"x" * (watchdog.PULL_MAX_BYTES + 1))
    (root / "__marimo__").mkdir()
    (root / "__marimo__" / "session.json").write_bytes(b"{}")  # excluded
    (root / (watchdog._TEMP_PREFIX + "x.py")).write_bytes(b"temp")
    assert sorted(watchdog.unsynced_paths(str(root), manifest, cache)) == ["nb.py", "new.py"]
    # No manifest: every pullable file counts.
    assert sorted(watchdog.unsynced_paths(str(root), None, cache)) == ["nb.py", "new.py", "sub/util.py"]


def test_hash_cache_reuses_unchanged_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "a.py"
    f.write_bytes(b"one")
    cache = watchdog.HashCache()
    assert cache.digest(str(f), os.lstat(f)) == _sha(b"one")
    opened: list[str] = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: (opened.append(str(p)), real_open(p, *a, **k))[1])
    assert cache.digest(str(f), os.lstat(f)) == _sha(b"one")
    assert opened == []


def test_internal_processes_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setattr(watchdog, "is_internal", lambda pid: pid == child.pid)
        assert child.pid not in watchdog.cpu_seconds("tree", os.getpid())
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs /proc")
def test_is_internal_reads_the_environment() -> None:
    env = {**os.environ, "NBGPU_INTERNAL": "1"}
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env=env)
    plain = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.2)
        assert watchdog.is_internal(child.pid) and not watchdog.is_internal(plain.pid)
    finally:
        for p in (child, plain):
            p.kill()
            p.wait()


def test_unsynced_edit_delays_the_stop_by_the_grace(tmp_path: Path, fake_marimo) -> None:
    fake = fake_marimo()
    project = tmp_path / "project"
    project.mkdir()
    (project / "nb.py").write_bytes(b"browser edit")
    old = time.time() - 600
    os.utime(project / "nb.py", (old, old))  # edited long ago: idle now
    manifest = tmp_path / "nbgpu-synced.json"
    write_manifest(manifest, {"nb.py": b"what the laptop has"})
    waiting = tmp_path / "nbgpu-idle-unsynced"
    beat = tmp_path / "beat"
    started = time.monotonic()
    wd = run_watchdog(tmp_path, fake, idle_timeout=1.0, extra=[
        "--manifest", str(manifest), "--unsynced-marker", str(waiting), "--unsynced-grace", "2.5",
        "--heartbeat", str(beat),
    ])
    try:
        wait_for_path(waiting, 10)
        assert json.loads(waiting.read_text())["unsynced"] == ["nb.py"]
        assert json.loads(beat.read_text())["pid"] == wd.pid  # alive and saying so
        assert not (tmp_path / "nbgpu-idle").exists() and wd.poll() is None
        wd.wait(timeout=30)
    finally:
        if wd.poll() is None:
            wd.kill()
            wd.wait()
    assert time.monotonic() - started >= 1.0 + 2.5
    info = json.loads((tmp_path / "nbgpu-idle").read_text())
    assert info["unsynced"] == ["nb.py"]  # recorded for the user
    assert not waiting.exists() and not beat.exists()


def test_a_sync_during_the_grace_lets_it_stop(tmp_path: Path, fake_marimo) -> None:
    fake = fake_marimo()
    project = tmp_path / "project"
    project.mkdir()
    (project / "nb.py").write_bytes(b"browser edit")
    old = time.time() - 600
    os.utime(project / "nb.py", (old, old))
    manifest = tmp_path / "nbgpu-synced.json"
    write_manifest(manifest, {"nb.py": b"stale"})
    waiting = tmp_path / "nbgpu-idle-unsynced"
    wd = run_watchdog(tmp_path, fake, idle_timeout=0.5, extra=[
        "--manifest", str(manifest), "--unsynced-marker", str(waiting), "--unsynced-grace", "60",
    ])
    try:
        wait_for_path(waiting, 10)
        write_manifest(manifest, {"nb.py": b"browser edit"})  # the laptop pulled it
        wd.wait(timeout=10)
    finally:
        if wd.poll() is None:
            wd.kill()
            wd.wait()
    info = json.loads((tmp_path / "nbgpu-idle").read_text())
    assert "unsynced" not in info and not waiting.exists()


def wait_for_path(path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.05)


class FakeMarimoServer:
    """Answers /api/sessions and /api/kernel/status like marimo 0.25.2."""

    def __init__(self, token: str) -> None:
        self.state = "running"
        self.seen_auth: list[str | None] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                outer.seen_auth.append(self.headers.get("Authorization"))
                if self.headers.get("Authorization") != f"Bearer {token}":
                    self.send_response(401)
                    self.end_headers()
                    return
                if self.path == "/api/sessions":
                    body = {"s1": {"session_id": "stable", "filename": "nb.py", "path": "nb.py"}}
                elif self.path == "/api/kernel/status" and self.headers.get("Marimo-Session-Id") == "s1":
                    body = {"state": outer.state}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:  # type: ignore[no-untyped-def]
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_marimo_busy_asks_the_server(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("secret-token")
    server = FakeMarimoServer("secret-token")
    try:
        assert watchdog.marimo_busy(server.url, str(token_file)) is True
        server.state = "idle"
        assert watchdog.marimo_busy(server.url, str(token_file)) is False
        assert "Bearer secret-token" in server.seen_auth
        token_file.write_text("wrong")
        assert watchdog.marimo_busy(server.url, str(token_file)) is None  # cannot ask: unknown
    finally:
        server.close()
    assert watchdog.marimo_busy(server.url, str(token_file)) is None


def test_a_running_cell_keeps_the_session_alive(tmp_path: Path, fake_marimo) -> None:
    fake = fake_marimo()  # uses no CPU at all
    token_file = tmp_path / "token"
    token_file.write_text("tok")
    server = FakeMarimoServer("tok")
    try:
        wd = run_watchdog(tmp_path, fake, idle_timeout=0.5, extra=[
            "--marimo-url", server.url, "--token-file", str(token_file),
        ])
        time.sleep(2.0)
        assert wd.poll() is None, "stopped while marimo reported a running cell"
        server.state = "idle"
        wd.wait(timeout=15)
        assert (tmp_path / "nbgpu-idle").exists()
    finally:
        if wd.poll() is None:
            wd.kill()
            wd.wait()
        server.close()


def test_ignored_outputs_written_by_the_notebook_are_not_unsynced(tmp_path: Path) -> None:
    """End to end with the real sync engine and local backend: a gitignored
    output the notebook writes in the runtime never counts as an unsynced
    edit (it is not pulled, so waiting for it would only keep billing)."""
    import posixpath

    from nbgpu.backends.base import RuntimeRequest
    from nbgpu.backends.local import LocalBackend
    from nbgpu.spec import ComputeSpec, NotebookSpec
    from nbgpu.sync import SyncEngine

    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitignore").write_text("out/\n")
    (project / "nb.py").write_text("print(1)\n")
    backend = LocalBackend(base_dir=tmp_path / "runtimes")
    spec = NotebookSpec(path=project / "nb.py", requires_python=None, dependencies=(),
                        compute=ComputeSpec(provider="local"))
    handle = backend.create(RuntimeRequest(spec=spec, tags={}))
    try:
        manifest = posixpath.join(posixpath.dirname(handle.remote_root), "nbgpu-synced.json")
        engine = SyncEngine(project, backend, handle, tmp_path / "state.json", manifest_path=manifest)
        engine.initial_push()
        backend.write_file(handle, "out/metrics.json", b"{}")
        backend.write_file(handle, "nb.py", b"print(2)\n")  # a real browser edit, not yet pulled
        assert watchdog.unsynced_paths(handle.remote_root, watchdog.read_manifest(manifest),
                                       watchdog.HashCache()) == ["nb.py", "out/metrics.json"]
        engine.poll_once()
        assert watchdog.unsynced_paths(handle.remote_root, watchdog.read_manifest(manifest),
                                       watchdog.HashCache()) == []
    finally:
        backend.terminate(handle)
