"""End-to-end: idle watchdog, orphan recovery and signal handling on the local backend."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import pytest
from click.testing import CliRunner

from nbgpu import cli, lifecycle, watchdog
from nbgpu.backends.local import LocalBackend
from nbgpu.session import TAG_OWNER, TAG_SESSION, TAG_TOOL, SessionStore
from tests.e2e.conftest import MarimoUI, marimo_pids_on_port, pid_alive, server_answers, wait_for
from tests.fakes import fake_request

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300)]


def watchdog_pids(handle) -> set[int]:  # type: ignore[no-untyped-def]
    import psutil

    marker = lifecycle.idle_marker_path(handle)
    pids = set()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
        except Exception:
            continue
        if any(part.endswith(lifecycle.WATCHDOG_NAME) for part in cmd) and marker in cmd:
            pids.add(proc.info["pid"])
    return pids


@pytest.fixture
def fast_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watchdog, "CHECK_INTERVAL_S", 0.3)
    monkeypatch.setattr(watchdog, "KILL_GRACE_S", 2.0)


def test_idle_session_is_stopped_by_the_watchdog(fast_watchdog, start_session, local_backend, store, notebook: Path) -> None:
    session = start_session(notebook, overrides={"idle-timeout": 2})
    handle = local_backend.get(session.runtime_id)
    assert handle is not None
    port = urllib.parse.urlsplit(session.url).port
    assert port is not None
    marimo = marimo_pids_on_port(port)
    assert marimo and watchdog_pids(handle)

    [entry] = lifecycle.status(backends={"local": local_backend}, store=store)
    assert entry.state == lifecycle.STATE_RUNNING

    # An open tab alone is not activity: nothing touches files or uses CPU.
    wait_for(lambda: lifecycle.read_idle_marker(local_backend, handle) is not None, 30, what="the idle marker")
    wait_for(lambda: not any(pid_alive(p) for p in marimo), 20, what="marimo to be stopped")
    wait_for(lambda: not watchdog_pids(handle), 20, what="the watchdog to exit")
    assert not server_answers(session.url)

    [entry] = lifecycle.status(backends={"local": local_backend}, store=store)
    assert entry.state == lifecycle.STATE_IDLE_STOPPED and entry.running  # the runtime itself is still up

    lifecycle.stop(session, backend=local_backend, store=store)
    assert local_backend.get(session.runtime_id) is None


def test_terminate_also_ends_the_watchdog(start_session, local_backend, store, notebook: Path) -> None:
    session = start_session(notebook)
    handle = local_backend.get(session.runtime_id)
    pids = watchdog_pids(handle)
    assert pids, "the watchdog should run next to marimo"
    # It lives outside the synced project and never reaches the local side.
    assert local_backend.read_file(handle, lifecycle.runtime_path(handle, lifecycle.WATCHDOG_NAME)) is not None
    assert lifecycle.WATCHDOG_NAME not in local_backend.hash_files(handle)
    lifecycle.stop(session, backend=local_backend, store=store)
    wait_for(lambda: not any(pid_alive(p) for p in pids), 20, what="the watchdog to be killed")


def test_cli_sync_loop_stops_an_idle_session(fast_watchdog, project: Path, notebook: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "SYNC_INTERVAL_S", 0.2)
    monkeypatch.chdir(project)
    result = CliRunner().invoke(
        cli.main, ["start", "notebook.py", "--backend", "local", "--idle-timeout", "2"], catch_exceptions=False
    )
    out = result.output
    try:
        assert result.exit_code == 0, out
        assert "stopped for idleness" in out
        store = SessionStore()
        assert store.list() == []
        assert LocalBackend().find({TAG_TOOL: "1"}) == []
    finally:
        backend = LocalBackend()
        for handle in backend.find({}):
            backend.terminate(handle)


def nbgpu(*args: str, cwd: Path, timeout_s: float = 180) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("NBGPU_BACKEND", None)
    return subprocess.run(
        [sys.executable, "-m", "nbgpu.cli", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s
    )


def popen_nbgpu(*args: str, cwd: Path) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.pop("NBGPU_BACKEND", None)
    return subprocess.Popen(
        [sys.executable, "-m", "nbgpu.cli", *args], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


@pytest.fixture
def cli_backend():
    backend = LocalBackend()  # what the CLI uses: NBGPU_STATE_DIR/local
    yield backend
    for handle in backend.find({}):
        backend.terminate(handle)


def test_detach_warns_about_unsynced_browser_edits(project: Path, cli_backend) -> None:
    proc = nbgpu("start", "notebook.py", "--backend", "local", "--detach", "--json", cwd=project)
    assert proc.returncode == 0, proc.stderr
    json.loads(proc.stdout)  # stdout stays one JSON document
    assert "nbgpu sync" in proc.stderr and "idle" in proc.stderr and "max runtime" in proc.stderr
    assert "unsynced" in proc.stderr and "can be lost" in proc.stderr
    assert nbgpu("stop", "notebook.py", "--force", cwd=project).returncode == 0


def test_orphans_are_listed_and_stopped(project: Path, cli_backend, monkeypatch) -> None:
    me = lifecycle.owner_id(SessionStore())
    orphan = cli_backend.create(fake_request(tags={TAG_TOOL: "1", TAG_OWNER: me, TAG_SESSION: "gone1"}, compute=None))
    # Not ours: never listed or touched.
    foreign = cli_backend.create(fake_request(tags={"other": "1"}))
    # Another laptop's / teammate's session in the same workspace: listed apart, never touched.
    teammate = cli_backend.create(fake_request(tags={TAG_TOOL: "1", TAG_OWNER: "teammate", TAG_SESSION: "t1"}))

    # No records and no --backend / $NBGPU_BACKEND: the backend is not asked.
    status = json.loads(nbgpu("status", "--json", cwd=project).stdout)
    assert status == {"sessions": [], "orphans": [], "other_owners": []}

    status = json.loads(nbgpu("status", "--json", "--backend", "local", cwd=project).stdout)
    assert [o["runtime_id"] for o in status["orphans"]] == [orphan.runtime_id]
    assert [o["runtime_id"] for o in status["other_owners"]] == [teammate.runtime_id]
    human = nbgpu("status", "--backend", "local", cwd=project)
    assert "untracked" in human.stdout and "still billing" in human.stdout
    assert "other owners" in human.stdout and teammate.runtime_id in human.stdout

    # --json without --yes is refused; nothing is terminated.
    refused = nbgpu("stop", "--orphans", "--backend", "local", "--json", cwd=project)
    assert refused.returncode != 0 and "--yes" in refused.stderr
    # Without --yes the user is asked; answering no keeps it.
    declined = subprocess.run(
        [sys.executable, "-m", "nbgpu.cli", "stop", "--orphans", "--backend", "local"],
        cwd=project, input="n\n", capture_output=True, text=True, timeout=60,
    )
    assert declined.returncode != 0 and orphan.runtime_id in declined.stdout
    # Just created: a concurrent start may own it, so it is skipped for now.
    young = json.loads(nbgpu("stop", "--orphans", "--backend", "local", "--json", "--yes", cwd=project).stdout)
    assert young["skipped"] == [orphan.runtime_id] and young["terminated"] == []
    assert cli_backend.get(orphan.runtime_id) is not None

    monkeypatch.setattr(lifecycle, "ORPHAN_GRACE_S", 0.0)
    result = CliRunner().invoke(cli.main, ["stop", "--orphans", "--backend", "local", "--yes"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert f"terminated {orphan.runtime_id}" in result.output
    assert cli_backend.get(orphan.runtime_id) is None
    assert cli_backend.get(foreign.runtime_id) is not None
    assert cli_backend.get(teammate.runtime_id) is not None
    assert teammate.runtime_id not in result.output


def test_sigterm_during_start_leaves_nothing(project: Path, cli_backend) -> None:
    proc = popen_nbgpu("start", "notebook.py", "--backend", "local", "--detach", cwd=project)
    try:
        wait_for(lambda: cli_backend.find({TAG_TOOL: "1"}), 60, interval_s=0.02, what="the runtime to be created")
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode == 0:
        pytest.skip("start finished before the signal arrived")
    assert proc.returncode == 128 + signal.SIGTERM, err
    assert "Traceback" not in err
    assert cli_backend.find({}) == []
    assert SessionStore().list() == []


def test_sigterm_ends_the_sync_loop_with_a_final_pull(project: Path, cli_backend) -> None:
    proc = popen_nbgpu("start", "notebook.py", "--backend", "local", cwd=project)
    try:
        wait_for(lambda: SessionStore().list() and not lifecycle.is_starting(SessionStore().list()[0]),
                 120, what="the session to start")
        time.sleep(1.5)  # the loop is running
        proc.send_signal(signal.SIGHUP)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert "final pull" in out and "still running" in out, out + err
    assert "nbgpu sync" in out and "idle" in out  # says how browser edits get saved, and what loses them
    [session] = SessionStore().list()
    assert cli_backend.get(session.runtime_id) is not None  # the loop never stops the runtime
    assert nbgpu("stop", "notebook.py", "--force", cwd=project).returncode == 0
    assert cli_backend.find({}) == []


# ---------------------------------------------------------------------------
# N2: unsynced browser edits; N3: running cells; dead watchdog


def test_a_sleeping_cell_keeps_the_session_alive(fast_watchdog, start_session, local_backend, store, notebook, tmp_path) -> None:
    """A cell waiting on something (no CPU, no file changes) is activity: marimo
    reports it as running through its authenticated status API."""
    session = start_session(notebook, overrides={"idle-timeout": 6})
    handle = local_backend.get(session.runtime_id)
    ui = MarimoUI(session.url, session.token, tmp_path / "pair-token")
    ui.connect()
    # `marimo pair` returns only when the cell has finished: run it aside.
    runner = threading.Thread(target=ui.create_cell, args=("import time as _t\n_t.sleep(14)\nslept = True",))
    try:
        runner.start()
        wait_for(lambda: "slept = True" in local_backend.read_file(handle, "notebook.py").decode(), 30,
                 what="the cell to be saved")
        lifecycle.sync_once(session, backend=local_backend, store=store)  # pull the new cell: nothing unsynced
        # Past the idle timeout (6s; 2s raced `marimo pair` on a loaded machine) while the cell sleeps.
        time.sleep(8)
        assert runner.is_alive(), "the cell should still be sleeping"
        assert lifecycle.read_idle_marker(local_backend, handle) is None
        assert lifecycle.read_unsynced_marker(local_backend, handle) is None
        assert server_answers(session.url)
        # Once the cell is done, the session idles out.
        runner.join(30)
        wait_for(lambda: lifecycle.read_idle_marker(local_backend, handle) is not None, 40, what="the idle stop")
    finally:
        runner.join(30)
        ui.close()
    lifecycle.stop(session, backend=local_backend, store=store)


def test_unsynced_browser_edit_delays_the_idle_stop_and_is_kept(
    fast_watchdog, start_session, local_backend, store, notebook, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(watchdog, "UNSYNCED_GRACE_S", 6.0)
    session = start_session(notebook, overrides={"idle-timeout": 6})
    handle = local_backend.get(session.runtime_id)
    ui = MarimoUI(session.url, session.token, tmp_path / "pair-token")
    ui.connect()
    try:
        ui.create_cell("w = 7  # unsynced-browser-edit")
    finally:
        ui.close()
    # Nothing pulls (detached / loop stopped): the watchdog waits instead of stopping.
    waiting = wait_for(lambda: lifecycle.read_unsynced_marker(local_backend, handle), 30, what="idle-unsynced")
    assert waiting["unsynced"] == ["notebook.py"]  # type: ignore[index]
    assert lifecycle.read_idle_marker(local_backend, handle) is None and server_answers(session.url)
    [entry] = lifecycle.status(backends={"local": local_backend}, store=store)
    assert entry.state == lifecycle.STATE_IDLE_UNSYNCED and entry.unsynced == ("notebook.py",)
    # Billing stays bounded: it stops after the grace and records what was unsynced.
    idle = wait_for(lambda: lifecycle.read_idle_marker(local_backend, handle), 30, what="the idle stop")
    assert idle["unsynced"] == ["notebook.py"]  # type: ignore[index]
    # The runtime is still there, so stop pulls the edit.
    lifecycle.stop(session, backend=local_backend, store=store)
    assert "unsynced-browser-edit" in notebook.read_text()


def test_a_sync_during_the_grace_lets_the_idle_stop_happen(fast_watchdog, start_session, local_backend, store, notebook, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(watchdog, "UNSYNCED_GRACE_S", 120.0)
    session = start_session(notebook, overrides={"idle-timeout": 6})
    handle = local_backend.get(session.runtime_id)
    ui = MarimoUI(session.url, session.token, tmp_path / "pair-token")
    ui.connect()
    try:
        ui.create_cell("v = 8  # pulled-later")
    finally:
        ui.close()
    wait_for(lambda: lifecycle.read_unsynced_marker(local_backend, handle), 30, what="idle-unsynced")
    lifecycle.sync_once(session, backend=local_backend, store=store)  # `nbgpu sync`
    assert "pulled-later" in notebook.read_text()
    idle = wait_for(lambda: lifecycle.read_idle_marker(local_backend, handle), 20, what="the idle stop")
    assert "unsynced" not in idle  # type: ignore[operator]
    lifecycle.stop(session, backend=local_backend, store=store)


def test_stop_and_status_warn_when_the_runtime_is_gone(project: Path, cli_backend) -> None:
    started = nbgpu("start", "notebook.py", "--backend", "local", "--detach", "--json", cwd=project)
    assert started.returncode == 0, started.stderr
    runtime_id = json.loads(started.stdout)["runtime_id"]
    assert nbgpu("sync", "notebook.py", cwd=project).returncode == 0
    cli_backend.terminate(cli_backend.get(runtime_id))  # e.g. the provider ended it after an idle stop
    status = nbgpu("status", cwd=project)
    assert "could not be recovered" in status.stdout and "last successful sync" in status.stdout, status.stdout
    [entry] = json.loads(nbgpu("status", "--json", cwd=project).stdout)["sessions"]
    assert entry["state"] == "not-running" and entry["last_sync_at"] and entry["warning"]
    stopped = nbgpu("stop", "notebook.py", cwd=project)
    assert stopped.returncode == 0, stopped.stderr
    assert "could not be recovered" in stopped.stderr and "already gone" in stopped.stdout
    assert "the runtime is terminated" not in stopped.stdout
    assert SessionStore().list() == []


def test_the_sync_loop_restarts_a_dead_watchdog(fast_watchdog, project: Path, notebook: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "SYNC_INTERVAL_S", 0.2)
    monkeypatch.setattr(cli, "WATCHDOG_STALE_SLACK_S", 1.0)
    monkeypatch.chdir(project)
    backend = LocalBackend()
    seen: dict[str, set[int]] = {"killed": set(), "restarted": set()}

    def killer() -> None:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not seen["restarted"]:
            records = SessionStore().list()
            if not records or lifecycle.is_starting(records[0]):
                time.sleep(0.05)  # kill it once the start is complete and the loop runs
                continue
            handles = backend.find({TAG_TOOL: "1"})
            pids = set().union(*(watchdog_pids(h) for h in handles)) if handles else set()
            if pids and not seen["killed"]:
                for pid in pids:
                    os.kill(pid, signal.SIGKILL)  # the watchdog dies (OOM killer, bug, ...)
                seen["killed"] = pids
            elif seen["killed"] and pids - seen["killed"]:
                seen["restarted"] = pids - seen["killed"]
            time.sleep(0.05)

    thread = threading.Thread(target=killer, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(
            cli.main, ["start", "notebook.py", "--backend", "local", "--idle-timeout", "4"], catch_exceptions=False
        )
        thread.join(5)
        assert result.exit_code == 0, result.output
        assert seen["killed"] and seen["restarted"], (seen, result.output)
        assert "restarted it" in result.output
        assert "stopped for idleness" in result.output  # the restarted watchdog still stops billing
        assert backend.find({TAG_TOOL: "1"}) == []
    finally:
        for handle in backend.find({}):
            backend.terminate(handle)


def test_cli_start_serves_a_local_proxy_and_exits_cleanly_after_an_external_stop(project: Path, cli_backend) -> None:
    """`start` (not detached) prints a localhost link whose assets come from
    the local marimo; `nbgpu stop` elsewhere ends the loop without an error."""
    from tests.e2e.conftest import http_get

    proc = popen_nbgpu("start", "notebook.py", "--backend", "local", "--json", cwd=project)
    try:
        assert proc.stdout is not None
        lines: list[str] = []
        while not (lines and lines[-1].startswith("}")):
            line = proc.stdout.readline()
            assert line, f"start ended early: {proc.stderr.read() if proc.stderr else ''}"
            lines.append(line)
        started = json.loads("".join(lines))
        local = started["local_url"]
        assert local and local.startswith("http://127.0.0.1:") and "access_token=" in local
        page = http_get(local, follow_redirects=True)
        assert page.status == 200 and b"data-version" in page.body
        asset = re.search(rb'(?:src|href)="\./(assets/[^"]+\.js)"', page.body)
        assert asset, "the page references its assets"
        base = local.split("/?", 1)[0]
        served = http_get(f"{base}/{asset.group(1).decode()}")
        assert served.status == 200 and "immutable" in served.headers.get("cache-control", "")

        assert nbgpu("stop", "notebook.py", cwd=project).returncode == 0
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
        assert "was stopped" in err and "could not be recovered" not in err
    finally:
        if proc.poll() is None:
            proc.kill()
