"""End-to-end: real marimo sessions on the local backend.

What these tests rely on about marimo 0.25.2 (confirmed against its source
and a running server):

- GET /health and /healthz answer 200 {"status": "healthy"} without auth:
  the readiness probe.
- With a token, GET / without credentials is a 303 to /auth/login?next=/;
  /api/status without credentials is 401. `?access_token=<token>` (or an
  `Authorization: Bearer <token>` header) authenticates and sets a session
  cookie; on / it then 303-redirects to strip the token from the URL.
- The browser opens a notebook session over the websocket /ws with
  `session_id=<client id>` (+ the token). Headless `marimo edit` has no
  session until a client connects.
- `marimo pair` code-mode edits (create_cell/edit_cell) are saved to the
  notebook file by the server immediately; this is the UI path to disk.
- `--watch` reloads the notebook in open sessions when the file changes on
  disk (observable through `marimo pair`'s view of the cells).
- marimo writes a session cache to __marimo__/session/<notebook>.json in the
  project directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import pytest

from nbgpu import lifecycle
from nbgpu.lifecycle import MARIMO_PORT, StartError, StopRefused
from nbgpu.session import TAG_NOTEBOOK, TAG_SESSION, TAG_TOOL, notebook_tag
from tests.e2e.conftest import (
    CHATTY_NOTEBOOK_TEXT,
    FAILING_NOTEBOOK_TEXT,
    MarimoUI,
    http_get,
    marimo_pids_on_port,
    pid_alive,
    server_answers,
    wait_for,
)

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300)]


def remote_bytes(backend, session, rel: str) -> bytes | None:
    handle = backend.get(session.runtime_id)
    assert handle is not None, "runtime is not running"
    return backend.read_file(handle, rel)


def write_remote(backend, session, rel: str, data: bytes) -> None:
    handle = backend.get(session.runtime_id)
    assert handle is not None
    backend.write_file(handle, rel, data)


def sync(session, backend, store):
    return lifecycle.sync_once(session, backend=backend, store=store)


# --------------------------------------------------------------------------
# start


def test_start_returns_a_running_session(session, local_backend, store, notebook: Path, project: Path) -> None:
    assert session.notebook == notebook.resolve()
    assert session.project_root == project.resolve()
    assert session.backend == "local"
    assert session.gpu is None
    assert len(session.session_id) == 8 and all(c in "0123456789abcdef" for c in session.session_id)
    assert session.token and len(session.token) >= 32

    handle = local_backend.get(session.runtime_id)
    assert handle is not None
    assert handle.remote_root == session.remote_root
    assert handle.tags[TAG_TOOL] == "1"
    assert handle.tags[TAG_SESSION] == session.session_id
    assert handle.tags[TAG_NOTEBOOK] == notebook_tag(notebook.resolve())
    assert [h.runtime_id for h in local_backend.find({TAG_SESSION: session.session_id})] == [session.runtime_id]

    # URL: the exposed marimo port, no trailing slash, no token in it.
    assert session.url == local_backend.expose(handle, MARIMO_PORT)
    assert not session.url.endswith("/")
    assert "access_token" not in session.url and session.token not in session.url

    # The record is saved.
    loaded = store.load(session.session_id)
    assert loaded is not None
    assert loaded.url == session.url and loaded.token == session.token and loaded.runtime_id == session.runtime_id

    # Initial push: project files are on the runtime, excluded ones are not.
    remote = local_backend.hash_files(handle)
    assert {"notebook.py", "helper.py", "pyproject.toml", "data/input.csv"} <= set(remote)
    assert not any(p.startswith((".venv/", "__pycache__/")) for p in remote)
    assert remote_bytes(local_backend, session, "notebook.py") == notebook.read_bytes()

    # The token file lives outside the project directory on both sides.
    for rel in remote:
        data = local_backend.read_file(handle, rel) or b""
        assert session.token.encode() not in data, f"token leaked into remote {rel}"
    for path in project.rglob("*"):
        if path.is_file():
            assert session.token.encode() not in path.read_bytes(), f"token leaked into local {path}"


def test_health_is_open_and_the_ui_needs_the_token(session) -> None:
    # Readiness endpoint: no auth.
    health = http_get(session.url + "/health")
    assert health.status == 200
    assert json.loads(health.body) == {"status": "healthy"}

    # No token: the page redirects to the login form, the API is rejected.
    root = http_get(session.url + "/")
    assert root.status in (302, 303, 307)
    assert root.headers["location"].startswith("/auth/login")
    assert http_get(session.url + "/api/status").status == 401

    # Wrong token: still rejected.
    wrong = session.url + "/api/status?" + urllib.parse.urlencode({"access_token": "not-the-token"})
    assert http_get(wrong).status == 401

    # Right token as a query parameter: accepted.
    api = http_get(session.url + "/api/status?" + urllib.parse.urlencode({"access_token": session.token}))
    assert api.status == 200
    assert json.loads(api.body)["mode"] == "edit"

    # Right token as a bearer header: accepted.
    assert http_get(session.url + "/api/status", headers={"Authorization": f"Bearer {session.token}"}).status == 200

    # The URL users open logs them in and lands on the editor.
    with_token = session.url_with_token()
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(with_token).query)["access_token"] == [session.token]
    page = http_get(with_token, follow_redirects=True)
    assert page.status == 200
    assert b"marimo" in page.body.lower()


def test_status_reports_running_then_gone(session, local_backend, store) -> None:
    [entry] = lifecycle.status(backends={"local": local_backend}, store=store)
    assert entry.session.session_id == session.session_id
    assert entry.running is True
    assert entry.uptime_s >= 0
    assert entry.conflicts == ()

    # Runtime disappears behind our back: reported as not running, record kept.
    handle = local_backend.get(session.runtime_id)
    local_backend.terminate(handle)
    [entry] = lifecycle.status(backends={"local": local_backend}, store=store)
    assert entry.running is False
    assert store.load(session.session_id) is not None


# --------------------------------------------------------------------------
# sync


def test_local_edit_reaches_the_runtime_and_marimo_reloads(
    session, local_backend, store, notebook: Path, marimo_ui: MarimoUI, tmp_path: Path
) -> None:
    assert any("x = 1" in code for code in marimo_ui.cell_codes())

    edited = notebook.read_text().replace("x = 1", "x = 1001  # edited-locally")
    notebook.write_text(edited)
    report = sync(session, local_backend, store)

    assert report.pushed == ["notebook.py"]
    # (marimo's own __marimo__/session cache may be pulled too; see module doc.)
    assert "notebook.py" not in report.pulled and report.conflicts == []
    assert remote_bytes(local_backend, session, "notebook.py") == edited.encode()
    assert notebook.read_text() == edited
    # The engine's base is persisted in the store's directory (the `store`
    # fixture uses tmp_path / "state").
    assert (tmp_path / "state" / f"{session.session_id}.sync.json").exists()

    # marimo --watch picked up the file.
    wait_for(
        lambda: any("edited-locally" in code for code in marimo_ui.cell_codes()),
        timeout_s=30,
        interval_s=0.5,
        what="marimo to reload the pushed notebook",
    )

    # A second pass with nothing changed moves nothing for the notebook.
    again = sync(session, local_backend, store)
    assert "notebook.py" not in again.pushed + again.pulled + again.merged
    assert again.conflicts == []


def test_new_local_file_is_pushed(session, local_backend, store, project: Path) -> None:
    (project / "pkg").mkdir()
    (project / "pkg" / "util.py").write_text("def f():\n    return 1\n")
    report = sync(session, local_backend, store)
    assert "pkg/util.py" in report.pushed
    assert remote_bytes(local_backend, session, "pkg/util.py") == b"def f():\n    return 1\n"


def test_ui_edit_reaches_the_local_file(session, local_backend, store, notebook: Path, marimo_ui: MarimoUI) -> None:
    """The core requirement: an edit made in the marimo UI (here through
    `marimo pair` code mode, which the server saves to the notebook file)
    is pulled into the local file."""
    original = notebook.read_text()
    marimo_ui.create_cell("z = 42  # added-in-the-ui")

    def pulled() -> bool:
        sync(session, local_backend, store)
        return "added-in-the-ui" in notebook.read_text()

    wait_for(pulled, timeout_s=60, interval_s=0.5, what="the UI edit to be pulled into the local file")

    local = notebook.read_bytes()
    assert local == remote_bytes(local_backend, session, "notebook.py")
    # The rest of the notebook (header included) is still there.
    text = local.decode()
    assert "[tool.nbgpu]" in text and 'print("hello-from-notebook")' in text
    assert not (notebook.parent / "notebook.remote.py").exists()
    assert original != text

    # Settled: nothing left to move for the notebook.
    report = sync(session, local_backend, store)
    assert "notebook.py" not in report.pulled + report.pushed + report.conflicts


def test_remote_file_change_is_pulled(session, local_backend, store, project: Path) -> None:
    """A file changed on the runtime by something other than the UI
    (e.g. code in a cell) is pulled; new remote files obey pull_allowed."""
    write_remote(local_backend, session, "helper.py", b"VALUE = 2  # changed-remotely\n")
    write_remote(local_backend, session, "results/summary.md", b"# results\n")
    write_remote(local_backend, session, "results/weights.bin", b"\x00" * 1024)
    report = sync(session, local_backend, store)

    assert "helper.py" in report.pulled
    assert (project / "helper.py").read_text() == "VALUE = 2  # changed-remotely\n"
    assert "results/summary.md" in report.pulled
    assert (project / "results" / "summary.md").read_text() == "# results\n"
    assert not (project / "results" / "weights.bin").exists()
    assert "results/weights.bin" in report.skipped


def test_concurrent_conflicting_edits(session, local_backend, store, notebook: Path) -> None:
    original = notebook.read_text()
    local_text = original.replace("x = 1", "x = 2  # local-version")
    remote_text = original.replace("x = 1", "x = 3  # remote-version")
    notebook.write_text(local_text)
    write_remote(local_backend, session, "notebook.py", remote_text.encode())

    report = sync(session, local_backend, store)
    assert report.conflicts == ["notebook.py"]
    assert "notebook.py" not in report.pushed + report.pulled + report.merged

    # Neither side is overwritten; the remote version is kept next to the local file.
    assert notebook.read_text() == local_text
    conflict_copy = notebook.parent / "notebook.remote.py"
    assert conflict_copy.read_text() == remote_text
    assert remote_bytes(local_backend, session, "notebook.py") == remote_text.encode()
    # Conflict copies are never synced.
    assert remote_bytes(local_backend, session, "notebook.remote.py") is None

    [entry] = lifecycle.status(backends={"local": local_backend}, store=store)
    assert entry.conflicts == ("notebook.py",)

    # stop refuses while the conflict is open, and leaves everything running.
    with pytest.raises(StopRefused) as info:
        lifecycle.stop(session, backend=local_backend, store=store)
    assert "notebook.py" in str(info.value)
    assert local_backend.get(session.runtime_id) is not None
    assert store.load(session.session_id) is not None
    assert server_answers(session.url)

    # Forced: terminates, keeps the local version and the conflict copy.
    port = urllib.parse.urlsplit(session.url).port
    report = lifecycle.stop(session, backend=local_backend, store=store, force=True)
    assert "notebook.py" in report.conflicts
    assert local_backend.get(session.runtime_id) is None
    assert store.load(session.session_id) is None
    assert notebook.read_text() == local_text
    assert conflict_copy.read_text() == remote_text
    wait_for(lambda: not server_answers(session.url), 20, what="marimo to stop answering")
    assert port is not None


def test_clean_concurrent_edits_merge(session, local_backend, store, notebook: Path) -> None:
    original = notebook.read_text()
    local_text = original.replace("x = 1", "x = 10  # local-change")
    remote_text = original.replace("y = 2", "y = 20  # remote-change")
    notebook.write_text(local_text)
    write_remote(local_backend, session, "notebook.py", remote_text.encode())

    report = sync(session, local_backend, store)
    assert report.merged == ["notebook.py"] and report.conflicts == []
    merged = notebook.read_text()
    assert "local-change" in merged and "remote-change" in merged
    assert remote_bytes(local_backend, session, "notebook.py") == merged.encode()


def test_resolving_a_conflict_unblocks_stop(session, local_backend, store, notebook: Path) -> None:
    original = notebook.read_text()
    notebook.write_text(original.replace("x = 1", "x = 2  # local"))
    write_remote(local_backend, session, "notebook.py", original.replace("x = 1", "x = 3  # remote").encode())
    assert sync(session, local_backend, store).conflicts == ["notebook.py"]

    # The user resolves: picks a version, deletes the copy, makes both sides equal.
    resolved = original.replace("x = 1", "x = 4  # resolved")
    notebook.write_text(resolved)
    (notebook.parent / "notebook.remote.py").unlink()
    write_remote(local_backend, session, "notebook.py", resolved.encode())
    assert sync(session, local_backend, store).conflicts == []

    lifecycle.stop(session, backend=local_backend, store=store)
    assert local_backend.get(session.runtime_id) is None


# --------------------------------------------------------------------------
# stop


def test_stop_pulls_terminates_and_removes_the_record(session, local_backend, store, project: Path) -> None:
    port = urllib.parse.urlsplit(session.url).port
    assert port is not None
    pids = marimo_pids_on_port(port)
    assert pids, "expected to find the marimo server process"

    # Changes made on the runtime since the last sync.
    write_remote(local_backend, session, "helper.py", b"VALUE = 3  # before-stop\n")
    write_remote(local_backend, session, "notes.md", b"remote notes\n")

    report = lifecycle.stop(session, backend=local_backend, store=store)

    # Final pull happened.
    assert {"helper.py", "notes.md"} <= set(report.pulled)
    assert (project / "helper.py").read_text() == "VALUE = 3  # before-stop\n"
    assert (project / "notes.md").read_text() == "remote notes\n"
    # Runtime gone, record removed, marimo dead.
    assert local_backend.get(session.runtime_id) is None
    assert local_backend.find({TAG_SESSION: session.session_id}) == []
    assert store.load(session.session_id) is None
    assert store.list() == []
    wait_for(lambda: not any(pid_alive(p) for p in pids), 20, what="marimo process to exit")
    assert not server_answers(session.url)


# --------------------------------------------------------------------------
# failed starts


def test_start_times_out_and_leaves_nothing(local_backend, store, notebook: Path) -> None:
    with pytest.raises(StartError):
        lifecycle.start(notebook, backend=local_backend, store=store, ready_timeout_s=0.001)
    assert local_backend.find({}) == []
    assert local_backend.find({TAG_TOOL: "1"}) == []
    assert store.list() == []


def test_marimo_failing_to_start_leaves_nothing(local_backend, store, notebook: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the server process dies, start fails with marimo's last output
    lines in the error, and the runtime is terminated."""
    monkeypatch.setattr(
        lifecycle,
        "MARIMO_ARGV",
        ("python", "-c", "import sys; print('marimo-could-not-start-marker'); sys.exit(3)"),
    )
    with pytest.raises(StartError) as info:
        lifecycle.start(notebook, backend=local_backend, store=store, ready_timeout_s=30)
    assert "marimo-could-not-start-marker" in str(info.value)
    assert local_backend.find({}) == []
    assert store.list() == []


# --------------------------------------------------------------------------
# run


def test_run_success(local_backend, notebook: Path, tmp_path: Path) -> None:
    result = lifecycle.run(notebook, backend=local_backend, output_dir=tmp_path / "out")
    assert result.exit_code == 0
    assert "hello-from-notebook" in result.output
    assert result.truncated is False
    assert result.full_output_path is None
    # A one-off run creates and terminates its own runtime.
    assert local_backend.find({}) == []


def test_run_failure_exit_code_and_output(local_backend, project: Path, tmp_path: Path) -> None:
    bad = project / "bad.py"
    bad.write_text(FAILING_NOTEBOOK_TEXT)
    result = lifecycle.run(bad, backend=local_backend, output_dir=tmp_path / "out")
    assert result.exit_code != 0
    assert "line-0004" in result.output
    assert "notebook-failed" in result.output  # the traceback is part of the output
    assert local_backend.find({}) == []


def test_run_truncates_long_output_and_keeps_the_full_output(local_backend, project: Path, tmp_path: Path) -> None:
    chatty = project / "chatty.py"
    chatty.write_text(CHATTY_NOTEBOOK_TEXT)
    out_dir = tmp_path / "out"
    result = lifecycle.run(chatty, backend=local_backend, max_output_lines=20, output_dir=out_dir)

    assert result.exit_code == 0
    assert result.truncated is True
    lines = result.output.splitlines()
    assert len(lines) <= 20
    # First and last halves, with a marker line between.
    assert lines[0] == "line-0000"
    assert lines[-1] == "line-0999"
    assert "line-0500" not in result.output
    assert any(not line.startswith("line-") for line in lines), "expected a truncation marker line"

    assert result.full_output_path is not None
    full = result.full_output_path.read_text()
    assert result.full_output_path.resolve().is_relative_to(out_dir.resolve())
    for i in (0, 1, 500, 998, 999):
        assert f"line-{i:04d}" in full
    assert local_backend.find({}) == []


def test_run_in_a_session_syncs_first_and_keeps_the_runtime(session, local_backend, notebook: Path, tmp_path: Path) -> None:
    notebook.write_text(notebook.read_text().replace('print("hello-from-notebook")', 'print("synced-before-run")'))
    result = lifecycle.run(notebook, backend=local_backend, session=session, output_dir=tmp_path / "out")
    assert result.exit_code == 0
    assert "synced-before-run" in result.output
    assert local_backend.get(session.runtime_id) is not None
    assert server_answers(session.url)


# --------------------------------------------------------------------------
# CLI (subprocesses, as a user runs it; state shared through NBGPU_STATE_DIR)


def nbgpu(*args: str, cwd: Path, timeout_s: float = 180) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("NBGPU_BACKEND", None)
    return subprocess.run(
        [sys.executable, "-m", "nbgpu.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def json_out(proc: subprocess.CompletedProcess[str]):
    assert proc.returncode == 0, f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(proc.stdout)


SYNC_REPORT_KEYS = {"pushed", "pulled", "merged", "conflicts", "skipped"}


@pytest.fixture
def cli_cleanup(project: Path):
    """Stop anything the CLI left running (the CLI's local backend shares
    state through NBGPU_STATE_DIR, set per test by tests/conftest.py)."""
    yield
    nbgpu("stop", "notebook.py", "--force", cwd=project, timeout_s=120)
    try:
        from nbgpu.backends import get_backend

        backend = get_backend("local")
        for handle in backend.find({TAG_TOOL: "1"}):
            backend.terminate(handle)
    except Exception:
        pass


def test_cli_start_status_sync_stop(project: Path, notebook: Path, cli_cleanup) -> None:
    # start --detach returns once the server is ready and leaves it running.
    started = json_out(nbgpu("start", "notebook.py", "--backend", "local", "--detach", "--json", cwd=project))
    assert isinstance(started, dict)
    assert {"session_id", "notebook", "backend", "runtime_id", "url", "rate"} <= set(started)
    assert started["backend"] == "local"
    assert Path(started["notebook"]) == notebook.resolve()
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(started["url"]).query)
    assert len(query.get("access_token", [""])[0]) >= 32, "url must be openable directly (token included)"
    assert started["rate"] is None or isinstance(started["rate"]["usd_per_hour"], (int, float))
    base_url = urllib.parse.urlunsplit(urllib.parse.urlsplit(started["url"])._replace(query="", fragment="")).rstrip("/")
    assert server_answers(base_url), "the detached session must keep serving after the CLI exits"
    assert http_get(started["url"], follow_redirects=True).status == 200

    # status --json: one running session.
    status = json_out(nbgpu("status", "--json", cwd=project))
    assert set(status) >= {"sessions"}
    [entry] = status["sessions"]
    assert {"session_id", "notebook", "backend", "gpu", "running", "uptime_s", "cost_usd", "conflicts"} <= set(entry)
    assert entry["session_id"] == started["session_id"]
    assert entry["running"] is True
    assert entry["backend"] == "local"
    assert entry["gpu"] is None
    assert entry["uptime_s"] >= 0
    assert entry["conflicts"] == []

    # sync --json: a local edit is pushed.
    notebook.write_text(notebook.read_text().replace("x = 1", "x = 5  # cli-edit"))
    synced = json_out(nbgpu("sync", "notebook.py", "--json", cwd=project))
    assert SYNC_REPORT_KEYS <= set(synced)
    assert synced["pushed"] == ["notebook.py"]
    assert synced["conflicts"] == []
    synced_again = json_out(nbgpu("sync", "notebook.py", "--json", cwd=project))
    assert synced_again["pushed"] == [] and "notebook.py" not in synced_again["pulled"]

    # stop --json: final pull report; then nothing is running or recorded.
    stopped = json_out(nbgpu("stop", "notebook.py", "--json", cwd=project))
    assert SYNC_REPORT_KEYS <= set(stopped)
    wait_for(lambda: not server_answers(base_url), 20, what="marimo to stop answering")
    assert json_out(nbgpu("status", "--json", cwd=project)) == {"sessions": [], "orphans": [], "other_owners": []}


def test_cli_human_output_and_errors(project: Path, cli_cleanup) -> None:
    # No session: sync and stop fail with a one-line error on stderr.
    for args in (("sync", "notebook.py"), ("stop", "notebook.py")):
        proc = nbgpu(*args, cwd=project)
        assert proc.returncode != 0
        assert proc.stderr.strip() and len(proc.stderr.strip().splitlines()) == 1, proc.stderr
        assert "Traceback" not in proc.stderr

    # Missing notebook.
    proc = nbgpu("start", "missing.py", "--backend", "local", "--detach", cwd=project)
    assert proc.returncode != 0
    assert "missing.py" in proc.stderr and "Traceback" not in proc.stderr

    # Unsupported GPU on the local backend: refused before anything starts.
    proc = nbgpu("start", "notebook.py", "--backend", "local", "--gpu", "A10G", "--detach", cwd=project)
    assert proc.returncode != 0 and "Traceback" not in proc.stderr
    assert json_out(nbgpu("status", "--json", cwd=project)) == {"sessions": [], "orphans": [], "other_owners": []}

    # Human start output shows the URL with the token.
    proc = nbgpu("start", "notebook.py", "--backend", "local", "--detach", cwd=project)
    assert proc.returncode == 0, proc.stderr
    assert "access_token=" in proc.stdout
    assert nbgpu("stop", "notebook.py", cwd=project).returncode == 0


def test_cli_run_exit_code_and_output(project: Path, cli_cleanup) -> None:
    (project / "bad.py").write_text(FAILING_NOTEBOOK_TEXT)
    proc = nbgpu("run", "bad.py", "--backend", "local", cwd=project)
    assert proc.returncode == 1
    assert "line-0004" in proc.stdout + proc.stderr

    (project / "chatty.py").write_text(CHATTY_NOTEBOOK_TEXT)
    proc = nbgpu("run", "chatty.py", "--backend", "local", "--max-lines", "10", cwd=project)
    assert proc.returncode == 0
    out_lines = proc.stdout.splitlines()
    assert "line-0000" in out_lines and "line-0999" in out_lines
    assert "line-0500" not in proc.stdout
    assert json_out(nbgpu("status", "--json", cwd=project)) == {"sessions": [], "orphans": [], "other_owners": []}


OUTPUT_NOTEBOOK_TEXT = '''\
import marimo

app = marimo.App()


@app.cell
def _():
    import json, os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    with open("metrics.json", "w") as f:
        json.dump({"loss": 0.1}, f)
    with open("plots/curve.png", "wb") as f:
        f.write(b"\\x89PNG\\x00" + os.urandom(16))
    with open("outputs/model.bin", "wb") as f:
        f.write(os.urandom(64))
    print("trained")
    return


if __name__ == "__main__":
    app.run()
'''


def test_cli_run_brings_back_what_the_run_wrote(project: Path, notebook: Path, cli_cleanup) -> None:
    (project / ".gitignore").write_text("outputs/\n")
    (project / "notebook.py").write_text(OUTPUT_NOTEBOOK_TEXT)

    # One-off: small text files come back by default; --pull brings the rest,
    # ignored ones included.
    done = json_out(nbgpu("run", "notebook.py", "--backend", "local", "--json", cwd=project))
    assert done["exit_code"] == 0 and done["pulled"] == ["metrics.json"]
    assert not (project / "plots").exists()
    (project / "metrics.json").unlink()
    done = json_out(nbgpu("run", "notebook.py", "--backend", "local", "--pull", "plots/", "--pull", "outputs/",
                          "--json", cwd=project))
    assert sorted(done["pulled"]) == ["metrics.json", "outputs/model.bin", "plots/curve.png"]
    first_model = (project / "outputs" / "model.bin").read_bytes()

    # In a session: a run brings back changes to existing outputs, too.
    json_out(nbgpu("start", "notebook.py", "--backend", "local", "--detach", "--json", cwd=project))
    done = json_out(nbgpu("run", "notebook.py", "--pull", "outputs/", "--json", cwd=project))
    assert done["exit_code"] == 0 and done["output"] == "trained"
    assert {"plots/curve.png", "outputs/model.bin"} <= set(done["pulled"])
    assert (project / "outputs" / "model.bin").read_bytes() != first_model
