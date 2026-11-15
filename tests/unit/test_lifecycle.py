"""Lifecycle behaviour that needs no real processes (FakeBackend)."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nbgpu import lifecycle
from nbgpu.backends.base import ExecResult
from nbgpu.lifecycle import StartError
from nbgpu.session import TAG_OWNER, TAG_SESSION, TAG_TOOL, Session, SessionStore
from tests.fakes import FakeBackend, fake_request

NOTEBOOK = '''\
# /// script
# [tool.nbgpu]
# provider = "local"
# ///
import marimo
app = marimo.App()
'''


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='p'\n")
    (root / "nb.py").write_text(NOTEBOOK)
    return root


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "state")


# ---------------------------------------------------------------------------
# project root and upload limits


def test_project_root_search_stops_at_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    nb_dir = home / "work" / "nbs"
    nb_dir.mkdir(parents=True)
    (home / ".git").mkdir()  # a dotfiles repo in home must not make home the project
    (nb_dir / "nb.py").write_text(NOTEBOOK)
    monkeypatch.setenv("HOME", str(home))
    assert lifecycle.find_project_root(nb_dir / "nb.py") == nb_dir.resolve()

    (home / "work" / "pyproject.toml").write_text("")
    assert lifecycle.find_project_root(nb_dir / "nb.py") == (home / "work").resolve()


def test_notebook_directly_in_home_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "nb.py").write_text(NOTEBOOK)
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(StartError, match="--project-root"):
        lifecycle.find_project_root(home / "nb.py")


def test_push_size_cap(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (project / ".venv").mkdir()
    for i in range(20):
        (project / ".venv" / f"f{i}").write_text("excluded")
    assert lifecycle.check_push_size(project) == (2, len(NOTEBOOK) + len("[project]\nname='p'\n"))

    monkeypatch.setattr(lifecycle, "PUSH_MAX_FILES", 3)
    for i in range(3):
        (project / f"data{i}.txt").write_text("x")
    with pytest.raises(StartError, match=r"at least 4 files.*--project-root"):
        lifecycle.check_push_size(project)

    monkeypatch.setattr(lifecycle, "PUSH_MAX_FILES", 1000)
    monkeypatch.setattr(lifecycle, "PUSH_MAX_TOTAL_BYTES", 1000)
    (project / "big.bin").write_bytes(b"\0" * 2000)
    with pytest.raises(StartError, match="MB"):
        lifecycle.check_push_size(project)


def test_start_refuses_a_too_large_project_before_creating(project: Path, store: SessionStore, monkeypatch) -> None:
    monkeypatch.setattr(lifecycle, "PUSH_MAX_FILES", 1)
    backend = FakeBackend()
    with pytest.raises(StartError, match="too large"):
        lifecycle.start(project / "nb.py", backend=backend, store=store)
    assert backend.requests == []


def test_state_dir_inside_the_project_is_refused(project: Path) -> None:
    backend = FakeBackend()
    with pytest.raises(StartError, match="NBGPU_STATE_DIR"):
        lifecycle.start(project / "nb.py", backend=backend, store=SessionStore(project / "state"))
    assert backend.requests == []
    # Under an excluded directory it is never synced, so it is fine.
    lifecycle._check_state_dir_outside(project, project / ".nbgpu" / "state")


# ---------------------------------------------------------------------------
# small helpers


def test_marimo_argv_substitutes_in_one_pass() -> None:
    argv = lifecycle._marimo_argv(notebook="dir/{host}-{port}.py", host="0.0.0.0", port=1234, token_file="/t/{notebook}")
    assert "dir/{host}-{port}.py" in argv
    assert "/t/{notebook}" in argv
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--port") + 1] == "1234"


def test_private_dir_modes(tmp_path: Path) -> None:
    created = tmp_path / "new"
    old = os.umask(0o022)
    try:
        lifecycle._ensure_private_dir(created)
    finally:
        os.umask(old)
    assert stat.S_IMODE(created.stat().st_mode) == 0o700

    existing = tmp_path / "existing"
    existing.mkdir(mode=0o755)
    existing.chmod(0o755)
    lifecycle._ensure_private_dir(existing)
    assert stat.S_IMODE(existing.stat().st_mode) == 0o700


class ExecBackend(FakeBackend):
    """A FakeBackend whose exec returns canned output."""

    def __init__(self, stdout: str) -> None:
        super().__init__()
        self.stdout = stdout

    def exec(self, handle, argv, *, cwd=None, env=None, timeout_s=None):  # type: ignore[override]
        return ExecResult(0, self.stdout, "")


def test_run_output_and_sync_state_are_private(project: Path, store: SessionStore, tmp_path: Path) -> None:
    backend = ExecBackend("\n".join(f"line-{i}" for i in range(50)))
    out_dir = tmp_path / "runs"
    result = lifecycle.run(project / "nb.py", backend=backend, max_output_lines=10, output_dir=out_dir, store=store)
    assert result.truncated and result.full_output_path is not None
    assert stat.S_IMODE(result.full_output_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
    assert backend.get("fake-1") is None  # one-off runtime terminated
    assert store.list() == []  # its run record is gone too


# ---------------------------------------------------------------------------
# start: provisional record, cleanup, signals


class SpawnFails(FakeBackend):
    bind_host = "127.0.0.1"

    def __init__(self, on_spawn) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.on_spawn = on_spawn

    def spawn(self, handle, argv, *, cwd=None, env=None):  # type: ignore[override]
        return self.on_spawn(handle)


def test_provisional_record_is_saved_right_after_create(project: Path, store: SessionStore) -> None:
    seen: list[Session | None] = []

    def on_spawn(handle):  # type: ignore[no-untyped-def]
        [record] = store.list()
        seen.append(record)
        assert record.runtime_id == handle.runtime_id
        assert lifecycle.is_starting(record)
        [entry] = lifecycle.status(backends={"fake": backend}, store=store)
        assert entry.state == lifecycle.STATE_STARTING
        raise RuntimeError("spawn failed")

    backend = SpawnFails(on_spawn)
    with pytest.raises(RuntimeError, match="spawn failed"):
        lifecycle.start(project / "nb.py", backend=backend, store=store)
    assert seen and seen[0] is not None
    assert backend.terminated == ["fake-1"]
    assert store.list() == []
    assert not list(store.directory.glob("*.sync.json"))


def test_failed_terminate_keeps_the_record(project: Path, store: SessionStore) -> None:
    def on_spawn(handle):  # type: ignore[no-untyped-def]
        raise RuntimeError("spawn failed")

    backend = SpawnFails(on_spawn)
    backend.terminate = lambda handle: (_ for _ in ()).throw(RuntimeError("provider down"))  # type: ignore[method-assign]
    with pytest.raises(StartError, match="could not be terminated"):
        lifecycle.start(project / "nb.py", backend=backend, store=store)
    [record] = store.list()
    assert lifecycle.is_starting(record) and record.runtime_id == "fake-1"


def test_sigterm_during_start_cleans_up(project: Path, store: SessionStore) -> None:
    def on_spawn(handle):  # type: ignore[no-untyped-def]
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)
        raise AssertionError("SIGTERM was not turned into an exception")

    backend = SpawnFails(on_spawn)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(lifecycle.Terminated):
        lifecycle.start(project / "nb.py", backend=backend, store=store)
    assert backend.terminated == ["fake-1"]
    assert store.list() == []
    assert signal.getsignal(signal.SIGTERM) == before


def test_terminate_is_shielded_from_ctrl_c() -> None:
    backend = FakeBackend()
    handle = backend.create(fake_request())
    real = backend.terminate

    def interrupted_terminate(h):  # type: ignore[no-untyped-def]
        os.kill(os.getpid(), signal.SIGINT)  # the user presses Ctrl-C again
        time.sleep(0.2)
        real(h)

    backend.terminate = interrupted_terminate  # type: ignore[method-assign]
    lifecycle._terminate(backend, handle)
    assert backend.terminated == [handle.runtime_id]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_stop_of_a_starting_session_terminates_without_pull(project: Path, store: SessionStore) -> None:
    backend = FakeBackend()
    handle = backend.create(fake_request(tags={TAG_TOOL: "1"}))
    session = _record(store, handle, extra={"state": lifecycle.STATE_STARTING})
    report = lifecycle.stop(session, backend=backend, store=store)
    assert backend.hash_calls == 0 and not report.changed
    assert backend.terminated == [handle.runtime_id]
    assert store.list() == []


# ---------------------------------------------------------------------------
# status and orphans


def _record(store: SessionStore, handle, *, extra=None, session_id: str | None = None) -> Session:  # type: ignore[no-untyped-def]
    session = Session(
        session_id=session_id or handle.runtime_id.replace("fake-", "s"),
        notebook=Path("/tmp/nb.py"),
        project_root=Path("/tmp"),
        backend="fake",
        runtime_id=handle.runtime_id,
        remote_root=handle.remote_root,
        url="https://x.invalid",
        token="t",
        created_at=handle.created_at,
        extra=dict(extra or {}),
    )
    store.save(session)
    return session


def test_status_reports_idle_stopped(store: SessionStore) -> None:
    backend = FakeBackend()
    handle = backend.create(fake_request(tags={TAG_TOOL: "1"}))
    _record(store, handle)
    [entry] = lifecycle.status(backends={"fake": backend}, store=store)
    assert entry.state == lifecycle.STATE_RUNNING
    backend.remote_write(handle, lifecycle.idle_marker_path(handle), '{"reason": "idle", "idle_timeout_s": 60}')
    [entry] = lifecycle.status(backends={"fake": backend}, store=store)
    assert entry.state == lifecycle.STATE_IDLE_STOPPED
    assert lifecycle.read_idle_marker(backend, handle) == {"reason": "idle", "idle_timeout_s": 60}


def _ours(store: SessionStore, **tags: str) -> dict[str, str]:
    return {TAG_TOOL: "1", TAG_OWNER: lifecycle.owner_id(store), **tags}


def test_find_and_stop_orphans(store: SessionStore) -> None:
    clock = [1000.0]
    backend = FakeBackend(clock=lambda: clock[0])
    tracked = backend.create(fake_request(tags=_ours(store, **{TAG_SESSION: "s1"})))
    orphan = backend.create(fake_request(tags=_ours(store, **{TAG_SESSION: "b"})))
    backend.create(fake_request(tags={}))  # not ours: never touched
    # Another laptop / teammate / state directory: listed apart, never an orphan.
    other = backend.create(fake_request(tags={TAG_TOOL: "1", TAG_OWNER: "someone-else", TAG_SESSION: "c"}))
    legacy = backend.create(fake_request(tags={TAG_TOOL: "1", TAG_SESSION: "d"}))  # no owner tag
    _record(store, tracked)

    others: list[lifecycle.Orphan] = []
    found = lifecycle.find_orphans(backends={"fake": backend}, store=store, others=others)
    assert [o.runtime_id for o in found] == [orphan.runtime_id]
    assert {o.runtime_id for o in others} == {other.runtime_id, legacy.runtime_id}
    refused = lifecycle.stop_orphans(others, backends={"fake": backend}, store=store, min_age_s=0)
    assert refused.terminated == [] and set(refused.failed) == {other.runtime_id, legacy.runtime_id}

    # Too young: a start may still own it.
    result = lifecycle.stop_orphans(found, backends={"fake": backend}, store=store, min_age_s=1e12)
    assert [o.runtime_id for o in result.skipped_young] == [orphan.runtime_id]
    assert backend.terminated == []

    result = lifecycle.stop_orphans(found, backends={"fake": backend}, store=store, min_age_s=0)
    assert [o.runtime_id for o in result.terminated] == [orphan.runtime_id]
    assert backend.terminated == [orphan.runtime_id]
    assert backend.get(tracked.runtime_id) is not None


def test_run_records_track_only_while_their_owner_lives(store: SessionStore) -> None:
    backend = FakeBackend()
    live = backend.create(fake_request(tags=_ours(store, **{TAG_SESSION: "s1"})))
    dead = backend.create(fake_request(tags=_ours(store, **{TAG_SESSION: "s2"})))
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    _record(store, live, extra={"kind": "run", "owner_pid": str(os.getpid())})
    _record(store, dead, extra={"kind": "run", "owner_pid": str(gone.pid)})

    assert lifecycle.status(backends={"fake": backend}, store=store) == []  # runs are not sessions
    found = lifecycle.find_orphans(backends={"fake": backend}, store=store)
    assert [o.runtime_id for o in found] == [dead.runtime_id]
    lifecycle.stop_orphans(found, backends={"fake": backend}, store=store, min_age_s=0)
    assert [s.runtime_id for s in store.list()] == [live.runtime_id]  # the stale run record went too


def test_find_orphans_collects_backend_errors(store: SessionStore) -> None:
    backend = FakeBackend()
    backend.find = lambda tags: (_ for _ in ()).throw(RuntimeError("no credentials"))  # type: ignore[method-assign]
    errors: dict[str, str] = {}
    assert lifecycle.find_orphans(backends={"fake": backend}, store=store, errors=errors) == []
    assert errors == {"fake": "no credentials"}
    with pytest.raises(RuntimeError):
        lifecycle.find_orphans(backends={"fake": backend}, store=store)


def _dead_pid() -> int:
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    return gone.pid


def test_owner_id_is_stable_and_private(store: SessionStore) -> None:
    first = lifecycle.owner_id(store)
    assert first and lifecycle.owner_id(store) == first
    assert lifecycle.owner_id(SessionStore(store.directory)) == first
    path = store.directory / lifecycle.OWNER_ID_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert lifecycle.owner_id(SessionStore(store.directory.parent / "other")) != first
    assert store.list() == []  # not mistaken for a record


class SlowCreate(FakeBackend):
    """create() takes a while (image build, GPU scheduling) and checks what is
    visible meanwhile."""

    def __init__(self, store: SessionStore) -> None:
        super().__init__()
        self.store = store
        self.seen: list[tuple] = []

    def create(self, request):  # type: ignore[no-untyped-def]
        handle = super().create(request)  # exists at the provider already
        [record] = self.store.list()
        orphans = lifecycle.find_orphans(backends={"fake": self}, store=self.store)
        [entry] = lifecycle.status(backends={"fake": self}, store=self.store)
        self.seen.append((record, orphans, entry.state, dict(request.tags)))
        raise RuntimeError("scheduling failed")


def test_provisional_record_exists_before_create(project: Path, store: SessionStore) -> None:
    backend = SlowCreate(store)
    with pytest.raises(RuntimeError, match="scheduling failed"):
        lifecycle.start(project / "nb.py", backend=backend, store=store)
    [(record, orphans, state, tags)] = backend.seen
    assert lifecycle.is_starting(record) and record.runtime_id == ""
    assert record.extra["owner_pid"] == str(os.getpid())
    assert tags[TAG_SESSION] == record.session_id and tags[TAG_OWNER] == lifecycle.owner_id(store)
    assert orphans == []  # matched by session tag, not runtime id
    assert state == lifecycle.STATE_STARTING
    # The runtime created before the failure was found by its tag and terminated.
    assert backend.terminated == ["fake-1"]
    assert store.list() == []


def test_a_crashed_start_is_not_running_but_its_runtime_is_tracked(store: SessionStore) -> None:
    backend = FakeBackend()
    handle = backend.create(fake_request(tags=_ours(store, **{TAG_SESSION: "s9"})))
    session = Session(
        session_id="s9", notebook=Path("/tmp/nb.py"), project_root=Path("/tmp"), backend="fake",
        runtime_id="", remote_root="", url="", token="", created_at=time.time(),
        extra={"state": lifecycle.STATE_STARTING, "owner_pid": str(_dead_pid())},
    )
    store.save(session)
    assert lifecycle.find_orphans(backends={"fake": backend}, store=store) == []
    [entry] = lifecycle.status(backends={"fake": backend}, store=store)
    assert entry.state == lifecycle.STATE_STARTING and entry.running
    lifecycle.stop(session, backend=backend, store=store)  # found by tag, terminated without a pull
    assert backend.terminated == [handle.runtime_id] and store.list() == []


def test_terminate_has_a_deadline(monkeypatch: pytest.MonkeyPatch, store: SessionStore, tmp_path: Path) -> None:
    monkeypatch.setattr(lifecycle, "TERMINATE_TIMEOUT_S", 0.3)
    backend = FakeBackend()
    handle = backend.create(fake_request(tags=_ours(store)))
    session = _record(store, handle)
    session.project_root = tmp_path / "empty"
    session.project_root.mkdir()
    backend.terminate = lambda h: time.sleep(5)  # type: ignore[method-assign]  # network outage
    before = signal.getsignal(signal.SIGINT)
    started = time.monotonic()
    with pytest.raises(lifecycle.TerminateTimeout, match=r"stop --orphans.*console"):
        lifecycle.stop(session, backend=backend, store=store)
    assert time.monotonic() - started < 3
    assert signal.getsignal(signal.SIGINT) == before
    assert store.load(session.session_id) is not None  # kept: the runtime may still bill


def test_failed_final_pull_keeps_the_runtime_unless_terminate_on_error(store: SessionStore) -> None:
    backend = FakeBackend()
    handle = backend.create(fake_request(tags=_ours(store)))
    session = _record(store, handle)

    class Broken:
        def final_pull(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        lifecycle.stop(session, backend=backend, store=store, ignore_conflicts=True, engine=Broken())  # type: ignore[arg-type]
    assert backend.terminated == [] and store.load(session.session_id) is not None

    with pytest.raises(Exception, match="terminated anyway"):
        lifecycle.stop(session, backend=backend, store=store, terminate_on_error=True, engine=Broken())  # type: ignore[arg-type]
    assert backend.terminated == [handle.runtime_id] and store.list() == []


def test_stop_of_a_gone_runtime_warns_about_lost_browser_edits(tmp_path: Path, store: SessionStore) -> None:
    root = tmp_path / "p"
    root.mkdir()
    (root / "nb.py").write_text("x = 1\n")
    backend = FakeBackend()
    handle = backend.create(fake_request(tags=_ours(store)))
    session = _record(store, handle)
    session.project_root = root
    store.save(session)
    lifecycle.sync_once(session, backend=backend, store=store)
    synced_at = lifecycle.last_sync_at(session, store)
    assert synced_at is not None and abs(synced_at - time.time()) < 60
    # The engine wrote the manifest next to the project, for the watchdog.
    manifest = json.loads(backend.remote_read(handle, lifecycle.manifest_path(handle)))
    assert set(manifest["files"]) == {"nb.py"}

    backend.terminate(handle)  # e.g. Modal ended the sandbox after an idle stop
    [entry] = lifecycle.status(backends={"fake": backend}, store=store)
    assert entry.state == lifecycle.STATE_NOT_RUNNING and entry.last_sync_at == synced_at
    assert entry.warning and lifecycle.format_time(synced_at) in entry.warning
    report = lifecycle.stop(session, backend=backend, store=store)
    [warning] = report.warnings
    assert "could not be recovered" in warning and lifecycle.format_time(synced_at) in warning
    assert store.list() == [] and lifecycle.last_sync_at(session, store) is None


def test_status_reports_idle_unsynced(store: SessionStore) -> None:
    backend = FakeBackend()
    handle = backend.create(fake_request(tags=_ours(store)))
    _record(store, handle)
    backend.remote_write(handle, lifecycle.runtime_path(handle, lifecycle.UNSYNCED_MARKER_NAME),
                         '{"reason": "idle-unsynced", "unsynced": ["nb.py"]}')
    [entry] = lifecycle.status(backends={"fake": backend}, store=store)
    assert entry.state == lifecycle.STATE_IDLE_UNSYNCED and entry.unsynced == ("nb.py",)
