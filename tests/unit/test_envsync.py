"""EnvWatcher: dependency changes during a session."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from nbgpu.backends.base import CAP_LIVE_INSTALL, ExecResult
from nbgpu.envsync import EXTRA_ENV, INSTALL_ARGV, EnvWatcher, _EnvLock, applied_extra
from nbgpu.session import Session, SessionStore
from nbgpu.spec import load_spec
from tests.fakes import FakeBackend, fake_request


class Proc:
    def __init__(self, lines: Sequence[str] = (), code: int | None = None) -> None:
        self._lines = list(lines)
        self.code = code
        self.terminated = False

    def lines(self) -> Iterator[str]:
        yield from self._lines

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.terminated = True


class InstallBackend(FakeBackend):
    def __init__(self, live: bool = True) -> None:
        super().__init__()
        self.capabilities = frozenset({CAP_LIVE_INSTALL}) if live else frozenset()
        self.spawned: list[list[str]] = []
        self.next = Proc()
        self.executed: list[list[str]] = []
        self.next_exec = ExecResult(0, "", "")

    def spawn(self, handle, argv, *, cwd=None, env=None):  # type: ignore[no-untyped-def]
        self.spawned.append(list(argv))
        proc, self.next = self.next, Proc()
        return proc

    def exec(self, handle, argv, *, cwd=None, env=None, timeout_s=None):  # type: ignore[no-untyped-def]
        self.executed.append(list(argv))
        return self.next_exec


def header(deps: Sequence[str], extra: str = "") -> str:
    return (
        "# /// script\n"
        f"# dependencies = {json.dumps(list(deps))}\n"
        "# [tool.nbgpu]\n"
        '# gpu = "A10"\n'
        f"{extra}"
        "# ///\n\nimport marimo\napp = marimo.App()\n"
    )


def write(path: Path, text: str) -> None:
    path.write_text(text)
    # make sure the fingerprint changes even within the mtime granularity
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def env(tmp_path: Path):
    def make(live: bool = True, deps: Sequence[str] = ("marimo",), extra: str = ""):
        nb = tmp_path / "nb.py"
        nb.write_text(header(deps, extra))
        backend = InstallBackend(live)
        handle = backend.create(fake_request())
        store = SessionStore(tmp_path / "state")
        session = Session(
            session_id="s1", notebook=nb, project_root=tmp_path, backend="fake", runtime_id=handle.runtime_id,
            remote_root=handle.remote_root, url="https://x", token="t", created_at=time.time(), gpu="A10",
            extra=dict(applied_extra(load_spec(nb))),
        )
        store.save(session)
        return nb, backend, EnvWatcher(session, backend, handle, store), store
    return make


def test_nothing_changed_is_silent(env) -> None:
    nb, backend, watcher, _ = env()
    assert watcher.tick() == [] and backend.spawned == []


def test_an_added_dependency_is_installed_in_the_background(env) -> None:
    nb, backend, watcher, store = env()
    backend.next = Proc(["Resolved 3 packages"], code=None)
    write(nb, header(["marimo", "polars>=1"]))
    [msg] = watcher.tick()
    assert msg.startswith("↻") and "polars>=1" in msg
    assert backend.spawned == [[*INSTALL_ARGV, "marimo", "polars>=1"]]  # the whole list, so constraints hold
    assert watcher.installing and watcher.tick() == []  # still running
    proc = watcher._install.proc  # type: ignore[union-attr]
    proc.code = 0
    [done] = watcher.tick()
    assert done.startswith("✓") and "restart the kernel" in done
    saved = json.loads(store.load("s1").extra[EXTRA_ENV])  # type: ignore[union-attr]
    assert saved["deps"] == ["marimo", "polars>=1"]
    assert watcher.tick() == [] and len(backend.spawned) == 1  # applied: no reinstall


def test_a_failed_install_shows_the_output_and_keeps_the_old_state(env) -> None:
    nb, backend, watcher, store = env()
    backend.next = Proc(["  × No solution found", "  ╰─▶ nope==99 was not found"], code=1)
    write(nb, header(["marimo", "nope==99"]))
    watcher.tick()
    [failed] = watcher.tick()
    assert failed.startswith("✗") and "nope==99 was not found" in failed and "exit 1" in failed
    assert json.loads(store.load("s1").extra[EXTRA_ENV])["deps"] == ["marimo"]  # type: ignore[union-attr]
    backend.next = Proc(code=0)
    write(nb, header(["marimo", "polars"]))  # fixed: retried on the next edit
    assert watcher.tick()[0].startswith("↻")


def test_removals_and_compute_changes_are_reported_once(env) -> None:
    nb, backend, watcher, _ = env(deps=["marimo", "torch"])
    write(nb, header(["marimo"]).replace('gpu = "A10"', 'gpu = "H100"'))
    msgs = watcher.tick()
    assert any(m.startswith("·") and "torch" in m and "next start" in m for m in msgs)
    assert any(m.startswith("!") and "gpu changed (A10 → H100)" in m for m in msgs)
    assert backend.spawned == []
    write(nb, header(["marimo"]).replace('gpu = "A10"', 'gpu = "H100"') + "\n# touched\n")
    assert watcher.tick() == []


def test_a_version_change_is_an_install_not_a_removal(env) -> None:
    nb, backend, watcher, _ = env(deps=["marimo", "torch==2.4"])
    write(nb, header(["marimo", "torch==2.5"]))
    msgs = watcher.tick()
    assert [m[0] for m in msgs] == ["↻"] and "torch==2.5" in msgs[0]


def test_an_invalid_header_is_reported_once_and_applied_when_fixed(env) -> None:
    nb, backend, watcher, _ = env()
    write(nb, header(["marimo"]).replace('gpu = "A10"', "gpu = ["))
    [msg] = watcher.tick()
    assert msg.startswith("!header")
    write(nb, header(["marimo"]).replace('gpu = "A10"', "gpu = [") + "\n")
    assert watcher.tick() == []
    backend.next = Proc(code=0)
    write(nb, header(["marimo", "polars"]))
    assert watcher.tick()[0].startswith("↻")


def test_backends_without_live_install_only_report(env) -> None:
    nb, backend, watcher, _ = env(live=False)
    write(nb, header(["marimo", "polars"]))
    [msg] = watcher.tick()
    assert msg.startswith("!") and "restart the session" in msg and backend.spawned == []


def test_requirements_file_changes_are_watched(env, tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("numpy\n")
    nb, backend, watcher, _ = env(extra='# requirements = "requirements.txt"\n')
    assert watcher.tick() == []
    backend.next = Proc(code=0)
    write(tmp_path / "requirements.txt", "numpy\nscipy>=1.11\n")
    [msg] = watcher.tick()
    assert "scipy>=1.11" in msg
    assert backend.spawned[-1] == [*INSTALL_ARGV, "marimo", "numpy", "scipy>=1.11"]


def test_adopt_takes_ui_installs_as_applied(env) -> None:
    nb, backend, watcher, store = env()
    write(nb, header(["marimo", "emoji"]))  # recorded after marimo installed it
    watcher.adopt()
    assert watcher.tick() == [] and backend.spawned == []
    assert json.loads(store.load("s1").extra[EXTRA_ENV])["deps"] == ["marimo", "emoji"]  # type: ignore[union-attr]


def test_a_resumed_loop_applies_edits_made_while_it_was_off(env, tmp_path: Path) -> None:
    nb, backend, watcher, store = env()
    nb.write_text(header(["marimo", "polars"]))  # edited while no loop ran
    session = store.load("s1")
    handle = backend.get(session.runtime_id)  # type: ignore[union-attr]
    backend.next = Proc(code=0)
    resumed = EnvWatcher(session, backend, handle, store)  # type: ignore[arg-type]
    [msg] = resumed.tick()
    assert msg.startswith("↻") and "polars" in msg


def test_stop_lists_installs_the_notebook_does_not_name(tmp_path: Path, monkeypatch) -> None:
    from nbgpu import envsync

    nb = tmp_path / "nb.py"
    nb.write_text(header(["marimo", "emoji"]))
    backend = InstallBackend()
    handle = backend.create(fake_request())

    class R:
        exit_code = 0
        stdout = '["Emoji==2.16.0", "polars==1.2.0"]\n'

    monkeypatch.setattr(backend, "exec", lambda *a, **k: R())
    spec = load_spec(nb)
    assert envsync.unrecorded_installs(backend, handle, "/x/baseline.json", spec) == ["polars==1.2.0"]
    assert envsync.unrecorded_installs(InstallBackend(live=False), handle, "/x", spec) is None


# --------------------------------------------------------------------------
# apply(): `nbgpu run` and `nbgpu sync` without a loop
# --------------------------------------------------------------------------


def _watcher(store, backend, nb):  # another process's view of the same session
    session = store.load("s1")
    return EnvWatcher(session, backend, backend.get(session.runtime_id), store)


def test_apply_installs_and_waits(env) -> None:
    nb, backend, watcher, store = env()
    write(nb, header(["marimo", "polars"]))
    applied = watcher.apply()
    assert not applied.failed and applied.messages[0].startswith("✓") and "polars" in applied.messages[0]
    assert backend.executed == [[*INSTALL_ARGV, "marimo", "polars"]] and backend.spawned == []
    assert json.loads(store.load("s1").extra[EXTRA_ENV])["deps"] == ["marimo", "polars"]  # type: ignore[union-attr]
    assert watcher.apply().messages == [] and len(backend.executed) == 1


def test_apply_reports_a_failed_install(env) -> None:
    nb, backend, watcher, store = env()
    backend.next_exec = ExecResult(1, "", "  × No solution found\n  ╰─▶ nope==99 was not found")
    write(nb, header(["marimo", "nope==99"]))
    applied = watcher.apply()
    assert applied.failed and "nope==99 was not found" in applied.messages[0]
    assert json.loads(store.load("s1").extra[EXTRA_ENV])["deps"] == ["marimo"]  # type: ignore[union-attr]


def test_apply_without_live_install_only_reports(env) -> None:
    nb, backend, watcher, _ = env(live=False)
    write(nb, header(["marimo", "polars"]))
    applied = watcher.apply()
    assert not applied.failed and "restart the session" in applied.messages[0] and backend.executed == []


def test_apply_gives_up_while_another_install_holds_the_lock(env, tmp_path: Path) -> None:
    nb, backend, watcher, store = env()
    other = _EnvLock(store, "s1")
    assert other.acquire()
    write(nb, header(["marimo", "polars"]))
    applied = watcher.apply(timeout_s=0.3)
    assert applied.failed and "still running" in applied.messages[0] and backend.executed == []
    other.release()
    assert not watcher.apply().failed


def test_the_loop_waits_for_a_run_and_does_not_reinstall_its_work(env) -> None:
    nb, backend, loop, store = env()
    run = _watcher(store, backend, nb)
    lock = _EnvLock(store, "s1")
    assert lock.acquire()  # `nbgpu run` is installing
    write(nb, header(["marimo", "polars"]))
    assert loop.tick() == [] and backend.spawned == []  # busy: look again next tick
    lock.release()
    assert not run.apply().failed and len(backend.executed) == 1
    assert loop.tick() == [] and backend.spawned == []  # the run's install is seen in the record


def test_a_run_waits_for_the_loops_install(env) -> None:
    nb, backend, loop, store = env()
    backend.next = Proc(code=None)
    write(nb, header(["marimo", "polars"]))
    assert loop.tick()[0].startswith("↻")
    run = _watcher(store, backend, nb)
    assert run.apply(timeout_s=0.2).failed  # the loop holds the lock while installing
    loop._install.proc.code = 0  # type: ignore[union-attr]
    assert loop.tick()[0].startswith("✓")
    assert run.apply().messages == [] and backend.executed == []  # already installed by the loop


def test_a_backend_override_is_not_a_provider_change(env) -> None:
    nb, backend, watcher, store = env()
    session = store.load("s1")
    applied = json.loads(session.extra[EXTRA_ENV])  # type: ignore[union-attr]
    applied["compute"]["provider"] = "local"  # started with NBGPU_BACKEND=local; header says modal
    session.extra = {**session.extra, EXTRA_ENV: json.dumps(applied)}  # type: ignore[union-attr]
    store.save(session)  # type: ignore[arg-type]
    assert _watcher(store, backend, nb).apply().messages == []
