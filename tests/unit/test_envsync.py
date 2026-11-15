"""EnvWatcher: dependency changes during a session."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from nbgpu.backends.base import CAP_LIVE_INSTALL
from nbgpu.envsync import EXTRA_ENV, INSTALL_ARGV, EnvWatcher, applied_extra
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

    def spawn(self, handle, argv, *, cwd=None, env=None):  # type: ignore[no-untyped-def]
        self.spawned.append(list(argv))
        proc, self.next = self.next, Proc()
        return proc


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
    from nbgpu import envsync, lifecycle

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
