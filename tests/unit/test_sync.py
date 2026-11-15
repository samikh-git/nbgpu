"""Tests for nbgpu.sync: the decision table, merge, naming helpers, and the
SyncEngine against an in-memory FakeBackend with tmp_path as the project."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from nbgpu import sync
from nbgpu.backends.base import RuntimeHandle
from nbgpu.sync import (
    PULL_MAX_BYTES,
    Action,
    SyncEngine,
    SyncReport,
    conflict_copy_name,
    decide,
    is_conflict_copy,
    is_excluded,
    merge3,
    pull_allowed,
)
from tests.fakes import FakeBackend, fake_request

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH; merge3 uses git merge-file")

# A realistic marimo notebook. Cells are separated by blank lines so edits in
# different cells are distinct hunks for a three-way merge.
NOTEBOOK = '''import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(np):
    x = np.linspace(0, 1, 100)
    return (x,)


@app.cell
def _(mo):
    mo.md("# Training run")
    return


@app.cell
def _(np, x):
    lr = 0.001
    loss = np.exp(-x / lr)
    return (loss,)


@app.cell
def _(loss, mo):
    mo.md(f"final loss: {loss[-1]:.4f}")
    return


if __name__ == "__main__":
    app.run()
'''

# Local editor changes cell 2, the marimo UI changes cell 4.
LOCAL_EDIT = NOTEBOOK.replace("np.linspace(0, 1, 100)", "np.linspace(0, 1, 1000)")
REMOTE_EDIT = NOTEBOOK.replace("lr = 0.001", "lr = 0.01")
MERGED = NOTEBOOK.replace("np.linspace(0, 1, 100)", "np.linspace(0, 1, 1000)").replace(
    "lr = 0.001", "lr = 0.01"
)
# Both sides change the same line differently.
LOCAL_CLASH = NOTEBOOK.replace("lr = 0.001", "lr = 0.005")
REMOTE_CLASH = NOTEBOOK.replace("lr = 0.001", "lr = 0.02")

NEW_CELL = '''

@app.cell
def _(loss):
    loss.mean()
    return
'''
REMOTE_ADDS_CELL = NOTEBOOK.replace('\n\nif __name__ == "__main__":', NEW_CELL + '\n\nif __name__ == "__main__":')

assert LOCAL_EDIT != NOTEBOOK and REMOTE_EDIT != NOTEBOOK and MERGED not in (LOCAL_EDIT, REMOTE_EDIT)
assert REMOTE_ADDS_CELL != NOTEBOOK


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

B, L, R, X = "h-base", "h-local", "h-remote", "h-same"


@pytest.mark.parametrize(
    ("base", "local", "remote", "expected"),
    [
        # any | x | x -> NOOP (both sides agree, whatever the base)
        pytest.param(B, X, X, Action.NOOP, id="both-changed-to-same"),
        pytest.param(B, B, B, Action.NOOP, id="unchanged"),
        pytest.param(None, X, X, Action.NOOP, id="both-new-same-content"),
        # b | b | r -> PULL
        pytest.param(B, B, R, Action.PULL, id="remote-changed"),
        # b | l | b -> PUSH
        pytest.param(B, L, B, Action.PUSH, id="local-changed"),
        # b | l | r -> MERGE
        pytest.param(B, L, R, Action.MERGE, id="both-changed"),
        # None | l | None -> PUSH
        pytest.param(None, L, None, Action.PUSH, id="new-local"),
        # None | None | r -> PULL (pull_allowed is the engine's job)
        pytest.param(None, None, R, Action.PULL, id="new-remote"),
        # None | l | r -> MERGE with empty base
        pytest.param(None, L, R, Action.MERGE, id="both-new-different"),
        # b | None | * -> NOOP (deletions never propagate)
        pytest.param(B, None, B, Action.NOOP, id="local-deleted"),
        pytest.param(B, None, R, Action.NOOP, id="local-deleted-remote-changed"),
        pytest.param(B, None, None, Action.NOOP, id="deleted-both-sides"),
        # b | * | None -> NOOP
        pytest.param(B, B, None, Action.NOOP, id="remote-deleted"),
        pytest.param(B, L, None, Action.NOOP, id="remote-deleted-local-changed"),
        # Degenerate: nothing anywhere.
        pytest.param(None, None, None, Action.NOOP, id="absent-everywhere"),
    ],
)
def test_decide_table(base: str | None, local: str | None, remote: str | None, expected: Action) -> None:
    assert decide(base, local, remote) is expected


# ---------------------------------------------------------------------------
# merge3
# ---------------------------------------------------------------------------


@needs_git
def test_merge3_non_overlapping_cells_merge() -> None:
    out = merge3(NOTEBOOK.encode(), LOCAL_EDIT.encode(), REMOTE_EDIT.encode())
    assert out == MERGED.encode()


@needs_git
def test_merge3_added_cell_and_edited_cell_merge() -> None:
    out = merge3(NOTEBOOK.encode(), LOCAL_EDIT.encode(), REMOTE_ADDS_CELL.encode())
    expected = REMOTE_ADDS_CELL.replace("np.linspace(0, 1, 100)", "np.linspace(0, 1, 1000)")
    assert out == expected.encode()


@needs_git
def test_merge3_overlapping_edits_conflict() -> None:
    assert merge3(NOTEBOOK.encode(), LOCAL_CLASH.encode(), REMOTE_CLASH.encode()) is None


@needs_git
def test_merge3_identical_edits_merge() -> None:
    assert merge3(NOTEBOOK.encode(), LOCAL_EDIT.encode(), LOCAL_EDIT.encode()) == LOCAL_EDIT.encode()


@needs_git
def test_merge3_one_side_unchanged_takes_other() -> None:
    assert merge3(NOTEBOOK.encode(), NOTEBOOK.encode(), REMOTE_EDIT.encode()) == REMOTE_EDIT.encode()
    assert merge3(NOTEBOOK.encode(), LOCAL_EDIT.encode(), NOTEBOOK.encode()) == LOCAL_EDIT.encode()


@needs_git
def test_merge3_empty_base() -> None:
    assert merge3(b"", NOTEBOOK.encode(), NOTEBOOK.encode()) == NOTEBOOK.encode()
    assert merge3(b"", LOCAL_CLASH.encode(), REMOTE_CLASH.encode()) is None


def test_merge3_without_git_raises_runtime_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))  # an empty directory: no git
    with pytest.raises(RuntimeError):
        merge3(NOTEBOOK.encode(), LOCAL_EDIT.encode(), REMOTE_EDIT.encode())


# ---------------------------------------------------------------------------
# Naming and filtering helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("notebook.py", "notebook.remote.py"),
        ("notebooks/train.py", "notebooks/train.remote.py"),
        ("a/b/c/config.json", "a/b/c/config.remote.json"),
        ("Makefile", "Makefile.remote"),
        ("docs/LICENSE", "docs/LICENSE.remote"),
        (".gitignore", ".gitignore.remote"),
        ("sub/.env", "sub/.env.remote"),
        ("archive.tar.gz", "archive.tar.remote.gz"),  # pathlib stem/suffix
        ("dir.v2/file", "dir.v2/file.remote"),  # a dot in a directory is not a suffix
    ],
)
def test_conflict_copy_name(path: str, expected: str) -> None:
    assert conflict_copy_name(path) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("notebook.remote.py", True),
        ("notebooks/train.remote.py", True),
        ("Makefile.remote", True),
        (".gitignore.remote", True),
        ("archive.tar.remote.gz", True),
        ("notebook.py", False),
        ("notebooks/train.py", False),
        ("Makefile", False),
        ("remote.py", False),  # its own copy would be remote.remote.py
        ("remote/notebook.py", False),  # directory named remote
        ("notebook.remotes.py", False),
        ("notebook.remote.py.bak", False),
        ("my_remote.py", False),
    ],
)
def test_is_conflict_copy(path: str, expected: bool) -> None:
    assert is_conflict_copy(path) is expected


@pytest.mark.parametrize(
    "path",
    ["notebook.py", "notebooks/train.py", "Makefile", ".gitignore", "deep/nested/dir/a.ext"],
)
def test_is_conflict_copy_of_conflict_copy_name(path: str) -> None:
    assert is_conflict_copy(conflict_copy_name(path))
    assert not is_conflict_copy(path)


@pytest.mark.parametrize(
    ("path", "size", "expected"),
    [
        ("notebook.py", 10, True),
        ("notebooks/new.py", 10, True),
        ("README.md", 0, True),
        ("pyproject.toml", 10, True),
        ("notes.txt", 10, True),
        ("cfg/settings.json", 10, True),
        ("cfg/a.yaml", 10, True),
        ("cfg/a.yml", 10, True),
        ("setup.cfg", 10, True),
        ("tox.ini", 10, True),
        ("notebook.py", PULL_MAX_BYTES, True),  # exactly the limit is allowed
        ("notebook.py", PULL_MAX_BYTES + 1, False),
        ("outputs/plot.png", 10, False),
        ("data/train.csv", 10, False),
        ("model.pt", 10, False),
        ("Makefile", 10, False),  # no suffix
        (".env", 10, False),  # dotfile without a suffix
        ("big/table.json", PULL_MAX_BYTES + 1, False),
    ],
)
def test_pull_allowed(path: str, size: int, expected: bool) -> None:
    assert pull_allowed(path, size) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (".git/HEAD", True),
        (".venv/lib/python3.12/site.py", True),
        ("src/__pycache__/mod.cpython-312.pyc", True),
        ("web/node_modules/pkg/index.js", True),
        (".nbgpu/state.json", True),
        ("a/b/.git/config", True),
        ("notebook.py", False),
        (".gitignore", False),
        ("venv/lib/x.py", False),
        ("my.git/x", False),
        ("notes.git", False),
        ("pkg/__init__.py", False),
    ],
)
def test_is_excluded(path: str, expected: bool) -> None:
    assert is_excluded(path) is expected


def test_report_changed() -> None:
    assert not SyncReport().changed
    assert not SyncReport(conflicts=["a.py"], skipped=["b.png"]).changed
    assert SyncReport(pushed=["a.py"]).changed
    assert SyncReport(pulled=["a.py"]).changed
    assert SyncReport(merged=["a.py"]).changed


# ---------------------------------------------------------------------------
# SyncEngine
# ---------------------------------------------------------------------------


@dataclass
class Env:
    root: Path
    backend: FakeBackend
    handle: RuntimeHandle
    state_path: Path

    def engine(self) -> SyncEngine:
        return SyncEngine(self.root, self.backend, self.handle, self.state_path)

    def write_local(self, rel: str, data: str | bytes) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode() if isinstance(data, str) else data)
        return p

    def read_local(self, rel: str) -> bytes | None:
        p = self.root / rel
        return p.read_bytes() if p.exists() else None

    def remote_write(self, rel: str, data: str | bytes) -> None:
        self.backend.remote_write(self.handle, rel, data)

    def remote(self, rel: str) -> bytes | None:
        return self.backend.remote_read(self.handle, rel)

    def local_files(self) -> set[str]:
        return {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    # Keep the size limit small so tests never write 50 MB.
    monkeypatch.setattr(sync, "PUSH_MAX_BYTES", 4096)
    backend = FakeBackend()
    handle = backend.create(fake_request())
    root = tmp_path / "project"
    root.mkdir()
    return Env(root=root, backend=backend, handle=handle, state_path=tmp_path / "sync-state.json")


def _started(env: Env, files: dict[str, str | bytes] | None = None) -> SyncEngine:
    """Write the given local files, run initial_push, and reset counters."""
    for rel, data in (files or {"notebook.py": NOTEBOOK}).items():
        env.write_local(rel, data)
    engine = env.engine()
    engine.initial_push()
    env.backend.reset_counters()
    return engine


def _assert_quiet(report: SyncReport, env: Env) -> None:
    assert report.pushed == [] and report.pulled == [] and report.merged == []
    assert not report.changed
    assert env.backend.writes == [], "no remote writes expected"


# -- initial_push -------------------------------------------------------------


def test_initial_push_skips_excluded_and_oversized(env: Env) -> None:
    env.write_local("notebook.py", NOTEBOOK)
    env.write_local("pkg/util.py", "def f():\n    return 1\n")
    env.write_local("data/small.txt", "hello\n")
    env.write_local("assets/fig.png", b"\x89PNG\r\n\x1a\n" + bytes(range(50)))
    env.write_local(".gitignore", ".venv\n")
    for excluded in [
        ".git/HEAD",
        ".venv/lib/site.py",
        "pkg/__pycache__/util.cpython-312.pyc",
        "node_modules/x/index.js",
        ".nbgpu/state.json",
    ]:
        env.write_local(excluded, "excluded")
    env.write_local("weights.bin", b"\0" * 5000)  # over the patched 4096 limit

    report = env.engine().initial_push()

    expected = {"notebook.py", "pkg/util.py", "data/small.txt", "assets/fig.png", ".gitignore"}
    assert set(env.backend.remote_files(env.handle)) == expected
    assert set(report.pushed) == expected
    assert "weights.bin" in report.skipped
    assert env.remote("notebook.py") == NOTEBOOK.encode()
    assert env.remote("assets/fig.png") == env.read_local("assets/fig.png")
    assert report.pulled == [] and report.merged == [] and report.conflicts == []


def test_poll_after_initial_push_is_quiet(env: Env) -> None:
    engine = _started(env, {"notebook.py": NOTEBOOK, "pkg/util.py": "x = 1\n", "weights.bin": b"\0" * 5000})
    report = engine.poll_once()
    _assert_quiet(report, env)
    assert env.backend.reads == []
    assert env.remote("weights.bin") is None  # still too large to push
    assert engine.conflicts == []


def test_second_poll_with_no_changes_does_nothing(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)
    assert engine.poll_once().pushed == ["notebook.py"]
    nb = env.root / "notebook.py"
    before = nb.stat()
    env.backend.reset_counters()

    report = engine.poll_once()

    _assert_quiet(report, env)
    assert env.backend.reads == []
    after = nb.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns), "local file was rewritten"


# -- one side changed ---------------------------------------------------------


def test_local_edit_is_pushed(env: Env) -> None:
    engine = _started(env, {"notebook.py": NOTEBOOK, "other.py": "y = 2\n"})
    env.write_local("notebook.py", LOCAL_EDIT)

    report = engine.poll_once()

    assert report.pushed == ["notebook.py"]
    assert report.pulled == [] and report.merged == [] and report.conflicts == []
    assert env.remote("notebook.py") == LOCAL_EDIT.encode()
    assert env.backend.writes == ["notebook.py"]


def test_new_local_file_is_pushed_but_not_excluded_ones(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebooks/eval.py", NOTEBOOK)
    env.write_local(".nbgpu/cache.json", "{}")
    env.write_local("__pycache__/x.pyc", "junk")

    report = engine.poll_once()

    assert report.pushed == ["notebooks/eval.py"]
    assert set(env.backend.remote_files(env.handle)) == {"notebook.py", "notebooks/eval.py"}


def test_remote_edit_is_pulled_atomically(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _started(env)
    nb = env.root / "notebook.py"
    ino_before = nb.stat().st_ino
    env.remote_write("notebook.py", REMOTE_EDIT)

    renames: list[tuple[Path, Path]] = []
    real_replace, real_rename = os.replace, os.rename

    def spy_replace(src, dst, *a, **kw):  # type: ignore[no-untyped-def]
        renames.append((Path(src), Path(dst)))
        return real_replace(src, dst, *a, **kw)

    def spy_rename(src, dst, *a, **kw):  # type: ignore[no-untyped-def]
        renames.append((Path(src), Path(dst)))
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(os, "rename", spy_rename)

    report = engine.poll_once()

    assert report.pulled == ["notebook.py"]
    assert report.pushed == [] and report.merged == [] and report.conflicts == []
    assert nb.read_bytes() == REMOTE_EDIT.encode()
    assert env.backend.writes == []
    # Atomic: a temp file in the same directory renamed over the target.
    to_nb = [(s, d) for s, d in renames if Path(os.path.abspath(d)) == nb.absolute()]
    assert to_nb, f"local write did not rename onto the target: {renames}"
    src, _ = to_nb[-1]
    assert Path(os.path.abspath(src)).parent == nb.parent.absolute()
    assert Path(os.path.abspath(src)) != nb.absolute()
    assert nb.stat().st_ino != ino_before
    assert env.local_files() == {"notebook.py"}, "temp files left behind"


def test_existing_file_syncs_both_ways_even_if_not_pull_allowed(env: Env) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"a" * 32
    engine = _started(env, {"notebook.py": NOTEBOOK, "assets/fig.png": png})
    new_png = b"\x89PNG\r\n\x1a\n" + b"b" * 64
    env.remote_write("assets/fig.png", new_png)

    report = engine.poll_once()

    assert report.pulled == ["assets/fig.png"]
    assert env.read_local("assets/fig.png") == new_png


# -- both changed -------------------------------------------------------------


@needs_git
def test_both_edited_non_overlapping_merges_on_both_sides(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)
    env.remote_write("notebook.py", REMOTE_EDIT)

    report = engine.poll_once()

    assert report.merged == ["notebook.py"]
    assert report.conflicts == [] and engine.conflicts == []
    assert env.read_local("notebook.py") == MERGED.encode()
    assert env.remote("notebook.py") == MERGED.encode()
    assert env.read_local("notebook.remote.py") is None

    env.backend.reset_counters()
    _assert_quiet(engine.poll_once(), env)

    # The merged content is the new base: a later remote-only edit is a pull.
    env.remote_write("notebook.py", MERGED.replace("# Training run", "# Run 2"))
    assert engine.poll_once().pulled == ["notebook.py"]


@needs_git
def test_overlapping_edits_conflict_without_overwriting(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)

    report = engine.poll_once()

    assert report.conflicts == ["notebook.py"]
    assert engine.conflicts == ["notebook.py"]
    assert report.merged == [] and report.pushed == [] and report.pulled == []
    assert env.read_local("notebook.py") == LOCAL_CLASH.encode()
    assert env.remote("notebook.py") == REMOTE_CLASH.encode()
    assert env.read_local("notebook.remote.py") == REMOTE_CLASH.encode()
    assert env.backend.writes == []
    assert env.remote("notebook.remote.py") is None

    # Reported on every poll until resolved; still nothing overwritten or pushed.
    for _ in range(2):
        env.backend.reset_counters()
        again = engine.poll_once()
        assert again.conflicts == ["notebook.py"]
        assert engine.conflicts == ["notebook.py"]
        assert env.backend.writes == []
        assert env.read_local("notebook.py") == LOCAL_CLASH.encode()
        assert env.remote("notebook.py") == REMOTE_CLASH.encode()
        assert env.read_local("notebook.remote.py") == REMOTE_CLASH.encode()
        assert env.remote("notebook.remote.py") is None


@needs_git
def test_conflict_resolution_clears_it(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)
    assert engine.poll_once().conflicts == ["notebook.py"]

    # The user merges by hand in the local file and deletes the copy:
    # the local version wins and is pushed.
    resolved = NOTEBOOK.replace("lr = 0.001", "lr = 0.003")
    env.write_local("notebook.py", resolved)
    (env.root / "notebook.remote.py").unlink()
    env.backend.reset_counters()

    report = engine.poll_once()

    assert report.conflicts == [] and engine.conflicts == []
    assert report.pushed == ["notebook.py"]
    assert env.remote("notebook.py") == resolved.encode()
    assert env.read_local("notebook.remote.py") is None

    # The resolved content is the new base: a local edit is now a plain push.
    env.write_local("notebook.py", resolved.replace("# Training run", "# Resolved"))
    assert engine.poll_once().pushed == ["notebook.py"]


@needs_git
def test_deleting_copy_when_sides_already_equal_is_quiet(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)
    engine.poll_once()

    env.remote_write("notebook.py", LOCAL_CLASH)
    (env.root / "notebook.remote.py").unlink()
    env.backend.reset_counters()

    report = engine.poll_once()
    assert report.conflicts == [] and engine.conflicts == []
    _assert_quiet(report, env)


@needs_git
def test_conflict_stays_open_while_copy_exists(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)
    engine.poll_once()

    # Sides made equal, but the conflict copy is still there: not resolved yet.
    env.remote_write("notebook.py", LOCAL_CLASH)
    report = engine.poll_once()
    assert report.conflicts == ["notebook.py"]
    assert env.remote("notebook.remote.py") is None


def test_user_files_named_like_conflict_copies_are_synced(env: Env) -> None:
    # Changed for finding 9: only copies the engine created are excluded; a
    # user's own file that merely has a copy-shaped name is a project file.
    engine = _started(env, {"notebook.py": NOTEBOOK, "notes.remote.md": "my notes\n"})
    assert env.backend.remote_files(env.handle).keys() == {"notebook.py", "notes.remote.md"}
    env.write_local("train.remote.py", NOTEBOOK)

    report = engine.poll_once()

    assert report.pushed == ["train.remote.py"]
    assert env.remote("train.remote.py") == NOTEBOOK.encode()


def test_remote_files_named_like_conflict_copies_are_pulled(env: Env) -> None:
    # Changed for finding 9 (see above): the engine never pushes its copies,
    # so a copy-shaped remote file is the user's and is pulled normally.
    engine = _started(env)
    env.remote_write("other.remote.py", NOTEBOOK)

    report = engine.poll_once()

    assert report.pulled == ["other.remote.py"]
    assert env.read_local("other.remote.py") == NOTEBOOK.encode()


def test_both_new_with_same_content_is_noop(env: Env) -> None:
    engine = _started(env)
    env.write_local("shared.py", "z = 3\n")
    env.remote_write("shared.py", "z = 3\n")

    report = engine.poll_once()

    _assert_quiet(report, env)
    assert report.conflicts == []
    # Recorded as base: a later remote edit is a pull.
    env.remote_write("shared.py", "z = 4\n")
    assert engine.poll_once().pulled == ["shared.py"]
    assert env.read_local("shared.py") == b"z = 4\n"


@needs_git
def test_both_new_with_different_content_conflicts(env: Env) -> None:
    engine = _started(env)
    env.write_local("shared.py", "z = 3\n")
    env.remote_write("shared.py", "z = 999\n")

    report = engine.poll_once()

    assert report.conflicts == ["shared.py"]
    assert env.read_local("shared.py") == b"z = 3\n"
    assert env.remote("shared.py") == b"z = 999\n"
    assert env.read_local("shared.remote.py") == b"z = 999\n"


# -- new remote files ---------------------------------------------------------


def test_new_remote_files_pulled_only_if_allowed(env: Env) -> None:
    engine = _started(env)
    env.remote_write("scratch.py", NOTEBOOK)
    env.remote_write("notebooks/sub/new.md", "# notes\n")
    env.remote_write("exact.json", b"x" * PULL_MAX_BYTES)
    env.remote_write("outputs/plot.png", b"\x89PNG" + b"\0" * 100)
    env.remote_write("too_big.json", b"x" * (PULL_MAX_BYTES + 1))
    env.remote_write(".nbgpu/runtime.json", "{}")

    report = engine.poll_once()

    assert sorted(report.pulled) == ["exact.json", "notebooks/sub/new.md", "scratch.py"]
    assert env.read_local("scratch.py") == NOTEBOOK.encode()
    assert env.read_local("notebooks/sub/new.md") == b"# notes\n"
    assert env.read_local("exact.json") == b"x" * PULL_MAX_BYTES
    assert env.read_local("outputs/plot.png") is None
    assert env.read_local("too_big.json") is None
    assert env.read_local(".nbgpu/runtime.json") is None
    assert {"outputs/plot.png", "too_big.json"} <= set(report.skipped)
    assert env.backend.writes == []

    # The pulled file is now tracked: a local edit to it is pushed.
    env.write_local("scratch.py", LOCAL_EDIT)
    assert engine.poll_once().pushed == ["scratch.py"]


# -- deletions ----------------------------------------------------------------


def test_local_deletion_is_not_propagated(env: Env) -> None:
    engine = _started(env, {"notebook.py": NOTEBOOK, "old.py": "a = 1\n"})
    (env.root / "old.py").unlink()

    report = engine.poll_once()

    _assert_quiet(report, env)
    assert env.remote("old.py") == b"a = 1\n"
    assert env.read_local("old.py") is None, "deleted local file must not be restored"


def test_local_deletion_with_remote_edit_is_noop(env: Env) -> None:
    engine = _started(env, {"notebook.py": NOTEBOOK, "old.py": "a = 1\n"})
    (env.root / "old.py").unlink()
    env.remote_write("old.py", "a = 2\n")

    report = engine.poll_once()

    _assert_quiet(report, env)
    assert env.read_local("old.py") is None
    assert env.remote("old.py") == b"a = 2\n"


def test_remote_deletion_is_not_propagated(env: Env) -> None:
    engine = _started(env, {"notebook.py": NOTEBOOK, "old.py": "a = 1\n"})
    env.backend.remote_delete(env.handle, "old.py")

    report = engine.poll_once()

    _assert_quiet(report, env)
    assert env.read_local("old.py") == b"a = 1\n"
    assert env.remote("old.py") is None, "remotely deleted file must not be re-pushed"

    # Even a local edit afterwards is not pushed (b, l, None) -> NOOP.
    env.write_local("old.py", "a = 5\n")
    _assert_quiet(engine.poll_once(), env)
    assert env.remote("old.py") is None


# -- persistence --------------------------------------------------------------


def test_base_persists_across_engines(env: Env) -> None:
    _started(env, {"notebook.py": NOTEBOOK, "pkg/util.py": "x = 1\n"})
    assert env.state_path.exists()

    report = env.engine().poll_once()

    _assert_quiet(report, env)
    assert report.conflicts == []
    assert env.backend.reads == []


def test_restart_then_remote_edit_is_a_pull_not_a_conflict(env: Env) -> None:
    _started(env)
    env.remote_write("notebook.py", REMOTE_EDIT)

    report = env.engine().poll_once()

    assert report.pulled == ["notebook.py"]
    assert report.conflicts == []
    assert env.read_local("notebook.remote.py") is None


@needs_git
def test_restart_keeps_base_content_for_merges(env: Env) -> None:
    _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)
    env.remote_write("notebook.py", REMOTE_EDIT)

    report = env.engine().poll_once()

    assert report.merged == ["notebook.py"]
    assert env.read_local("notebook.py") == MERGED.encode()
    assert env.remote("notebook.py") == MERGED.encode()


@needs_git
def test_restart_with_open_conflict_still_reports_it(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)
    engine.poll_once()
    env.backend.reset_counters()

    report = env.engine().poll_once()

    assert report.conflicts == ["notebook.py"]
    assert env.backend.writes == []
    assert env.read_local("notebook.py") == LOCAL_CLASH.encode()
    assert env.read_local("notebook.remote.py") == REMOTE_CLASH.encode()


# -- final_pull ---------------------------------------------------------------


def test_final_pull_pulls_remote_edits(env: Env) -> None:
    engine = _started(env)
    env.remote_write("notebook.py", REMOTE_EDIT)
    env.remote_write("new.py", "n = 1\n")

    report = engine.final_pull()

    assert sorted(report.pulled) == ["new.py", "notebook.py"]
    assert env.read_local("notebook.py") == REMOTE_EDIT.encode()
    assert env.read_local("new.py") == b"n = 1\n"


def test_final_pull_pushes_local_edits_like_poll(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)

    report = engine.final_pull()

    assert report.pushed == ["notebook.py"]
    assert env.remote("notebook.py") == LOCAL_EDIT.encode()


@needs_git
def test_final_pull_reports_conflicts_without_overwriting(env: Env) -> None:
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)

    report = engine.final_pull()

    assert report.conflicts == ["notebook.py"]
    assert env.read_local("notebook.py") == LOCAL_CLASH.encode()
    assert env.remote("notebook.py") == REMOTE_CLASH.encode()
    assert env.read_local("notebook.remote.py") == REMOTE_CLASH.encode()
    assert env.backend.writes == []


def test_final_pull_with_no_changes_is_quiet(env: Env) -> None:
    engine = _started(env)
    _assert_quiet(engine.final_pull(), env)


# ---------------------------------------------------------------------------
# FakeBackend compare-and-swap (the engine relies on these semantics)
# ---------------------------------------------------------------------------


def test_fake_write_file_expect_semantics() -> None:
    import hashlib

    from nbgpu.backends.base import ABSENT

    fb = FakeBackend()
    h = fb.create(fake_request())
    assert fb.write_file(h, "a.py", b"one", expect=ABSENT) is True
    assert fb.write_file(h, "a.py", b"two", expect=ABSENT) is False
    assert fb.write_file(h, "a.py", b"two", expect=hashlib.sha256(b"nope").hexdigest()) is False
    assert fb.remote_read(h, "a.py") == b"one"
    assert fb.write_file(h, "a.py", b"two", expect=hashlib.sha256(b"one").hexdigest()) is True
    assert fb.write_file(h, "missing.py", b"x", expect=hashlib.sha256(b"").hexdigest()) is False
    assert fb.write_file(h, "a.py", b"three") is True
    assert fb.remote_read(h, "a.py") == b"three"
    assert fb.rejected_writes == ["a.py", "a.py", "missing.py"]


# ---------------------------------------------------------------------------
# Races: an edit lands between hashing and writing (findings 1-4)
# ---------------------------------------------------------------------------

# A third, non-overlapping edit (cell 3) the user saves locally mid-poll.
LOCAL_LATER = "# Local run"


def _later(text: str) -> str:
    return text.replace("# Training run", LOCAL_LATER)


@needs_git
def test_local_save_during_pull_is_not_overwritten(env: Env) -> None:
    """Finding 1 (repro scenario_pull_race): the remote changed; the user
    saves locally after the scan but before the pull writes."""
    engine = _started(env)
    env.remote_write("notebook.py", REMOTE_EDIT)
    env.backend.after_hash_files(lambda: env.write_local("notebook.py", LOCAL_EDIT))

    report = engine.poll_once()

    assert report.pulled == [] and "notebook.py" in report.skipped
    assert env.read_local("notebook.py") == LOCAL_EDIT.encode()
    assert env.remote("notebook.py") == REMOTE_EDIT.encode()

    # Next poll sees both changes and merges them.
    report = engine.poll_once()
    assert report.merged == ["notebook.py"]
    assert env.read_local("notebook.py") == env.remote("notebook.py") == MERGED.encode()


def test_local_file_created_during_new_file_pull_is_not_overwritten(env: Env) -> None:
    engine = _started(env)
    env.remote_write("scratch.py", "remote = 1\n")
    env.backend.after_hash_files(lambda: env.write_local("scratch.py", "local = 1\n"))

    report = engine.poll_once()

    assert report.pulled == [] and "scratch.py" in report.skipped
    assert env.read_local("scratch.py") == b"local = 1\n"
    assert env.local_files() == {"notebook.py", "scratch.py"}, "temp files left behind"


@needs_git
def test_local_save_during_merge_is_not_overwritten(env: Env) -> None:
    """Finding 1 for _merge: the local file changes after the merge was
    computed. The remote gets the merge; local keeps the save; the next poll
    merges again and nothing is lost."""
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)
    env.remote_write("notebook.py", REMOTE_EDIT)
    env.backend.before_write("notebook.py", lambda: env.write_local("notebook.py", _later(LOCAL_EDIT)))

    report = engine.poll_once()

    assert report.merged == [] and "notebook.py" in report.skipped
    assert env.read_local("notebook.py") == _later(LOCAL_EDIT).encode()

    report = engine.poll_once()
    assert report.merged == ["notebook.py"]
    expected = _later(MERGED).encode()
    assert env.read_local("notebook.py") == expected
    assert env.remote("notebook.py") == expected


@needs_git
def test_remote_edit_during_push_is_not_overwritten(env: Env) -> None:
    """Finding 2 (repro scenario_push_race): the UI saves after hashing."""
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)
    env.backend.after_hash_files(lambda: env.remote_write("notebook.py", REMOTE_EDIT))

    report = engine.poll_once()

    assert report.pushed == [] and "notebook.py" in report.skipped
    assert env.backend.rejected_writes == ["notebook.py"]
    assert env.remote("notebook.py") == REMOTE_EDIT.encode()

    report = engine.poll_once()
    assert report.merged == ["notebook.py"]
    assert env.read_local("notebook.py") == env.remote("notebook.py") == MERGED.encode()


def test_remote_edit_right_before_push_write_is_not_overwritten(env: Env) -> None:
    """Finding 2: the check is inside write_file, so even an edit landing
    immediately before the write is kept."""
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)
    env.backend.before_write("notebook.py", lambda: env.remote_write("notebook.py", REMOTE_EDIT))

    report = engine.poll_once()

    assert report.pushed == []
    assert env.remote("notebook.py") == REMOTE_EDIT.encode()
    assert env.read_local("notebook.py") == LOCAL_EDIT.encode()


def test_remote_file_created_during_new_file_push_is_not_overwritten(env: Env) -> None:
    engine = _started(env)
    env.write_local("new.py", "local = 1\n")
    env.backend.after_hash_files(lambda: env.remote_write("new.py", "remote = 1\n"))

    report = engine.poll_once()

    assert report.pushed == [] and "new.py" in report.skipped
    assert env.remote("new.py") == b"remote = 1\n"


@needs_git
def test_remote_edit_during_merge_is_not_overwritten(env: Env) -> None:
    """Finding 2 for _merge: the UI saves again while merging."""
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)
    env.remote_write("notebook.py", REMOTE_EDIT)
    env.backend.before_write("notebook.py", lambda: env.remote_write("notebook.py", _later(REMOTE_EDIT)))

    report = engine.poll_once()

    assert report.merged == []
    assert env.remote("notebook.py") == _later(REMOTE_EDIT).encode()
    assert env.read_local("notebook.py") == LOCAL_EDIT.encode(), "local must not get a half-applied merge"

    assert engine.poll_once().merged == ["notebook.py"]
    assert env.read_local("notebook.py") == env.remote("notebook.py") == _later(MERGED).encode()


def test_remote_changed_between_hash_and_read_is_skipped(env: Env) -> None:
    engine = _started(env)
    env.remote_write("notebook.py", REMOTE_EDIT)
    env.backend.before_read("notebook.py", lambda: env.remote_write("notebook.py", _later(REMOTE_EDIT)))

    report = engine.poll_once()

    assert report.pulled == [] and "notebook.py" in report.skipped
    assert env.read_local("notebook.py") == NOTEBOOK.encode()
    assert engine.poll_once().pulled == ["notebook.py"]
    assert env.read_local("notebook.py") == _later(REMOTE_EDIT).encode()


def test_noop_base_is_the_hashed_content_not_a_reread(env: Env) -> None:
    """Finding 3 (repro scenario_noop_reread): both sides converge on v1, then
    the user saves v2 mid-poll. v2 must not become the base, or the next poll
    would pull v1 over it."""
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_EDIT)
    env.remote_write("notebook.py", LOCAL_EDIT)
    env.backend.after_hash_files(lambda: env.write_local("notebook.py", _later(LOCAL_EDIT)))

    engine.poll_once()
    assert env.read_local("notebook.py") == _later(LOCAL_EDIT).encode()

    # The base is v1 (read back from the remote, which still had it), so the
    # local save is a plain push, not reverted and not a conflict.
    report = engine.poll_once()
    assert report.pushed == ["notebook.py"], report
    assert report.pulled == [] and report.conflicts == []
    assert env.read_local("notebook.py") == env.remote("notebook.py") == _later(LOCAL_EDIT).encode()


@needs_git
def test_resolution_base_is_the_hashed_content(env: Env) -> None:
    """Finding 3 in conflict resolution: sides equal when the copy is deleted,
    but the user saves again mid-poll; that save must be pushed later, not
    reverted."""
    engine = _started(env)
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)
    engine.poll_once()
    env.remote_write("notebook.py", LOCAL_CLASH)
    (env.root / "notebook.remote.py").unlink()
    env.backend.after_hash_files(lambda: env.write_local("notebook.py", _later(LOCAL_CLASH)))

    engine.poll_once()
    report = engine.poll_once()

    assert report.pushed == ["notebook.py"] and report.pulled == [] and report.conflicts == []
    assert env.read_local("notebook.py") == env.remote("notebook.py") == _later(LOCAL_CLASH).encode()


# -- conflicts: remote edits while open (finding 4) ---------------------------


def _clash(env: Env, engine: SyncEngine) -> None:
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)
    assert engine.poll_once().conflicts == ["notebook.py"]


@needs_git
def test_open_conflict_copy_follows_remote_edits(env: Env) -> None:
    """Repro scenario_conflict_then_more_remote_edits: UI work after the
    conflict is shown in the copy, and resolving keeps the local file only
    against the remote the user has seen."""
    engine = _started(env)
    _clash(env, engine)
    more = REMOTE_CLASH + "# more remote work\n"
    env.remote_write("notebook.py", more)

    env.backend.reset_counters()
    report = engine.poll_once()

    assert report.conflicts == ["notebook.py"]
    assert env.backend.writes == []
    assert env.read_local("notebook.remote.py") == more.encode()
    assert env.read_local("notebook.py") == LOCAL_CLASH.encode()
    assert env.remote("notebook.py") == more.encode()

    # The user has seen `more` and resolves: the local version wins.
    (env.root / "notebook.remote.py").unlink()
    report = engine.poll_once()
    assert report.pushed == ["notebook.py"] and engine.conflicts == []
    assert env.remote("notebook.py") == LOCAL_CLASH.encode()


@needs_git
def test_remote_edit_after_copy_deleted_reraises_conflict(env: Env) -> None:
    engine = _started(env)
    _clash(env, engine)
    (env.root / "notebook.remote.py").unlink()
    newer = REMOTE_CLASH + "# typed in the UI after the copy was made\n"
    env.remote_write("notebook.py", newer)  # lands before the next poll

    report = engine.poll_once()

    assert report.pushed == [] and report.conflicts == ["notebook.py"]
    assert env.remote("notebook.py") == newer.encode(), "remote edit must not be overwritten"
    assert env.read_local("notebook.remote.py") == newer.encode()
    assert env.read_local("notebook.py") == LOCAL_CLASH.encode()

    # Deleting the fresh copy now resolves in favour of local.
    (env.root / "notebook.remote.py").unlink()
    assert engine.poll_once().pushed == ["notebook.py"]
    assert env.remote("notebook.py") == LOCAL_CLASH.encode()


@needs_git
def test_remote_edit_during_resolution_push_keeps_conflict(env: Env) -> None:
    engine = _started(env)
    _clash(env, engine)
    (env.root / "notebook.remote.py").unlink()
    newer = REMOTE_CLASH + "# saved during the push\n"
    env.backend.before_write("notebook.py", lambda: env.remote_write("notebook.py", newer))

    report = engine.poll_once()

    assert report.pushed == [] and engine.conflicts == ["notebook.py"]
    assert env.remote("notebook.py") == newer.encode()

    report = engine.poll_once()
    assert report.conflicts == ["notebook.py"]
    assert env.read_local("notebook.remote.py") == newer.encode()


@needs_git
def test_user_edited_copy_is_not_refreshed(env: Env) -> None:
    engine = _started(env)
    _clash(env, engine)
    env.write_local("notebook.remote.py", "my scratch merge\n")
    newer = REMOTE_CLASH + "# more\n"
    env.remote_write("notebook.py", newer)

    assert engine.poll_once().conflicts == ["notebook.py"]
    assert env.read_local("notebook.remote.py") == b"my scratch merge\n"

    # On resolution the remote differs from what the copy last showed:
    # re-raised with a fresh copy, remote untouched.
    (env.root / "notebook.remote.py").unlink()
    assert engine.poll_once().conflicts == ["notebook.py"]
    assert env.remote("notebook.py") == newer.encode()
    assert env.read_local("notebook.remote.py") == newer.encode()


# -- conflict copy names (finding 9) -------------------------------------------


def test_conflict_copy_name_numbered() -> None:
    assert conflict_copy_name("nb.py", 2) == "nb.remote-2.py"
    assert conflict_copy_name("Makefile", 3) == "Makefile.remote-3"


@needs_git
def test_conflict_copy_never_overwrites_user_file(env: Env) -> None:
    """Repro scenario_existing_remote_named_file."""
    mine = b"MY OWN FILE, not a conflict copy\n"
    engine = _started(env, {"notebook.py": NOTEBOOK, "notebook.remote.py": mine})
    assert env.remote("notebook.remote.py") == mine  # a normal project file

    _clash(env, engine)

    assert env.read_local("notebook.remote.py") == mine
    assert env.read_local("notebook.remote-2.py") == REMOTE_CLASH.encode()
    # The engine's copy is not synced; the user's file still is.
    env.write_local("notebook.remote.py", mine + b"edit\n")
    rebuilt = env.engine()  # copies are tracked in the state file
    report = rebuilt.poll_once()
    assert report.pushed == ["notebook.remote.py"] and report.conflicts == ["notebook.py"]
    assert env.remote("notebook.remote-2.py") is None

    (env.root / "notebook.remote-2.py").unlink()
    assert rebuilt.poll_once().pushed == ["notebook.py"]
    assert rebuilt.conflicts == []


@needs_git
def test_conflict_copy_avoids_remote_only_names(env: Env) -> None:
    engine = _started(env)
    env.remote_write("notebook.remote.py", "remote user file\n")
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)

    engine.poll_once()

    assert env.read_local("notebook.remote-2.py") == REMOTE_CLASH.encode()
    assert env.read_local("notebook.remote.py") == b"remote user file\n"  # pulled as a normal file


# -- path safety (finding 5) ---------------------------------------------------


@pytest.mark.parametrize(
    ("path", "ok"),
    [
        ("a.py", True),
        ("dir/sub/a.py", True),
        ("..", False),
        ("../escaped.py", False),
        ("a/../../b.py", False),
        ("/etc/passwd", False),
        ("", False),
        ("a//b.py", False),
        ("./a.py", False),
        ("a/", False),
        ("a\0b", False),
    ],
)
def test_is_safe_relpath(path: str, ok: bool) -> None:
    assert sync.is_safe_relpath(path) is ok


def test_unsafe_remote_paths_are_never_written(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repro repro_traversal.py."""
    engine = _started(env)
    outside_abs = str(tmp_path / "outside_abs.py")
    evil = ["../escaped.py", outside_abs, "a/../../x.py", ""]
    real_hash, real_read = env.backend.hash_files, env.backend.read_file
    monkeypatch.setattr(
        env.backend, "hash_files", lambda h, root=".": {**real_hash(h, root), **{p: "beef" for p in evil}}
    )
    monkeypatch.setattr(env.backend, "read_file", lambda h, p: b"pwned\n" if p in evil else real_read(h, p))

    report = engine.poll_once()

    assert set(evil) <= set(report.skipped)
    assert report.pulled == []
    assert not (tmp_path / "escaped.py").exists()
    assert not Path(outside_abs).exists()
    assert not (tmp_path / "x.py").exists()


def test_remote_path_under_local_symlinked_dir_is_not_written(env: Env, tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    engine = _started(env)
    (env.root / "data").symlink_to(datasets, target_is_directory=True)
    env.remote_write("data/x.json", "{}")

    report = engine.poll_once()

    assert "data/x.json" in report.skipped and report.pulled == []
    assert list(datasets.iterdir()) == []


def test_remote_change_to_local_symlinked_file_is_not_written(env: Env, tmp_path: Path) -> None:
    real = tmp_path / "real.py"
    real.write_text("r = 1\n")
    (env.root / "link.py").symlink_to(real)
    engine = _started(env)
    assert env.remote("link.py") == b"r = 1\n"  # pushed through the link
    env.remote_write("link.py", "r = 2\n")

    report = engine.poll_once()

    assert "link.py" in report.skipped
    assert (env.root / "link.py").is_symlink()
    assert real.read_text() == "r = 1\n"


# -- per-tick cost (finding 10) --------------------------------------------------


def test_rejected_remote_cache_persists_across_engines(env: Env) -> None:
    """Repro repro_reread.py: the CLI rebuilds the engine every tick."""
    _started(env)
    env.remote_write("big_output.json", b"0" * (PULL_MAX_BYTES + 1))

    for _ in range(3):
        report = env.engine().poll_once()
        assert "big_output.json" in report.skipped

    assert env.backend.reads == ["big_output.json"]


def _age(env: Env, *rels: str) -> None:
    """Make files' mtimes old (ctime cannot be set back; see _settled_clock)."""
    old = os.stat(env.root).st_mtime - 100
    for rel in rels:
        os.utime(env.root / rel, (old, old))


def _settled_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the engine's clock run 10 s ahead, so files just written count as
    settled (their ctime older than RECENT_MTIME_NS) and get cached."""
    real = sync._now_ns
    monkeypatch.setattr(sync, "_now_ns", lambda: real() + 10_000_000_000)


def _spy_local_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = Path.read_bytes

    def spy(self: Path) -> bytes:
        seen.append(self.name)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    return seen


def test_unchanged_local_files_are_not_reread(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.write_local("notebook.py", NOTEBOOK)
    env.write_local("other.py", "o = 1\n")
    _age(env, "notebook.py", "other.py")
    _settled_clock(monkeypatch)
    env.engine().initial_push()
    seen = _spy_local_reads(monkeypatch)

    report = env.engine().poll_once()  # a fresh engine: the cache is in the state file

    assert not report.changed
    assert seen == []

    env.write_local("other.py", "o = 2\n")  # same size, new mtime
    report = env.engine().poll_once()
    assert report.pushed == ["other.py"]
    assert "notebook.py" not in seen


def test_oversized_local_file_is_never_read(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.write_local("notebook.py", NOTEBOOK)
    env.write_local("weights.bin", b"\0" * 5000)  # over the patched 4096 limit
    seen = _spy_local_reads(monkeypatch)

    report = env.engine().initial_push()
    report2 = env.engine().poll_once()

    assert "weights.bin" in report.skipped and "weights.bin" in report2.skipped
    assert "weights.bin" not in seen


def test_oversized_local_file_is_not_overwritten_by_pull(env: Env) -> None:
    engine = _started(env, {"notebook.py": NOTEBOOK, "weights.bin": b"\0" * 5000})
    env.remote_write("weights.bin", b"remote")

    report = engine.poll_once()

    assert "weights.bin" in report.skipped
    assert env.read_local("weights.bin") == b"\0" * 5000


# -- unreadable files, state file (findings 12, 13) ------------------------------


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read anything")
def test_unreadable_local_file_does_not_break_poll(env: Env) -> None:
    engine = _started(env, {"notebook.py": NOTEBOOK, "secret.py": "s = 1\n"})
    secret = env.root / "secret.py"
    secret.chmod(0)
    try:
        env.write_local("notebook.py", LOCAL_EDIT)
        env.remote_write("secret.py", "s = 2\n")

        report = engine.poll_once()

        assert report.pushed == ["notebook.py"]
        assert "secret.py" in report.skipped
    finally:
        secret.chmod(0o644)
    assert secret.read_text() == "s = 1\n"


def test_unwritable_local_dir_does_not_break_poll(env: Env) -> None:
    engine = _started(env)
    env.write_local("blocker", "a file where the remote has a directory\n")
    env.remote_write("blocker/inner.py", "x = 1\n")
    env.remote_write("fine.py", "f = 1\n")

    report = engine.poll_once()

    assert "blocker/inner.py" in report.skipped
    assert "fine.py" in report.pulled


def test_state_file_is_private(env: Env) -> None:
    env.state_path.write_text("{}")
    env.state_path.chmod(0o644)
    _started(env)
    assert env.state_path.stat().st_mode & 0o777 == 0o600


def test_version1_state_is_loaded(env: Env) -> None:
    import base64
    import hashlib
    import json

    env.write_local("notebook.py", NOTEBOOK)
    env.write_local("notebook.remote.py", "copy\n")
    env.remote_write("notebook.py", NOTEBOOK)
    env.state_path.write_text(json.dumps({
        "version": 1,
        "bases": {"notebook.py": {
            "hash": hashlib.sha256(NOTEBOOK.encode()).hexdigest(),
            "content": base64.b64encode(NOTEBOOK.encode()).decode(),
        }},
        "conflicts": ["notebook.py"],
    }))
    engine = env.engine()
    assert engine.conflicts == ["notebook.py"]
    assert engine.poll_once().conflicts == ["notebook.py"]
    assert env.remote("notebook.remote.py") is None


# -- local stat cache: ctime and inode (N8) -------------------------------------


def _local_index(env: Env) -> dict[str, list[object]]:
    import json

    return json.loads(env.state_path.read_text())["local_index"]


def test_ctime_only_change_is_detected(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """An in-place edit that keeps size and mtime (os.utime, cp -p into an
    existing file) still changes ctime, so the cache is not trusted."""
    env.write_local("notebook.py", NOTEBOOK)
    path = env.write_local("other.py", "o = 1\n")
    _age(env, "notebook.py", "other.py")
    _settled_clock(monkeypatch)
    env.engine().initial_push()
    assert "other.py" in _local_index(env)
    before = os.stat(path)

    with open(path, "r+b") as f:  # same inode, same size
        f.write(b"o = 2\n")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (before.st_size, before.st_mtime_ns, before.st_ino)
    assert after.st_ctime_ns != before.st_ctime_ns

    report = env.engine().poll_once()

    assert report.pushed == ["other.py"]
    assert env.remote("other.py") == b"o = 2\n"


def test_replaced_file_with_same_size_and_mtime_is_detected(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """rsync -t style: a new file renamed over the old one, mtime copied."""
    env.write_local("notebook.py", NOTEBOOK)
    path = env.write_local("other.py", "o = 1\n")
    _age(env, "notebook.py", "other.py")
    _settled_clock(monkeypatch)
    env.engine().initial_push()
    before = os.stat(path)

    tmp = env.root / ".tmp-other"
    tmp.write_bytes(b"o = 3\n")
    os.utime(tmp, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(tmp, path)

    report = env.engine().poll_once()

    assert report.pushed == ["other.py"]
    assert env.remote("other.py") == b"o = 3\n"


def test_recent_ctime_is_not_cached(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.write_local("notebook.py", NOTEBOOK)
    _age(env, "notebook.py")  # old mtime, but ctime is now
    env.engine().initial_push()
    assert _local_index(env) == {}

    seen = _spy_local_reads(monkeypatch)
    env.engine().poll_once()
    assert "notebook.py" in seen


def test_version2_state_is_loaded(env: Env) -> None:
    import base64
    import hashlib
    import json

    env.write_local("notebook.py", LOCAL_EDIT)
    env.remote_write("notebook.py", NOTEBOOK)
    digest = hashlib.sha256(NOTEBOOK.encode()).hexdigest()
    env.state_path.write_text(json.dumps({
        "version": 2,
        "bases": {"notebook.py": {"hash": digest, "content": base64.b64encode(NOTEBOOK.encode()).decode()}},
        "conflicts": {},
        "rejected_remote": {},
        # A stale v2 entry that would hide the local edit if it were trusted.
        "local_index": {"notebook.py": [len(LOCAL_EDIT), 1, digest]},
    }))

    report = env.engine().poll_once()

    assert report.pushed == ["notebook.py"] and report.conflicts == []
    assert env.remote("notebook.py") == LOCAL_EDIT.encode()
    assert json.loads(env.state_path.read_text())["version"] == sync.STATE_VERSION


# -- several engines on one state file (N1) -------------------------------------


@needs_git
def test_conflict_copy_made_by_another_engine_is_not_pushed(env: Env) -> None:
    """The CLI loop's long-lived engine must see the conflict `nbgpu sync`
    created through its own engine on the same state file."""
    loop = _started(env)
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)

    other = env.engine().poll_once()  # e.g. `nbgpu sync`
    assert other.conflicts == ["notebook.py"]
    assert env.read_local("notebook.remote.py") == REMOTE_CLASH.encode()

    report = loop.poll_once()

    assert report.conflicts == ["notebook.py"]
    assert report.pushed == [] and report.merged == []
    assert env.remote("notebook.remote.py") is None, "the copy was pushed as a project file"
    assert env.read_local("notebook.remote-2.py") is None, "a second copy was made"
    assert env.read_local("notebook.py") == LOCAL_CLASH.encode()
    assert env.remote("notebook.py") == REMOTE_CLASH.encode()
    assert env.engine().conflicts == ["notebook.py"], "the state file dropped the conflict"
    assert loop.conflicts == ["notebook.py"]

    # Resolved by deleting the copy: whichever engine polls next pushes the local file.
    (env.root / "notebook.remote.py").unlink()
    assert env.engine().poll_once().pushed == ["notebook.py"]
    assert loop.poll_once().conflicts == []
    assert env.remote("notebook.py") == LOCAL_CLASH.encode()


def test_engines_interleaving_share_bases(env: Env) -> None:
    """A base recorded by one engine is used by the other: a remote edit after
    another engine pushed is a plain pull, not a merge against a stale base."""
    a = _started(env)
    b = env.engine()

    env.write_local("notebook.py", LOCAL_EDIT)
    assert b.poll_once().pushed == ["notebook.py"]

    remote_next = LOCAL_EDIT.replace("lr = 0.001", "lr = 0.01")
    env.remote_write("notebook.py", remote_next)
    report = a.poll_once()
    assert report.pulled == ["notebook.py"] and report.merged == [] and report.conflicts == []

    env.write_local("new.py", "n = 1\n")
    assert b.poll_once().pushed == ["new.py"]
    assert not a.poll_once().changed
    assert env.read_local("notebook.py") == remote_next.encode()


def test_operations_exclude_each_other(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync, "LOCK_TIMEOUT_S", 0.2)
    a = _started(env)
    b = env.engine()
    errors: list[BaseException] = []

    def other_engine_polls_mid_poll() -> None:
        try:
            b.poll_once()
        except sync.SyncLockTimeout as exc:
            errors.append(exc)

    env.backend.after_hash_files(other_engine_polls_mid_poll)
    a.poll_once()

    assert len(errors) == 1 and "sync-state.json.lock" in str(errors[0])
    assert not b.poll_once().changed  # the lock is free again


def test_lock_is_released_and_state_saved_when_a_poll_fails(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbgpu.backends.base import BackendError

    monkeypatch.setattr(sync, "LOCK_TIMEOUT_S", 0.2)
    engine = _started(env)
    env.write_local("new.py", "n = 1\n")
    real = env.backend.hash_files

    def boom(handle: RuntimeHandle, root: str = ".") -> dict[str, str]:
        raise BackendError("runtime went away")

    monkeypatch.setattr(env.backend, "hash_files", boom)
    with pytest.raises(BackendError):
        engine.poll_once()
    monkeypatch.setattr(env.backend, "hash_files", real)

    assert env.engine().poll_once().pushed == ["new.py"]


# -- synced manifest --------------------------------------------------------------

MANIFEST = "/runtimes/nbgpu-meta/sync-manifest.json"


class _Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(sync, "_now", c)
    return c


def _menv(env: Env) -> SyncEngine:
    return SyncEngine(env.root, env.backend, env.handle, env.state_path, manifest_path=MANIFEST)


def _manifest(env: Env) -> dict[str, object] | None:
    import json

    raw = env.backend.files[env.handle.runtime_id].get(MANIFEST)
    return None if raw is None else json.loads(raw)


def _sha(data: str | bytes) -> str:
    import hashlib

    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def test_manifest_written_on_change_and_refresh_only(env: Env, clock: _Clock) -> None:
    env.write_local("notebook.py", NOTEBOOK)
    report = _menv(env).initial_push()
    assert report.warnings == []
    assert _manifest(env) == {"version": 1, "written_at": clock.t, "files": {"notebook.py": _sha(NOTEBOOK)}}

    # Quiet polls (fresh engines, as the CLI does) inside the refresh interval: no write.
    env.backend.reset_counters()
    clock.t += 30
    _menv(env).poll_once()
    clock.t += 29.9
    _menv(env).poll_once()
    assert env.backend.writes == []

    # Refresh interval reached: rewritten with a new timestamp.
    clock.t += 0.1
    _menv(env).poll_once()
    assert env.backend.writes == [MANIFEST]
    assert _manifest(env)["written_at"] == clock.t  # type: ignore[index]

    # A change in the bases: written at once.
    env.backend.reset_counters()
    clock.t += 1
    env.remote_write("notebook.py", REMOTE_EDIT)
    assert _menv(env).poll_once().pulled == ["notebook.py"]
    assert env.backend.writes == [MANIFEST]
    assert _manifest(env) == {"version": 1, "written_at": clock.t, "files": {"notebook.py": _sha(REMOTE_EDIT)}}

    env.backend.reset_counters()
    clock.t += 1
    _menv(env).poll_once()
    assert env.backend.writes == []


def test_engine_without_manifest_path_never_writes_one(env: Env, clock: _Clock) -> None:
    _started(env).poll_once()
    assert _manifest(env) is None


@needs_git
def test_manifest_lists_conflicts_by_remote_hash_and_rejected_files(env: Env, clock: _Clock) -> None:
    env.write_local("notebook.py", NOTEBOOK)
    _menv(env).initial_push()
    env.write_local("notebook.py", LOCAL_CLASH)
    env.remote_write("notebook.py", REMOTE_CLASH)
    env.remote_write("plot.png", b"\x89PNG")

    report = _menv(env).poll_once()

    assert report.conflicts == ["notebook.py"]
    files = _manifest(env)["files"]  # type: ignore[index]
    assert files == {"notebook.py": _sha(REMOTE_CLASH), "plot.png": _sha(b"\x89PNG")}

    # The remote moves on while the conflict is open: the manifest follows it.
    remote_next = REMOTE_CLASH.replace("lr = 0.02", "lr = 0.03")
    env.remote_write("notebook.py", remote_next)
    _menv(env).poll_once()
    assert _manifest(env)["files"]["notebook.py"] == _sha(remote_next)  # type: ignore[index]


def test_manifest_write_failure_is_a_warning(env: Env, clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbgpu.backends.base import BackendError

    env.write_local("notebook.py", NOTEBOOK)
    real = env.backend.write_file

    def flaky(handle: RuntimeHandle, path: str, data: bytes, *, expect: str | None = None) -> bool:
        if path == MANIFEST:
            raise BackendError("disk full")
        return real(handle, path, data, expect=expect)

    monkeypatch.setattr(env.backend, "write_file", flaky)
    report = _menv(env).initial_push()
    assert report.pushed == ["notebook.py"]
    assert len(report.warnings) == 1 and "disk full" in report.warnings[0]
    assert env.engine().conflicts == []  # the state was saved

    # Retried on the next operation, though nothing changed.
    monkeypatch.setattr(env.backend, "write_file", real)
    clock.t += 1
    report = _menv(env).poll_once()
    assert report.warnings == []
    assert _manifest(env)["files"] == {"notebook.py": _sha(NOTEBOOK)}  # type: ignore[index]


@pytest.mark.parametrize("path", ["relative/manifest.json", "/runtimes/fake-1/project/m.json", "/runtimes/fake-1/project"])
def test_manifest_path_must_be_absolute_and_outside_project(env: Env, path: str) -> None:
    with pytest.raises(ValueError, match="manifest_path"):
        SyncEngine(env.root, env.backend, env.handle, env.state_path, manifest_path=path)


# --------------------------------------------------------------------------
# deferred manifest (the sync loop's engine)


def _denv(env: Env) -> SyncEngine:
    return SyncEngine(env.root, env.backend, env.handle, env.state_path, manifest_path=MANIFEST, defer_manifest=True)


def test_deferred_manifest_rides_on_the_next_hash(env: Env, clock) -> None:
    env.write_local("notebook.py", NOTEBOOK)
    _menv(env).initial_push()  # one-shot: written right away
    first = _manifest(env)
    assert first is not None and "notebook.py" in first["files"]  # type: ignore[operator]

    engine = _denv(env)
    env.write_local("helper.py", "x = 1\n")
    writes_before = len(env.backend.writes)
    engine.poll_once()  # pushes helper.py; the new manifest is only queued
    assert env.backend.writes[writes_before:] == ["helper.py"]  # no separate manifest write
    assert "helper.py" not in _manifest(env)["files"]  # type: ignore[index,operator]
    hashes = env.backend.hash_calls
    engine.poll_once()  # the next hash carries it
    assert env.backend.hash_calls == hashes + 1
    assert "helper.py" in _manifest(env)["files"]  # type: ignore[index,operator]
    clock.t += 1
    writes_before = len(env.backend.writes)
    engine.poll_once()  # nothing changed: nothing queued or written
    assert env.backend.writes[writes_before:] == [] and engine.flush_manifest() is None


def test_flush_writes_a_queued_manifest(env: Env, clock) -> None:
    env.write_local("notebook.py", NOTEBOOK)
    engine = _denv(env)
    engine.initial_push()
    assert _manifest(env) is None  # queued, not written
    engine.flush_manifest()
    assert "notebook.py" in _manifest(env)["files"]  # type: ignore[index,operator]
    assert engine.flush_manifest() is None
