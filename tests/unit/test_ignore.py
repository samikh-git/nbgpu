"""Ignore rules (.gitignore + [tool.nbgpu] exclude), the batched initial push,
and the Modal unpack script run locally."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nbgpu import lifecycle, sync
from nbgpu.backends.base import ABSENT, RuntimeHandle
from nbgpu.ignore import IgnoreRules
from nbgpu.spec import SpecError, load_spec
from nbgpu.sync import SyncEngine
from tests.fakes import FakeBackend, fake_request


def _write(root: Path, rel: str, data: str = "x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data)


# --------------------------------------------------------------------------
# IgnoreRules


def test_root_gitignore_files_dirs_and_negation(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "*.log\n!keep.log\nbuild/\n/top.txt\n.env\n")
    rules = IgnoreRules(tmp_path)
    assert rules.is_ignored("a.log")
    assert rules.is_ignored("deep/a.log")
    assert not rules.is_ignored("keep.log")
    assert rules.is_ignored("build", is_dir=True)
    assert rules.is_ignored("build/x.py")
    assert rules.is_ignored("src/build/x.py")
    assert rules.is_ignored("top.txt")
    assert not rules.is_ignored("sub/top.txt")  # anchored to the root
    assert rules.is_ignored(".env")
    assert not rules.is_ignored("notebook.py")
    assert not rules.is_ignored(".gitignore")


def test_nested_gitignore_is_relative_and_overrides(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "*.csv\n")
    _write(tmp_path, "data/.gitignore", "!small.csv\n/raw\n")
    rules = IgnoreRules(tmp_path)
    assert rules.is_ignored("big.csv")
    assert rules.is_ignored("data/big.csv")
    assert not rules.is_ignored("data/small.csv")
    assert rules.is_ignored("data/raw/x.py")
    assert not rules.is_ignored("raw/x.py")  # /raw is anchored to data/


def test_nothing_is_reincluded_under_an_ignored_directory(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "out/\n!out/keep.py\n")
    assert IgnoreRules(tmp_path).is_ignored("out/keep.py")


def test_exclude_patterns_win_over_gitignore(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "!data/\n")
    rules = IgnoreRules(tmp_path, ["data/", "*.ckpt", "secrets.toml"])
    assert rules.is_ignored("data/train.parquet")
    assert rules.is_ignored("models/a.ckpt")
    assert rules.is_ignored("secrets.toml")
    assert not rules.is_ignored("notebook.py")


def test_git_info_exclude_is_read(tmp_path: Path) -> None:
    _write(tmp_path, ".git/info/exclude", "scratch/\n")
    assert IgnoreRules(tmp_path).is_ignored("scratch/a.py")


def test_gitignore_edits_take_effect_after_refresh(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "")
    rules = IgnoreRules(tmp_path)
    assert not rules.is_ignored("out/a.py")
    _write(tmp_path, ".gitignore", "out/\n# changed size\n")
    rules.refresh()
    assert rules.is_ignored("out/a.py")


def test_git_info_exclude_edits_take_effect_after_refresh(tmp_path: Path) -> None:
    rules = IgnoreRules(tmp_path)  # no .git yet
    assert not rules.is_ignored("scratch/a.py")
    _write(tmp_path, ".git/info/exclude", "scratch/\n")
    rules.refresh()
    assert rules.is_ignored("scratch/a.py")


# --------------------------------------------------------------------------
# spec


def test_spec_exclude(tmp_path: Path) -> None:
    nb = tmp_path / "n.py"
    nb.write_text('# /// script\n# [tool.nbgpu]\n# gpu = "A10"\n# exclude = ["data/", "*.ckpt"]\n# ///\n')
    spec = load_spec(nb)
    assert spec.exclude == ("data/", "*.ckpt")
    assert spec.compute.gpu == "A10"
    nb.write_text("# /// script\n# [tool.nbgpu]\n# ///\n")
    assert load_spec(nb).exclude == ()


@pytest.mark.parametrize("value", ['"data/"', '[1]', '[""]'])
def test_spec_exclude_rejects_non_lists(tmp_path: Path, value: str) -> None:
    nb = tmp_path / "n.py"
    nb.write_text(f"# /// script\n# [tool.nbgpu]\n# exclude = {value}\n# ///\n")
    with pytest.raises(SpecError, match="exclude"):
        load_spec(nb)


# --------------------------------------------------------------------------
# sync engine


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, FakeBackend, RuntimeHandle, Path]:
    backend = FakeBackend()
    handle = backend.create(fake_request())
    root = tmp_path / "project"
    root.mkdir()
    return root, backend, handle, tmp_path / "state.json"


def test_ignored_files_are_neither_pushed_nor_pulled(setup) -> None:
    root, backend, handle, state = setup
    _write(root, ".gitignore", ".env\ncheckpoints/\n")
    _write(root, "notebook.py", "print(1)\n")
    _write(root, ".env", "SECRET=1")
    _write(root, "checkpoints/a.txt", "big")
    _write(root, "data/x.csv", "1,2")
    engine = SyncEngine(root, backend, handle, state, exclude=["data/"])
    report = engine.initial_push()
    assert sorted(report.pushed) == [".gitignore", "notebook.py"]
    assert backend.remote_read(handle, ".env") is None
    assert backend.remote_read(handle, "data/x.csv") is None

    backend.remote_write(handle, "checkpoints/new.txt", b"out")
    backend.remote_write(handle, "data/y.json", b"{}")
    report = engine.poll_once()
    assert report.pulled == []
    assert not (root / "checkpoints/new.txt").exists()
    assert not (root / "data/y.json").exists()


def test_a_newly_ignored_synced_file_is_left_alone_on_both_sides(setup) -> None:
    root, backend, handle, state = setup
    _write(root, "notebook.py", "print(1)\n")
    _write(root, "notes.txt", "v1")
    engine = SyncEngine(root, backend, handle, state)
    engine.initial_push()
    _write(root, ".gitignore", "notes.txt\n")
    _write(root, "notes.txt", "v2 local")
    engine.poll_once()
    assert backend.remote_read(handle, "notes.txt") == b"v1"  # not pushed, not deleted
    assert (root / "notes.txt").read_text() == "v2 local"


class BulkFake(FakeBackend):
    """A FakeBackend with write_new_files, recording each batch."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    def write_new_files(self, handle: RuntimeHandle, files: dict[str, bytes]) -> set[str]:
        self.batches.append(sorted(files))
        written = set()
        for rel, data in files.items():
            if self.remote_read(handle, rel) is None:
                FakeBackend.write_file(self, handle, rel, data, expect=ABSENT)
                written.add(rel)
        return written


def test_initial_push_uses_write_new_files_in_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync, "BULK_PUSH_BATCH_BYTES", 10)
    backend = BulkFake()
    handle = backend.create(fake_request())
    root = tmp_path / "p"
    for i in range(5):
        _write(root, f"f{i}.py", "123456")  # 6 bytes: two files per batch
    backend.remote_write(handle, "f4.py", b"already there")
    report = SyncEngine(root, backend, handle, tmp_path / "s.json").initial_push()
    assert backend.batches == [["f0.py", "f1.py"], ["f2.py", "f3.py"], ["f4.py"]]
    assert sorted(report.pushed) == ["f0.py", "f1.py", "f2.py", "f3.py"]
    assert "f4.py" in report.skipped
    assert backend.remote_read(handle, "f4.py") == b"already there"
    # Bases were recorded: an unchanged project is a no-op afterwards.
    again = SyncEngine(root, backend, handle, tmp_path / "s.json").poll_once()
    assert again.pushed == [] and again.pulled == []


# --------------------------------------------------------------------------
# lifecycle checks


def test_check_push_size_skips_ignored_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "PUSH_MAX_FILES", 3)
    _write(tmp_path, "notebook.py")
    _write(tmp_path, ".gitignore", "data/\n")
    for i in range(10):
        _write(tmp_path, f"data/{i}.csv")
        _write(tmp_path, f"ckpt/{i}.bin")
    with pytest.raises(lifecycle.StartError):
        lifecycle.check_push_size(tmp_path)
    assert lifecycle.check_push_size(tmp_path, ["ckpt/"])[0] == 2


def test_an_ignored_notebook_is_refused(tmp_path: Path) -> None:
    with pytest.raises(lifecycle.StartError, match="ignored"):
        lifecycle._check_notebook_not_ignored("nb/a.py", tmp_path, ["nb/"])
    lifecycle._check_notebook_not_ignored("nb/a.py", tmp_path, ["data/"])


# --------------------------------------------------------------------------
# Modal unpack script, run with the local python


def test_modal_bulk_code_is_create_only_and_safe(tmp_path: Path) -> None:
    from nbgpu.backends.modal import BULK_CODE, build_tar, parse_bulk_output

    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "existing.txt").write_text("old")
    (tmp_path / "outside").mkdir()
    (root / "link").symlink_to(tmp_path / "outside")
    files = {
        "a.py": b"print(1)",
        "sub/b.txt": b"b",
        "existing.txt": b"new",
        "../evil": b"x",
        "link/escape.txt": b"x",
        "deep/x/y.bin": bytes(range(256)),
    }
    archive = tmp_path / "a.tar"
    archive.write_bytes(build_tar(files))
    res = subprocess.run(
        [sys.executable, "-c", BULK_CODE, str(archive), str(root), ".nbgpu-tmp-"],
        capture_output=True, text=True, check=True,
    )
    assert parse_bulk_output(res.stdout, files) == {"a.py", "sub/b.txt", "deep/x/y.bin"}
    assert (root / "existing.txt").read_text() == "old"
    assert (root / "deep/x/y.bin").read_bytes() == bytes(range(256))
    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / "outside" / "escape.txt").exists()
    assert not archive.exists()
    assert not [p for p in root.rglob("*") if p.name.startswith(".nbgpu-tmp-")]


def test_modal_hash_code_writes_first_in_chunks(tmp_path: Path) -> None:
    import json
    import os

    from nbgpu.backends import modal as mb

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("print(1)\n")
    manifest = tmp_path / "manifest.json"
    payload = json.dumps({"files": {f"dir/file{i}.py": os.urandom(32).hex() for i in range(6000)}}).encode()
    argv = mb.hash_argv(str(root), str(tmp_path / "cache.json"), (str(manifest), payload))
    assert int(argv[argv.index("--write-first") + 2]) > 1  # needed several chunks
    assert max(len(a) for a in argv[3:]) <= mb.ARG_CHUNK
    res = subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True, check=True)
    assert json.loads(res.stdout) == {"a.py": __import__("hashlib").sha256(b"print(1)\n").hexdigest()}
    assert manifest.read_bytes() == payload
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".nbgpu-tmp-")]
    # no write-first: the usual argv still works
    res = subprocess.run([sys.executable, *mb.hash_argv(str(root), str(tmp_path / "cache.json"))[1:]],
                         capture_output=True, text=True, check=True)
    assert "a.py" in json.loads(res.stdout)
