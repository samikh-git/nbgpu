"""Unit tests for the Modal backend's pure helpers. Nothing here contacts Modal."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nbgpu.backends import modal as mb
from nbgpu.backends.base import (
    ABSENT,
    CAP_GPU,
    CAP_LIVE_INSTALL,
    CAP_IDLE_TIMEOUT,
    CAP_MAX_RUNTIME,
    CAP_VOLUMES,
    TEMP_PREFIX,
    BackendError,
    UnsupportedError,
)
from nbgpu.spec import ComputeSpec, NotebookSpec


def _spec(**compute: object) -> NotebookSpec:
    return NotebookSpec(
        path=Path("/tmp/nb.py"),
        requires_python=compute.pop("requires_python", None),  # type: ignore[arg-type]
        dependencies=tuple(compute.pop("dependencies", ())),  # type: ignore[arg-type]
        compute=ComputeSpec(**compute),  # type: ignore[arg-type]
    )


# -- construction ------------------------------------------------------------


def test_init_does_not_contact_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    import modal

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("App.lookup must be lazy")

    monkeypatch.setattr(modal.App, "lookup", boom)
    backend = mb.ModalBackend()
    assert backend.app_name == "nbgpu"
    assert backend.name == "modal"
    assert backend.bind_host == "0.0.0.0"
    assert backend.capabilities == frozenset({CAP_GPU, CAP_VOLUMES, CAP_MAX_RUNTIME, CAP_LIVE_INSTALL})
    assert CAP_IDLE_TIMEOUT not in backend.capabilities


def test_create_uses_idle_grace_not_user_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sandbox.create gets idle_timeout=IDLE_GRACE_S and timeout=max_runtime_s.
    Modal is replaced by stand-ins; nothing is created."""
    import modal

    seen: dict[str, object] = {}

    class FakeSandbox:
        object_id = "sb-fake"

    def fake_create(*args: object, **kwargs: object) -> FakeSandbox:
        seen.update(kwargs, entrypoint=args)
        return FakeSandbox()

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not reach Modal")

    monkeypatch.setattr(modal.Sandbox, "create", fake_create)
    monkeypatch.setattr(modal.App, "lookup", boom)
    monkeypatch.setattr(modal.Image, "debian_slim", lambda **k: type("I", (), {"uv_pip_install": lambda self, *p: self})())
    backend = mb.ModalBackend()
    monkeypatch.setattr(backend, "_get_app", lambda: object())
    monkeypatch.setattr(backend, "exec", lambda *a, **k: mb.ExecResult(0, "", ""))
    from nbgpu.backends.base import RuntimeRequest

    backend.create(RuntimeRequest(spec=_spec(idle_timeout_s=7200, max_runtime_s=3 * 3600), tags={"t": "1"}, ports=(8000,)))
    assert seen["idle_timeout"] == mb.IDLE_GRACE_S == 300
    assert seen["timeout"] == 3 * 3600
    assert seen["entrypoint"] == mb.ENTRYPOINT


def test_create_rejects_unknown_gpu_before_any_modal_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import modal

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not reach Modal")

    monkeypatch.setattr(modal.App, "lookup", boom)
    monkeypatch.setattr(modal.Sandbox, "create", boom)
    monkeypatch.setattr(modal.Volume, "from_name", boom)
    from nbgpu.backends.base import RuntimeRequest

    with pytest.raises(UnsupportedError) as info:
        mb.ModalBackend().create(RuntimeRequest(spec=_spec(gpu="NOT-A-REAL-GPU-9000", volumes={"/d": "v"}), tags={}))
    assert "A10G" in str(info.value) and "H100" in str(info.value)


# -- GPU ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [(None, None), ("A10G", "A10G"), ("h100", "H100"), ("H100:2", "H100:2"), ("H100!:8", "H100!:8"),
     ("a100-80gb:1", "A100-80GB"), ("B200+", "B200+"), ("T4", "T4"), ("A10:4", "A10:4")],
)
def test_parse_gpu_valid(given: str | None, expected: str | None) -> None:
    assert mb.parse_gpu(given) == expected


@pytest.mark.parametrize("bad", ["NOT-A-REAL-GPU-9000", "", "H100:0", "H100:9", "A10:5", "H100:x", "H100:-1", "H100,A100", "any"])
def test_parse_gpu_invalid(bad: str) -> None:
    with pytest.raises(UnsupportedError):
        mb.parse_gpu(bad)


def test_unknown_gpu_message_lists_known() -> None:
    with pytest.raises(UnsupportedError) as info:
        mb.parse_gpu("RTX4090")
    for name in ("A10G", "H100", "T4", "L40S"):
        assert name in str(info.value)


# -- image -------------------------------------------------------------------


def test_image_packages_adds_marimo_and_watchdog() -> None:
    assert mb.image_packages(()) == ["marimo", "watchdog", "uv"]
    assert mb.image_packages(["torch", "numpy>=2"]) == ["torch", "numpy>=2", "marimo", "watchdog", "uv"]


def test_image_packages_pins_an_unversioned_marimo() -> None:
    assert mb.image_packages(["torch"], "0.25.2") == ["torch", "marimo==0.25.2", "watchdog", "uv"]
    assert mb.image_packages(["marimo", "torch"], "0.25.2") == ["marimo==0.25.2", "torch", "watchdog", "uv"]
    assert mb.image_packages(["marimo[sql]"], "0.25.2") == ["marimo[sql]==0.25.2", "watchdog", "uv"]
    assert mb.image_packages(["marimo>=0.20"], "0.25.2") == ["marimo>=0.20", "watchdog", "uv"]  # the user's choice
    assert mb.image_packages(["torch"], None) == ["torch", "marimo", "watchdog", "uv"]


def test_image_packages_does_not_duplicate_marimo() -> None:
    assert mb.image_packages(["marimo>=0.25", "torch"]) == ["marimo>=0.25", "torch", "watchdog", "uv"]
    assert mb.image_packages(["Marimo[recommended]==0.25.2"]) == ["Marimo[recommended]==0.25.2", "watchdog", "uv"]
    assert mb.image_packages(["watchdog", "marimo", "marimo"]) == ["watchdog", "marimo", "uv"]


@pytest.mark.parametrize(
    ("requires", "local", "expected"),
    [
        (None, "3.12", "3.12"),
        (">=3.11", "3.12", "3.12"),
        (">=3.13", "3.12", "3.14"),
        ("==3.11.*", "3.12", "3.11"),
        ("==3.11", "3.12", "3.11"),
        ("~=3.12.1", "3.11", "3.12"),
        (">=3.11,<3.13", "3.13", "3.12"),
        (">3.12", "3.12", "3.12"),  # 3.12.1 satisfies >3.12
        ("<3.13", "3.14", "3.12"),
        (None, "3.9", "3.13"),
    ],
)
def test_choose_python_version(requires: str | None, local: str, expected: str) -> None:
    assert mb.choose_python_version(requires, local=local) == expected


@pytest.mark.parametrize("requires", ["<3.10", "==3.9.*", ">=4", "banana"])
def test_choose_python_version_unsatisfiable(requires: str) -> None:
    with pytest.raises(UnsupportedError):
        mb.choose_python_version(requires, local="3.12")


def test_validate_spec() -> None:
    settings = mb.validate_spec(_spec(gpu="a10g:2", dependencies=["torch"], requires_python=">=3.12"))
    assert settings["gpu"] == "A10G:2"
    # marimo is pinned to the laptop's version, whose frontend the proxy serves
    assert settings["packages"] == ["torch", f"marimo=={mb.local_marimo_version()}", "watchdog", "uv"]
    assert settings["python_version"] in mb.SUPPORTED_PYTHONS
    with pytest.raises(UnsupportedError):
        mb.validate_spec(_spec(max_runtime_s=25 * 3600))
    with pytest.raises(UnsupportedError):
        mb.validate_spec(_spec(volumes={"relative": "v"}))


# -- paths, tags, timeouts ---------------------------------------------------


def test_resolve_path() -> None:
    root = mb.REMOTE_ROOT
    assert mb.resolve_path(root, ".") == root
    assert mb.resolve_path(root, "a/b.txt") == f"{root}/a/b.txt"
    assert mb.resolve_path(root, "a/../c") == f"{root}/c"
    assert mb.resolve_path(root, "/data/x") == "/data/x"


def test_temp_path_is_in_same_directory_with_temp_prefix() -> None:
    tmp = mb.temp_path_for("/workspace/project/d/f.py", token="abc")
    assert tmp == f"/workspace/project/d/{TEMP_PREFIX}abc"
    name = mb.temp_path_for("/workspace/project/d/f.py").rsplit("/", 1)[1]
    assert name.startswith(TEMP_PREFIX)
    assert mb.temp_path_for("/workspace/project/d/f.py") != mb.temp_path_for("/workspace/project/d/f.py")


def test_expect_arg() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    assert mb.expect_arg(None) == mb.EXPECT_ANY
    assert mb.expect_arg(ABSENT) == mb.EXPECT_ABSENT
    assert mb.expect_arg(digest) == digest
    assert mb.expect_arg(digest.upper()) == digest
    for bad in ("absent", "*", "abc", "g" * 64, digest + "0"):
        with pytest.raises(BackendError):
            mb.expect_arg(bad)
    # The one-liner hard-codes the mode strings; keep them in step.
    assert f"expect == {mb.EXPECT_ANY!r}" in mb.CAS_CODE
    assert f"expect == {mb.EXPECT_ABSENT!r}" in mb.CAS_CODE
    assert f"{mb.CAS_WRITTEN!r} if done else {mb.CAS_CONFLICT!r}" in mb.CAS_CODE


def test_parse_cas_output() -> None:
    assert mb.parse_cas_output("written\n") is True
    assert mb.parse_cas_output("conflict\n") is False
    with pytest.raises(BackendError):
        mb.parse_cas_output("")
    with pytest.raises(BackendError):
        mb.parse_cas_output("Traceback ...")


def test_safe_hash_paths() -> None:
    raw = {
        "a.txt": "1",
        "d/b.py": "2",
        "/etc/passwd": "3",
        "../up.txt": "4",
        "d/../e": "5",
        "./x": "6",
        "d//y": "7",
        "": "8",
        f"{TEMP_PREFIX}abc": "9",
        f"d/{TEMP_PREFIX}abc": "10",
        f"{TEMP_PREFIX}dir/kept.txt": "11",  # only the file name matters
    }
    assert mb.safe_hash_paths(raw) == {"a.txt": "1", "d/b.py": "2", f"{TEMP_PREFIX}dir/kept.txt": "11"}


def test_tags_roundtrip_and_hide_internal() -> None:
    tags = mb.build_tags({"role": "x", "nbgpu.test": "t"}, 1234.5, (8000, 2718))
    handle = mb.handle_from_tags("sb-1", tags)
    assert handle.tags == {"role": "x", "nbgpu.test": "t"}
    assert handle.created_at == 1234.5
    assert handle.port_map == {8000: 8000, 2718: 2718}
    assert handle.remote_root == mb.REMOTE_ROOT
    assert handle.backend == "modal"
    with pytest.raises(BackendError):
        mb.build_tags({mb.TAG_PORTS: "1"}, 0, ())


def test_tags_match() -> None:
    tags = {"group": "g", "which": "a"}
    assert mb.tags_match(tags, {})
    assert mb.tags_match(tags, {"group": "g"})
    assert mb.tags_match(tags, {"group": "g", "which": "a"})
    assert not mb.tags_match(tags, {"group": "g", "which": "c"})
    assert not mb.tags_match(tags, {"absent-key": "x"})


def test_exec_timeout_seconds() -> None:
    assert mb.exec_timeout_seconds(None) is None
    assert mb.exec_timeout_seconds(0.2) == 1
    assert mb.exec_timeout_seconds(1.0) == 1
    assert mb.exec_timeout_seconds(1.5) == 2
    with pytest.raises(BackendError):
        mb.exec_timeout_seconds(0)


def test_split_lines() -> None:
    assert list(mb.split_lines(iter(["a\nb", "c\r\n", "", "d"]))) == ["a", "bc", "d"]
    assert list(mb.split_lines(iter(["x\n"]))) == ["x"]
    assert list(mb.split_lines(iter([]))) == []


# -- remote one-liners, run against the local Python -------------------------


def _local(argv: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    assert argv[0] == mb.REMOTE_PYTHON
    return subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True, **kw)  # type: ignore[call-overload]


def _cas(tmp_path: Path, dest: Path, data: bytes, expect: str | None) -> subprocess.CompletedProcess[str]:
    """Stage `data` as a temp file next to `dest` (as write_bytes would) and
    run the CAS one-liner on it with the local interpreter."""
    tmp = Path(mb.temp_path_for(str(dest)))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    res = _local(mb.cas_argv(str(tmp), str(dest), expect))
    assert not tmp.exists(), "the temp file must never survive"
    return res


def _temps(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.startswith(TEMP_PREFIX)]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_cas_one_liner_unconditional(tmp_path: Path) -> None:
    dest = tmp_path / "f"
    res = _cas(tmp_path, dest, b"new", None)
    assert res.returncode == 0, res.stderr
    assert mb.parse_cas_output(res.stdout) is True
    assert dest.read_bytes() == b"new"
    dest.write_bytes(b"old")
    assert mb.parse_cas_output(_cas(tmp_path, dest, b"newer", None).stdout) is True
    assert dest.read_bytes() == b"newer"
    assert _temps(tmp_path) == []


def test_cas_one_liner_absent(tmp_path: Path) -> None:
    dest = tmp_path / "f"
    res = _cas(tmp_path, dest, b"first", ABSENT)
    assert res.returncode == 0, res.stderr
    assert mb.parse_cas_output(res.stdout) is True
    assert dest.read_bytes() == b"first"
    assert dest.stat().st_nlink == 1  # the temp hard link is gone
    res = _cas(tmp_path, dest, b"second", ABSENT)
    assert res.returncode == 0, res.stderr
    assert mb.parse_cas_output(res.stdout) is False
    assert dest.read_bytes() == b"first"
    # A dangling symlink counts as existing.
    link = tmp_path / "dangling"
    os.symlink(tmp_path / "nowhere", link)
    assert mb.parse_cas_output(_cas(tmp_path, link, b"x", ABSENT).stdout) is False
    assert link.is_symlink() and not (tmp_path / "nowhere").exists()
    assert _temps(tmp_path) == []


def test_cas_one_liner_hash(tmp_path: Path) -> None:
    dest = tmp_path / "f"
    dest.write_bytes(b"v1")
    res = _cas(tmp_path, dest, b"v2", _sha(b"v1"))
    assert res.returncode == 0, res.stderr
    assert mb.parse_cas_output(res.stdout) is True
    assert dest.read_bytes() == b"v2"
    # Mismatch: untouched, reports conflict.
    res = _cas(tmp_path, dest, b"v3", _sha(b"v1"))
    assert res.returncode == 0, res.stderr
    assert mb.parse_cas_output(res.stdout) is False
    assert dest.read_bytes() == b"v2"
    # Expected hash but the file is gone: conflict, nothing created.
    missing = tmp_path / "missing"
    assert mb.parse_cas_output(_cas(tmp_path, missing, b"x", _sha(b"")).stdout) is False
    assert not missing.exists()
    # A large file is hashed in chunks.
    big = os.urandom(3 * 1024 * 1024 + 7)
    dest.write_bytes(big)
    assert mb.parse_cas_output(_cas(tmp_path, dest, b"small", _sha(big)).stdout) is True
    assert dest.read_bytes() == b"small"
    assert _temps(tmp_path) == []


def test_cas_one_liner_failure_exits_nonzero_and_removes_temp(tmp_path: Path) -> None:
    target_dir = tmp_path / "adir"
    (target_dir / "child").mkdir(parents=True)
    # Replacing a non-empty directory fails: an error, not a conflict.
    res = _cas(tmp_path, target_dir, b"x", None)
    assert res.returncode != 0
    assert CAS_OUTPUTS.isdisjoint(res.stdout.split())
    # Hashing a directory fails too.
    res = _cas(tmp_path, target_dir, b"x", _sha(b""))
    assert res.returncode != 0
    assert target_dir.is_dir()
    assert _temps(tmp_path) == []


CAS_OUTPUTS = {mb.CAS_WRITTEN, mb.CAS_CONFLICT}


class _FakeFS:
    def __init__(self, root: Path, fail_upload: bool = False) -> None:
        self.root = root
        self.fail_upload = fail_upload
        self.removed: list[str] = []

    def write_bytes(self, data: bytes, remote_path: str) -> None:
        p = Path(remote_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data[: len(data) // 2] if self.fail_upload else data)
        if self.fail_upload:
            import modal.exception

            raise modal.exception.Error("upload interrupted")

    def remove(self, remote_path: str, *, recursive: bool = False) -> None:
        self.removed.append(remote_path)
        Path(remote_path).unlink()


def _fake_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fs: _FakeFS) -> tuple[mb.ModalBackend, object]:
    """A ModalBackend whose sandbox filesystem writes into tmp_path and whose
    exec runs argv locally: exercises write_file end to end without Modal."""
    from nbgpu.backends.base import ExecResult, RuntimeHandle

    backend = mb.ModalBackend()
    sandbox = type("SB", (), {"filesystem": fs})()
    monkeypatch.setattr(backend, "_sandbox", lambda runtime_id: sandbox)
    execs: list[list[str]] = []

    def fake_exec(handle: object, argv: list[str], **kw: object) -> ExecResult:
        execs.append(list(argv))
        res = _local(list(argv))
        return ExecResult(res.returncode, res.stdout, res.stderr)

    monkeypatch.setattr(backend, "exec", fake_exec)
    handle = RuntimeHandle(backend="modal", runtime_id="sb-x", remote_root=str(tmp_path), created_at=0.0)
    return backend, (handle, execs)


def test_write_file_end_to_end_with_local_stand_ins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, (handle, execs) = _fake_backend(tmp_path, monkeypatch, _FakeFS(tmp_path))
    assert backend.write_file(handle, "d/f.txt", b"one", expect=ABSENT) is True
    assert backend.write_file(handle, "d/f.txt", b"two", expect=ABSENT) is False
    assert backend.write_file(handle, "d/f.txt", b"two", expect=_sha(b"nope")) is False
    assert backend.write_file(handle, "d/f.txt", b"two", expect=_sha(b"one")) is True
    assert backend.write_file(handle, "d/f.txt", b"three") is True
    assert (tmp_path / "d/f.txt").read_bytes() == b"three"
    assert _temps(tmp_path / "d") == []
    assert len(execs) == 5  # exactly one exec per write: check and replace together
    with pytest.raises(BackendError):
        backend.write_file(handle, "d/f.txt", b"x", expect="bogus")
    assert len(execs) == 5  # rejected before anything was uploaded or run


def test_write_file_removes_temp_when_upload_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fs = _FakeFS(tmp_path, fail_upload=True)
    backend, (handle, execs) = _fake_backend(tmp_path, monkeypatch, fs)
    with pytest.raises(BackendError):
        backend.write_file(handle, "d/f.txt", b"payload")
    assert execs == []
    assert len(fs.removed) == 1 and Path(fs.removed[0]).name.startswith(TEMP_PREFIX)
    assert _temps(tmp_path / "d") == []


def test_hash_one_liner(tmp_path: Path) -> None:
    files = {"x.txt": b"x", "d/e/y.bin": bytes(range(256)), ".hidden/k": b"", "a.git.txt": b"kept"}
    excluded = {".git/HEAD": b"r", "d/__pycache__/q.pyc": b"\0", "node_modules/i.js": b"", ".venv/s.py": b""}
    for rel, data in {**files, **excluded}.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    os.symlink(tmp_path / "x.txt", tmp_path / "link")
    (tmp_path / f"{TEMP_PREFIX}abc").write_bytes(b"partial")
    (tmp_path / "d" / f"{TEMP_PREFIX}def").write_bytes(b"partial")
    os.symlink(tmp_path / "d", tmp_path / "dirlink")  # not followed
    cache = str(tmp_path.parent / f"{tmp_path.name}-cache.json")
    res = _local(mb.hash_argv(str(tmp_path), cache))
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    assert got == {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    assert all(not k.startswith("/") and ".." not in k.split("/") for k in got)
    # Through a symlinked root, keys are still relative to the root given.
    os.symlink(tmp_path, tmp_path.parent / f"{tmp_path.name}-link")
    via_link = _local(mb.hash_argv(str(tmp_path.parent / f"{tmp_path.name}-link"), cache))
    assert json.loads(via_link.stdout) == got
    missing = _local(mb.hash_argv(str(tmp_path / "nope"), cache))
    assert json.loads(missing.stdout) == {}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_launch_and_kill_one_liners(tmp_path: Path) -> None:
    pidfile = tmp_path / "p.pgid"
    proc = subprocess.Popen(
        [sys.executable, *mb.launch_argv(str(pidfile), ["sh", "-c", "echo out; echo err >&2; sleep 600 & wait"])[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert proc.stdout is not None
        first = {proc.stdout.readline().strip(), proc.stdout.readline().strip()}
        assert first == {"out", "err"}  # stderr is merged into stdout
        deadline = time.time() + 10
        while not pidfile.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert int(pidfile.read_text()) == proc.pid  # its own process group
        res = _local(mb.kill_argv(str(pidfile), wait_s=1, grace_s=2))
        assert res.returncode == 0, res.stderr
        assert proc.wait(10) != 0
        assert not pidfile.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
    # A missing pid file is not an error.
    assert _local(mb.kill_argv(str(pidfile), wait_s=0.1, grace_s=0.1)).returncode == 0


# -- hash_files stat cache (the one-liner, run locally) ----------------------


def _hash_with_cache(root: Path, cache: Path) -> dict[str, str]:
    res = _local(mb.hash_argv(str(root), str(cache)))
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _tamper(cache: Path, rel: str, fake: str) -> None:
    """Replace the cached digest of `rel` (keeping its stat key): a result
    equal to `fake` proves the file was not re-read."""
    data = json.loads(cache.read_text())
    data["files"][rel][4] = fake
    cache.write_text(json.dumps(data))


def test_hash_cache_path_is_next_to_remote_root() -> None:
    assert mb.hash_cache_path(mb.REMOTE_ROOT) == "/workspace/nbgpu-hash-cache.json"
    assert mb.hash_cache_path("/a/b/project/") == "/a/b/nbgpu-hash-cache.json"


def test_hash_one_liner_reuses_cache_and_detects_ctime_change(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    cache = tmp_path / mb.HASH_CACHE_NAME
    f = root / "big.bin"
    f.write_bytes(b"A" * 1000)
    (root / "other.txt").write_bytes(b"o")
    fake = "f" * 64

    # Just written: within the 2 s window, the cached digest is not trusted.
    first = _hash_with_cache(root, cache)
    assert first["big.bin"] == _sha(b"A" * 1000)
    entry = json.loads(cache.read_text())["files"]["big.bin"]
    st = f.stat()
    assert entry == [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, first["big.bin"]]
    _tamper(cache, "big.bin", fake)
    assert _hash_with_cache(root, cache)["big.bin"] == first["big.bin"]

    time.sleep(2.2)
    _hash_with_cache(root, cache)
    _tamper(cache, "big.bin", fake)
    # Unchanged and older than 2 s: served from the cache, file not read.
    assert _hash_with_cache(root, cache)["big.bin"] == fake

    # Same size and mtime, new content: only ctime differs.
    mtime_ns = st.st_mtime_ns
    with open(f, "r+b") as fh:
        fh.write(b"B" * 1000)
    os.utime(f, ns=(st.st_atime_ns, mtime_ns))
    assert f.stat().st_mtime_ns == mtime_ns and f.stat().st_size == 1000
    time.sleep(2.2)
    assert _hash_with_cache(root, cache)["big.bin"] == _sha(b"B" * 1000)
    # No temp files left next to the cache, and the cache is not in the output.
    assert _temps(tmp_path) == []


@pytest.mark.parametrize(
    "garbage",
    [b"{not json", b"[]", b'{"root": 1}', b'{"files": []}', b"", b"\xff\xfe"],
)
def test_hash_one_liner_tolerates_corrupt_cache(tmp_path: Path, garbage: bytes) -> None:
    root = tmp_path / "project"
    (root / "d").mkdir(parents=True)
    (root / "d" / "x.txt").write_bytes(b"x")
    cache = tmp_path / mb.HASH_CACHE_NAME
    cache.write_bytes(garbage)
    assert _hash_with_cache(root, cache) == {"d/x.txt": _sha(b"x")}
    assert json.loads(cache.read_text())["files"]["d/x.txt"][4] == _sha(b"x")


def test_hash_one_liner_bad_entries_and_root_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "x.txt").write_bytes(b"x")
    cache = tmp_path / mb.HASH_CACHE_NAME
    cache.write_text(json.dumps({"root": str(root), "files": {"x.txt": "bad", "gone": [1, 2, 3, 4, "h"]}}))
    assert _hash_with_cache(root, cache) == {"x.txt": _sha(b"x")}
    assert set(json.loads(cache.read_text())["files"]) == {"x.txt"}  # pruned
    st = (root / "x.txt").stat()
    other = {"root": "/elsewhere", "files": {"x.txt": [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, "f" * 64]}}
    cache.write_text(json.dumps(other))
    assert _hash_with_cache(root, cache) == {"x.txt": _sha(b"x")}


def test_hash_one_liner_skips_cache_inside_root(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"a")
    cache = tmp_path / mb.HASH_CACHE_NAME
    assert _hash_with_cache(tmp_path, cache) == {"a.txt": _sha(b"a")}
    assert cache.exists()
    assert _hash_with_cache(tmp_path, cache) == {"a.txt": _sha(b"a")}


def test_hash_one_liner_unwritable_cache_dir_still_hashes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"a")
    cache = tmp_path / "no-such-dir" / mb.HASH_CACHE_NAME
    assert _hash_with_cache(tmp_path, cache) == {"a.txt": _sha(b"a")}


# -- NBGPU_INTERNAL on helper commands ---------------------------------------


def test_helper_execs_are_marked_internal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbgpu.backends.base import ExecResult, RuntimeHandle

    backend = mb.ModalBackend()
    sandbox = type("SB", (), {"filesystem": _FakeFS(tmp_path)})()
    monkeypatch.setattr(backend, "_sandbox", lambda runtime_id: sandbox)
    calls: list[tuple[list[str], object]] = []

    def fake_exec(handle: object, argv: list[str], **kw: object) -> ExecResult:
        calls.append((list(argv), kw.get("env")))
        res = _local(list(argv))
        return ExecResult(res.returncode, res.stdout, res.stderr)

    monkeypatch.setattr(backend, "exec", fake_exec)
    project = tmp_path / "project"
    project.mkdir()
    handle = RuntimeHandle(backend="modal", runtime_id="sb-x", remote_root=str(project), created_at=0.0)
    assert backend.write_file(handle, "f.txt", b"1") is True
    assert backend.hash_files(handle) == {"f.txt": _sha(b"1")}
    assert (tmp_path / mb.HASH_CACHE_NAME).exists()  # cache path passed: next to project
    proc = mb._ModalProcess(backend, handle, type("P", (), {"poll": lambda self: None})(), str(tmp_path / "none.pgid"))
    monkeypatch.setattr(mb, "kill_argv", lambda pidfile: [mb.REMOTE_PYTHON, "-c", mb.KILL_CODE, pidfile, "0", "0"])
    proc.terminate()
    assert len(calls) == 3
    for argv, env in calls:
        assert env == {"NBGPU_INTERNAL": "1"}, argv


class _RecordingSandbox:
    object_id = "sb-rec"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def exec(self, *argv: object, **kw: object) -> object:
        self.calls.append((argv, kw))
        reader = type("R", (), {"read": lambda self: "", "__iter__": lambda self: iter(())})()
        return type("Proc", (), {"stdout": reader, "stderr": reader, "wait": lambda self: 0, "poll": lambda self: 0})()


def test_user_exec_and_spawn_are_not_marked_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    from nbgpu.backends.base import RuntimeHandle

    backend = mb.ModalBackend()
    sb = _RecordingSandbox()
    monkeypatch.setattr(backend, "_sandbox", lambda runtime_id: sb)
    handle = RuntimeHandle(backend="modal", runtime_id="sb-rec", remote_root=mb.REMOTE_ROOT, created_at=0.0)
    backend.exec(handle, ["echo", "hi"])
    backend.exec(handle, ["echo", "hi"], env={"A": "b"})
    backend.spawn(handle, ["marimo", "edit"])
    backend.spawn(handle, ["marimo", "edit"], env={"A": "b"})
    envs = [kw.get("env") for _, kw in sb.calls]
    assert envs == [None, {"A": "b"}, None, {"A": "b"}]


def test_create_marks_mkdir_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    import modal

    from nbgpu.backends.base import ExecResult, RuntimeRequest

    monkeypatch.setattr(modal.Sandbox, "create", lambda *a, **k: type("S", (), {"object_id": "sb-new"})())
    monkeypatch.setattr(modal.Image, "debian_slim", lambda **k: type("I", (), {"uv_pip_install": lambda self, *p: self})())
    backend = mb.ModalBackend()
    monkeypatch.setattr(backend, "_get_app", lambda: object())
    seen: list[object] = []
    monkeypatch.setattr(backend, "exec", lambda h, argv, **kw: seen.append((argv, kw.get("env"))) or ExecResult(0, "", ""))
    backend.create(RuntimeRequest(spec=_spec(), tags={}))
    assert seen == [(["mkdir", "-p", mb.REMOTE_ROOT], {"NBGPU_INTERNAL": "1"})]


# -- create cleanup ----------------------------------------------------------


def _failing_create_backend(monkeypatch: pytest.MonkeyPatch, terminate: object) -> mb.ModalBackend:
    import modal

    monkeypatch.setattr(modal.Sandbox, "create", lambda *a, **k: type("S", (), {"object_id": "sb-half"})())
    monkeypatch.setattr(modal.Image, "debian_slim", lambda **k: type("I", (), {"uv_pip_install": lambda self, *p: self})())
    backend = mb.ModalBackend()
    monkeypatch.setattr(backend, "_get_app", lambda: object())

    def interrupted(*a: object, **k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(backend, "exec", interrupted)
    monkeypatch.setattr(backend, "terminate", terminate)
    return backend


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_create_cleanup_ignores_signals_and_restores_them(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal

    from nbgpu.backends.base import RuntimeRequest

    during: dict[str, object] = {}

    def terminate(handle: object) -> None:
        during.update(
            id=getattr(handle, "runtime_id"),
            **{name: signal.getsignal(getattr(signal, name)) for name in ("SIGINT", "SIGTERM", "SIGHUP")},
        )
        raise BackendError("terminate failed")  # swallowed: the original error wins

    before = {name: signal.getsignal(getattr(signal, name)) for name in ("SIGINT", "SIGTERM", "SIGHUP")}
    backend = _failing_create_backend(monkeypatch, terminate)
    with pytest.raises(KeyboardInterrupt):
        backend.create(RuntimeRequest(spec=_spec(), tags={}))
    assert during["id"] == "sb-half"
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        assert during[name] == signal.SIG_IGN
        assert signal.getsignal(getattr(signal, name)) == before[name]


def test_create_cleanup_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal
    import threading

    from nbgpu.backends.base import RuntimeHandle

    release = threading.Event()
    backend = _failing_create_backend(monkeypatch, lambda handle: release.wait(30))
    handle = RuntimeHandle(backend="modal", runtime_id="sb-hung", remote_root=mb.REMOTE_ROOT, created_at=0.0)
    before = signal.getsignal(signal.SIGINT)
    start = time.monotonic()
    try:
        with pytest.raises(BackendError) as info:
            backend._terminate_uninterruptibly(handle, timeout_s=0.3)
    finally:
        release.set()
    assert time.monotonic() - start < 5
    assert "sb-hung" in str(info.value)
    assert signal.getsignal(signal.SIGINT) == before
    assert mb.CLEANUP_TERMINATE_TIMEOUT_S == 60


def test_create_cleanup_off_main_thread_leaves_signals_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from nbgpu.backends.base import RuntimeHandle

    calls: list[str] = []
    backend = _failing_create_backend(monkeypatch, lambda handle: calls.append(handle.runtime_id))
    handle = RuntimeHandle(backend="modal", runtime_id="sb-t", remote_root=mb.REMOTE_ROOT, created_at=0.0)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            backend._terminate_uninterruptibly(handle, timeout_s=5)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=run)
    t.start()
    t.join(10)
    assert errors == [] and calls == ["sb-t"]
