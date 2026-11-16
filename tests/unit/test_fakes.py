"""Sanity checks for the FakeBackend test double."""

from __future__ import annotations

import hashlib

import pytest

from nbgpu.backends.base import BackendError
from tests.fakes import FakeBackend, fake_request


def test_create_get_find_terminate() -> None:
    fb = FakeBackend()
    a = fb.create(fake_request(tags={"project": "p", "nb": "a"}))
    b = fb.create(fake_request(tags={"project": "p", "nb": "b"}))
    assert a.runtime_id != b.runtime_id
    assert a.remote_root.startswith("/") and a.remote_root != b.remote_root
    assert fb.get(a.runtime_id) == a
    assert {h.runtime_id for h in fb.find({"project": "p"})} == {a.runtime_id, b.runtime_id}
    assert fb.find({"nb": "b"}) == [b]
    assert fb.find({"nb": "zzz"}) == []
    fb.terminate(a)
    fb.terminate(a)  # no error if already gone
    assert fb.get(a.runtime_id) is None
    assert fb.find({"project": "p"}) == [b]
    with pytest.raises(BackendError):
        fb.read_file(a, "x.py")


def test_read_write_relative_and_absolute() -> None:
    fb = FakeBackend()
    h = fb.create(fake_request())
    assert fb.read_file(h, "nb.py") is None
    fb.write_file(h, "dir/nb.py", b"one")
    assert fb.read_file(h, f"{h.remote_root}/dir/nb.py") == b"one"
    fb.write_file(h, f"{h.remote_root}/dir/nb.py", b"two")
    assert fb.read_file(h, "dir/nb.py") == b"two"
    assert fb.writes == ["dir/nb.py", "dir/nb.py"]
    assert fb.reads == ["nb.py", "dir/nb.py", "dir/nb.py"]


def test_hash_files_exclusions_and_root() -> None:
    fb = FakeBackend()
    h = fb.create(fake_request())
    for p in ["a.py", "sub/b.txt", ".git/HEAD", ".venv/x.py", "sub/__pycache__/c.pyc",
              "node_modules/m/i.js", ".nbgpu/state.json"]:
        fb.remote_write(h, p, p.encode())
    fb.write_file(h, "/elsewhere/outside.py", b"no")
    hashes = fb.hash_files(h)
    assert hashes == {
        "a.py": hashlib.sha256(b"a.py").hexdigest(),
        "sub/b.txt": hashlib.sha256(b"sub/b.txt").hexdigest(),
    }  # .nbgpu is in EXCLUDED_PARTS like the others
    assert fb.hash_files(h, "sub") == {"b.txt": hashlib.sha256(b"sub/b.txt").hexdigest()}
    assert fb.hash_calls == 2


def test_hooks_are_not_counted_and_runtimes_isolated() -> None:
    fb = FakeBackend()
    a = fb.create(fake_request())
    b = fb.create(fake_request())
    fb.remote_write(a, "nb.py", "edited in UI")
    assert fb.remote_read(a, "nb.py") == b"edited in UI"
    assert fb.remote_read(b, "nb.py") is None
    assert fb.remote_files(a) == {"nb.py": b"edited in UI"}
    assert fb.reads == [] and fb.writes == []
    fb.remote_delete(a, "nb.py")
    assert fb.remote_files(a) == {}
    fb.read_file(a, "x")
    fb.reset_counters()
    assert fb.reads == [] and fb.writes == [] and fb.hash_calls == 0


def test_expose_and_unsupported_process_ops() -> None:
    fb = FakeBackend()
    h = fb.create(fake_request(ports=(2718,)))
    url = fb.expose(h, 2718)
    assert url.startswith("https://") and not url.endswith("/")
    with pytest.raises(BackendError):
        fb.expose(h, 9999)
    assert h.port_for(2718) == 2718
    with pytest.raises(NotImplementedError):
        fb.exec(h, ["true"])
