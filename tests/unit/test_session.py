"""Tests for nbgpu.session: ids, tokens, tags and the local session store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import fields
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from nbgpu.session import (
    Session,
    SessionStore,
    default_state_dir,
    new_session_id,
    new_token,
    notebook_tag,
)

URL_SAFE = re.compile(r"^[A-Za-z0-9_-]+$")


def make_session(
    session_id: str = "0123abcd",
    *,
    notebook: Path = Path("/home/user/proj/analysis.py"),
    created_at: float = 1_700_000_000.0,
    gpu: str | None = "A10G",
    extra: dict[str, str] | None = None,
    url: str = "https://example-runtime.test/",
) -> Session:
    return Session(
        session_id=session_id,
        notebook=notebook,
        project_root=notebook.parent,
        backend="modal",
        runtime_id=f"sb-{session_id}",
        remote_root="/root/project",
        url=url,
        token="tok_" + session_id,
        created_at=created_at,
        gpu=gpu,
        extra={"app": "nbgpu"} if extra is None else extra,
    )


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --------------------------------------------------------------------------
# ids, tokens, tags
# --------------------------------------------------------------------------


def test_new_session_id_format() -> None:
    for _ in range(50):
        sid = new_session_id()
        assert re.fullmatch(r"[0-9a-f]{8}", sid), sid


def test_new_session_id_is_random() -> None:
    ids = {new_session_id() for _ in range(200)}
    assert len(ids) > 190


def test_new_token_is_url_safe() -> None:
    for _ in range(50):
        token = new_token()
        assert URL_SAFE.match(token), token


def test_new_token_has_at_least_32_bytes_of_entropy() -> None:
    # 32 bytes in URL-safe base64 is 43 characters; hex would be 64.
    for _ in range(20):
        assert len(new_token()) >= 43


def test_new_token_is_random() -> None:
    tokens = {new_token() for _ in range(200)}
    assert len(tokens) == 200


def test_notebook_tag_is_first_16_hex_of_sha256() -> None:
    path = Path("/home/user/proj/analysis.py")
    tag = notebook_tag(path)
    assert re.fullmatch(r"[0-9a-f]{16}", tag)
    assert tag == hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def test_notebook_tag_is_stable() -> None:
    path = Path("/home/user/proj/analysis.py")
    assert notebook_tag(path) == notebook_tag(Path("/home/user/proj/analysis.py"))


def test_notebook_tag_differs_by_path() -> None:
    assert notebook_tag(Path("/a/nb.py")) != notebook_tag(Path("/b/nb.py"))


def test_notebook_tag_does_not_reveal_path() -> None:
    path = Path("/home/secretuser/topsecret/analysis.py")
    tag = notebook_tag(path)
    for part in ("home", "secretuser", "topsecret", "analysis", ".py", "/"):
        assert part not in tag


# --------------------------------------------------------------------------
# Session.url_with_token
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected_query"),
    [
        ("https://host.test/", {"access_token": ["tok"]}),
        ("https://host.test", {"access_token": ["tok"]}),
        ("http://localhost:2718/", {"access_token": ["tok"]}),
        ("https://host.test/?file=nb.py", {"file": ["nb.py"], "access_token": ["tok"]}),
        (
            "https://host.test/edit?a=1&b=two",
            {"a": ["1"], "b": ["two"], "access_token": ["tok"]},
        ),
    ],
)
def test_url_with_token(url: str, expected_query: dict[str, list[str]]) -> None:
    session = make_session(url=url)
    session.token = "tok"
    result = session.url_with_token()
    original, got = urlsplit(url), urlsplit(result)
    assert (got.scheme, got.netloc) == (original.scheme, original.netloc)
    assert got.path in (original.path, original.path or "/")
    assert parse_qs(got.query) == expected_query


def test_url_with_token_exact_simple_case() -> None:
    session = make_session(url="http://localhost:2718/")
    session.token = "abc_DEF-123"
    assert session.url_with_token() == "http://localhost:2718/?access_token=abc_DEF-123"


def test_url_with_token_keeps_existing_query_order() -> None:
    session = make_session(url="https://host.test/?file=nb.py")
    session.token = "abc"
    assert session.url_with_token() == "https://host.test/?file=nb.py&access_token=abc"


def test_url_with_token_does_not_change_url_field() -> None:
    session = make_session(url="https://host.test/")
    session.url_with_token()
    assert session.url == "https://host.test/"
    assert "access_token" not in session.url


# --------------------------------------------------------------------------
# default_state_dir
# --------------------------------------------------------------------------


def test_default_state_dir_prefers_nbgpu_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NBGPU_STATE_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_state_dir() == tmp_path / "explicit"


def test_default_state_dir_uses_xdg_state_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NBGPU_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_state_dir() == tmp_path / "xdg" / "nbgpu"


def test_default_state_dir_falls_back_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NBGPU_STATE_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_state_dir() == tmp_path / "home" / ".local" / "state" / "nbgpu"


def test_default_state_dir_returns_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBGPU_STATE_DIR", str(tmp_path))
    assert isinstance(default_state_dir(), Path)


def test_default_state_dir_does_not_create_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NBGPU_STATE_DIR", str(tmp_path / "not-yet"))
    default_state_dir()
    assert not (tmp_path / "not-yet").exists()


# --------------------------------------------------------------------------
# SessionStore
# --------------------------------------------------------------------------


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "state" / "nbgpu"


@pytest.fixture
def store(store_dir: Path) -> SessionStore:
    return SessionStore(store_dir)


class TestStoreRoundTrip:
    def test_round_trip_every_field(self, store: SessionStore) -> None:
        original = make_session(extra={"app_id": "ap-123", "region": "us-east"})
        store.save(original)
        loaded = store.load(original.session_id)
        assert loaded is not None
        assert loaded == original
        for f in fields(Session):
            assert getattr(loaded, f.name) == getattr(original, f.name), f.name

    def test_path_fields_come_back_as_path(self, store: SessionStore) -> None:
        store.save(make_session())
        loaded = store.load("0123abcd")
        assert loaded is not None
        assert isinstance(loaded.notebook, Path)
        assert isinstance(loaded.project_root, Path)
        assert loaded.notebook == Path("/home/user/proj/analysis.py")
        assert loaded.project_root == Path("/home/user/proj")

    def test_round_trip_gpu_none_and_empty_extra(self, store: SessionStore) -> None:
        original = make_session(gpu=None, extra={})
        store.save(original)
        loaded = store.load(original.session_id)
        assert loaded == original
        assert loaded is not None
        assert loaded.gpu is None
        assert loaded.extra == {}

    def test_round_trip_float_created_at(self, store: SessionStore) -> None:
        original = make_session(created_at=1_700_000_123.456789)
        store.save(original)
        loaded = store.load(original.session_id)
        assert loaded is not None
        assert loaded.created_at == original.created_at

    def test_load_missing_returns_none(self, store: SessionStore) -> None:
        assert store.load("deadbeef") is None

    def test_load_missing_before_directory_exists(self, store: SessionStore, store_dir: Path) -> None:
        assert not store_dir.exists()
        assert store.load("deadbeef") is None

    def test_save_overwrites_same_id(self, store: SessionStore) -> None:
        store.save(make_session(gpu="A10G"))
        store.save(make_session(gpu="H100:2"))
        loaded = store.load("0123abcd")
        assert loaded is not None
        assert loaded.gpu == "H100:2"
        assert len(store.list()) == 1

    def test_records_are_json_files(self, store: SessionStore, store_dir: Path) -> None:
        store.save(make_session())
        files = list(store_dir.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".json"
        json.loads(files[0].read_text())  # valid JSON

    def test_default_directory_comes_from_env(self) -> None:
        # conftest points NBGPU_STATE_DIR at a temp dir.
        store = SessionStore()
        store.save(make_session())
        assert any(default_state_dir().glob("*.json"))
        assert SessionStore().load("0123abcd") == make_session()

    def test_two_store_instances_share_directory(self, store_dir: Path) -> None:
        SessionStore(store_dir).save(make_session())
        assert SessionStore(store_dir).load("0123abcd") == make_session()


class TestStorePermissions:
    def test_directory_created_lazily(self, store_dir: Path) -> None:
        SessionStore(store_dir)
        assert not store_dir.exists()

    def test_directory_mode_0700(self, store: SessionStore, store_dir: Path) -> None:
        store.save(make_session())
        assert store_dir.is_dir()
        assert mode(store_dir) == 0o700

    def test_file_mode_0600(self, store: SessionStore, store_dir: Path) -> None:
        store.save(make_session())
        files = list(store_dir.iterdir())
        assert files
        for f in files:
            assert mode(f) == 0o600, f

    def test_file_mode_0600_after_overwrite(self, store: SessionStore, store_dir: Path) -> None:
        store.save(make_session(gpu="A10G"))
        store.save(make_session(gpu="T4"))
        for f in store_dir.iterdir():
            assert mode(f) == 0o600

    def test_modes_ignore_permissive_umask(self, store: SessionStore, store_dir: Path) -> None:

        old = os.umask(0o000)
        try:
            store.save(make_session())
        finally:
            os.umask(old)
        assert mode(store_dir) == 0o700
        for f in store_dir.iterdir():
            assert mode(f) == 0o600

    def test_atomic_save_leaves_no_temp_files(self, store: SessionStore, store_dir: Path) -> None:
        for i in range(5):
            store.save(make_session(f"0000000{i}"))
        store.save(make_session("00000000", gpu="T4"))
        names = sorted(p.name for p in store_dir.iterdir())
        assert len(names) == 5
        assert all(name.endswith(".json") for name in names)
        assert not any(name.startswith(".") for name in names)


class TestStoreList:
    def test_list_empty_when_directory_missing(self, store: SessionStore) -> None:
        assert store.list() == []

    def test_list_oldest_first(self, store: SessionStore) -> None:
        # Save in an order unrelated to created_at or id.
        store.save(make_session("bbbbbbbb", created_at=300.0))
        store.save(make_session("cccccccc", created_at=100.0))
        store.save(make_session("aaaaaaaa", created_at=200.0))
        assert [s.session_id for s in store.list()] == ["cccccccc", "aaaaaaaa", "bbbbbbbb"]

    def test_list_returns_sessions(self, store: SessionStore) -> None:
        original = make_session()
        store.save(original)
        assert store.list() == [original]

    @pytest.mark.parametrize(
        "content",
        ["{not json", "", "[]", "{}", '{"session_id": "x"}', "\x00\x01binary"],
    )
    def test_list_skips_corrupt_files(
        self, store: SessionStore, store_dir: Path, content: str
    ) -> None:
        good = make_session("11111111")
        store.save(good)
        bad = store_dir / "22222222.json"
        bad.write_text(content)
        assert store.list() == [good]

    def test_list_never_opens_other_json_files(
        self, store: SessionStore, store_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nbgpu.session as session_mod

        good = make_session("11111111")
        store.save(good)
        (store_dir / "11111111.sync.json").write_text("{}")  # sync state: can be many MB
        (store_dir / "settings.v2.json").write_text("{}")
        opened: list[str] = []
        real = session_mod._read_record
        monkeypatch.setattr(session_mod, "_read_record", lambda p: opened.append(p.name) or real(p))
        assert store.list() == [good]
        assert opened == ["11111111.json"]

    def test_list_skips_unreadable_file(self, store: SessionStore, store_dir: Path) -> None:

        good = make_session("11111111")
        store.save(good)
        bad = store_dir / "33333333.json"
        bad.write_text(json.dumps({"session_id": "33333333"}))
        bad.chmod(0o000)
        try:
            if os.access(bad, os.R_OK):  # running as root
                pytest.skip("file permissions not enforced for this user")
            assert store.list() == [good]
        finally:
            bad.chmod(0o600)


class TestStoreFindByNotebook:
    def test_find_by_notebook_newest_first(self, store: SessionStore) -> None:
        nb = Path("/home/user/proj/analysis.py")
        other = Path("/home/user/proj/other.py")
        store.save(make_session("aaaaaaaa", notebook=nb, created_at=100.0))
        store.save(make_session("bbbbbbbb", notebook=other, created_at=150.0))
        store.save(make_session("cccccccc", notebook=nb, created_at=300.0))
        store.save(make_session("dddddddd", notebook=nb, created_at=200.0))
        found = store.find_by_notebook(nb)
        assert [s.session_id for s in found] == ["cccccccc", "dddddddd", "aaaaaaaa"]

    def test_find_by_notebook_none_match(self, store: SessionStore) -> None:
        store.save(make_session(notebook=Path("/a/nb.py")))
        assert store.find_by_notebook(Path("/b/nb.py")) == []

    def test_find_by_notebook_empty_store(self, store: SessionStore) -> None:
        assert store.find_by_notebook(Path("/a/nb.py")) == []

    def test_find_by_notebook_same_name_different_dir(self, store: SessionStore) -> None:
        store.save(make_session("aaaaaaaa", notebook=Path("/x/nb.py")))
        store.save(make_session("bbbbbbbb", notebook=Path("/y/nb.py")))
        assert [s.session_id for s in store.find_by_notebook(Path("/y/nb.py"))] == ["bbbbbbbb"]


class TestStoreRemove:
    def test_remove_deletes_record(self, store: SessionStore) -> None:
        store.save(make_session("aaaaaaaa"))
        store.save(make_session("bbbbbbbb"))
        store.remove("aaaaaaaa")
        assert store.load("aaaaaaaa") is None
        assert [s.session_id for s in store.list()] == ["bbbbbbbb"]

    def test_remove_is_idempotent(self, store: SessionStore) -> None:
        store.save(make_session("aaaaaaaa"))
        store.remove("aaaaaaaa")
        store.remove("aaaaaaaa")
        assert store.load("aaaaaaaa") is None

    def test_remove_missing_is_not_an_error(self, store: SessionStore) -> None:
        store.remove("deadbeef")  # directory does not exist yet

    def test_remove_missing_with_existing_directory(self, store: SessionStore) -> None:
        store.save(make_session("aaaaaaaa"))
        store.remove("deadbeef")
        assert store.load("aaaaaaaa") is not None
