"""Filesystem snapshots: saved by stop, matched and used by start (FakeBackend)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from nbgpu import envsync, lifecycle
from nbgpu.backends.base import CAP_SNAPSHOT, BackendError, ExecResult
from nbgpu.session import TAG_TOOL, Session, SessionStore
from nbgpu.spec import load_spec
from tests.fakes import FakeBackend, fake_request


def notebook_text(deps: list[str], extra: str = "") -> str:
    return (
        "# /// script\n"
        f"# dependencies = {json.dumps(deps)}\n"
        "# [tool.nbgpu]\n"
        f"{extra}"
        "# ///\nimport marimo\napp = marimo.App()\n"
    )


class SnapBackend(FakeBackend):
    capabilities = frozenset({CAP_SNAPSHOT})

    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.snapshots: list[str] = []
        self.fail = fail

    def snapshot(self, handle) -> str:  # type: ignore[no-untyped-def]
        if self.fail:
            raise BackendError("Modal: snapshotting failed: boom")
        self.snapshots.append(handle.runtime_id)
        return f"im-{len(self.snapshots)}"

    def exec(self, handle, argv, *, cwd=None, env=None, timeout_s=None):  # type: ignore[no-untyped-def]
        if argv[:2] == ["sh", "-c"] and argv[2].startswith("rm -rf "):
            parent = argv[2].split()[2].strip("'")
            files = self.files[handle.runtime_id]
            for path in [p for p in files if p.startswith(parent + "/")]:
                del files[path]
            return ExecResult(0, "", "")
        raise NotImplementedError(argv)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='p'\n")
    (root / "nb.py").write_text(notebook_text(["marimo", "polars"]))
    return root


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "state")


def _session(store: SessionStore, backend, project: Path, applied: list[str] | None = None) -> Session:  # type: ignore[no-untyped-def]
    handle = backend.create(fake_request(tags={TAG_TOOL: "1"}))
    spec = load_spec(project / "nb.py")
    extra = dict(envsync.applied_extra(spec))
    if applied is not None:
        extra[envsync.EXTRA_ENV] = json.dumps({"deps": applied, "compute": {}})
    session = Session(
        session_id="s1", notebook=spec.path, project_root=project, backend="fake",
        runtime_id=handle.runtime_id, remote_root=handle.remote_root, url="https://x.invalid", token="t",
        created_at=handle.created_at, extra=extra,
    )
    store.save(session)
    return session


def test_stop_with_snapshot_saves_a_record_for_the_next_start(project: Path, store: SessionStore) -> None:
    backend = SnapBackend()
    session = _session(store, backend, project)
    report = lifecycle.stop(session, backend=backend, store=store, snapshot=True)
    assert report.snapshot is not None and report.snapshot["image"] == "im-1"
    assert backend.terminated == [session.runtime_id]  # snapshot first, then terminate
    record = lifecycle.snapshot_for(project / "nb.py", store)
    assert record == report.snapshot
    assert record["key"]["deps"] == ["marimo", "polars"]
    assert (store.directory / lifecycle.SNAPSHOTS_NAME).stat().st_mode & 0o777 == 0o600
    assert lifecycle.matching_snapshot(load_spec(project / "nb.py"), backend, store) == record


def test_the_header_key_turns_snapshots_on(project: Path, store: SessionStore) -> None:
    (project / "nb.py").write_text(notebook_text(["marimo", "polars"], "# snapshot = true\n"))
    backend = SnapBackend()
    report = lifecycle.stop(_session(store, backend, project), backend=backend, store=store)
    assert report.snapshot is not None
    backend2 = SnapBackend()
    report = lifecycle.stop(_session(store, backend2, project), backend=backend2, store=store, snapshot=False)
    assert report.snapshot is None and backend2.snapshots == []


def test_no_snapshot_by_default(project: Path, store: SessionStore) -> None:
    backend = SnapBackend()
    report = lifecycle.stop(_session(store, backend, project), backend=backend, store=store)
    assert report.snapshot is None and backend.snapshots == []


def test_live_installs_are_part_of_the_key(project: Path, store: SessionStore) -> None:
    backend = SnapBackend()
    session = _session(store, backend, project, applied=["marimo", "polars", "rich"])
    lifecycle.stop(session, backend=backend, store=store, snapshot=True)
    assert lifecycle.matching_snapshot(load_spec(project / "nb.py"), backend, store) is None
    (project / "nb.py").write_text(notebook_text(["rich", "marimo", "polars"]))  # order does not matter
    assert lifecycle.matching_snapshot(load_spec(project / "nb.py"), backend, store) is not None


def test_a_failed_snapshot_still_terminates(project: Path, store: SessionStore) -> None:
    backend = SnapBackend(fail=True)
    session = _session(store, backend, project)
    report = lifecycle.stop(session, backend=backend, store=store, snapshot=True)
    assert backend.terminated == [session.runtime_id] and report.snapshot is None
    assert any("snapshot failed" in w and "boom" in w for w in report.warnings)
    assert lifecycle.snapshot_for(project / "nb.py", store) is None


def test_backends_without_snapshots_say_so(project: Path, store: SessionStore) -> None:
    backend = FakeBackend()
    report = lifecycle.stop(_session(store, backend, project), backend=backend, store=store, snapshot=True)
    assert any("cannot snapshot" in w for w in report.warnings)


def test_matching_rejects_changes_age_and_other_backends(project: Path, store: SessionStore) -> None:
    backend = SnapBackend()
    lifecycle.stop(_session(store, backend, project), backend=backend, store=store, snapshot=True)
    spec = load_spec(project / "nb.py")
    assert lifecycle.matching_snapshot(spec, backend, store) is not None
    assert lifecycle.matching_snapshot(spec, FakeBackend(), store) is None  # cannot boot snapshots
    later = time.time() + lifecycle.SNAPSHOT_MAX_AGE_S + 60
    assert lifecycle.matching_snapshot(spec, backend, store, now=later) is None
    (project / "nb.py").write_text(notebook_text(["marimo", "polars"]).replace(
        "# dependencies", '# requires-python = ">=3.12"\n# dependencies'))
    assert lifecycle.matching_snapshot(load_spec(project / "nb.py"), backend, store) is None
    (project / "nb.py").write_text(notebook_text(["marimo"]))
    assert lifecycle.matching_snapshot(load_spec(project / "nb.py"), backend, store) is None
    assert lifecycle.forget_snapshot(project / "nb.py", store) is True
    assert lifecycle.forget_snapshot(project / "nb.py", store) is False


def test_compute_changes_keep_the_snapshot(project: Path, store: SessionStore) -> None:
    backend = SnapBackend()
    lifecycle.stop(_session(store, backend, project), backend=backend, store=store, snapshot=True)
    (project / "nb.py").write_text(notebook_text(["marimo", "polars"], '# gpu = "H100"\n# memory = "64GiB"\n'))
    assert lifecycle.matching_snapshot(load_spec(project / "nb.py"), backend, store) is not None


def test_a_booted_snapshot_loses_its_old_project_and_token(store: SessionStore) -> None:
    backend = SnapBackend()
    handle = backend.create(fake_request())
    backend.remote_write(handle, f"{handle.remote_root}/old.py", "stale")
    backend.remote_write(handle, lifecycle.token_path(handle), "old-token")
    lifecycle._reset_runtime_dir(backend, handle)
    assert backend.files[handle.runtime_id] == {}


# --------------------------------------------------------------------------
# cell-cache: marimo's cache on the cache volume, per project
# --------------------------------------------------------------------------

from nbgpu.backends.base import CACHE_MOUNT, CAP_VOLUMES  # noqa: E402


class VolumeBackend(FakeBackend):
    capabilities = frozenset({CAP_VOLUMES})


def test_cell_cache_links_marimos_cache_per_project(project: Path, tmp_path: Path) -> None:
    (project / "nb.py").write_text(notebook_text(["marimo"], "# cell-cache = true\n"))
    spec = load_spec(project / "nb.py")
    backend = VolumeBackend()
    handle = backend.create(fake_request())
    link, target = lifecycle.cell_cache_paths(spec, backend, handle, project, "nb.py")  # type: ignore[misc]
    assert link == f"{handle.remote_root}/__marimo__/cache"
    assert target == f"{CACHE_MOUNT}/marimo/{lifecycle.project_cache_id(project)}"
    _, sub_target = lifecycle.cell_cache_paths(spec, backend, handle, project, "nbs/a/nb.py")  # type: ignore[misc]
    assert sub_target == f"{target}/nbs/a"
    other = tmp_path / "other"
    other.mkdir()
    _, other_target = lifecycle.cell_cache_paths(spec, backend, handle, other, "nb.py")  # type: ignore[misc]
    assert other_target != target  # projects never share a folder


def test_no_link_unless_asked_or_without_volumes(project: Path) -> None:
    backend = VolumeBackend()
    handle = backend.create(fake_request())
    assert lifecycle.cell_cache_paths(load_spec(project / "nb.py"), backend, handle, project, "nb.py") is None
    (project / "nb.py").write_text(notebook_text(["marimo"], "# cell-cache = true\n"))
    spec = load_spec(project / "nb.py")
    assert lifecycle.cell_cache_paths(spec, FakeBackend(), handle, project, "nb.py") is None
