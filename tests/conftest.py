"""Shared test configuration.

Tests marked `modal` create real (billed) Modal sandboxes and are skipped
unless NBGPU_MODAL_TESTS=1.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("NBGPU_MODAL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="creates Modal sandboxes; set NBGPU_MODAL_TESTS=1")
    for item in items:
        if "modal" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the real ~/.local/state/nbgpu from tests."""
    monkeypatch.setenv("NBGPU_STATE_DIR", str(tmp_path_factory.mktemp("nbgpu-state")))
