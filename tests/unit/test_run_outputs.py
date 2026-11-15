"""Bringing back what a run wrote (lifecycle.pull_matching), what a provider
billed (lifecycle.spend) and its live prices (lifecycle.refresh_live_prices)."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from nbgpu import cost, lifecycle
from nbgpu.backends.base import CAP_BILLING, ExecResult, ProviderPrices, SpendRow, UnsupportedError
from tests.fakes import FakeBackend, fake_request


class SizedBackend(FakeBackend):
    """Answers pull_matching's size query from the fake file system."""

    def exec(self, handle, argv: Sequence[str], *, cwd=None, env=None, timeout_s=None):  # type: ignore[no-untyped-def]
        assert list(argv[:2]) == ["python", "-c"]
        files = self.files[handle.runtime_id]
        root = handle.remote_root.rstrip("/") + "/"
        sizes = {p: len(files[root + p]) for p in argv[3:] if root + p in files}
        return ExecResult(0, json.dumps(sizes), "")


@pytest.fixture
def remote(tmp_path: Path):
    backend = SizedBackend()
    handle = backend.create(fake_request())
    root = tmp_path / "proj"
    root.mkdir()

    def put(rel: str, data: bytes) -> None:
        backend.write_file(handle, f"{handle.remote_root}/{rel}", data)

    return root, backend, handle, put


def test_matching_new_files_of_any_type_are_pulled(remote) -> None:
    root, backend, handle, put = remote
    put("plots/a.png", b"\x89PNG\x00binary")
    put("plots/b.svg", b"<svg/>")
    put("model.bin", b"\x00" * 10)
    pulled, warnings = lifecycle.pull_matching(root, backend, handle, ["plots/"])
    assert pulled == ["plots/a.png", "plots/b.svg"] and warnings == []
    assert (root / "plots" / "a.png").read_bytes() == b"\x89PNG\x00binary"
    assert not (root / "model.bin").exists()


def test_no_patterns_pulls_nothing(remote) -> None:
    root, backend, handle, put = remote
    put("a.png", b"x")
    assert lifecycle.pull_matching(root, backend, handle, []) == ([], [])


def test_a_synced_local_file_is_left_to_sync(remote) -> None:
    root, backend, handle, put = remote
    (root / "data.csv").write_text("local edits")
    put("data.csv", b"remote version")
    assert lifecycle.pull_matching(root, backend, handle, ["*.csv"]) == ([], [])
    assert (root / "data.csv").read_text() == "local edits"


def test_an_ignored_output_from_an_earlier_run_is_replaced(remote) -> None:
    root, backend, handle, put = remote
    (root / ".gitignore").write_text("outputs/\n")
    (root / "outputs").mkdir()
    (root / "outputs" / "m.bin").write_bytes(b"old")
    put("outputs/m.bin", b"new")
    pulled, _ = lifecycle.pull_matching(root, backend, handle, ["outputs/"])
    assert pulled == ["outputs/m.bin"] and (root / "outputs" / "m.bin").read_bytes() == b"new"


def test_identical_files_are_not_rewritten(remote) -> None:
    root, backend, handle, put = remote
    (root / "a.png").write_bytes(b"same")
    put("a.png", b"same")
    assert lifecycle.pull_matching(root, backend, handle, ["*.png"]) == ([], [])


def test_files_over_the_limit_are_reported(remote, monkeypatch: pytest.MonkeyPatch) -> None:
    root, backend, handle, put = remote
    monkeypatch.setattr(lifecycle, "RUN_PULL_MAX_BYTES", 4)
    put("big.bin", b"12345")
    pulled, [warning] = lifecycle.pull_matching(root, backend, handle, ["*.bin"])
    assert pulled == [] and "big.bin" in warning and "volume" in warning
    assert not (root / "big.bin").exists()


def test_nothing_is_written_through_a_local_symlink(remote, tmp_path: Path) -> None:
    root, backend, handle, put = remote
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / "out")
    put("out/x.png", b"x")
    pulled, [warning] = lifecycle.pull_matching(root, backend, handle, ["out/"])
    assert pulled == [] and "unsafe" in warning and not (outside / "x.png").exists()


# --------------------------------------------------------------------------
# billing


class BillingBackend(FakeBackend):
    capabilities = frozenset({CAP_BILLING})

    def __init__(self, rows: Sequence[SpendRow] = (), prices: ProviderPrices | None = None) -> None:
        super().__init__()
        self.rows = list(rows)
        self.prices = prices
        self.price_calls = 0
        self.spend_calls: list[tuple[dt.datetime, dt.datetime, str]] = []

    def current_prices(self) -> ProviderPrices:
        self.price_calls += 1
        if self.prices is None:
            raise RuntimeError("no access")
        return self.prices

    def spend(self, start: dt.datetime, end: dt.datetime, resolution: str) -> list[SpendRow]:
        self.spend_calls.append((start, end, resolution))
        return [r for r in self.rows if start <= r.interval_start < end]


def _row(day: int, hour: int, usd: float, **by: float) -> SpendRow:
    return SpendRow(dt.datetime(2026, 11, day, hour, tzinfo=dt.timezone.utc), usd, by)


def test_spend_adds_up_days_and_resources() -> None:
    now = dt.datetime(2026, 11, 15, 14, 30, tzinfo=dt.timezone.utc)
    backend = BillingBackend([_row(12, 0, 1.0, A10G=0.9, CPU=0.1), _row(14, 0, 2.0, H100=2.0), _row(15, 9, 0.5, A10G=0.5)])
    report = lifecycle.spend(backend, 3, now=now)
    assert report.days == [("2026-11-12", 1.0), ("2026-11-14", 2.0), ("2026-11-15", 0.5)]
    assert report.total == pytest.approx(3.5)
    assert list(report.by_resource) == ["H100", "A10G", "CPU"]
    assert report.by_resource["A10G"] == pytest.approx(1.4)
    # complete days daily, today hourly up to the last complete hour
    assert [c[2] for c in backend.spend_calls] == ["d", "h"]
    assert backend.spend_calls[1][1] == dt.datetime(2026, 11, 15, 14, tzinfo=dt.timezone.utc)


def test_spend_needs_billing() -> None:
    with pytest.raises(UnsupportedError):
        lifecycle.spend(FakeBackend(), 7)


def test_refresh_caches_live_prices_once_a_day(tmp_path: Path) -> None:
    backend = BillingBackend(prices=ProviderPrices({"T4": 0.5}, 0.2, 0.03))
    assert lifecycle.refresh_live_prices(backend, tmp_path) is None
    assert lifecycle.refresh_live_prices(backend, tmp_path) is None
    assert backend.price_calls == 1  # fresh enough
    assert lifecycle.refresh_live_prices(backend, tmp_path, force=True) is None
    assert backend.price_calls == 2
    table = cost.current_prices(tmp_path)
    assert table.kind == cost.KIND_LIVE
    assert table.gpu_hourly[("fake", "T4")] == 0.5 and table.cpu_core_hourly["fake"] == 0.2


def test_refresh_errors_are_returned_not_raised(tmp_path: Path) -> None:
    assert "no access" in (lifecycle.refresh_live_prices(BillingBackend(), tmp_path) or "")
    assert lifecycle.refresh_live_prices(FakeBackend(), tmp_path) is None  # no billing: nothing to do
    assert cost.current_prices(tmp_path) is cost.DEFAULT_PRICES
