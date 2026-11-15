"""Tests for nbgpu.cost. They build their own PriceTable and do not depend on
the contents of DEFAULT_PRICES, except the DEFAULT_PRICES section at the end."""

from __future__ import annotations

import pytest

from nbgpu.cost import PriceTable, Rate, cost_so_far, format_usd, hourly_rate, normalise_gpu
from nbgpu.spec import ComputeSpec

PRICES = PriceTable(
    gpu_hourly={
        ("modal", "A10G"): 1.10,
        ("modal", "H100"): 4.00,
        ("modal", "T4"): 0.60,
        ("runpod", "A10G"): 0.50,
        ("runpod", "H100"): 2.50,
    },
    cpu_core_hourly={"modal": 0.05},
    memory_gib_hourly={"modal": 0.01},
    source="test fixture",
    checked="2026-01-01",
)


# --------------------------------------------------------------------------
# normalise_gpu
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        ("A10G", ("A10G", 1)),
        ("a10g", ("A10G", 1)),
        ("H100:2", ("H100", 2)),
        ("h100:2", ("H100", 2)),
        ("H100:1", ("H100", 1)),
        ("A100-80GB:8", ("A100-80GB", 8)),
        ("t4", ("T4", 1)),
    ],
)
def test_normalise_gpu(gpu: str, expected: tuple[str, int]) -> None:
    assert normalise_gpu(gpu) == expected


@pytest.mark.parametrize("gpu", ["H100:0", "H100:-1", "A10G:-4"])
def test_normalise_gpu_count_below_one_raises(gpu: str) -> None:
    with pytest.raises(ValueError):
        normalise_gpu(gpu)


def test_normalise_gpu_non_integer_count_raises() -> None:
    with pytest.raises(ValueError):
        normalise_gpu("H100:two")


# --------------------------------------------------------------------------
# hourly_rate
# --------------------------------------------------------------------------


def test_gpu_only_rate_is_complete() -> None:
    rate = hourly_rate(ComputeSpec(provider="modal", gpu="A10G"), PRICES)
    assert isinstance(rate, Rate)
    assert rate.usd_per_hour == pytest.approx(1.10)
    assert rate.complete is True
    assert rate.note == ""


def test_gpu_count_multiplies() -> None:
    rate = hourly_rate(ComputeSpec(provider="modal", gpu="H100:2"), PRICES)
    assert rate.usd_per_hour == pytest.approx(8.00)
    assert rate.complete is True


def test_gpu_count_eight() -> None:
    rate = hourly_rate(ComputeSpec(provider="runpod", gpu="H100:8"), PRICES)
    assert rate.usd_per_hour == pytest.approx(20.00)


@pytest.mark.parametrize("gpu", ["a10g", "A10g", "A10G"])
def test_gpu_case_is_normalised(gpu: str) -> None:
    rate = hourly_rate(ComputeSpec(provider="modal", gpu=gpu), PRICES)
    assert rate.usd_per_hour == pytest.approx(1.10)
    assert rate.complete is True


def test_prices_are_per_provider() -> None:
    modal = hourly_rate(ComputeSpec(provider="modal", gpu="A10G"), PRICES)
    runpod = hourly_rate(ComputeSpec(provider="runpod", gpu="A10G"), PRICES)
    assert modal.usd_per_hour == pytest.approx(1.10)
    assert runpod.usd_per_hour == pytest.approx(0.50)


def test_gpu_cpu_and_memory_add_up() -> None:
    compute = ComputeSpec(provider="modal", gpu="A10G", cpu=8, memory_mib=32 * 1024)
    rate = hourly_rate(compute, PRICES)
    # 1.10 + 8 * 0.05 + 32 * 0.01
    assert rate.usd_per_hour == pytest.approx(1.10 + 0.40 + 0.32)
    assert rate.complete is True
    assert rate.note == ""


def test_fractional_cpu_and_memory() -> None:
    compute = ComputeSpec(provider="modal", gpu="T4", cpu=0.5, memory_mib=512)
    rate = hourly_rate(compute, PRICES)
    # 0.60 + 0.5 * 0.05 + 0.5 GiB * 0.01
    assert rate.usd_per_hour == pytest.approx(0.60 + 0.025 + 0.005)
    assert rate.complete is True


def test_gpu_count_with_cpu_and_memory() -> None:
    compute = ComputeSpec(provider="modal", gpu="H100:2", cpu=4, memory_mib=2048)
    rate = hourly_rate(compute, PRICES)
    assert rate.usd_per_hour == pytest.approx(2 * 4.00 + 4 * 0.05 + 2 * 0.01)


def test_unknown_gpu_makes_rate_unknown() -> None:
    compute = ComputeSpec(provider="modal", gpu="H200", cpu=8, memory_mib=1024)
    rate = hourly_rate(compute, PRICES)
    assert rate.usd_per_hour == 0.0
    assert rate.complete is False
    assert rate.note != ""
    assert "H200" in rate.note.upper()


def test_known_gpu_on_unknown_provider_is_unknown() -> None:
    rate = hourly_rate(ComputeSpec(provider="lambda", gpu="A10G"), PRICES)
    assert rate.usd_per_hour == 0.0
    assert rate.complete is False
    assert rate.note != ""


def test_missing_cpu_price_is_skipped_and_incomplete() -> None:
    compute = ComputeSpec(provider="runpod", gpu="A10G", cpu=8)
    rate = hourly_rate(compute, PRICES)
    assert rate.usd_per_hour == pytest.approx(0.50)
    assert rate.complete is False
    assert "cpu" in rate.note.lower()
    assert "runpod" in rate.note


def test_missing_memory_price_is_skipped_and_incomplete() -> None:
    prices = PriceTable(
        gpu_hourly={("modal", "A10G"): 1.10},
        cpu_core_hourly={"modal": 0.05},
        memory_gib_hourly={},
        source="test",
        checked="2026-01-01",
    )
    compute = ComputeSpec(provider="modal", gpu="A10G", cpu=2, memory_mib=4096)
    rate = hourly_rate(compute, prices)
    assert rate.usd_per_hour == pytest.approx(1.10 + 2 * 0.05)
    assert rate.complete is False
    assert "memory" in rate.note.lower()
    assert "modal" in rate.note


def test_missing_cpu_and_memory_prices_both_noted() -> None:
    compute = ComputeSpec(provider="runpod", gpu="H100", cpu=4, memory_mib=8192)
    rate = hourly_rate(compute, PRICES)
    assert rate.usd_per_hour == pytest.approx(2.50)
    assert rate.complete is False
    assert "cpu" in rate.note.lower()
    assert "memory" in rate.note.lower()


def test_unrequested_cpu_and_memory_need_no_price() -> None:
    # runpod has no cpu/memory prices, but the spec asks for neither.
    rate = hourly_rate(ComputeSpec(provider="runpod", gpu="A10G"), PRICES)
    assert rate.usd_per_hour == pytest.approx(0.50)
    assert rate.complete is True
    assert rate.note == ""


def test_rate_is_frozen() -> None:
    rate = Rate(1.0, True, "")
    with pytest.raises(AttributeError):
        rate.usd_per_hour = 2.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# cost_so_far
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("usd_per_hour", "seconds", "expected"),
    [
        (1.10, 3600, 1.10),
        (1.10, 1800, 0.55),
        (1.10, 0, 0.0),
        (2.00, 90, 0.05),
        (1.00, 30, 0.01),  # 0.00833 rounds to 0.01
        (1.00, 10, 0.0),  # 0.00278 rounds to 0.00
        (4.00, 7200, 8.00),
        (6.00, 61, 0.10),  # 0.10166 rounds to 0.10
        (6.00, 65, 0.11),  # 0.10833 rounds to 0.11
        (0.0, 3600, 0.0),
        (12.00, 0.5, 0.0),
    ],
)
def test_cost_so_far_is_rounded_to_cents(
    usd_per_hour: float, seconds: float, expected: float
) -> None:
    result = cost_so_far(Rate(usd_per_hour, True, ""), seconds)
    assert result == pytest.approx(expected, abs=1e-9)
    # Already a whole number of cents.
    assert round(result, 2) == pytest.approx(result, abs=1e-9)


def test_cost_so_far_uses_rate_even_if_incomplete() -> None:
    assert cost_so_far(Rate(2.0, False, "no price for cpu on 'x'"), 3600) == pytest.approx(2.0)


@pytest.mark.parametrize("seconds", [-1, -0.001, -3600])
def test_cost_so_far_negative_seconds_raise(seconds: float) -> None:
    with pytest.raises(ValueError):
        cost_so_far(Rate(1.0, True, ""), seconds)


# --------------------------------------------------------------------------
# format_usd
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (0, "$0.00"),
        (0.0, "$0.00"),
        (1.1, "$1.10"),
        (1.10, "$1.10"),
        (12.349, "$12.35"),
        (12.35, "$12.35"),
        (0.01, "$0.01"),
        (0.5, "$0.50"),
        (3, "$3.00"),
        (0.001, "<$0.01"),
        (0.004, "<$0.01"),
        (0.0001, "<$0.01"),
    ],
)
def test_format_usd(amount: float, expected: str) -> None:
    assert format_usd(amount) == expected


def test_cpu_only_runtime_costs_cpu_and_memory() -> None:
    rate = hourly_rate(ComputeSpec(provider="modal", gpu=None, cpu=2, memory_mib=4096), PRICES)
    assert rate.usd_per_hour == pytest.approx(2 * 0.05 + 4 * 0.01)
    assert rate.complete is True


def test_cpu_only_runtime_without_shape_is_free_and_complete() -> None:
    rate = hourly_rate(ComputeSpec(provider="modal"), PRICES)
    assert rate.usd_per_hour == 0.0
    assert rate.complete is True


# --------------------------------------------------------------------------
# DEFAULT_PRICES (the only tests that read the shipped table)
# --------------------------------------------------------------------------


def test_default_prices_cover_every_known_modal_gpu() -> None:
    from nbgpu.backends.modal import KNOWN_GPUS
    from nbgpu.cost import DEFAULT_PRICES

    missing = [name for name in KNOWN_GPUS if ("modal", normalise_gpu(name)[0]) not in DEFAULT_PRICES.gpu_hourly]
    assert missing == []
    assert all(price > 0 for price in DEFAULT_PRICES.gpu_hourly.values())
    assert DEFAULT_PRICES.source
    assert DEFAULT_PRICES.checked


def test_default_prices_give_complete_rate_for_typical_modal_shape() -> None:
    from nbgpu.cost import DEFAULT_PRICES

    rate = hourly_rate(ComputeSpec(provider="modal", gpu="A10G", cpu=8, memory_mib=32 * 1024), DEFAULT_PRICES)
    assert rate.complete is True
    assert rate.note == ""
    expected = (
        DEFAULT_PRICES.gpu_hourly[("modal", "A10G")]
        + 8 * DEFAULT_PRICES.cpu_core_hourly["modal"]
        + 32 * DEFAULT_PRICES.memory_gib_hourly["modal"]
    )
    assert rate.usd_per_hour == pytest.approx(expected)
    assert rate.usd_per_hour > 0


# --------------------------------------------------------------------------
# price files: parse_prices, $NBGPU_PRICES, staleness
# --------------------------------------------------------------------------

import datetime as dt  # noqa: E402

from nbgpu.cost import PRICES_ENV, current_prices, disclaimer, is_stale, parse_prices  # noqa: E402

_PRICE_FILE = """
checked = "2026-03-01"
sources = ["https://example.com/pricing"]

[acme]
cpu-core = 0.0001
memory-gib = 0.00001

[acme.gpu]
h100 = 0.001
"B200+" = 0.002
"""


def test_parse_prices_converts_per_second_to_hourly_and_normalises_gpus() -> None:
    prices = parse_prices(_PRICE_FILE)
    assert prices.checked == "2026-03-01"
    assert prices.source == "https://example.com/pricing"
    assert prices.gpu_hourly == {("acme", "H100"): pytest.approx(3.6), ("acme", "B200+"): pytest.approx(7.2)}
    assert prices.cpu_core_hourly["acme"] == pytest.approx(0.36)
    assert prices.memory_gib_hourly["acme"] == pytest.approx(0.036)


def test_parse_prices_accepts_unquoted_toml_date() -> None:
    assert parse_prices("checked = 2026-03-01\n").checked == "2026-03-01"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[acme.gpu]\nT4 = 1\n", "`checked`"),
        ('checked = "March"\n', "ISO date"),
        ('checked = "2026-03-01"\n[acme.gpu]\nT4 = -1\n', "acme.gpu.T4"),
        ('checked = "2026-03-01"\n[acme.gpu]\nT4 = "cheap"\n', "acme.gpu.T4"),
        ('checked = "2026-03-01"\n[acme]\ncpu = 1\n', "unknown key 'cpu'"),
        ('checked = "2026-03-01"\ncurrency = "USD"\n', "unknown top-level key"),
        ("checked = [", "my.toml"),
    ],
)
def test_parse_prices_rejects_malformed_files(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(")):
        parse_prices(text, "my.toml")


def test_shipped_file_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from nbgpu.cost import DEFAULT_PRICES

    monkeypatch.delenv(PRICES_ENV, raising=False)
    assert current_prices() is DEFAULT_PRICES
    dt.date.fromisoformat(DEFAULT_PRICES.checked)


def test_env_override_replaces_the_table(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "prices.toml"
    path.write_text(_PRICE_FILE)
    monkeypatch.setenv(PRICES_ENV, str(path))
    assert hourly_rate(ComputeSpec(provider="acme", gpu="H100:2")).usd_per_hour == pytest.approx(7.2)
    assert hourly_rate(ComputeSpec(provider="modal", gpu="T4")).complete is False


def test_env_override_missing_file_is_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(PRICES_ENV, str(tmp_path / "nope.toml"))
    with pytest.raises(ValueError, match=PRICES_ENV):
        current_prices()


def test_rate_carries_the_tables_checked_date() -> None:
    prices = parse_prices(_PRICE_FILE)
    assert hourly_rate(ComputeSpec(provider="acme", gpu="H100"), prices).prices_checked == "2026-03-01"
    assert hourly_rate(ComputeSpec(provider="acme", gpu="T4"), prices).prices_checked == "2026-03-01"


def test_disclaimer_names_the_date_and_flags_stale_tables() -> None:
    assert disclaimer("2026-03-01", today=dt.date(2026, 3, 2)) == "estimate from list prices as of 2026-03-01"
    assert not is_stale("2026-03-01", today=dt.date(2026, 6, 29))
    assert is_stale("2026-03-01", today=dt.date(2026, 12, 1))
    assert disclaimer("2026-03-01", today=dt.date(2026, 12, 1)).endswith("; may be out of date")
    assert disclaimer("") == "estimate from list prices"
    assert is_stale("not a date")


# --------------------------------------------------------------------------
# live prices
# --------------------------------------------------------------------------

from nbgpu.backends.base import ProviderPrices  # noqa: E402
from nbgpu.cost import KIND_FILE, KIND_LIST, KIND_LIVE, LIVE_MAX_AGE_S, save_live_prices  # noqa: E402


def test_live_prices_override_the_shipped_ones_they_cover(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbgpu.cost import DEFAULT_PRICES

    monkeypatch.delenv(PRICES_ENV, raising=False)
    save_live_prices(tmp_path, "modal", ProviderPrices({"T4": 0.5, "h100!": 4.0}, 0.15, None), now=1_000_000.0)
    table = current_prices(tmp_path, now=1_000_000.0 + 60)
    assert table.kind == KIND_LIVE and table.checked == "1970-01-12"
    assert table.gpu_hourly[("modal", "T4")] == 0.5
    assert table.gpu_hourly[("modal", "H100!")] == 4.0  # names normalised
    assert table.gpu_hourly[("modal", "L4")] == DEFAULT_PRICES.gpu_hourly[("modal", "L4")]  # not covered: shipped
    assert table.cpu_core_hourly["modal"] == 0.15
    assert table.memory_gib_hourly["modal"] == DEFAULT_PRICES.memory_gib_hourly["modal"]
    rate = hourly_rate(ComputeSpec(provider="modal", gpu="T4"), table)
    assert (rate.prices_source, rate.prices_checked) == (KIND_LIVE, "1970-01-12")
    assert (tmp_path / "prices-live.json").stat().st_mode & 0o777 == 0o600


def test_old_live_prices_are_ignored(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbgpu.cost import DEFAULT_PRICES

    monkeypatch.delenv(PRICES_ENV, raising=False)
    save_live_prices(tmp_path, "modal", ProviderPrices({"T4": 0.5}, None, None), now=0.0)
    assert current_prices(tmp_path, now=LIVE_MAX_AGE_S + 1) is DEFAULT_PRICES


def test_a_broken_live_cache_is_ignored(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbgpu.cost import DEFAULT_PRICES

    monkeypatch.delenv(PRICES_ENV, raising=False)
    (tmp_path / "prices-live.json").write_text("{not json")
    assert current_prices(tmp_path) is DEFAULT_PRICES


def test_the_env_file_beats_live_prices(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_live_prices(tmp_path, "acme", ProviderPrices({"H100": 99.0}, None, None))
    path = tmp_path / "mine.toml"
    path.write_text(_PRICE_FILE)
    monkeypatch.setenv(PRICES_ENV, str(path))
    table = current_prices(tmp_path)
    assert table.kind == KIND_FILE and table.gpu_hourly[("acme", "H100")] == pytest.approx(3.6)


def test_disclaimer_names_the_source() -> None:
    today = dt.date(2026, 11, 15)
    assert disclaimer("2026-11-15", KIND_LIVE, today) == "estimate from your account's rates as of 2026-11-15"
    assert disclaimer("2026-11-15", KIND_LIST, today) == "estimate from list prices as of 2026-11-15"
    assert disclaimer("2026-01-01", KIND_FILE, today) == "estimate from $NBGPU_PRICES as of 2026-01-01; may be out of date"
