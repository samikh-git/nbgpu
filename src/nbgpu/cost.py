"""Hourly prices and running totals, so cost is always visible.

Prices are per hour, in US dollars, for a runtime of a given shape. A price
table maps (provider, gpu) to the GPU's hourly rate. CPU and memory are
priced separately per provider when known; when they are not, they are left
out and the estimate says so. A CPU-only runtime (gpu None) costs only its
cpu and memory; there is no separate base rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from nbgpu.spec import ComputeSpec


@dataclass(frozen=True)
class PriceTable:
    # (provider, gpu name as normalised by `normalise_gpu`) -> USD per hour
    gpu_hourly: Mapping[tuple[str, str], float]
    # provider -> USD per CPU core per hour
    cpu_core_hourly: Mapping[str, float]
    # provider -> USD per GiB of memory per hour
    memory_gib_hourly: Mapping[str, float]
    # where the numbers came from and when they were checked
    source: str
    checked: str  # ISO date


@dataclass(frozen=True)
class Rate:
    usd_per_hour: float
    complete: bool  # False when part of the shape had no known price
    note: str  # e.g. "no price for cpu on 'runpod'"; "" when complete


# Filled from the providers' pricing pages; see `source` and `checked`.
#
# Modal lists prices per second; each value below is that list price x 3600.
# nbgpu runs on Modal Sandboxes, which are "billed at non-preemptible rates of
# 3x list price" for CPU and memory, so the sandbox CPU and memory rates are
# used here. GPU prices for sandboxes are the standard ones ("See standard
# pricing" on the pricing page).
#
# CPU is priced per physical core ("Physical core (2 vCPU equivalent)"), and
# `cpu=` in Sandbox.create is also in physical cores (fractional allowed), so
# ComputeSpec.cpu multiplies the per-core rate directly. Modal bills
# max(request, usage) and has a 0.125-core minimum; this table prices the
# request only.
#
# Aliases, per https://modal.com/docs/guide/gpu:
# - A10G: the older name for Modal's A10; priced as "Nvidia A10".
# - A100: a 40 GB request; Modal may upgrade it to 80 GB, which "does not
#   change the cost", so it is priced as A100-40GB.
# - H100!: an H100 with automatic H200 upgrade disabled; priced as H100.
# - B200+: "billed as B200, regardless of which GPU is used".
# Region selection (1.15 - 1.75x base prices) is not included.
_MODAL_GPU_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A10G": 0.000306,
    "L40S": 0.000542,
    "A100": 0.000583,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100": 0.001097,
    "H100!": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "B200+": 0.001736,
    "B300": 0.001972,
}

DEFAULT_PRICES = PriceTable(
    gpu_hourly={("modal", name): per_s * 3600 for name, per_s in _MODAL_GPU_PER_SECOND.items()},
    cpu_core_hourly={"modal": 0.00003942 * 3600},  # sandbox, per physical core
    memory_gib_hourly={"modal": 0.00000667 * 3600},  # sandbox, per GiB
    source="https://modal.com/pricing; https://modal.com/docs/guide/gpu",
    checked="2026-11-14",
)

_CENT = Decimal("0.01")


def _to_cents(amount: float) -> Decimal:
    # Round half up on the decimal form, so 0.125 -> 0.13 rather than
    # whatever the binary float happens to be closest to.
    return Decimal(repr(amount)).quantize(_CENT, rounding=ROUND_HALF_UP)


def normalise_gpu(gpu: str) -> tuple[str, int]:
    """Split "H100:2" into ("H100", 2) and upper-case the name.
    A missing count is 1. Raises ValueError for a count below 1."""
    name, sep, count_text = gpu.partition(":")
    name = name.strip().upper()
    if not name:
        raise ValueError(f"GPU {gpu!r} has no name")
    count = int(count_text) if sep else 1  # int() raises ValueError for "two"
    if count < 1:
        raise ValueError(f"GPU count in {gpu!r} must be at least 1")
    return name, count


def hourly_rate(compute: ComputeSpec, prices: PriceTable = DEFAULT_PRICES) -> Rate:
    """GPU rate x count, plus cpu x cores, plus memory x GiB, for the provider.
    An unknown GPU makes the whole rate unknown: Rate(0.0, False, note).
    Unknown cpu or memory prices are skipped with complete=False."""
    provider = compute.provider
    total = 0.0
    missing: list[str] = []

    if compute.gpu is not None:
        try:
            name, count = normalise_gpu(compute.gpu)
        except ValueError:
            return Rate(0.0, False, f"cannot price malformed gpu {compute.gpu!r}")
        gpu_price = prices.gpu_hourly.get((provider, name))
        if gpu_price is None:
            return Rate(0.0, False, f"no price for gpu {name!r} on {provider!r}")
        total += gpu_price * count

    if compute.cpu is not None:
        cpu_price = prices.cpu_core_hourly.get(provider)
        if cpu_price is None:
            missing.append(f"no price for cpu on {provider!r}")
        else:
            total += cpu_price * compute.cpu

    if compute.memory_mib is not None:
        memory_price = prices.memory_gib_hourly.get(provider)
        if memory_price is None:
            missing.append(f"no price for memory on {provider!r}")
        else:
            total += memory_price * compute.memory_mib / 1024

    return Rate(total, not missing, "; ".join(missing))


def cost_so_far(rate: Rate, seconds: float) -> float:
    """rate x elapsed hours, rounded to cents. Negative seconds raise ValueError."""
    if seconds < 0:
        raise ValueError(f"elapsed seconds must not be negative, got {seconds}")
    return float(_to_cents(rate.usd_per_hour * seconds / 3600))


def format_usd(amount: float) -> str:
    """"$0.00", "$1.10", "$12.35"; amounts under one cent but above zero show "<$0.01"."""
    if 0 < amount < 0.01:
        return "<$0.01"
    cents = _to_cents(amount)
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents):.2f}"
