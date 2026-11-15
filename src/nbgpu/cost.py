"""Hourly prices and running totals, so cost is always visible.

Prices are per hour, in US dollars, for a runtime of a given shape. A price
table maps (provider, gpu) to the GPU's hourly rate. CPU and memory are
priced separately per provider when known; when they are not, they are left
out and the estimate says so. A CPU-only runtime (gpu None) costs only its
cpu and memory; there is no separate base rate.

Where prices come from, first match wins:

1. $NBGPU_PRICES: a price file that replaces everything (kind "file").
2. Live rates: backends with CAP_BILLING report the account's current prices
   (Modal: Workspace.billing.rates()). lifecycle caches them in the state
   directory (LIVE_CACHE_NAME) at most once per LIVE_REFRESH_AFTER_S; a cache
   younger than LIVE_MAX_AGE_S overrides the shipped prices it covers
   (kind "live").
3. prices.toml next to this module: list prices copied by hand from the
   providers' pricing pages, with the date they were checked (kind "list").

Every displayed rate carries its source and date (Rate.prices_source,
Rate.prices_checked; see `disclaimer`).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import tempfile
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from importlib import resources
from pathlib import Path

from nbgpu.backends.base import ProviderPrices
from nbgpu.spec import ComputeSpec


KIND_LIST = "list"  # the shipped prices.toml
KIND_LIVE = "live"  # fetched from the provider for this account
KIND_FILE = "file"  # $NBGPU_PRICES

LIVE_CACHE_NAME = "prices-live.json"
LIVE_REFRESH_AFTER_S = 24 * 3600.0
LIVE_MAX_AGE_S = 7 * 24 * 3600.0


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
    kind: str = KIND_LIST  # KIND_LIST, KIND_LIVE or KIND_FILE


@dataclass(frozen=True)
class Rate:
    usd_per_hour: float
    complete: bool  # False when part of the shape had no known price
    note: str  # e.g. "no price for cpu on 'runpod'"; "" when complete
    prices_checked: str = ""  # the price table's `checked` date; "" if unknown
    prices_source: str = ""  # the price table's kind (KIND_*); "" if unknown


# Prices older than this are flagged as possibly out of date.
STALE_AFTER_DAYS = 120

PRICES_ENV = "NBGPU_PRICES"


def parse_prices(text: str, origin: str = "prices.toml") -> PriceTable:
    """Read a price file (see the shipped prices.toml for the format).

    Top level: `checked` (ISO date) and `sources` (list of URLs). One table
    per provider with optional `cpu-core` and `memory-gib` and a `gpu`
    sub-table; every price is USD per second. GPU names are normalised the
    same way as in the header. Raises ValueError naming `origin` on any
    malformed entry, so a broken override fails loudly instead of silently
    pricing at zero."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{origin}: {exc}") from None

    def price(value: object, where: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{origin}: {where} must be a non-negative number, got {value!r}")
        return float(value) * 3600

    checked = data.pop("checked", None)
    if isinstance(checked, dt.date):  # an unquoted TOML date
        checked = checked.isoformat()
    if not isinstance(checked, str):
        raise ValueError(f"{origin}: `checked` (the date the prices were last verified) is required")
    try:
        dt.date.fromisoformat(checked)
    except ValueError:
        raise ValueError(f"{origin}: `checked` must be an ISO date, got {checked!r}") from None
    sources = data.pop("sources", [])
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise ValueError(f"{origin}: `sources` must be a list of strings")

    gpu_hourly: dict[tuple[str, str], float] = {}
    cpu_core_hourly: dict[str, float] = {}
    memory_gib_hourly: dict[str, float] = {}
    for provider, table in data.items():
        if not isinstance(table, dict):
            raise ValueError(f"{origin}: unknown top-level key {provider!r}")
        for key, value in table.items():
            if key == "cpu-core":
                cpu_core_hourly[provider] = price(value, f"{provider}.cpu-core")
            elif key == "memory-gib":
                memory_gib_hourly[provider] = price(value, f"{provider}.memory-gib")
            elif key == "gpu" and isinstance(value, dict):
                for gpu, per_s in value.items():
                    gpu_hourly[(provider, normalise_gpu(gpu)[0])] = price(per_s, f"{provider}.gpu.{gpu}")
            else:
                raise ValueError(f"{origin}: unknown key {key!r} under [{provider}] "
                                 "(expected cpu-core, memory-gib or [gpu])")
    return PriceTable(gpu_hourly, cpu_core_hourly, memory_gib_hourly, "; ".join(sources), checked)


def _shipped_prices() -> PriceTable:
    text = resources.files("nbgpu").joinpath("prices.toml").read_text(encoding="utf-8")
    return parse_prices(text, "nbgpu's prices.toml")


def current_prices(state_dir: Path | None = None, now: float | None = None) -> PriceTable:
    """The table to price with (see the module docstring for the order).
    `state_dir` holds the live-rates cache; None skips it. Raises ValueError
    for a missing or malformed $NBGPU_PRICES file; a bad cache is ignored."""
    path = os.environ.get(PRICES_ENV)
    if path:
        try:
            text = Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{PRICES_ENV}={path}: {exc.strerror or exc}") from None
        table = parse_prices(text, f"{PRICES_ENV}={path}")
        return PriceTable(table.gpu_hourly, table.cpu_core_hourly, table.memory_gib_hourly, table.source,
                          table.checked, KIND_FILE)
    if state_dir is None:
        return DEFAULT_PRICES
    return _with_live(DEFAULT_PRICES, _read_live_cache(state_dir), time.time() if now is None else now)


def _read_live_cache(state_dir: Path) -> dict:
    try:
        data = json.loads((state_dir / LIVE_CACHE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _with_live(base: PriceTable, cache: Mapping[str, object], now: float) -> PriceTable:
    """`base` with every fresh provider entry of the cache applied."""
    gpu = dict(base.gpu_hourly)
    cpu = dict(base.cpu_core_hourly)
    memory = dict(base.memory_gib_hourly)
    fetched: list[float] = []
    for provider, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        at = entry.get("fetched_at")
        if not isinstance(at, (int, float)) or not 0 <= now - at <= LIVE_MAX_AGE_S:
            continue
        gpus = entry.get("gpu_hourly")
        if isinstance(gpus, dict):
            gpu.update({(provider, normalise_gpu(name)[0]): float(v) for name, v in gpus.items()
                        if isinstance(v, (int, float)) and v >= 0})
        for key, target in (("cpu_core_hourly", cpu), ("memory_gib_hourly", memory)):
            value = entry.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                target[provider] = float(value)
        fetched.append(float(at))
    if not fetched:
        return base
    checked = dt.datetime.fromtimestamp(min(fetched), dt.timezone.utc).date().isoformat()
    return PriceTable(gpu, cpu, memory, "the provider's current rates for this account", checked, KIND_LIVE)


def live_cache_age(state_dir: Path, provider: str, now: float | None = None) -> float | None:
    """Seconds since `provider`'s live rates were cached, or None if never."""
    entry = _read_live_cache(state_dir).get(provider)
    at = entry.get("fetched_at") if isinstance(entry, dict) else None
    if not isinstance(at, (int, float)):
        return None
    return (time.time() if now is None else now) - at


def save_live_prices(state_dir: Path, provider: str, prices: ProviderPrices, now: float | None = None) -> None:
    """Cache `provider`'s current prices (atomic replace, mode 0600)."""
    cache = _read_live_cache(state_dir)
    cache[provider] = {
        "fetched_at": time.time() if now is None else now,
        "gpu_hourly": dict(prices.gpu_hourly),
        "cpu_core_hourly": prices.cpu_core_hourly,
        "memory_gib_hourly": prices.memory_gib_hourly,
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".prices-", dir=state_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), 0o600)
            json.dump(cache, f, indent=1, sort_keys=True)
        os.replace(tmp, state_dir / LIVE_CACHE_NAME)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def is_stale(checked: str, today: dt.date | None = None) -> bool:
    """True when `checked` (an ISO date) is more than STALE_AFTER_DAYS ago.
    An empty or malformed date counts as stale."""
    try:
        day = dt.date.fromisoformat(checked)
    except ValueError:
        return True
    return ((today or dt.date.today()) - day).days > STALE_AFTER_DAYS


def disclaimer(checked: str, source: str = KIND_LIST, today: dt.date | None = None) -> str:
    """What a displayed rate is, e.g. "estimate from list prices as of
    2026-11-15" or "estimate from your account's rates as of 2026-11-15".
    List and file prices add "; may be out of date" when stale. `checked` ""
    (a session recorded before dates were kept) leaves the date out."""
    what = {KIND_LIVE: "your account's rates", KIND_FILE: f"${PRICES_ENV}"}.get(source, "list prices")
    if not checked:
        return f"estimate from {what}"
    text = f"estimate from {what} as of {checked}"
    if source != KIND_LIVE and is_stale(checked, today):
        text += "; may be out of date"
    return text

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


def hourly_rate(compute: ComputeSpec, prices: PriceTable | None = None) -> Rate:
    """GPU rate x count, plus cpu x cores, plus memory x GiB, for the provider.
    An unknown GPU makes the whole rate unknown: Rate(0.0, False, note).
    Unknown cpu or memory prices are skipped with complete=False.
    `prices` defaults to current_prices()."""
    prices = current_prices() if prices is None else prices
    provider = compute.provider
    total = 0.0
    missing: list[str] = []

    if compute.gpu is not None:
        try:
            name, count = normalise_gpu(compute.gpu)
        except ValueError:
            return Rate(0.0, False, f"cannot price malformed gpu {compute.gpu!r}", prices.checked, prices.kind)
        gpu_price = prices.gpu_hourly.get((provider, name))
        if gpu_price is None:
            return Rate(0.0, False, f"no price for gpu {name!r} on {provider!r}", prices.checked, prices.kind)
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

    return Rate(total, not missing, "; ".join(missing), prices.checked, prices.kind)


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


# The table shipped with nbgpu, loaded once at import (after normalise_gpu
# exists). Callers should use `current_prices()`, which honours $NBGPU_PRICES.
DEFAULT_PRICES = _shipped_prices()
