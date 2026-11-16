"""Read what a notebook needs from its PEP 723 header.

A notebook declares its compute next to its dependencies:

    # /// script
    # requires-python = ">=3.12"
    # dependencies = ["marimo", "torch"]
    #
    # [tool.nbgpu]
    # provider = "modal"
    # gpu = "A10G"
    # cpu = 8
    # memory = "32GiB"
    # idle-timeout = "20m"
    # max-runtime = "8h"
    # volumes = { "/data" = "my-datasets" }
    # exclude = ["data/", "*.ckpt"]
    # requirements = "requirements.txt"
    # ///

`exclude` lists gitignore-style patterns (relative to the project root) for
files that should not be uploaded or pulled even though git tracks them;
`.gitignore` files are always respected (see nbgpu.ignore).

`requirements` names requirements files (a path or a list, relative to the
notebook's directory) whose packages are installed with the header's
`dependencies`. Only package lines (and `-r` includes) are supported.

Every key under [tool.nbgpu] is optional. A notebook with no header, or a
header with no [tool.nbgpu] table, gets the defaults below (a CPU-only
runtime on Modal).

Overrides (from CLI flags) use the same key names as the header, win over
the header, and go through the same validation. An override whose value is
None means "not given" and is ignored, so the CLI can pass every flag.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

TOOL_TABLE = "nbgpu"

DEFAULT_PROVIDER = "modal"
DEFAULT_IDLE_TIMEOUT_S = 20 * 60
DEFAULT_MAX_RUNTIME_S = 8 * 60 * 60
# The volume `cache` uses unless the header names another (or turns it off).
DEFAULT_CACHE_VOLUME = "nbgpu-cache"

KNOWN_KEYS = frozenset(
    {"provider", "gpu", "cpu", "memory", "idle-timeout", "max-runtime", "volumes", "exclude", "requirements",
     "cache", "snapshot", "cell-cache"}
)


class SpecError(ValueError):
    """The header or an override is invalid. The message names the key."""


@dataclass(frozen=True)
class ComputeSpec:
    provider: str = DEFAULT_PROVIDER
    # Provider-specific GPU string, kept as written (e.g. "A10G", "H100:2").
    # None means no GPU. The backend validates it, not this module.
    gpu: str | None = None
    cpu: float | None = None
    memory_mib: int | None = None
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S
    max_runtime_s: int = DEFAULT_MAX_RUNTIME_S
    # Absolute mount path in the runtime -> provider volume name.
    volumes: Mapping[str, str] = field(default_factory=dict)
    # Volume for download and compile caches, kept between sessions
    # (backends.base.CACHE_MOUNT); None turns it off. Backends without
    # volumes ignore it: their disk already persists.
    cache_volume: str | None = DEFAULT_CACHE_VOLUME


@dataclass(frozen=True)
class NotebookSpec:
    path: Path  # absolute path of the local notebook
    requires_python: str | None
    dependencies: tuple[str, ...]
    compute: ComputeSpec
    # gitignore-style patterns from [tool.nbgpu] exclude, relative to the project root
    exclude: tuple[str, ...] = ()
    # requirements files read into `dependencies` (absolute paths)
    requirement_files: tuple[Path, ...] = ()
    # Snapshot the runtime's filesystem at stop so the next start resumes
    # from it ([tool.nbgpu] snapshot; `nbgpu stop --snapshot` for one stop).
    snapshot: bool = False
    # Keep marimo's own cache (mo.persistent_cache, `__marimo__/cache`) on the
    # cache volume so it survives sessions ([tool.nbgpu] cell-cache). Opt-in:
    # it also moves Python's bytecode cache there.
    cell_cache: bool = False


# The PEP 723 reference regex.
_BLOCK_RE = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)

# "1h30m15s", "2h 15m", "90s": each unit at most once, in h/m/s order.
_DURATION_RE = re.compile(
    r"(?=\d)(?:(?P<h>\d+)h)?\s?(?:(?P<m>\d+)m)?\s?(?:(?P<s>\d+)s)?"
)

_MEMORY_RE = re.compile(r"(?P<n>\d+)\s*(?P<unit>[A-Za-z]+)")
_MIB = 1 << 20
_MEMORY_UNITS_BYTES = {
    "mib": 1 << 20,
    "gib": 1 << 30,
    "mb": 10**6,
    "gb": 10**9,
}


def requirement_name(requirement: str) -> str:
    """The normalized distribution name of a requirement string (PEP 503):
    "Torch_Vision[cuda]>=0.1" -> "torch-vision"."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else requirement.strip().lower()


def script_block(text: str) -> re.Match[str] | None:
    """The `# /// script` block (its "content" group is the commented TOML),
    or None if absent. Follows the PEP 723 reference regex: the block starts
    with a line that is exactly `# /// script`, ends with a line that is
    exactly `# ///`, and every line in between starts with `#`. Raises
    SpecError if there is more than one."""
    blocks = [m for m in _BLOCK_RE.finditer(text) if m.group("type") == "script"]
    if len(blocks) > 1:
        raise SpecError("more than one `# /// script` block")
    return blocks[0] if blocks else None


def block_toml(block: re.Match[str]) -> str:
    """The TOML text of a script block: each line without its `# ` or `#`."""
    return "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in block.group("content").splitlines(keepends=True)
    )


def read_script_metadata(text: str) -> dict | None:
    """Return the parsed TOML of the `# /// script` block, or None if absent.
    Raises SpecError if more than one script block exists or the TOML is invalid."""
    block = script_block(text)
    if block is None:
        return None
    try:
        return tomllib.loads(block_toml(block))
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"invalid TOML in `# /// script` block: {exc}") from exc


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_duration(value: str | int) -> int:
    """Return seconds. Accepts an int (seconds) or strings like "90s", "20m",
    "8h", "1h30m", "2h 15m". Raises SpecError for zero, negative, or anything
    else ("", "soon", "5d", 1.5)."""
    if _is_int(value):
        seconds = int(value)
    elif isinstance(value, str) and (m := _DURATION_RE.fullmatch(value)) and any(m.groups()):
        h, mins, s = (int(g) if g else 0 for g in m.group("h", "m", "s"))
        seconds = h * 3600 + mins * 60 + s
    else:
        raise SpecError(f"invalid duration {value!r}; use seconds or e.g. '20m', '1h30m'")
    if seconds <= 0:
        raise SpecError(f"duration must be positive, got {value!r}")
    return seconds


def parse_memory(value: str | int) -> int:
    """Return MiB. Accepts an int (MiB) or strings with a unit: "512MiB",
    "32GiB" (binary) or "500MB", "32GB" (decimal, converted to MiB and rounded
    up). Units are case-insensitive. Raises SpecError for zero, negative, a
    missing unit on a string, or an unknown unit."""
    if _is_int(value):
        mib = int(value)
    elif isinstance(value, str) and (m := _MEMORY_RE.fullmatch(value)):
        unit = _MEMORY_UNITS_BYTES.get(m.group("unit").lower())
        if unit is None:
            raise SpecError(f"unknown memory unit in {value!r}; use MiB, GiB, MB or GB")
        mib = -(-int(m.group("n")) * unit // _MIB)  # ceiling division
    else:
        raise SpecError(f"invalid memory {value!r}; use MiB as an integer or e.g. '32GiB'")
    if mib <= 0:
        raise SpecError(f"memory must be positive, got {value!r}")
    return mib


def _check_keys(table: Mapping[str, object], where: str) -> None:
    for key in table:
        if key not in KNOWN_KEYS:
            known = ", ".join(sorted(KNOWN_KEYS))
            raise SpecError(f"unknown key {key!r} {where} (known keys: {known})")


def _non_empty_str(key: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _positive_number(key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise SpecError(f"{key} must be a positive number, got {value!r}")
    return float(value)


def _with_key(key: str, parse: Callable[[str | int], int], value: object) -> int:
    try:
        return parse(value)  # type: ignore[arg-type]  # the parsers reject other types
    except SpecError as exc:
        raise SpecError(f"{key}: {exc}") from exc


def _volumes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise SpecError(f"volumes must be a table of mount path -> volume name, got {value!r}")
    volumes: dict[str, str] = {}
    for mount, name in value.items():
        if not isinstance(mount, str) or not PurePosixPath(mount).is_absolute():
            raise SpecError(f"volumes: mount path {mount!r} must be absolute")
        if not isinstance(name, str) or not name:
            raise SpecError(f"volumes: volume name for {mount!r} must be a non-empty string")
        volumes[mount] = name
    return volumes


def _compute(table: Mapping[str, object]) -> ComputeSpec:
    kwargs: dict[str, object] = {}
    if "provider" in table:
        kwargs["provider"] = _non_empty_str("provider", table["provider"])
    if "gpu" in table:
        kwargs["gpu"] = _non_empty_str("gpu", table["gpu"])
    if "cpu" in table:
        kwargs["cpu"] = _positive_number("cpu", table["cpu"])
    if "memory" in table:
        kwargs["memory_mib"] = _with_key("memory", parse_memory, table["memory"])
    if "idle-timeout" in table:
        kwargs["idle_timeout_s"] = _with_key("idle-timeout", parse_duration, table["idle-timeout"])
    if "max-runtime" in table:
        kwargs["max_runtime_s"] = _with_key("max-runtime", parse_duration, table["max-runtime"])
    if "volumes" in table:
        kwargs["volumes"] = _volumes(table["volumes"])
    if "cache" in table:
        kwargs["cache_volume"] = _cache(table["cache"])
    return ComputeSpec(**kwargs)


def _cache(value: object) -> str | None:
    if value is True:
        return DEFAULT_CACHE_VOLUME
    if value is False:
        return None
    if isinstance(value, str) and value:
        return value
    raise SpecError(f"cache must be true, false or a volume name, got {value!r}")


def parse_requirements_file(path: Path, _seen: frozenset[Path] = frozenset()) -> list[str]:
    """Package lines of a requirements file, following `-r` includes.
    Comments and blank lines are skipped; any other option (`-e`, `--index-url`,
    ...) raises SpecError naming the file and line, rather than being ignored."""
    path = Path(path).resolve()
    if path in _seen:
        raise SpecError(f"requirements: {path} includes itself")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"requirements: cannot read {str(path)!r}: {exc}") from exc
    out: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        include = re.fullmatch(r"(?:-r|--requirement)\s*=?\s*(\S+)", line)
        if include:
            out += parse_requirements_file(path.parent / include.group(1), _seen | {path})
        elif line.startswith("-"):
            raise SpecError(f"requirements: unsupported line {number} in {path.name}: {line!r} "
                            "(only package requirements and -r includes are supported)")
        else:
            out.append(line)
    return out


def _requirement_files(value: object, base: Path) -> tuple[Path, ...]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(p, str) and p.strip() for p in items):
        raise SpecError(f"requirements must be a path or a list of paths, got {value!r}")
    return tuple((base / p).resolve() for p in items)


def _exclude(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(p, str) and p.strip() for p in value):
        raise SpecError(f"exclude must be a list of non-empty pattern strings, got {value!r}")
    return tuple(value)  # type: ignore[arg-type]


def load_spec(path: Path, overrides: Mapping[str, object] | None = None) -> NotebookSpec:
    """Read the notebook at `path` and return its spec.

    - `path` is resolved to an absolute path; a missing file raises SpecError.
    - Unknown keys under [tool.nbgpu] (or in `overrides`) raise SpecError
      naming the key, so a typo like `gpus` is not silently ignored.
    - `provider` and `gpu` must be non-empty strings; `cpu` a positive number;
      `volumes` a table of absolute paths to non-empty names.
    - `dependencies` defaults to () and `requires_python` to None.
    """
    path = Path(path).resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SpecError(f"cannot read notebook {str(path)!r}: {exc}") from exc

    meta = read_script_metadata(text) or {}

    requires_python = meta.get("requires-python")
    if requires_python is not None and not isinstance(requires_python, str):
        raise SpecError(f"requires-python must be a string, got {requires_python!r}")

    dependencies = meta.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(isinstance(d, str) for d in dependencies):
        raise SpecError(f"dependencies must be a list of strings, got {dependencies!r}")

    tool = meta.get("tool", {})
    table = tool.get(TOOL_TABLE, {}) if isinstance(tool, Mapping) else None
    if not isinstance(table, Mapping):
        raise SpecError(f"[tool.{TOOL_TABLE}] must be a table")
    _check_keys(table, f"under [tool.{TOOL_TABLE}]")

    given = {k: v for k, v in (overrides or {}).items() if v is not None}
    _check_keys(given, "in overrides")

    merged = {**table, **given}
    files = _requirement_files(merged["requirements"], path.parent) if "requirements" in merged else ()
    for file in files:
        dependencies = [*dependencies, *parse_requirements_file(file)]
    compute = _compute({k: v for k, v in merged.items() if k not in ("exclude", "requirements", "snapshot", "cell-cache")})
    cell_cache = _bool("cell-cache", merged["cell-cache"]) if "cell-cache" in merged else False
    if cell_cache and compute.cache_volume is None:
        raise SpecError("cell-cache keeps marimo's cache on the cache volume; it needs `cache` (remove `cache = false`)")
    return NotebookSpec(
        path=path,
        requires_python=requires_python,
        dependencies=tuple(dependencies),
        compute=compute,
        exclude=_exclude(merged["exclude"]) if "exclude" in merged else (),
        requirement_files=files,
        snapshot=_bool("snapshot", merged["snapshot"]) if "snapshot" in merged else False,
        cell_cache=cell_cache,
    )


def _bool(key: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise SpecError(f"{key} must be true or false, got {value!r}")
    return value
