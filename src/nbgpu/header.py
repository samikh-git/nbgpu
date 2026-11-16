"""Edit the `dependencies` of a notebook's PEP 723 header, touching nothing else.

Used to record packages installed from marimo's UI (seen by nbgpu.proxy) in
the notebook on the laptop. Requirements are matched by distribution name: adding
`emoji==2.16` replaces an `emoji` entry, removing `emoji` drops any entry
for it. The rest of the header ([tool.nbgpu], comments, other keys) and of
the file is kept byte for byte; only the `dependencies = [...]` assignment
is rewritten (one line when short, else one entry per line). A file with no
header gets one at the top.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Sequence

from nbgpu.spec import block_toml, requirement_name, script_block

__all__ = ["requirement_name", "update_dependencies"]

_DEPS_START = re.compile(r"(?m)^# dependencies\s*=")
_ONE_LINE_MAX = 88


def _format(deps: Sequence[str]) -> str:
    one = "# dependencies = " + json.dumps(list(deps))
    if len(one) <= _ONE_LINE_MAX:
        return one + "\n"
    lines = ["# dependencies = ["] + [f"#     {json.dumps(d)}," for d in deps] + ["# ]"]
    return "\n".join(lines) + "\n"


def _deps_span(block_text: str, start: int) -> int:
    """End offset of the `dependencies = [...]` assignment starting at `start`
    in the header block text (bracket matching outside strings)."""
    depth, i, in_str = 0, start, None
    while i < len(block_text):
        ch = block_text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = block_text.find("\n", i)
                return len(block_text) if end < 0 else end + 1
        i += 1
    raise ValueError("unterminated dependencies list")


def update_dependencies(text: str, add: Sequence[str] = (), remove: Sequence[str] = ()) -> tuple[str, list[str]]:
    """Return (new text, the dependencies after the change). Raises
    ValueError if the header cannot be parsed (the file is then left alone)."""
    match = script_block(text)  # SpecError (a ValueError) for several blocks
    if match is None:
        deps = _merge([], add, remove)
        header = "# /// script\n" + _format(deps) + "# ///\n"
        return (header + "\n" + text, deps) if deps else (text, deps)
    try:
        meta = tomllib.loads(block_toml(match))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in the script header: {exc}") from exc
    current = meta.get("dependencies", [])
    if not isinstance(current, list) or not all(isinstance(d, str) for d in current):
        raise ValueError("dependencies is not a list of strings")
    deps = _merge(current, add, remove)
    if deps == current:
        return text, deps
    block_start, block_end = match.start("content"), match.end("content")
    block = text[block_start:block_end]
    found = _DEPS_START.search(block)
    if found is None:
        # Insert after requires-python if present, else first in the block.
        rp = re.search(r"(?m)^# requires-python\s*=.*\n", block)
        at = rp.end() if rp else 0
        new_block = block[:at] + _format(deps) + block[at:]
    else:
        end = _deps_span(block, found.start())
        new_block = block[: found.start()] + _format(deps) + block[end:]
    return text[:block_start] + new_block + text[block_end:], deps


def _merge(current: Sequence[str], add: Sequence[str], remove: Sequence[str]) -> list[str]:
    out = list(current)
    drop = {requirement_name(r) for r in remove}
    out = [d for d in out if requirement_name(d) not in drop]
    for req in add:
        name = requirement_name(req)
        same = [i for i, d in enumerate(out) if requirement_name(d) == name]
        if not same:
            out.append(req)
        elif req != out[same[0]] and any(op in req for op in "=<>~!@"):
            out[same[0]] = req  # an explicit version replaces the old entry
    return out
