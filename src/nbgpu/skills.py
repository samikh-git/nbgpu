"""The agent skills that ship inside the package, and installing them.

The `nbgpu` and `nbgpu-notebook` skills live in `nbgpu/skills/<name>/`
(a `SKILL.md` folder each) so that every install, including Homebrew's,
carries the copy that matches its CLI. `install_skills` puts them in an
agent's skills directory, `~/.claude/skills` by default.

Contract:
- Each skill becomes `<dest>/<name>`. With `link=False` (the default) it is
  a copy containing a marker file (`MARKER`); with `link=True` a symlink to
  the packaged folder.
- An existing target is replaced only when nbgpu owns it: a symlink (removing
  one loses nothing) or a directory holding `MARKER`. Anything else is left
  alone and reported in `skipped`, unless `force=True`.
- Replacing is not atomic, but a copy is built next to the target first, so a
  failure mid-copy leaves the old version in place.

Decisions beyond the contract:
- Copy is the default because a package manager's install path moves on
  every upgrade (Homebrew's Cellar is versioned), which would leave links
  dangling. Copies go stale instead, and a re-run after upgrading fixes
  them; the marker records the version that wrote them.
- `nbgpu-dev` is not shipped: it is for working on this repository and lives
  in its `.claude/skills/`.
"""

from __future__ import annotations

import importlib.resources
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from nbgpu import __version__

SKILL_NAMES: tuple[str, ...] = ("nbgpu", "nbgpu-notebook")
MARKER = ".nbgpu-installed"
DEFAULT_DEST = Path("~/.claude/skills")


@dataclass
class InstallResult:
    dest: Path
    installed: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)


def packaged_skills_dir() -> Path:
    return Path(str(importlib.resources.files("nbgpu").joinpath("skills")))


def _owned(target: Path) -> bool:
    return target.is_symlink() or (target.is_dir() and (target / MARKER).is_file())


def _remove(target: Path) -> None:
    if target.is_symlink() or target.is_file():
        target.unlink()
    else:
        shutil.rmtree(target)


def install_skills(dest: Path | None = None, *, link: bool = False, force: bool = False) -> InstallResult:
    dest = (dest or DEFAULT_DEST).expanduser()
    source_root = packaged_skills_dir()
    result = InstallResult(dest=dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in SKILL_NAMES:
        source = source_root / name
        if not (source / "SKILL.md").is_file():
            raise FileNotFoundError(f"packaged skill missing: {source}")
        target = dest / name
        exists = target.exists() or target.is_symlink()
        if exists and not force and not _owned(target):
            result.skipped[name] = f"{target} exists and was not installed by nbgpu (use --force to replace it)"
            continue
        if link:
            if exists:
                _remove(target)
            target.symlink_to(source.resolve(), target_is_directory=True)
        else:
            staging = dest / f".{name}.nbgpu-tmp"
            if staging.exists() or staging.is_symlink():
                _remove(staging)
            shutil.copytree(source, staging, ignore=shutil.ignore_patterns("__pycache__"))
            (staging / MARKER).write_text(f"{__version__}\n", encoding="utf-8")
            if exists:
                _remove(target)
            staging.rename(target)
        (result.replaced if exists else result.installed).append(name)
    return result
