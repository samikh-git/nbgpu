"""Which project files are left out of sync, gitignore-style.

Three layers, each in gitignore syntax and evaluated like git does:

1. `.git/info/exclude` of the project, if any (lowest priority).
2. Every `.gitignore` in the project, a deeper file overriding a shallower
   one; patterns are relative to the directory holding the `.gitignore`.
3. `exclude = [...]` from the notebook's [tool.nbgpu] table, relative to the
   project root (highest priority), for files git tracks but that should not
   be uploaded (datasets, checkpoints, secrets).

As in git, nothing inside an ignored directory can be re-included.
The rules come from the local project only and apply to both directions:
an ignored path is neither pushed nor pulled. Files the user's global git
config excludes are not consulted, so a session behaves the same on any
machine.

`.gitignore` files are re-read when they change (size or mtime), so editing
one during a session takes effect on the next sync pass.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

import pathspec

GITIGNORE = ".gitignore"
# Cache key for .git/info/exclude; "\0" cannot be part of a directory path.
_INFO_EXCLUDE_KEY = "\0info/exclude"


def _spec(lines: Iterable[str]) -> pathspec.GitIgnoreSpec | None:
    spec = pathspec.GitIgnoreSpec.from_lines(list(lines))
    return spec if len(spec.patterns) else None


class IgnoreRules:
    def __init__(self, root: Path, exclude: Sequence[str] = ()) -> None:
        self._root = Path(root)
        self._exclude = _spec(exclude)
        # directory rel path ("" for root), or _INFO_EXCLUDE_KEY ->
        # ((size, mtime_ns) or None, spec or None)
        self._cache: dict[str, tuple[tuple[int, int] | None, pathspec.GitIgnoreSpec | None]] = {}
        self._dir_memo: dict[str, bool] = {}

    @property
    def exclude(self) -> tuple[str, ...]:
        return tuple(str(p.pattern) for p in self._exclude.patterns) if self._exclude else ()

    def refresh(self) -> None:
        """Forget memoised directory answers; call once per scan so edits to
        .gitignore files (and .git/info/exclude) are picked up."""
        self._dir_memo.clear()

    @staticmethod
    def _read(path: Path) -> pathspec.GitIgnoreSpec | None:
        try:
            return _spec(path.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            return None

    def _gitignore(self, dir_rel: str) -> pathspec.GitIgnoreSpec | None:
        path = self._root / dir_rel / GITIGNORE if dir_rel else self._root / GITIGNORE
        return self._cached(dir_rel, path)

    def _info_exclude(self) -> pathspec.GitIgnoreSpec | None:
        return self._cached(_INFO_EXCLUDE_KEY, self._root / ".git" / "info" / "exclude")

    def _cached(self, key_name: str, path: Path) -> pathspec.GitIgnoreSpec | None:
        """The rules in `path`, re-read only when its size or mtime changes."""
        try:
            st = os.stat(path)
            key: tuple[int, int] | None = (st.st_size, st.st_mtime_ns)
        except OSError:
            key = None
        cached = self._cache.get(key_name)
        if cached is not None and cached[0] == key:
            return cached[1]
        spec = self._read(path) if key is not None else None
        self._cache[key_name] = (key, spec)
        return spec

    def _match(self, rel: str, is_dir: bool) -> bool:
        """Whether `rel` itself is matched (ignoring its parents)."""
        parts = PurePosixPath(rel).parts
        verdict = False
        layers: list[tuple[str, pathspec.GitIgnoreSpec | None]] = [("", self._info_exclude())]
        for depth in range(len(parts)):  # the directories holding rel, root first
            layers.append(("/".join(parts[:depth]), self._gitignore("/".join(parts[:depth]))))
        layers.append(("", self._exclude))
        for base, spec in layers:
            if spec is None:
                continue
            sub = rel[len(base) + 1 :] if base else rel
            result = spec.check_file(sub + "/" if is_dir else sub)
            if result.include is not None:
                verdict = result.include
        return verdict

    def _dir_ignored(self, dir_rel: str) -> bool:
        """Whether the directory or any of its parents is ignored."""
        if not dir_rel:
            return False
        memo = self._dir_memo.get(dir_rel)
        if memo is None:
            parent = dir_rel.rpartition("/")[0]
            memo = self._dir_ignored(parent) or self._match(dir_rel, True)
            self._dir_memo[dir_rel] = memo
        return memo

    def is_ignored(self, rel: str, is_dir: bool = False) -> bool:
        """Whether the relative POSIX path is left out of sync."""
        rel = rel.strip("/")
        if not rel:
            return False
        if is_dir:
            return self._dir_ignored(rel)
        parent = rel.rpartition("/")[0]
        return self._dir_ignored(parent) or self._match(rel, False)
