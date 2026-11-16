"""Two-way file sync between the local project and the runtime.

Editing in the marimo UI is a first-class way to work, so remote edits are
pulled back as reliably as local edits are pushed. The engine is polling and
deterministic: each `poll_once` compares content hashes on both sides with
the last version both sides agreed on (the base) and acts per file.

Per file, with hashes (None = file absent):

| base | local | remote | action |
|------|-------|--------|--------|
| any  | x     | x      | NOOP (record x as base if it differs) |
| b    | b     | r      | PULL |
| b    | l     | b      | PUSH |
| b    | l     | r      | MERGE (three-way); CONFLICT if it does not merge cleanly |
| None | l     | None   | PUSH (new local file) |
| None | None  | r      | PULL (new remote file), only if `pull_allowed` |
| None | l     | r      | MERGE with an empty base; CONFLICT if not clean |
| b    | None  | *      | NOOP: deletions are never propagated in v1 |
| b    | *     | None   | NOOP: deletions are never propagated in v1 |

Race safety
-----------
Both sides keep changing while a poll runs: the user saves in an editor, and
the marimo UI autosaves on the runtime about a second after typing. A poll
decides on hashes taken at one moment and writes a moment later, so every
write is a compare-and-swap against the hash the decision was made on:

* Remote writes use `write_file(expect=<hash seen this poll>)` (or ABSENT for
  a new file). The check and the replace are one atomic step in the runtime.
  If it fails, a UI edit landed in between: the path is skipped this round
  (reported in `skipped`, no base recorded) and the next poll sees both
  changes and merges or conflicts.
* Local writes re-hash the local file immediately before replacing it and
  skip the path if it no longer matches what was scanned (or a file appeared
  where none was). New local files (pulls of new remote files, conflict
  copies) are created with a hard link, which fails if the name exists, so
  they never replace anything.
* A base is only ever recorded from the exact bytes whose hash was compared
  (content read back is verified against the scanned hash), so a save made
  mid-poll can never become the base and be reverted by the next poll.
* Remote content that was read is verified against the hash it was decided
  on; a mismatch skips the path this round.

Residual window: the local check-then-replace is not atomic. An editor that
saves in the microseconds between the re-hash and the rename loses that save
(the remote side has no such window). Local directories are also checked for
symlinks just before writing, with the same kind of window.

Conflicts
---------
CONFLICT never overwrites either side. The remote version is written next
to the local file as a conflict copy, `<stem>.remote<suffix>` (e.g.
notebook.remote.py), or `<stem>.remote-2<suffix>` and so on if that name is
taken: an existing file is never overwritten. The engine records the copies
it created in its state; only those are treated as copies and left out of
sync, so a user file that merely looks like one syncs normally.

While a conflict is open, the base is left unchanged, the path is reported on
every poll, and the copy is refreshed from the remote whenever the remote
changes (unless the user edited the copy). Deleting the copy is how the user
resolves it: on the next poll the local file wins and is pushed, but only if
the remote still equals the content last shown in the copy. If the remote
changed since, the conflict is raised again with a fresh copy instead.

A clean MERGE writes the merged bytes to both sides and records the merged
hash as the base.

Remote-to-local is restricted (`pull_allowed`): a new remote file is pulled
only if its suffix is in PULL_SUFFIXES and it is at most PULL_MAX_BYTES, so
cell outputs and data written by notebooks do not flood the project. Files
that already exist locally are always synced both ways. A remote path is
never written locally if it is absolute, contains `..`, or would land outside
the project or pass through a symlink (`skipped`).

Local files over PUSH_MAX_BYTES, and files that cannot be read, are reported
in `skipped` and left alone that round; one bad file never stops a poll.
Unchanged local files are not re-read: their hash is reused while their size,
mtime, ctime and inode all match the state. ctime changes on every content or
metadata write and cannot be set back by `utime`, so edits that keep size and
mtime (`cp -p`, `rsync -t`, `os.utime`) are still seen. Files whose mtime or
ctime is in the last couple of seconds are always re-read, so same-tick edits
are not missed.

Local writes are atomic (temp file in the same directory, then rename). The
state file holds copies of project files, so it is written with mode 0600.

Several engines, one state file
-------------------------------
The CLI's sync loop keeps one long-lived engine while `nbgpu run` / `nbgpu
sync` build their own on the same state file. Every operation (initial_push,
poll_once, final_pull) therefore holds an exclusive `fcntl.flock` on
`<state_path>.lock` for its whole duration, reloads the state from disk when
it starts and saves it before releasing the lock. The in-memory state is only
a cache within one operation, so a conflict copy created by another engine is
always known (never pushed as a project file, never copied again) and no
engine overwrites the state with a stale view. Waiting for the lock gives up
after LOCK_TIMEOUT_S with SyncLockTimeout.

Synced manifest
---------------
With `manifest_path` (an absolute remote path outside the project), the
engine writes, after a successful operation, JSON
`{"version": 1, "written_at": <unix seconds>, "files": {rel: sha256}}` with an
unconditional `write_file`. The idle watchdog compares the runtime's project
files with it: a remote file whose hash is not the one listed has edits the
laptop has not pulled yet. Per path, the hash listed is:

* the agreed base, for synced paths;
* for a path with an open conflict, the remote hash seen by the last poll
  (else the hash last written to the conflict copy, else the base): the
  remote version has been shown to the user in the copy, so an open conflict
  is not mistaken for "unpulled";
* for remote files that are never pulled (not `pull_allowed`, e.g. notebook
  outputs), the hash they had when rejected: there is nothing to pull, and
  listing them keeps them from looking like pending edits forever;
* for remote files that .gitignore or [tool.nbgpu] exclude leave out of
  sync, the hash the last poll saw, for the same reason (the watchdog does
  not know the ignore rules).

Paths skipped in a round keep their old base, so they correctly look
unpulled. The manifest is rewritten only when its file list changes or at
least every MANIFEST_REFRESH_S, so a quiet poll does no remote write. A
failure to write it never fails the operation; it is reported in
`SyncReport.warnings` and retried on the next operation.

With `defer_manifest=True` (the long-lived engine of the sync loop) a due
manifest is not written by itself: it is queued and sent with the next
poll's `hash_files(write_first=...)`, saving a round trip on every change.
It then lags by one tick, which is safe: a stale manifest can only list
versions the laptop really had, so the watchdog may think files are
unsynced (and wait) but never that unsynced files are synced.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from nbgpu.backends.base import ABSENT, EXCLUDED_PARTS, TEMP_PREFIX, Backend, RuntimeHandle
from nbgpu.ignore import IgnoreRules

PULL_SUFFIXES = frozenset({".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini"})
PULL_MAX_BYTES = 1024 * 1024
PUSH_MAX_BYTES = 50 * 1024 * 1024  # larger local files are skipped with a warning
# The initial push sends files in archives of about this size (a single
# file larger than this goes in an archive of its own).
BULK_PUSH_BATCH_BYTES = 64 * 1024 * 1024

CONFLICT_MARKER = ".remote"
MAX_CONFLICT_COPIES = 100  # notebook.remote.py, notebook.remote-2.py, ... -100
# A cached local hash is trusted only for files whose mtime and ctime are
# older than this, so an edit within the filesystem's timestamp granularity is
# not missed.
RECENT_MTIME_NS = 2_000_000_000
# 3: local_index entries are (size, mtime_ns, ctime_ns, ino, hash); manifest.
STATE_VERSION = 3
STATE_FILE_MODE = 0o600
LOCK_TIMEOUT_S = 60.0  # waiting for another engine's operation on the same state
LOCK_POLL_S = 0.05
MANIFEST_VERSION = 1
MANIFEST_REFRESH_S = 60.0


# Clocks, as module functions so tests can replace them.
def _now() -> float:
    return time.time()


def _now_ns() -> int:
    return time.time_ns()


class SyncLockTimeout(RuntimeError):
    """Another sync operation held the state file's lock for too long."""


class Action(Enum):
    NOOP = "noop"
    PUSH = "push"
    PULL = "pull"
    MERGE = "merge"
    CONFLICT = "conflict"  # only produced by the engine after a failed merge


def decide(base: str | None, local: str | None, remote: str | None) -> Action:
    """The table above, on hashes. Never returns CONFLICT: a both-changed case
    is MERGE, and the engine downgrades it to CONFLICT if the merge fails.
    For (None, None, r) this returns PULL; the engine applies `pull_allowed`."""
    if local == remote:
        return Action.NOOP  # includes "absent everywhere"
    if base is not None and (local is None or remote is None):
        return Action.NOOP  # a deletion on either side is never propagated
    if local is None:
        return Action.PULL  # new remote file
    if remote is None:
        return Action.PUSH  # new local file
    if local == base:
        return Action.PULL
    if remote == base:
        return Action.PUSH
    return Action.MERGE  # both changed, or both new with different content


def merge3(base: bytes, local: bytes, remote: bytes) -> bytes | None:
    """Three-way merge of text with `git merge-file -p`. Returns the merged
    bytes, or None if there are conflicting hunks. Identical changes on both
    sides merge cleanly. Raises RuntimeError if git is not installed."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is not installed; it is needed for three-way merges")

    with tempfile.TemporaryDirectory(prefix="nbgpu-merge-") as tmp:
        tmp_dir = Path(tmp)
        local_file, base_file, remote_file = tmp_dir / "local", tmp_dir / "base", tmp_dir / "remote"
        local_file.write_bytes(local)
        base_file.write_bytes(base)
        remote_file.write_bytes(remote)
        try:
            result = subprocess.run(
                [git, "merge-file", "-p", str(local_file), str(base_file), str(remote_file)],
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError(f"could not run git merge-file: {exc}") from exc

    # 0: clean. 1..127: number of conflicting hunks (git caps it at 127).
    # Anything else (git reports errors as -1, i.e. 255, or a signal) is a failure.
    if result.returncode == 0:
        return result.stdout
    if 0 < result.returncode < 128:
        return None
    stderr = result.stderr.decode(errors="replace").strip()
    raise RuntimeError(f"git merge-file failed (exit {result.returncode}): {stderr}")


def conflict_copy_name(path: str, n: int = 1) -> str:
    """"notebooks/train.py" -> "notebooks/train.remote.py";
    "Makefile" -> "Makefile.remote"; with n=2, "notebooks/train.remote-2.py"."""
    p = PurePosixPath(path)
    marker = CONFLICT_MARKER if n == 1 else f"{CONFLICT_MARKER}-{n}"
    return str(p.with_name(f"{p.stem}{marker}{p.suffix}"))


def _original_of_conflict_copy(path: str) -> str | None:
    """The path whose first conflict copy name `path` is, or None."""
    p = PurePosixPath(path)
    candidates: list[str] = []
    if p.suffix == CONFLICT_MARKER:  # "Makefile.remote" <- "Makefile"
        candidates.append(p.stem)
    if p.stem.endswith(CONFLICT_MARKER):  # "train.remote.py" <- "train.py"
        candidates.append(p.stem[: -len(CONFLICT_MARKER)] + p.suffix)
    for name in candidates:
        if not name:
            continue
        original = str(p.with_name(name))
        if conflict_copy_name(original) == path:
            return original
    return None


def is_conflict_copy(path: str) -> bool:
    """Whether `path` is shaped like a conflict copy name. Only a naming
    helper: the engine treats a file as a copy only if it created it."""
    return _original_of_conflict_copy(path) is not None


def is_excluded(path: str) -> bool:
    """True if any part of the POSIX relative path is in EXCLUDED_PARTS."""
    return any(part in EXCLUDED_PARTS for part in PurePosixPath(path).parts)


def pull_allowed(path: str, size: int) -> bool:
    return PurePosixPath(path).suffix in PULL_SUFFIXES and size <= PULL_MAX_BYTES


def local_target(root: Path, rel: str) -> Path | None:
    """Where `rel` lives under `root`, or None if writing there is unsafe:
    the path is not a plain relative path, a parent is a symlink or not a
    directory, or the target exists and is not a regular file (a symlink
    included). Nothing outside the root is ever reachable."""
    if not is_safe_relpath(rel):
        return None
    parts = rel.split("/")
    path = root
    for i, part in enumerate(parts):
        path = path / part
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return root.joinpath(*parts)  # the rest does not exist yet
        if stat.S_ISLNK(st.st_mode):
            return None
        is_last = i == len(parts) - 1
        if not is_last and not stat.S_ISDIR(st.st_mode):
            return None
        if is_last and not stat.S_ISREG(st.st_mode):
            return None
    return path


def write_local_file(target: Path, data: bytes) -> None:
    """Atomically create or replace `target` (see _atomic_write)."""
    _atomic_write(target, data)


def is_safe_relpath(path: str) -> bool:
    """A plain relative POSIX path: not empty or absolute, no "", "." or ".."
    parts, no NUL. Remote paths failing this are never written locally."""
    if not path or "\0" in path:
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def _syncable(path: str) -> bool:
    """Whether a relative path takes part in sync at all, on either side
    (engine-created conflict copies are filtered separately)."""
    return not is_excluded(path) and not PurePosixPath(path).name.startswith(TEMP_PREFIX)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _looks_binary(data: bytes) -> bool:
    # git merge-file refuses files containing NUL bytes (exit 255).
    return b"\0" in data


@dataclass
class SyncReport:
    pushed: list[str] = field(default_factory=list)
    pulled: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)  # paths with an open conflict
    # Too large, not pull_allowed, unreadable, unsafe remote path, or changed
    # during the poll (retried on the next one).
    skipped: list[str] = field(default_factory=list)
    # Problems that did not fail the operation (e.g. the manifest write).
    warnings: list[str] = field(default_factory=list)
    # Set by lifecycle.stop when it saved a filesystem snapshot: its record.
    snapshot: dict | None = None

    @property
    def changed(self) -> bool:
        return bool(self.pushed or self.pulled or self.merged)

    def skip(self, rel: str) -> None:
        if rel not in self.skipped:
            self.skipped.append(rel)


# ---------------------------------------------------------------------------
# Local filesystem helpers
# ---------------------------------------------------------------------------


def _write_temp(directory: Path, name: str, data: bytes, mode: int) -> str:
    """Write `data` to a new fsynced temp file in `directory`; return its path."""
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f"{TEMP_PREFIX}{name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, mode)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
    return tmp_name


def _atomic_write(target: Path, data: bytes, *, mode: int | None = None) -> None:
    """Write `data` to `target` via a temp file in the same directory and
    os.replace, so readers (and marimo's file watcher) never see a partial
    file. Keeps the mode of an existing target unless `mode` is given."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode is None:
        try:
            mode = target.stat().st_mode & 0o7777
        except FileNotFoundError:
            mode = 0o644
    tmp_name = _write_temp(target.parent, target.name, data, mode)
    try:
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _create_new_file(target: Path, data: bytes) -> bool:
    """Create `target` with `data` only if nothing exists at that name.
    Atomic and never overwrites: the content is written to a temp file and
    hard-linked into place. Returns False if the name is taken."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = _write_temp(target.parent, target.name, data, 0o644)
    try:
        os.link(tmp_name, target)
        return True
    except FileExistsError:
        return False
    except OSError:
        # Filesystems without hard links: exclusive create (still never
        # overwrites, but a reader could see a partial file).
        try:
            with open(target, "xb") as f:
                f.write(data)
        except FileExistsError:
            return False
        return True
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


@dataclass(frozen=True)
class _LocalFile:
    hash: str
    size: int


@dataclass
class _LocalScan:
    files: dict[str, _LocalFile] = field(default_factory=dict)
    # Present locally but not hashed (too large, unreadable): left alone this
    # round on both sides, since we cannot know what we would overwrite.
    unknown: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Base:
    """The last version both sides agreed on. `content` is None when the file
    was larger than PULL_MAX_BYTES when recorded (it then merges against an
    empty base)."""

    hash: str
    content: bytes | None


@dataclass
class _Conflict:
    """An open conflict: the copy the engine created (relative path) and the
    hash of the remote content last written to it (None if unknown)."""

    copy: str
    copy_hash: str | None


@dataclass
class _State:
    bases: dict[str, _Base] = field(default_factory=dict)
    conflicts: dict[str, _Conflict] = field(default_factory=dict)
    # Remote files not allowed to be pulled, by the hash they had when
    # checked, so an unchanged one is not read again on every poll.
    rejected_remote: dict[str, str] = field(default_factory=dict)
    # Remote files the ignore rules leave out, by hash, as of the last poll;
    # only listed in the manifest.
    ignored_remote: dict[str, str] = field(default_factory=dict)
    # rel -> (size, mtime_ns, ctime_ns, ino, hash) of the local file, to skip
    # re-hashing.
    local_index: dict[str, tuple[int, int, int, int, str]] = field(default_factory=dict)
    # The last manifest written: its remote path, the digest of its file
    # list, and when (unix seconds).
    manifest_path: str | None = None
    manifest_digest: str | None = None
    manifest_written_at: float | None = None

    def copy_paths(self) -> set[str]:
        return {c.copy for c in self.conflicts.values()}

    @classmethod
    def load(cls, path: Path) -> _State | None:
        """The saved state, or None if missing or unreadable. Starting empty
        is safe: with no base, differing sides conflict instead of being
        overwritten."""
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
            bases = {
                rel: _Base(
                    hash=entry["hash"],
                    content=None if entry.get("content") is None else base64.b64decode(entry["content"]),
                )
                for rel, entry in raw["bases"].items()
            }
            conflicts_raw = raw.get("conflicts", {})
            if isinstance(conflicts_raw, list):  # version 1: a list of paths
                conflicts = {rel: _Conflict(conflict_copy_name(rel), None) for rel in conflicts_raw}
            else:
                conflicts = {
                    rel: _Conflict(str(c["copy"]), c.get("copy_hash"))
                    for rel, c in conflicts_raw.items()
                }
            rejected = {str(k): str(v) for k, v in raw.get("rejected_remote", {}).items()}
            ignored = {str(k): str(v) for k, v in raw.get("ignored_remote", {}).items()}
            # Version 2 entries (size, mtime, hash) lack ctime and inode:
            # dropped, so those files are simply re-hashed once.
            index = {
                str(rel): (int(e[0]), int(e[1]), int(e[2]), int(e[3]), str(e[4]))
                for rel, e in raw.get("local_index", {}).items()
                if isinstance(e, list) and len(e) == 5
            }
            manifest = raw.get("manifest") or {}
            written_at = manifest.get("written_at")
            state = cls(
                bases=bases,
                conflicts=conflicts,
                rejected_remote=rejected,
                ignored_remote=ignored,
                local_index=index,
                manifest_path=manifest.get("path"),
                manifest_digest=manifest.get("digest"),
                manifest_written_at=None if written_at is None else float(written_at),
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            return None
        return state

    def to_json(self) -> bytes:
        raw = {
            "version": STATE_VERSION,
            "bases": {
                rel: {
                    "hash": b.hash,
                    "content": None if b.content is None else base64.b64encode(b.content).decode("ascii"),
                }
                for rel, b in sorted(self.bases.items())
            },
            "conflicts": {
                rel: {"copy": c.copy, "copy_hash": c.copy_hash} for rel, c in sorted(self.conflicts.items())
            },
            "rejected_remote": dict(sorted(self.rejected_remote.items())),
            "ignored_remote": dict(sorted(self.ignored_remote.items())),
            "local_index": {rel: list(entry) for rel, entry in sorted(self.local_index.items())},
            "manifest": {
                "path": self.manifest_path,
                "digest": self.manifest_digest,
                "written_at": self.manifest_written_at,
            },
        }
        return json.dumps(raw, indent=1).encode("utf-8")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SyncEngine:
    """Syncs `local_root` with `handle.remote_root`.

    The state (bases, open conflicts and their copies, rejected remote files,
    local stat cache) is persisted as JSON at `state_path`, so an engine
    rebuilt on every tick resumes without re-reading or re-deciding anything.
    Base content is kept so three-way merges have a real base; it is stored
    only for files under PULL_MAX_BYTES (larger files merge with an empty
    base, i.e. they conflict if both sides changed).

    Every operation locks `<state_path>.lock` and reloads the state (see
    "Several engines, one state file"). With `manifest_path`, the synced
    manifest is written there after each operation when due (see "Synced
    manifest").
    """

    def __init__(
        self,
        local_root: Path,
        backend: Backend,
        handle: RuntimeHandle,
        state_path: Path,
        *,
        manifest_path: str | None = None,
        exclude: Sequence[str] = (),
        defer_manifest: bool = False,
    ) -> None:
        if manifest_path is not None:
            remote_root = handle.remote_root.rstrip("/") + "/"
            if not manifest_path.startswith("/") or (manifest_path.rstrip("/") + "/").startswith(remote_root):
                raise ValueError(
                    f"manifest_path must be an absolute remote path outside {handle.remote_root}: {manifest_path!r}"
                )
        self._root = Path(local_root)
        self._ignore = IgnoreRules(self._root, exclude)
        self._backend = backend
        self._handle = handle
        self._state_path = Path(state_path)
        self._lock_path = self._state_path.with_name(self._state_path.name + ".lock")
        self._manifest_path = manifest_path
        self._defer_manifest = defer_manifest
        # A due manifest waiting for the next hash_files: (payload, digest, written_at).
        self._pending_manifest: tuple[bytes, str, float] | None = None
        # A cache, valid only within one locked operation (reloaded by each).
        self._state = _State()
        self._saved: bytes | None = None
        # Paths in use this poll on either side; conflict copies avoid them.
        self._in_use: set[str] = set()
        # Remote hashes seen by the current poll (for the manifest).
        self._remote_seen: dict[str, str] = {}

    # -- public API ---------------------------------------------------------

    def initial_push(self) -> SyncReport:
        """Push every non-excluded local file (up to PUSH_MAX_BYTES) and record
        bases. Used once right after the runtime is created, when the remote
        project directory is empty; a remote file that already exists is not
        overwritten (it is skipped and handled by the next poll)."""
        return self._operation(self._initial_push, fresh_state=True)

    def poll_once(self) -> SyncReport:
        """Hash both sides once and apply `decide` to every path seen on
        either side or in the state."""
        return self._operation(self._poll)

    def final_pull(self) -> SyncReport:
        """A poll_once, used before stop or detach. Same rules: nothing is
        overwritten, conflicts are reported."""
        return self.poll_once()

    def flush_manifest(self) -> SyncReport | None:
        """Write a queued manifest now (when the loop stops, so the watchdog
        does not keep a stale one). None if nothing was queued."""
        if self._pending_manifest is None:
            return None
        self._pending_manifest = None
        deferred, self._defer_manifest = self._defer_manifest, False
        try:
            return self._operation(lambda report: None)
        finally:
            self._defer_manifest = deferred

    @property
    def conflicts(self) -> list[str]:
        """Paths with an open conflict, sorted, as last saved by any engine
        on this state file."""
        loaded = _State.load(self._state_path)
        return sorted((loaded or self._state).conflicts)

    # -- operations ---------------------------------------------------------

    def _operation(self, body: Callable[[SyncReport], None], *, fresh_state: bool = False) -> SyncReport:
        """Run `body` holding the state lock, on state freshly loaded from
        disk; save before releasing (even if `body` raised: everything
        recorded so far is true). Then write the manifest if it is due."""
        report = SyncReport()
        with self._locked():
            self._reload()
            if fresh_state:
                self._state = _State()
            self._remote_seen = {}
            try:
                body(report)
            finally:
                self._save()
            if self._manifest_path is not None:
                self._maybe_write_manifest(report)
                self._save()
        return report

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive flock on `<state_path>.lock`, waiting up to
        LOCK_TIMEOUT_S. flock locks belong to the open file, so two engines
        in one process exclude each other too."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, STATE_FILE_MODE)
        try:
            deadline = time.monotonic() + LOCK_TIMEOUT_S
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise SyncLockTimeout(
                            f"another nbgpu sync held {self._lock_path} for over {LOCK_TIMEOUT_S:g}s; "
                            "if no other nbgpu process is running, retry"
                        ) from None
                    time.sleep(LOCK_POLL_S)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _reload(self) -> None:
        loaded = _State.load(self._state_path)
        self._state = loaded or _State()
        self._saved = self._state.to_json() if loaded is not None else None

    def _initial_push(self, report: SyncReport) -> None:
        scan = self._scan_local(report)
        bulk = getattr(self._backend, "write_new_files", None)
        if bulk is None:
            for rel, info in sorted(scan.files.items()):
                try:
                    self._push(rel, info, None, report)
                except OSError:
                    report.skip(rel)
            return
        # One upload per batch instead of one round trip per file. Same rule
        # as _push: create-only, bases recorded from the bytes sent.
        batch: dict[str, bytes] = {}
        size = 0
        for rel, info in sorted(scan.files.items()):
            try:
                data = self._read_local_verified(rel, info.hash)
            except OSError:
                data = None
            if data is None:
                report.skip(rel)
                continue
            batch[rel] = data
            size += len(data)
            if size >= BULK_PUSH_BATCH_BYTES:
                self._push_batch(bulk, batch, report)
                batch, size = {}, 0
        if batch:
            self._push_batch(bulk, batch, report)

    def _push_batch(self, bulk: Callable[..., set[str]], batch: dict[str, bytes], report: SyncReport) -> None:
        written = bulk(self._handle, batch)
        for rel, data in batch.items():
            if rel in written:
                self._record_base(rel, data)
                report.pushed.append(rel)
            else:
                report.skip(rel)

    def _poll(self, report: SyncReport) -> None:
        scan = self._scan_local(report)
        remote = self._scan_remote(report)
        self._remote_seen = dict(remote)
        self._in_use = set(scan.files) | scan.unknown | set(remote) | set(self._state.bases)
        paths = set(scan.files) | set(remote) | set(self._state.bases) | set(self._state.conflicts)
        for rel in sorted(paths):
            if rel in scan.unknown:
                if rel in self._state.conflicts:
                    report.conflicts.append(rel)
                continue
            try:
                self._sync_path(rel, scan.files.get(rel), remote.get(rel), report)
            except OSError:
                # A local filesystem problem with this one path (permissions,
                # a directory in the way, ...): skip it, keep syncing the rest.
                report.skip(rel)
                if rel in self._state.conflicts and rel not in report.conflicts:
                    report.conflicts.append(rel)

    # -- manifest -------------------------------------------------------------

    def _manifest_files(self) -> dict[str, str]:
        """rel -> hash for the manifest (see "Synced manifest")."""
        files = dict(self._state.ignored_remote)
        files.update(self._state.rejected_remote)
        files.update({rel: b.hash for rel, b in self._state.bases.items()})
        for rel, conflict in self._state.conflicts.items():
            base = self._state.bases.get(rel)
            seen = self._remote_seen.get(rel) or conflict.copy_hash or (base.hash if base else None)
            if seen is None:
                files.pop(rel, None)
            else:
                files[rel] = seen
        return dict(sorted(files.items()))

    def _maybe_write_manifest(self, report: SyncReport) -> None:
        """Write the manifest if its file list changed (or it went to another
        path) or MANIFEST_REFRESH_S passed since the last write. Never raises:
        a failure is a warning, and the next operation tries again."""
        assert self._manifest_path is not None
        files = self._manifest_files()
        digest = _sha256(json.dumps(files, sort_keys=True).encode("utf-8"))
        now = _now()
        state = self._state
        last = state.manifest_written_at
        due = (
            state.manifest_path != self._manifest_path
            or state.manifest_digest != digest
            or last is None
            or now - last >= MANIFEST_REFRESH_S
            or now < last  # the clock went back
        )
        if not due:
            return
        payload = json.dumps({"version": MANIFEST_VERSION, "written_at": now, "files": files}, indent=1)
        if self._defer_manifest:
            self._pending_manifest = (payload.encode("utf-8"), digest, now)
            return
        try:
            ok = self._backend.write_file(self._handle, self._manifest_path, payload.encode("utf-8"), expect=None)
        except Exception as exc:  # noqa: BLE001 - the manifest must never fail a sync
            report.warnings.append(f"could not write sync manifest {self._manifest_path}: {exc}")
            return
        if ok is False:
            report.warnings.append(f"could not write sync manifest {self._manifest_path}: write refused")
            return
        state.manifest_path, state.manifest_digest, state.manifest_written_at = self._manifest_path, digest, now

    # -- scanning -------------------------------------------------------------

    def _scan_local(self, report: SyncReport) -> _LocalScan:
        """Hash every syncable regular file under the root, reusing the cached
        hash when size, mtime, ctime and inode are unchanged. Symlinked
        directories are not descended into. Too-large and unreadable files go
        to `unknown`."""
        scan = _LocalScan()
        copies = self._state.copy_paths()
        old_index = self._state.local_index
        new_index: dict[str, tuple[int, int, int, int, str]] = {}
        recent = _now_ns() - RECENT_MTIME_NS
        ignore = self._ignore
        ignore.refresh()
        for dirpath, dirnames, filenames in os.walk(self._root):
            dir_rel = Path(dirpath).relative_to(self._root).as_posix()
            dir_rel = "" if dir_rel == "." else dir_rel
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in EXCLUDED_PARTS and not ignore.is_ignored(f"{dir_rel}/{d}" if dir_rel else d, True)
            )
            for name in sorted(filenames):
                abs_path = Path(dirpath) / name
                rel = abs_path.relative_to(self._root).as_posix()
                if not _syncable(rel) or rel in copies or ignore.is_ignored(rel):
                    continue
                try:
                    st = abs_path.stat()
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    if st.st_size > PUSH_MAX_BYTES:  # never read just to hash it
                        scan.unknown.add(rel)
                        report.skip(rel)
                        continue
                    key = (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
                    cached = old_index.get(rel)
                    if cached is not None and cached[:4] == key:
                        digest, size = cached[4], st.st_size
                    else:
                        data = abs_path.read_bytes()
                        digest, size = _sha256(data), len(data)
                except FileNotFoundError:  # deleted while scanning
                    continue
                except OSError:  # unreadable: leave it alone, keep going
                    scan.unknown.add(rel)
                    report.skip(rel)
                    continue
                scan.files[rel] = _LocalFile(hash=digest, size=size)
                # Cache only settled files whose content matched the stat.
                if size == key[0] and max(key[1], key[2]) < recent:
                    new_index[rel] = (*key, digest)
        self._state.local_index = new_index
        return scan

    def _scan_remote(self, report: SyncReport) -> dict[str, str]:
        """Remote syncable files -> hash. Unsafe paths are reported as skipped."""
        copies = self._state.copy_paths()
        remote: dict[str, str] = {}
        pending = self._pending_manifest if self._manifest_path is not None else None
        if pending is not None:
            hashed = self._backend.hash_files(self._handle, write_first=(self._manifest_path, pending[0]))
            self._pending_manifest = None
            state = self._state
            state.manifest_path, state.manifest_digest, state.manifest_written_at = (
                self._manifest_path, pending[1], pending[2])
        else:
            hashed = self._backend.hash_files(self._handle)
        ignored: dict[str, str] = {}
        for rel, digest in hashed.items():
            if not is_safe_relpath(rel):
                report.skip(rel)
            elif not _syncable(rel) or rel in copies:
                continue
            elif self._ignore.is_ignored(rel):
                ignored[rel] = digest
            else:
                remote[rel] = digest
        self._state.ignored_remote = ignored
        return remote

    # -- per-path logic -----------------------------------------------------

    def _sync_path(
        self,
        rel: str,
        local: _LocalFile | None,
        remote_hash: str | None,
        report: SyncReport,
    ) -> None:
        if rel in self._state.conflicts:
            self._sync_conflicted(rel, local, remote_hash, report)
            return

        local_hash = local.hash if local is not None else None
        base = self._state.bases.get(rel)
        action = decide(base.hash if base else None, local_hash, remote_hash)

        if action is Action.NOOP:
            if local is not None and local_hash == remote_hash and (base is None or base.hash != local_hash):
                # Both sides agreed on these bytes. If the local file moved on
                # mid-poll, the remote copy still has them.
                data = self._read_local_verified(rel, local.hash)
                if data is None:
                    data = self._read_remote_verified(rel, local.hash)
                if data is None:
                    report.skip(rel)
                else:
                    self._record_base(rel, data)
        elif action is Action.PUSH:
            assert local is not None
            self._push(rel, local, remote_hash, report)
        elif action is Action.PULL:
            assert remote_hash is not None
            self._pull(rel, local_hash, remote_hash, is_new=base is None, report=report)
        elif action is Action.MERGE:
            assert local is not None and remote_hash is not None
            self._merge(rel, base, local, remote_hash, report)

    def _sync_conflicted(
        self,
        rel: str,
        local: _LocalFile | None,
        remote_hash: str | None,
        report: SyncReport,
    ) -> None:
        conflict = self._state.conflicts[rel]
        if os.path.lexists(self._root.joinpath(*conflict.copy.split("/"))):
            self._refresh_conflict_copy(rel, conflict, remote_hash)
            report.conflicts.append(rel)  # still open: the original is not touched
            return
        self._resolve_conflict(rel, conflict, local, remote_hash, report)

    def _refresh_conflict_copy(self, rel: str, conflict: _Conflict, remote_hash: str | None) -> None:
        """Keep the copy showing the current remote version, so remote edits
        made while the conflict is open are visible. Uses the local
        compare-and-swap: a copy the user edited is left alone (resolving
        then re-raises the conflict with a fresh copy, so nothing is lost)."""
        if remote_hash is None or conflict.copy_hash is None or remote_hash == conflict.copy_hash:
            return
        data = self._read_remote_verified(rel, remote_hash)
        if data is not None and self._write_local_if_unchanged(conflict.copy, data, conflict.copy_hash):
            conflict.copy_hash = remote_hash

    def _resolve_conflict(
        self,
        rel: str,
        conflict: _Conflict,
        local: _LocalFile | None,
        remote_hash: str | None,
        report: SyncReport,
    ) -> None:
        """The user deleted the conflict copy: the local file wins, unless the
        remote changed after the copy was last written."""
        if local is None or remote_hash is None or local.hash == remote_hash:
            # Nothing to win with, nothing to overwrite, or already agreed:
            # back to the table (deletions are never propagated).
            del self._state.conflicts[rel]
            self._sync_path(rel, local, remote_hash, report)
            return

        if remote_hash != conflict.copy_hash:
            # The user resolved against content the remote no longer has.
            self._raise_conflict_again(rel, conflict, remote_hash, report)
            return

        data = self._read_local_verified(rel, local.hash)
        if data is None or not self._write_remote_if_unchanged(rel, data, remote_hash):
            # Changed during this poll. The conflict stays open with its copy
            # gone, so the next poll re-decides (and re-raises if the remote moved).
            report.skip(rel)
            report.conflicts.append(rel)
            return
        del self._state.conflicts[rel]
        self._record_base(rel, data)
        report.pushed.append(rel)

    def _raise_conflict_again(
        self, rel: str, conflict: _Conflict, remote_hash: str, report: SyncReport
    ) -> None:
        report.conflicts.append(rel)
        data = self._read_remote_verified(rel, remote_hash)
        copy = self._create_conflict_copy(rel, data) if data is not None else None
        if copy is None:
            report.skip(rel)
            return
        conflict.copy, conflict.copy_hash = copy, remote_hash

    # -- actions -------------------------------------------------------------

    def _push(self, rel: str, local: _LocalFile, remote_hash: str | None, report: SyncReport) -> None:
        data = self._read_local_verified(rel, local.hash)
        if data is None or not self._write_remote_if_unchanged(rel, data, remote_hash):
            report.skip(rel)
            return
        self._record_base(rel, data)
        report.pushed.append(rel)

    def _pull(
        self,
        rel: str,
        local_hash: str | None,
        remote_hash: str,
        *,
        is_new: bool,
        report: SyncReport,
    ) -> None:
        if is_new:
            if self._state.rejected_remote.get(rel) == remote_hash:
                report.skip(rel)
                return
            # Check the suffix before reading so outputs are never downloaded.
            if not pull_allowed(rel, 0):
                self._reject_remote(rel, remote_hash, report)
                return
        data = self._read_remote_verified(rel, remote_hash)
        if data is None:
            report.skip(rel)
            return
        if is_new and not pull_allowed(rel, len(data)):
            self._reject_remote(rel, remote_hash, report)
            return
        if not self._write_local_if_unchanged(rel, data, local_hash):
            report.skip(rel)
            return
        self._record_base(rel, data)
        report.pulled.append(rel)

    def _merge(
        self, rel: str, base: _Base | None, local: _LocalFile, remote_hash: str, report: SyncReport
    ) -> None:
        local_data = self._read_local_verified(rel, local.hash)
        remote_data = self._read_remote_verified(rel, remote_hash)
        if local_data is None or remote_data is None:
            report.skip(rel)
            return
        base_data = base.content if base is not None and base.content is not None else b""

        merged: bytes | None = None
        if not any(_looks_binary(d) for d in (base_data, local_data, remote_data)):
            merged = merge3(base_data, local_data, remote_data)

        if merged is None:
            copy = self._create_conflict_copy(rel, remote_data)
            if copy is None:
                report.skip(rel)
                return
            self._state.conflicts[rel] = _Conflict(copy=copy, copy_hash=remote_hash)
            report.conflicts.append(rel)
            return

        # Remote first (atomic compare-and-swap), then local. If either side
        # moved, no base is recorded: the next poll merges again from the old
        # base, and identical changes on both sides merge cleanly.
        if merged != remote_data and not self._write_remote_if_unchanged(rel, merged, remote_hash):
            report.skip(rel)
            return
        if merged != local_data and not self._write_local_if_unchanged(rel, merged, local.hash):
            report.skip(rel)
            return
        self._record_base(rel, merged)
        report.merged.append(rel)

    # -- compare-and-swap helpers -------------------------------------------

    def _write_remote_if_unchanged(self, rel: str, data: bytes, expected_hash: str | None) -> bool:
        """Write `rel` on the runtime only if it still has `expected_hash`
        (None: only if it does not exist). Atomic inside the runtime."""
        expect = ABSENT if expected_hash is None else expected_hash
        return self._backend.write_file(self._handle, rel, data, expect=expect)

    def _write_local_if_unchanged(self, rel: str, data: bytes, expected_hash: str | None) -> bool:
        """Replace the local `rel` with `data` only if it still has
        `expected_hash` (None: only if it does not exist, which is atomic).
        Refuses unsafe targets (outside the root, through a symlink)."""
        target = self._local_target(rel)
        if target is None:
            return False
        if expected_hash is None:
            return _create_new_file(target, data)
        if self._local_hash_now(target) != expected_hash:
            self._state.local_index.pop(rel, None)
            return False
        _atomic_write(target, data)  # residual window: see the module docstring
        return True

    def _read_local_verified(self, rel: str, expected_hash: str) -> bytes | None:
        """The local file's bytes if they still hash to `expected_hash`, else
        None. Bases are recorded from these bytes, never from a fresh read."""
        try:
            data = self._local_path(rel).read_bytes()
        except FileNotFoundError:
            return None
        if _sha256(data) != expected_hash:
            self._state.local_index.pop(rel, None)
            return None
        return data

    def _read_remote_verified(self, rel: str, expected_hash: str) -> bytes | None:
        """The remote file's bytes if they still hash to `expected_hash`."""
        data = self._backend.read_file(self._handle, rel)
        if data is None or _sha256(data) != expected_hash:
            return None
        return data

    @staticmethod
    def _local_hash_now(target: Path) -> str | None:
        try:
            return _sha256(target.read_bytes())
        except FileNotFoundError:
            return None

    # -- helpers -------------------------------------------------------------

    def _local_target(self, rel: str) -> Path | None:
        return local_target(self._root, rel)

    def _create_conflict_copy(self, rel: str, data: bytes) -> str | None:
        """Create a new conflict copy of `rel` holding `data` under the first
        free name, never overwriting anything. Returns its relative path."""
        taken = self._in_use | {c.copy for r, c in self._state.conflicts.items() if r != rel}
        for n in range(1, MAX_CONFLICT_COPIES + 1):
            name = conflict_copy_name(rel, n)
            if name in taken:
                continue
            target = self._local_target(name)
            if target is not None and _create_new_file(target, data):
                self._in_use.add(name)
                return name
        return None

    def _reject_remote(self, rel: str, remote_hash: str, report: SyncReport) -> None:
        self._state.rejected_remote[rel] = remote_hash
        report.skip(rel)

    def _record_base(self, rel: str, data: bytes) -> None:
        content = data if len(data) <= PULL_MAX_BYTES else None
        self._state.bases[rel] = _Base(hash=_sha256(data), content=content)

    def _local_path(self, rel: str) -> Path:
        return self._root.joinpath(*PurePosixPath(rel).parts)

    def _save(self) -> None:
        raw = self._state.to_json()
        if raw != self._saved:
            _atomic_write(self._state_path, raw, mode=STATE_FILE_MODE)
            self._saved = raw
