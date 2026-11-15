"""Local records of running sessions.

The provider is the source of truth for what is running (runtimes carry
tags); the local store only remembers what the provider cannot tell us,
mainly the access token and the URL. Records are JSON files in a directory
only the user can read (0700), each file 0600.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Tags put on every runtime so sessions can be found from the provider alone.
TAG_SESSION = "nbgpu.session"
TAG_NOTEBOOK = "nbgpu.notebook"  # see notebook_tag()
TAG_TOOL = "nbgpu"  # value "1"; marks every runtime this tool created
# A stable random id of the state directory that created the runtime (see
# lifecycle.owner_id), so other laptops' / teammates' runtimes in the same
# provider workspace are never taken for this directory's orphans.
TAG_OWNER = "nbgpu.owner"

_DIR_MODE = 0o700
_FILE_MODE = 0o600
_RECORD_SUFFIX = ".json"
# Session ids become file names, so they must not be able to escape the store.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]+")


def default_state_dir() -> Path:
    """$NBGPU_STATE_DIR if set, else $XDG_STATE_HOME/nbgpu, else ~/.local/state/nbgpu."""
    explicit = os.environ.get("NBGPU_STATE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME")
    # The XDG spec says relative values are invalid and must be ignored.
    if xdg and Path(xdg).is_absolute():
        return Path(xdg) / "nbgpu"
    return Path.home() / ".local" / "state" / "nbgpu"


def ensure_private_dir(directory: Path) -> None:
    """Make sure `directory` exists and only the user can use it: created
    0700, and an existing one we own has group/other permissions removed."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(mode=_DIR_MODE)
    except FileExistsError:
        st = directory.stat()
        if st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode) & 0o077:
            directory.chmod(stat.S_IMODE(st.st_mode) & _DIR_MODE)
        return
    directory.chmod(_DIR_MODE)  # mkdir's mode is filtered by the umask


def new_session_id() -> str:
    """8 lowercase hex characters, random."""
    return secrets.token_hex(4)


def new_token() -> str:
    """A URL-safe random token with at least 32 bytes of entropy."""
    return secrets.token_urlsafe(32)


def notebook_tag(notebook: Path) -> str:
    """A stable tag value for an absolute notebook path: the first 16 hex
    characters of its SHA-256. Same path -> same tag; it does not reveal the path."""
    return hashlib.sha256(str(notebook).encode("utf-8")).hexdigest()[:16]


@dataclass
class Session:
    session_id: str
    notebook: Path  # absolute local path
    project_root: Path  # absolute local directory synced to the runtime
    backend: str  # backend name, e.g. "modal" or "local"
    runtime_id: str
    remote_root: str  # project directory on the runtime
    url: str  # URL to open, without the token
    token: str
    created_at: float  # unix seconds
    gpu: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def url_with_token(self) -> str:
        """`url` with `access_token=<token>` added as a query parameter,
        keeping any existing query."""
        parts = urlsplit(self.url)
        params = parse_qsl(parts.query, keep_blank_values=True)
        query = parts.query
        if any(k == "access_token" for k, _ in params):
            # Replace a stale token rather than sending two.
            query = urlencode([(k, v) for k, v in params if k != "access_token"])
        token_param = urlencode({"access_token": self.token})
        query = f"{query}&{token_param}" if query else token_param
        return urlunsplit(parts._replace(query=query))


def _to_record(session: Session) -> dict[str, object]:
    record = asdict(session)
    record["notebook"] = str(session.notebook)
    record["project_root"] = str(session.project_root)
    return record


def _from_record(record: object) -> Session:
    """Build a Session from parsed JSON. Raises ValueError if it is not a
    complete, well-typed record."""
    if not isinstance(record, dict):
        raise ValueError("record is not an object")

    def text(key: str) -> str:
        value = record.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value

    created_at = record.get("created_at")
    if isinstance(created_at, bool) or not isinstance(created_at, int | float):
        raise ValueError("created_at must be a number")
    gpu = record.get("gpu")
    if gpu is not None and not isinstance(gpu, str):
        raise ValueError("gpu must be a string or null")
    extra = record.get("extra", {})
    if not isinstance(extra, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in extra.items()
    ):
        raise ValueError("extra must map strings to strings")

    return Session(
        session_id=text("session_id"),
        notebook=Path(text("notebook")),
        project_root=Path(text("project_root")),
        backend=text("backend"),
        runtime_id=text("runtime_id"),
        remote_root=text("remote_root"),
        url=text("url"),
        token=text("token"),
        created_at=float(created_at),
        gpu=gpu,
        extra=dict(extra),
    )


def _read_record(path: Path) -> Session | None:
    """The session in `path`, or None if it is missing, unreadable or corrupt."""
    try:
        return _from_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, ValueError):  # JSONDecodeError is a ValueError
        return None


class SessionStore:
    directory: Path  # resolved state directory (may not exist yet)

    def __init__(self, directory: Path | None = None) -> None:
        """Use `directory`, or default_state_dir() when None. Created lazily
        with mode 0700 on first write (see ensure_private_dir)."""
        base = default_state_dir() if directory is None else Path(directory)
        self.directory = base.expanduser().resolve()

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(f"invalid session id {session_id!r}")
        return self.directory / f"{session_id}{_RECORD_SUFFIX}"

    def save(self, session: Session) -> None:
        """Write atomically (temp file + rename), file mode 0600."""
        target = self._path(session.session_id)
        ensure_private_dir(self.directory)
        data = json.dumps(_to_record(session), indent=2, sort_keys=True) + "\n"
        # mkstemp creates the file 0600; the leading dot and .tmp suffix keep
        # it out of list() while it is being written.
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix=".", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                os.fchmod(f.fileno(), _FILE_MODE)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def load(self, session_id: str) -> Session | None:
        """The record for `session_id`, or None if there is none. A corrupt or
        unreadable record is treated as absent, as in list()."""
        try:
            path = self._path(session_id)
        except ValueError:
            return None
        return _read_record(path)

    def list(self) -> list[Session]:
        """All records, oldest first. Unreadable or corrupt files are skipped,
        not fatal. Other files in the directory (e.g. `<id>.sync.json`, which
        can be large) are never opened."""
        try:
            paths = [
                p
                for p in self.directory.iterdir()
                if p.suffix == _RECORD_SUFFIX and _SESSION_ID_RE.fullmatch(p.stem)
            ]
        except FileNotFoundError:
            return []
        sessions = [s for p in paths if (s := _read_record(p)) is not None]
        return sorted(sessions, key=lambda s: (s.created_at, s.session_id))

    def find_by_notebook(self, notebook: Path) -> list[Session]:
        """Records whose notebook is the same absolute path, newest first."""
        matches = [s for s in self.list() if s.notebook == notebook]
        return sorted(matches, key=lambda s: (s.created_at, s.session_id), reverse=True)

    def remove(self, session_id: str) -> None:
        """No error if it does not exist."""
        try:
            path = self._path(session_id)
        except ValueError:
            return  # no record can have this id
        path.unlink(missing_ok=True)
