"""Modal Sandboxes as runtimes.

- One Modal App, looked up or created by name (`app_name`, default "nbgpu").
- Image: debian_slim with the notebook's Python version (from
  requires-python when it pins one, else the local minor version), plus
  `uv_pip_install` of the spec's dependencies, marimo, and watchdog.
- `create` -> modal.Sandbox.create with gpu, cpu, memory, timeout
  (max_runtime_s), idle_timeout (IDLE_GRACE_S, see below), volumes looked up by name,
  encrypted_ports for request.ports, tags (passed to Sandbox.create, which
  sets them atomically; set_tags would replace the whole set later), and a
  long-lived entrypoint that keeps the sandbox alive (e.g. `sleep infinity`).
  remote_root is /workspace/project.
- `get` -> Sandbox.from_id, None when it has finished.
- `find` -> Sandbox.list(app_id=..., tags=...).
- `exec`/`spawn` -> sandbox.exec.
- `expose` -> sandbox.tunnels()[port].url.
- File operations use the sandbox filesystem API plus small exec'd Python
  one-liners; writes go to a TEMP_PREFIX temp file and are then checked and
  renamed in one exec (see write_file below).
- GPU strings are validated against a known list before anything is
  created; an unknown one raises UnsupportedError listing the known ones.

Capabilities: gpu, volumes, max-runtime (not idle-timeout; see below).

Implementation notes (checked against modal 1.6.4):

- Tags are passed to Sandbox.create(tags=...) so a sandbox is never visible
  untagged. Two internal tags (prefix "nbgpu.internal.") record the creation
  time and the requested ports; they are hidden from RuntimeHandle.tags.
- Idle shutdown: Modal counts a sandbox as active while any `sandbox.exec`
  command is running, stdin is being written, or a TCP connection over a
  tunnel is open (the `sleep infinity` entrypoint does not count). The marimo
  server is a spawned exec and the sync loop execs every second, so a Modal
  idle_timeout equal to the user's idle-timeout could never fire. Instead,
  the user's idle-timeout is enforced by nbgpu's own watchdog inside the
  runtime (lifecycle), which stops marimo once the user has been idle that
  long; it works the same on every backend, so this backend does not claim
  CAP_IDLE_TIMEOUT. The sandbox's Modal idle_timeout is the short
  IDLE_GRACE_S: once marimo is gone and the laptop has stopped polling,
  nothing is executing and Modal ends the sandbox within that grace period.
  timeout=max_runtime_s stays as the hard backstop.
- Files: read_file/write_file use `sandbox.filesystem` (read_bytes /
  write_bytes, not marked experimental; Modal documents reads up to 5 GB and
  writes of any size; write_bytes creates parent directories). Whole files
  are held in memory on the laptop. write_file uploads to a TEMP_PREFIX temp
  file in the target's directory, then runs CAS_CODE in one exec: it checks
  `expect` (ABSENT via os.link, which fails atomically if the target exists;
  a SHA-256 by hashing the current file) and either os.replaces the temp
  file onto the target or deletes it, printing the outcome. If the upload
  fails, the temp file is removed best-effort; any leftover is ignored by
  sync and hash_files because of its prefix.
- Spawned processes run under a tiny Python launcher that merges stderr
  into stdout, starts a new process group and records its id in a pid file,
  because ContainerProcess has no kill method; terminate() kills that group
  with an exec'd one-liner.
- hash_files keeps a stat cache at /workspace/nbgpu-hash-cache.json (next to
  remote_root, outside the synced tree) so unchanged files are not re-read
  on every poll; see HASH_CODE.
- Every helper exec (hash, CAS write, mkdir, kill) runs with
  NBGPU_INTERNAL=1 so the in-runtime idle watchdog can ignore it; user
  commands passed to exec/spawn never get it added.
- If create fails after the sandbox exists, the sandbox is terminated with
  SIGINT/SIGTERM/SIGHUP ignored and a bounded wait
  (CLEANUP_TERMINATE_TIMEOUT_S); on timeout BackendError names the sandbox.
"""

from __future__ import annotations

import base64
import io
import json
import math
import posixpath
import re
import signal
import sys
import tarfile
import threading
import time
import uuid
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from nbgpu.backends.base import (
    ABSENT,
    CAP_GPU,
    CAP_LIVE_INSTALL,
    CAP_MAX_RUNTIME,
    CAP_VOLUMES,
    TEMP_PREFIX,
    BackendError,
    ExecResult,
    RuntimeHandle,
    RuntimeRequest,
    UnsupportedError,
)
from nbgpu.spec import NotebookSpec

if TYPE_CHECKING:  # pragma: no cover
    import modal

REMOTE_ROOT = "/workspace/project"
ENTRYPOINT = ("sleep", "infinity")
REMOTE_PYTHON = "python"

# GPU type strings accepted by `gpu=` per https://modal.com/docs/guide/gpu
# (the SDK itself does not validate names; the server does). "A10G" is the
# older name of "A10" and is kept because existing notebooks use it.
# Value: max GPUs per container (docs: 8 for most types, 4 for A10).
KNOWN_GPUS: dict[str, int] = {
    "T4": 8,
    "L4": 8,
    "A10": 4,
    "A10G": 4,
    "L40S": 8,
    "A100": 8,
    "A100-40GB": 8,
    "A100-80GB": 8,
    "RTX-PRO-6000": 8,
    "H100": 8,
    "H100!": 8,
    "H200": 8,
    "B200": 8,
    "B200+": 8,
    "B300": 8,
}

# Python series supported by recent Modal image builders (2024.10 .. 2026.10)
# that nbgpu offers. Highest preferred when the local version does not fit.
SUPPORTED_PYTHONS: tuple[str, ...] = ("3.11", "3.12", "3.13", "3.14")

# Modal's documented maximum sandbox lifetime.
MAX_RUNTIME_LIMIT_S = 24 * 60 * 60

# The sandbox's Modal idle_timeout: how long a sandbox with no running exec,
# no stdin and no tunnel connection lives on. The user's idle-timeout is
# enforced by the in-runtime watchdog, not by this (see the module docstring).
IDLE_GRACE_S = 300

# uv installs packages into the running sandbox (nbgpu and marimo use it).
BASE_PACKAGES: tuple[str, ...] = ("marimo", "watchdog", "uv")

INTERNAL_TAG_PREFIX = "nbgpu.internal."
TAG_CREATED_AT = INTERNAL_TAG_PREFIX + "created_at"
TAG_PORTS = INTERNAL_TAG_PREFIX + "ports"

HASH_EXCLUDED_PARTS: tuple[str, ...] = (".git", ".venv", "__pycache__", "node_modules")
HASH_CACHE_NAME = "nbgpu-hash-cache.json"

# Set on every command this backend runs for its own purposes (hashing, the
# CAS write, mkdir, killing spawned groups) so the idle watchdog can ignore
# their CPU use. Never added to user commands passed to exec/spawn.
INTERNAL_ENV_VAR = "NBGPU_INTERNAL"
INTERNAL_ENV: Mapping[str, str] = {INTERNAL_ENV_VAR: "1"}

# Bound on the cleanup terminate in create (see _terminate_uninterruptibly).
CLEANUP_TERMINATE_TIMEOUT_S = 60.0

SPAWN_PID_DIR = "/tmp"
TERMINATE_GRACE_S = 5.0
PIDFILE_WAIT_S = 10.0

# ---------------------------------------------------------------------------
# Remote one-liners. Each is run as [python, "-c", CODE, *args].

# Compare-and-swap: move temp file argv[1] onto argv[2] if precondition
# argv[3] holds, else delete argv[1]. argv[3] is EXPECT_ANY (unconditional),
# EXPECT_ABSENT (only if argv[2] does not exist; os.link fails atomically if
# it does) or a lowercase SHA-256 hex (only if argv[2] hashes to it).
# Prints CAS_WRITTEN or CAS_CONFLICT and exits 0; any other failure exits
# non-zero. argv[1] never survives.
EXPECT_ANY = "*"
EXPECT_ABSENT = "absent"
CAS_WRITTEN = "written"
CAS_CONFLICT = "conflict"
CAS_CODE = (
    "import hashlib, os, sys\n"
    "tmp, dest, expect = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "done = False\n"
    "try:\n"
    "    if expect == '*':\n"
    "        os.replace(tmp, dest)\n"
    "        done = True\n"
    "    elif expect == 'absent':\n"
    "        try:\n"
    "            os.link(tmp, dest)\n"
    "            done = True\n"
    "        except FileExistsError:\n"
    "            pass\n"
    "        except OSError:\n"
    "            if not os.path.lexists(dest):\n"
    "                os.replace(tmp, dest)\n"
    "                done = True\n"
    "    else:\n"
    "        h = hashlib.sha256()\n"
    "        try:\n"
    "            with open(dest, 'rb') as fh:\n"
    "                for chunk in iter(lambda: fh.read(1 << 20), b''):\n"
    "                    h.update(chunk)\n"
    "            current = h.hexdigest()\n"
    "        except FileNotFoundError:\n"
    "            current = None\n"
    "        if current == expect:\n"
    "            os.replace(tmp, dest)\n"
    "            done = True\n"
    "finally:\n"
    "    try:\n"
    "        os.unlink(tmp)\n"
    "    except FileNotFoundError:\n"
    "        pass\n"
    "print('written' if done else 'conflict')\n"
)

# Unpack the tar archive argv[1] into the directory argv[2], create-only:
# each regular file goes to a TEMP_PREFIX (argv[3]) temp file in its target
# directory and is hard-linked into place, so an existing file is never
# overwritten (as CAS_CODE with expect 'absent'). Members with unsafe names,
# or whose directory resolves outside argv[2], are skipped. The archive is
# deleted. Prints the JSON list of relative paths written.
BULK_CODE = (
    "import json, os, shutil, sys, tarfile\n"
    "archive, root, prefix = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "real_root = os.path.realpath(root)\n"
    "written = []\n"
    "try:\n"
    "    with tarfile.open(archive) as tf:\n"
    "        for n, m in enumerate(tf):\n"
    "            rel = m.name\n"
    "            parts = rel.split('/')\n"
    "            if not m.isfile() or not rel or rel.startswith('/') or '\\0' in rel:\n"
    "                continue\n"
    "            if any(p in ('', '.', '..') for p in parts):\n"
    "                continue\n"
    "            dest = os.path.join(root, rel)\n"
    "            d = os.path.dirname(dest)\n"
    "            os.makedirs(d, exist_ok=True)\n"
    "            real_d = os.path.realpath(d)\n"
    "            if real_d != real_root and not real_d.startswith(real_root + os.sep):\n"
    "                continue\n"
    "            tmp = os.path.join(d, prefix + 'bulk-' + str(os.getpid()) + '-' + str(n))\n"
    "            try:\n"
    "                with tf.extractfile(m) as src, open(tmp, 'wb') as out:\n"
    "                    shutil.copyfileobj(src, out)\n"
    "                if m.mode & 0o111:\n"
    "                    os.chmod(tmp, 0o755)\n"
    "                try:\n"
    "                    os.link(tmp, dest)\n"
    "                    written.append(rel)\n"
    "                except FileExistsError:\n"
    "                    pass\n"
    "                except OSError:\n"
    "                    if not os.path.lexists(dest):\n"
    "                        os.replace(tmp, dest)\n"
    "                        written.append(rel)\n"
    "            finally:\n"
    "                try:\n"
    "                    os.unlink(tmp)\n"
    "                except FileNotFoundError:\n"
    "                    pass\n"
    "finally:\n"
    "    try:\n"
    "        os.unlink(archive)\n"
    "    except FileNotFoundError:\n"
    "        pass\n"
    "print(json.dumps(written))\n"
)


def build_tar(files: Mapping[str, bytes]) -> bytes:
    """An uncompressed tar of relpath -> bytes (mode 0644, mtime now)."""
    buf = io.BytesIO()
    now = time.time()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for rel, data in files.items():
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = now
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def parse_bulk_output(stdout: str, sent: Mapping[str, bytes]) -> set[str]:
    try:
        written = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else None
    except json.JSONDecodeError:
        written = None
    if not isinstance(written, list):
        raise BackendError(f"Modal: unexpected unpack output: {stdout[:200]!r}")
    return {w for w in written if isinstance(w, str) and w in sent}


# Print {relpath: sha256} as JSON for every regular file under argv[1],
# skipping files whose name starts with argv[2] (TEMP_PREFIX), the stat cache
# argv[3] itself, and any path with a part in argv[4:]. Symlinked directories
# are not followed. A missing root prints {}.
# Stat cache: argv[3] holds {"root": root, "files": {relpath: [size,
# mtime_ns, ctime_ns, inode, sha256]}}; a file is re-hashed unless all four
# stat fields match and neither timestamp is within the last 2 s (timestamp
# granularity). The cache is rewritten atomically (TEMP_PREFIX temp file +
# os.replace) only when it changed; a missing or corrupt cache just means
# re-hashing, and a failed write is ignored.
HASH_CODE = (
    "import hashlib, json, os, stat, sys, tempfile, time\n"
    "root, tmp_prefix, cache_path = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "skip = set(sys.argv[4:])\n"
    "recent = time.time_ns() - 2000000000\n"
    "try:\n"
    "    with open(cache_path, 'rb') as fh:\n"
    "        c = json.load(fh)\n"
    "    old = c['files'] if c.get('root') == root else {}\n"
    "    if not isinstance(old, dict):\n"
    "        old = {}\n"
    "except Exception:\n"
    "    old = {}\n"
    "cache_abs = os.path.abspath(cache_path)\n"
    "new = {}\n"
    "out = {}\n"
    "if os.path.isdir(root):\n"
    "    for d, dirs, files in os.walk(root):\n"
    "        dirs[:] = [x for x in dirs if x not in skip]\n"
    "        for f in files:\n"
    "            if f in skip or f.startswith(tmp_prefix):\n"
    "                continue\n"
    "            p = os.path.join(d, f)\n"
    "            if os.path.abspath(p) == cache_abs:\n"
    "                continue\n"
    "            rel = os.path.relpath(p, root).replace(os.sep, '/')\n"
    "            try:\n"
    "                st = os.lstat(p)\n"
    "                if not stat.S_ISREG(st.st_mode):\n"
    "                    continue\n"
    "                key = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]\n"
    "                e = old.get(rel)\n"
    "                if (isinstance(e, list) and len(e) == 5 and e[:4] == key and isinstance(e[4], str)\n"
    "                        and max(st.st_mtime_ns, st.st_ctime_ns) < recent):\n"
    "                    digest = e[4]\n"
    "                else:\n"
    "                    h = hashlib.sha256()\n"
    "                    with open(p, 'rb') as fh:\n"
    "                        for chunk in iter(lambda: fh.read(1 << 20), b''):\n"
    "                            h.update(chunk)\n"
    "                    digest = h.hexdigest()\n"
    "            except OSError:\n"
    "                continue\n"
    "            out[rel] = digest\n"
    "            new[rel] = key + [digest]\n"
    "if new != old and os.path.isdir(root):\n"
    "    tmp = None\n"
    "    try:\n"
    "        fd, tmp = tempfile.mkstemp(prefix=tmp_prefix, dir=os.path.dirname(cache_abs))\n"
    "        with os.fdopen(fd, 'w') as fh:\n"
    "            json.dump({'root': root, 'files': new}, fh)\n"
    "        os.replace(tmp, cache_abs)\n"
    "        tmp = None\n"
    "    except OSError:\n"
    "        pass\n"
    "    finally:\n"
    "        if tmp is not None:\n"
    "            try:\n"
    "                os.unlink(tmp)\n"
    "            except OSError:\n"
    "                pass\n"
    "json.dump(out, sys.stdout)\n"
)

# Launcher for spawn: merge stderr into stdout, become a process-group leader,
# record the group id in argv[1] (atomically), then exec argv[2:].
LAUNCH_CODE = (
    "import os, sys\n"
    "os.dup2(1, 2)\n"
    "try:\n"
    "    os.setsid()\n"
    "except OSError:\n"
    "    pass\n"
    "tmp = sys.argv[1] + '.tmp'\n"
    "with open(tmp, 'w') as fh:\n"
    "    fh.write(str(os.getpgrp()))\n"
    "os.replace(tmp, sys.argv[1])\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n"
)

# Kill the process group whose id is in argv[1]: wait up to argv[2] s for the
# file, SIGTERM, then SIGKILL after argv[3] s. Missing file/group is not an error.
KILL_CODE = (
    "import os, signal, sys, time\n"
    "path, wait_s, grace_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])\n"
    "pg = None\n"
    "end = time.time() + wait_s\n"
    "while True:\n"
    "    try:\n"
    "        pg = int(open(path).read())\n"
    "        break\n"
    "    except (OSError, ValueError):\n"
    "        if time.time() >= end:\n"
    "            break\n"
    "        time.sleep(0.1)\n"
    "if pg is not None:\n"
    "    try:\n"
    "        os.killpg(pg, signal.SIGTERM)\n"
    "        end = time.time() + grace_s\n"
    "        while time.time() < end:\n"
    "            os.killpg(pg, 0)\n"
    "            time.sleep(0.1)\n"
    "        os.killpg(pg, signal.SIGKILL)\n"
    "    except (ProcessLookupError, PermissionError):\n"  # gone (EPERM: only zombies left, on some kernels)

    "        pass\n"
    "    try:\n"
    "        os.unlink(path)\n"
    "    except OSError:\n"
    "        pass\n"
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without Modal).


def parse_gpu(gpu: str | None) -> str | None:
    """Validate a GPU string ("A10G", "h100:2") and return Modal's canonical
    form (upper-case name, ":n" only when n > 1). None means no GPU."""
    if gpu is None:
        return None
    known = ", ".join(KNOWN_GPUS)
    raw = gpu.strip()
    name, sep, count_str = raw.partition(":")
    name = name.strip().upper()
    if name not in KNOWN_GPUS:
        raise UnsupportedError(f"Unknown Modal GPU {gpu!r}. Known GPUs: {known} (optionally with ':<count>').")
    count = 1
    if sep:
        if not count_str.strip().isdigit():
            raise UnsupportedError(f"Invalid GPU count in {gpu!r}: expected a positive integer, e.g. '{name}:2'.")
        count = int(count_str)
    limit = KNOWN_GPUS[name]
    if not 1 <= count <= limit:
        raise UnsupportedError(f"GPU count for {name} must be between 1 and {limit}, got {count} in {gpu!r}.")
    return name if count == 1 else f"{name}:{count}"


def _requirement_name(requirement: str) -> str:
    """The normalized distribution name of a requirement string (PEP 503)."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if not match:
        return requirement.strip().lower()
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def local_marimo_version() -> str | None:
    """The marimo version installed next to nbgpu, or None."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version("marimo")
    except PackageNotFoundError:
        return None


def _pin_marimo(requirement: str, marimo_version: str | None) -> str:
    """A bare `marimo` (optionally with extras) becomes `marimo==<version>`,
    so the runtime matches the laptop (whose copy serves the frontend); a
    requirement with a version specifier or marker is the user's choice."""
    if marimo_version is None or not re.fullmatch(r"\s*[Mm]arimo\s*(\[[^\]]*\])?\s*", requirement):
        return requirement
    return f"{requirement.strip()}=={marimo_version}"


def image_packages(dependencies: Sequence[str], marimo_version: str | None = None) -> list[str]:
    """The spec's dependencies (in order, exact duplicates removed) plus
    marimo and watchdog unless a dependency already names them. With
    `marimo_version`, an unversioned marimo (given or added) is pinned to it."""
    out: list[str] = []
    seen_reqs: set[str] = set()
    names: set[str] = set()
    for dep in dependencies:
        dep = dep.strip()
        if not dep or dep in seen_reqs:
            continue
        seen_reqs.add(dep)
        names.add(_requirement_name(dep))
        out.append(_pin_marimo(dep, marimo_version))
    for base in BASE_PACKAGES:
        if base not in names:
            out.append(_pin_marimo(base, marimo_version))
    return out


_CLAUSE = re.compile(r"^\s*(~=|===|==|!=|<=|>=|<|>)\s*([0-9][0-9a-zA-Z.*+!-]*)\s*$")


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in text.split("."):
        m = re.match(r"\d+", piece)
        if not m:
            break
        parts.append(int(m.group(0)))
    return tuple(parts)


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def _clause_ok(version: tuple[int, ...], op: str, target: str) -> bool:
    wildcard = target.endswith(".*")
    tv = _version_tuple(target[:-2] if wildcard else target)
    if op in ("==", "!=") and wildcard:
        match = version[: len(tv)] == tv
        return match if op == "==" else not match
    c = _cmp(version, tv)
    if op in ("==", "==="):
        return c == 0
    if op == "!=":
        return c != 0
    if op == ">=":
        return c >= 0
    if op == "<=":
        return c <= 0
    if op == ">":
        return c > 0
    if op == "<":
        return c < 0
    if op == "~=":
        prefix = tv[:-1] if len(tv) > 1 else tv
        return c >= 0 and version[: len(prefix)] == prefix
    return False


def python_series_satisfies(series: str, requires_python: str) -> bool:
    """Whether some release in `series` ("3.12") meets `requires_python`.
    Raises ValueError for a specifier it cannot parse."""
    clauses = [c for c in requires_python.split(",") if c.strip()]
    parsed = []
    for clause in clauses:
        m = _CLAUSE.match(clause)
        if not m:
            raise ValueError(f"cannot parse requires-python clause {clause.strip()!r}")
        parsed.append((m.group(1), m.group(2)))
    base = _version_tuple(series)
    return any(all(_clause_ok(base + (micro,), op, t) for op, t in parsed) for micro in range(0, 40))


def choose_python_version(requires_python: str | None, local: str | None = None) -> str:
    """The Python series for the image: the local minor version when it meets
    requires-python (or none is given), else the highest supported series that
    does. Raises UnsupportedError when no supported series fits."""
    if local is None:
        local = f"{sys.version_info[0]}.{sys.version_info[1]}"
    if not requires_python or not requires_python.strip():
        return local if local in SUPPORTED_PYTHONS else SUPPORTED_PYTHONS[-2]
    try:
        if local in SUPPORTED_PYTHONS and python_series_satisfies(local, requires_python):
            return local
        for series in reversed(SUPPORTED_PYTHONS):
            if python_series_satisfies(series, requires_python):
                return series
    except ValueError as exc:
        raise UnsupportedError(f"requires-python {requires_python!r}: {exc}") from None
    raise UnsupportedError(
        f"requires-python {requires_python!r} matches none of the Python versions "
        f"available on Modal: {', '.join(SUPPORTED_PYTHONS)}."
    )


def resolve_path(remote_root: str, path: str) -> str:
    """Absolute POSIX path: `path` as-is when absolute, else under remote_root."""
    if posixpath.isabs(path):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(remote_root, path))


def temp_path_for(path: str, token: str | None = None) -> str:
    """A temp path next to `path` (same directory, so os.replace is atomic)
    whose name starts with TEMP_PREFIX, so sync and hash_files ignore it."""
    directory = posixpath.dirname(path)
    return posixpath.join(directory, f"{TEMP_PREFIX}{token or uuid.uuid4().hex}")


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def expect_arg(expect: str | None) -> str:
    """CAS_CODE's argv[3] for a write_file `expect` value."""
    if expect is None:
        return EXPECT_ANY
    if expect == ABSENT:
        return EXPECT_ABSENT
    lowered = expect.lower()
    if not _SHA256_RE.match(lowered):
        raise BackendError(f"write_file: expect must be None, ABSENT or a SHA-256 hex digest, got {expect!r}")
    return lowered


def parse_cas_output(stdout: str) -> bool:
    """True for CAS_WRITTEN, False for CAS_CONFLICT; anything else is an error."""
    outcome = stdout.strip()
    if outcome == CAS_WRITTEN:
        return True
    if outcome == CAS_CONFLICT:
        return False
    raise BackendError(f"Modal: unexpected write_file output: {stdout[:200]!r}")


def safe_hash_paths(result: Mapping[str, str]) -> dict[str, str]:
    """hash_files output with anything that is not a plain relative path
    (absolute, empty or `.`/`..` parts) or that is a TEMP_PREFIX file dropped."""
    out: dict[str, str] = {}
    for key, value in result.items():
        rel = str(key)
        parts = rel.split("/")
        if rel.startswith("/") or any(part in ("", ".", "..") for part in parts):
            continue
        if parts[-1].startswith(TEMP_PREFIX):
            continue
        out[rel] = str(value)
    return out


def validate_spec(spec: NotebookSpec) -> dict[str, Any]:
    """Check everything that can be checked without Modal and return the
    normalized compute settings (gpu, python_version, packages)."""
    compute = spec.compute
    gpu = parse_gpu(compute.gpu)
    if compute.max_runtime_s <= 0 or compute.max_runtime_s > MAX_RUNTIME_LIMIT_S:
        raise UnsupportedError(
            f"max-runtime must be between 1s and {MAX_RUNTIME_LIMIT_S // 3600}h on Modal, got {compute.max_runtime_s}s."
        )
    if compute.idle_timeout_s <= 0:
        raise UnsupportedError(f"idle-timeout must be positive, got {compute.idle_timeout_s}s.")
    for mount, name in compute.volumes.items():
        if not posixpath.isabs(mount):
            raise UnsupportedError(f"Volume mount path {mount!r} must be absolute.")
        if not name:
            raise UnsupportedError(f"Volume for {mount!r} has an empty name.")
    return {
        "gpu": gpu,
        "python_version": choose_python_version(spec.requires_python),
        "packages": image_packages(spec.dependencies, local_marimo_version()),
    }


def encode_ports(ports: Sequence[int]) -> str:
    return ",".join(str(p) for p in ports)


def decode_ports(text: str | None) -> dict[int, int]:
    out: dict[int, int] = {}
    for piece in (text or "").split(","):
        piece = piece.strip()
        if piece.isdigit():
            out[int(piece)] = int(piece)
    return out


def build_tags(user_tags: Mapping[str, str], created_at: float, ports: Sequence[int]) -> dict[str, str]:
    for key in user_tags:
        if key.startswith(INTERNAL_TAG_PREFIX):
            raise BackendError(f"Tag {key!r} uses the reserved prefix {INTERNAL_TAG_PREFIX!r}.")
    return {**dict(user_tags), TAG_CREATED_AT: repr(float(created_at)), TAG_PORTS: encode_ports(ports)}


def user_tags(tags: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in tags.items() if not k.startswith(INTERNAL_TAG_PREFIX)}


def tags_match(tags: Mapping[str, str], wanted: Mapping[str, str]) -> bool:
    return all(k in tags and tags[k] == v for k, v in wanted.items())


def handle_from_tags(runtime_id: str, tags: Mapping[str, str]) -> RuntimeHandle:
    try:
        created_at = float(tags.get(TAG_CREATED_AT, "0"))
    except ValueError:
        created_at = 0.0
    return RuntimeHandle(
        backend="modal",
        runtime_id=runtime_id,
        remote_root=REMOTE_ROOT,
        created_at=created_at,
        tags=user_tags(tags),
        port_map=decode_ports(tags.get(TAG_PORTS)),
    )


def exec_timeout_seconds(timeout_s: float | None) -> int | None:
    """Modal's exec timeout is whole seconds; round up, at least 1."""
    if timeout_s is None:
        return None
    if timeout_s <= 0:
        raise BackendError(f"timeout_s must be positive, got {timeout_s}")
    return max(1, math.ceil(timeout_s))


def cas_argv(tmp: str, dest: str, expect: str | None = None) -> list[str]:
    return [REMOTE_PYTHON, "-c", CAS_CODE, tmp, dest, expect_arg(expect)]


def hash_cache_path(remote_root: str) -> str:
    """hash_files' stat cache: next to the project directory, outside what is synced."""
    return posixpath.join(posixpath.dirname(posixpath.normpath(remote_root)), HASH_CACHE_NAME)


# Run before HASH_CODE when argv starts with `--write-first PATH N chunk...`:
# writes zlib(base64) of the N chunks to PATH atomically (best effort), then
# drops those arguments so HASH_CODE sees its usual argv.
WRITE_FIRST_CODE = (
    "import base64 as _b, os as _o, sys as _s, zlib as _z\n"
    "if len(_s.argv) > 3 and _s.argv[1] == '--write-first':\n"
    "    _dest, _n = _s.argv[2], int(_s.argv[3])\n"
    "    _tmp = _o.path.join(_o.path.dirname(_dest), %r + 'w' + str(_o.getpid()))\n"
    "    try:\n"
    "        with open(_tmp, 'wb') as _f:\n"
    "            _f.write(_z.decompress(_b.b64decode(''.join(_s.argv[4:4 + _n]))))\n"
    "        _o.replace(_tmp, _dest)\n"
    "    except Exception as _e:\n"
    "        print('nbgpu: write-first failed: %%s' %% _e, file=_s.stderr)\n"
    "        try:\n"
    "            _o.unlink(_tmp)\n"
    "        except OSError:\n"
    "            pass\n"
    "    _s.argv = _s.argv[:1] + _s.argv[4 + _n:]\n"
) % TEMP_PREFIX
# Linux limits one argument to 128 KiB; stay well under it.
ARG_CHUNK = 100_000


def write_first_args(path: str, data: bytes) -> list[str]:
    encoded = base64.b64encode(zlib.compress(data, 6)).decode("ascii")
    chunks = [encoded[i : i + ARG_CHUNK] for i in range(0, len(encoded), ARG_CHUNK)] or [""]
    return ["--write-first", path, str(len(chunks)), *chunks]


def hash_argv(root: str, cache_path: str, write_first: tuple[str, bytes] | None = None) -> list[str]:
    pre = write_first_args(*write_first) if write_first is not None else []
    return [REMOTE_PYTHON, "-c", WRITE_FIRST_CODE + HASH_CODE, *pre, root, TEMP_PREFIX, cache_path,
            *HASH_EXCLUDED_PARTS]


def launch_argv(pidfile: str, argv: Sequence[str]) -> list[str]:
    return [REMOTE_PYTHON, "-c", LAUNCH_CODE, pidfile, *argv]


def kill_argv(pidfile: str, wait_s: float = PIDFILE_WAIT_S, grace_s: float = TERMINATE_GRACE_S) -> list[str]:
    return [REMOTE_PYTHON, "-c", KILL_CODE, pidfile, str(wait_s), str(grace_s)]


def split_lines(chunks: Iterator[str]) -> Iterator[str]:
    """Re-chunk a text stream into lines without trailing newlines."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        while True:
            nl = buf.find("\n")
            if nl < 0:
                break
            line, buf = buf[:nl], buf[nl + 1 :]
            yield line[:-1] if line.endswith("\r") else line
    if buf:
        yield buf[:-1] if buf.endswith("\r") else buf


# ---------------------------------------------------------------------------
# Modal glue.


@contextmanager
def _modal_errors(what: str) -> Iterator[None]:
    """Turn Modal SDK errors into BackendError with a readable message."""
    import modal.exception

    try:
        yield
    except BackendError:
        raise
    except (modal.exception.Error, TimeoutError, ConnectionError) as exc:
        raise BackendError(f"Modal: {what} failed: {type(exc).__name__}: {exc}") from exc


class _ModalProcess:
    """RemoteProcess backed by a ContainerProcess started under LAUNCH_CODE."""

    def __init__(self, backend: ModalBackend, handle: RuntimeHandle, proc: Any, pidfile: str) -> None:
        self._backend = backend
        self._handle = handle
        self._proc = proc
        self._pidfile = pidfile
        self._lock = threading.Lock()
        self._exit: int | None = None
        self._terminated = False

    def lines(self) -> Iterator[str]:
        import modal.exception

        try:
            yield from split_lines(iter(self._proc.stdout))
        except (modal.exception.Error, ConnectionError):
            # The stream ends when the process or the sandbox goes away.
            return

    def poll(self) -> int | None:
        if self._exit is not None:
            return self._exit
        import modal.exception

        try:
            code = self._proc.poll()
        except (modal.exception.NotFoundError, modal.exception.SandboxTerminatedError):
            code = -1 if self._terminated else None
            if code is None and self._backend.get(self._handle.runtime_id) is None:
                code = -1
        except modal.exception.Error as exc:
            raise BackendError(f"Modal: polling process failed: {exc}") from exc
        if code is not None:
            self._exit = code
        return code

    def terminate(self) -> None:
        with self._lock:
            if self.poll() is not None:
                return
            self._terminated = True
            try:
                self._backend.exec(
                    self._handle,
                    kill_argv(self._pidfile),
                    cwd="/",
                    env=INTERNAL_ENV,
                    timeout_s=PIDFILE_WAIT_S + TERMINATE_GRACE_S + 30,
                )
            except BackendError:
                # The sandbox may be gone, which also ends the process.
                if self._backend.get(self._handle.runtime_id) is not None:
                    raise


class ModalBackend:
    name = "modal"
    bind_host = "0.0.0.0"
    capabilities: frozenset[str] = frozenset({CAP_GPU, CAP_VOLUMES, CAP_MAX_RUNTIME, CAP_LIVE_INSTALL})

    def __init__(self, app_name: str = "nbgpu") -> None:
        self.app_name = app_name
        self._app: modal.App | None = None
        self._sandboxes: dict[str, modal.Sandbox] = {}
        self._lock = threading.Lock()

    # -- internals ---------------------------------------------------------

    def _get_app(self) -> modal.App:
        with self._lock:
            if self._app is None:
                import modal

                with _modal_errors(f"looking up app {self.app_name!r}"):
                    self._app = modal.App.lookup(self.app_name, create_if_missing=True)
            return self._app

    def _sandbox(self, runtime_id: str) -> modal.Sandbox:
        sb = self._sandboxes.get(runtime_id)
        if sb is None:
            import modal

            with _modal_errors(f"looking up sandbox {runtime_id}"):
                sb = modal.Sandbox.from_id(runtime_id)
            self._sandboxes[runtime_id] = sb
        return sb

    def _build_image(self, python_version: str, packages: Sequence[str]) -> modal.Image:
        import modal

        image = modal.Image.debian_slim(python_version=python_version)
        if packages:
            image = image.uv_pip_install(*packages)
        return image

    def _lookup_volumes(self, volumes: Mapping[str, str]) -> dict[str, modal.Volume]:
        import modal
        import modal.exception

        out: dict[str, modal.Volume] = {}
        for mount, name in volumes.items():
            vol = modal.Volume.from_name(name)
            try:
                vol.hydrate()
            except modal.exception.NotFoundError:
                raise UnsupportedError(f"Modal volume {name!r} (for {mount}) does not exist.") from None
            except modal.exception.Error as exc:
                raise BackendError(f"Modal: looking up volume {name!r} failed: {exc}") from exc
            out[mount] = vol
        return out

    # -- Backend -----------------------------------------------------------

    def create(self, request: RuntimeRequest) -> RuntimeHandle:
        settings = validate_spec(request.spec)
        compute = request.spec.compute
        created_at = time.time()
        tags = build_tags(request.tags, created_at, request.ports)

        import modal

        volumes = self._lookup_volumes(compute.volumes) if compute.volumes else {}
        app = self._get_app()
        image = self._build_image(settings["python_version"], settings["packages"])
        with _modal_errors("creating sandbox"):
            sb = modal.Sandbox.create(
                *ENTRYPOINT,
                app=app,
                image=image,
                tags=tags,
                env=dict(request.env) or None,
                gpu=settings["gpu"],
                cpu=compute.cpu,
                memory=compute.memory_mib,
                timeout=compute.max_runtime_s,
                idle_timeout=IDLE_GRACE_S,
                volumes=volumes,
                encrypted_ports=list(request.ports),
            )
        runtime_id = sb.object_id
        self._sandboxes[runtime_id] = sb
        handle = RuntimeHandle(
            backend=self.name,
            runtime_id=runtime_id,
            remote_root=REMOTE_ROOT,
            created_at=created_at,
            tags=user_tags(tags),
            port_map={p: p for p in request.ports},
        )
        try:
            res = self.exec(handle, ["mkdir", "-p", REMOTE_ROOT], cwd="/", env=INTERNAL_ENV, timeout_s=300)
            if res.exit_code != 0:
                raise BackendError(f"Modal: creating {REMOTE_ROOT} failed: {res.stderr.strip()}")
        except BaseException:
            self._terminate_uninterruptibly(handle)
            raise
        return handle

    def _terminate_uninterruptibly(self, handle: RuntimeHandle, timeout_s: float | None = None) -> None:
        """Terminate a half-created sandbox without a second Ctrl-C cutting it
        short: SIGINT/SIGTERM/SIGHUP are ignored meanwhile (main thread only;
        the previous handlers are restored after), and terminate runs in a
        daemon thread joined with a timeout so a hung call cannot block
        forever. On timeout raises BackendError naming the sandbox. Other
        terminate errors are swallowed (the original error matters more)."""
        if timeout_s is None:
            timeout_s = CLEANUP_TERMINATE_TIMEOUT_S
        done = threading.Event()

        def run() -> None:
            try:
                self.terminate(handle)
            except BaseException:  # noqa: BLE001 - best effort cleanup
                pass
            finally:
                done.set()

        saved: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGINT", "SIGTERM", "SIGHUP"):
                sig = getattr(signal, name, None)
                if sig is None:
                    continue
                with suppress(ValueError, OSError):
                    saved[sig] = signal.signal(sig, signal.SIG_IGN)
        try:
            threading.Thread(target=run, name="nbgpu-cleanup-terminate", daemon=True).start()
            finished = done.wait(timeout_s)
        finally:
            for sig, previous in saved.items():
                with suppress(ValueError, OSError, TypeError):
                    signal.signal(sig, previous if previous is not None else signal.SIG_DFL)
        if not finished:
            raise BackendError(
                f"Modal: could not stop half-created sandbox {handle.runtime_id} within {timeout_s:g}s; "
                f"stop it yourself with `modal sandbox terminate {handle.runtime_id}` or in the Modal dashboard."
            )

    def get(self, runtime_id: str) -> RuntimeHandle | None:
        import modal
        import modal.exception

        try:
            sb = modal.Sandbox.from_id(runtime_id)
            if sb.poll() is not None:
                self._sandboxes.pop(runtime_id, None)
                return None
            tags = sb.get_tags()
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            self._sandboxes.pop(runtime_id, None)
            return None
        except modal.exception.Error as exc:
            raise BackendError(f"Modal: looking up sandbox {runtime_id} failed: {exc}") from exc
        self._sandboxes.setdefault(runtime_id, sb)
        return handle_from_tags(runtime_id, tags)

    def find(self, tags: Mapping[str, str]) -> list[RuntimeHandle]:
        import modal

        app = self._get_app()
        out: list[RuntimeHandle] = []
        with _modal_errors("listing sandboxes"):
            for sb in modal.Sandbox.list(app_id=app.app_id, tags=dict(tags) or None):
                sb_tags = sb.get_tags()
                if not tags_match(sb_tags, tags):
                    continue
                out.append(handle_from_tags(sb.object_id, sb_tags))
        return out

    def exec(
        self,
        handle: RuntimeHandle,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        if not argv:
            raise BackendError("exec: empty argv")
        import modal.exception

        sb = self._sandbox(handle.runtime_id)
        workdir = resolve_path(handle.remote_root, cwd) if cwd else handle.remote_root
        timeout = exec_timeout_seconds(timeout_s)
        what = f"running {argv[0]!r}"
        with _modal_errors(what):
            proc = sb.exec(*argv, workdir=workdir, env=dict(env) if env else None, timeout=timeout, text=True)
        stderr_box: list[str] = []
        errors: list[BaseException] = []

        def read_stderr() -> None:
            try:
                stderr_box.append(proc.stderr.read())
            except BaseException as exc:
                errors.append(exc)

        reader = threading.Thread(target=read_stderr, daemon=True)
        reader.start()
        timed_out = False
        try:
            stdout = proc.stdout.read()
            code = proc.wait()
        except modal.exception.ExecTimeoutError:
            timed_out, stdout, code = True, "", -1
        except (modal.exception.Error, ConnectionError) as exc:
            raise BackendError(f"Modal: {what} failed: {exc}") from exc
        reader.join(30)
        if timeout is not None and (timed_out or code == -1):
            raise BackendError(f"Command {argv[0]!r} timed out after {timeout_s}s")
        if errors and not isinstance(errors[0], modal.exception.ExecTimeoutError):
            raise BackendError(f"Modal: reading stderr of {argv[0]!r} failed: {errors[0]}")
        return ExecResult(exit_code=code, stdout=stdout, stderr=stderr_box[0] if stderr_box else "")

    def spawn(
        self,
        handle: RuntimeHandle,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> _ModalProcess:
        if not argv:
            raise BackendError("spawn: empty argv")
        from modal.stream_type import StreamType

        sb = self._sandbox(handle.runtime_id)
        workdir = resolve_path(handle.remote_root, cwd) if cwd else handle.remote_root
        pidfile = posixpath.join(SPAWN_PID_DIR, f"nbgpu-spawn-{uuid.uuid4().hex}.pgid")
        with _modal_errors(f"starting {argv[0]!r}"):
            proc = sb.exec(
                *launch_argv(pidfile, argv),
                workdir=workdir,
                env=dict(env) if env else None,
                text=True,
                bufsize=1,
                stderr=StreamType.DEVNULL,
            )
        return _ModalProcess(self, handle, proc, pidfile)

    def expose(self, handle: RuntimeHandle, port: int) -> str:
        sb = self._sandbox(handle.runtime_id)
        with _modal_errors(f"getting tunnel for port {port}"):
            tunnels = sb.tunnels()
        tunnel = tunnels.get(port)
        if tunnel is None:
            raise BackendError(
                f"Port {port} is not exposed by sandbox {handle.runtime_id} (exposed: {sorted(tunnels)})."
            )
        return tunnel.url.rstrip("/")

    def read_file(self, handle: RuntimeHandle, path: str) -> bytes | None:
        import modal.exception

        sb = self._sandbox(handle.runtime_id)
        target = resolve_path(handle.remote_root, path)
        try:
            return sb.filesystem.read_bytes(target)
        except (modal.exception.SandboxFilesystemNotFoundError, modal.exception.SandboxFilesystemNotADirectoryError):
            return None
        except modal.exception.Error as exc:
            raise BackendError(f"Modal: reading {target} failed: {type(exc).__name__}: {exc}") from exc

    def write_file(
        self,
        handle: RuntimeHandle,
        path: str,
        data: bytes,
        *,
        expect: str | None = None,
    ) -> bool:
        target = resolve_path(handle.remote_root, path)
        tmp = temp_path_for(target)
        argv = cas_argv(tmp, target, expect)  # validates `expect` before any upload
        sb = self._sandbox(handle.runtime_id)
        try:
            with _modal_errors(f"writing {target}"):
                sb.filesystem.write_bytes(bytes(data), tmp)
        except BaseException:
            with suppress(Exception):
                sb.filesystem.remove(tmp)
            raise
        res = self.exec(handle, argv, cwd="/", env=INTERNAL_ENV, timeout_s=120)
        if res.exit_code != 0:
            raise BackendError(f"Modal: replacing {target} failed: {res.stderr.strip()}")
        return parse_cas_output(res.stdout)

    def write_new_files(self, handle: RuntimeHandle, files: Mapping[str, bytes]) -> set[str]:
        """Create many files under remote_root in one upload and one exec;
        files that already exist are left alone. Returns the paths written."""
        if not files:
            return set()
        archive = temp_path_for(posixpath.join(posixpath.dirname(handle.remote_root.rstrip("/")), "x"))
        sb = self._sandbox(handle.runtime_id)
        try:
            with _modal_errors("uploading the project"):
                sb.filesystem.write_bytes(build_tar(files), archive)
        except BaseException:
            with suppress(Exception):
                sb.filesystem.remove(archive)
            raise
        argv = [REMOTE_PYTHON, "-c", BULK_CODE, archive, handle.remote_root, TEMP_PREFIX]
        res = self.exec(handle, argv, cwd="/", env=INTERNAL_ENV, timeout_s=600)
        if res.exit_code != 0:
            raise BackendError(f"Modal: unpacking the project failed: {res.stderr.strip()}")
        return parse_bulk_output(res.stdout, files)

    def hash_files(
        self, handle: RuntimeHandle, root: str = ".", *, write_first: tuple[str, bytes] | None = None
    ) -> dict[str, str]:
        target = resolve_path(handle.remote_root, root)
        if write_first is not None:
            write_first = (resolve_path(handle.remote_root, write_first[0]), write_first[1])
        argv = hash_argv(target, hash_cache_path(handle.remote_root), write_first)
        res = self.exec(handle, argv, cwd="/", env=INTERNAL_ENV, timeout_s=600)
        if res.exit_code != 0:
            raise BackendError(f"Modal: hashing files under {target} failed: {res.stderr.strip()}")
        try:
            result = json.loads(res.stdout)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Modal: unexpected hash output: {res.stdout[:200]!r}") from exc
        if not isinstance(result, dict):
            raise BackendError(f"Modal: unexpected hash output: {res.stdout[:200]!r}")
        return safe_hash_paths(result)

    def terminate(self, handle: RuntimeHandle) -> None:
        import modal
        import modal.exception

        self._sandboxes.pop(handle.runtime_id, None)
        try:
            sb = modal.Sandbox.from_id(handle.runtime_id)
            if sb.poll() is not None:
                return
            sb.terminate(wait=True)
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            return
        except modal.exception.Error as exc:
            raise BackendError(f"Modal: terminating {handle.runtime_id} failed: {exc}") from exc
