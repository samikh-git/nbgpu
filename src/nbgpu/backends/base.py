"""The provider contract.

A backend creates a runtime (a machine or container with the notebook's
environment), runs commands in it, exposes one port to the laptop, moves
files in and out, and terminates it. Everything else (marimo, sync, cost)
is built on these operations, so a new provider only implements this.

Paths given to file operations are POSIX paths relative to the runtime's
`remote_root` unless they are absolute; absolute paths are used as-is.
`env` passed to exec/spawn is added to the runtime's environment, never a
replacement for it.

Optional method, looked up with getattr by the sync engine's initial push:
`write_new_files(handle, files: Mapping[str, bytes]) -> set[str]` creates
many files under remote_root in one round trip, never overwriting an
existing one, and returns the paths it wrote. Without it, the push writes
file by file with write_file(expect=ABSENT).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from dataclasses import dataclass, field
from typing import Protocol

from nbgpu.spec import NotebookSpec

# Optional features a backend may support. Absence means the feature is
# missing, never approximated.
CAP_GPU = "gpu"
CAP_VOLUMES = "volumes"
CAP_IDLE_TIMEOUT = "idle-timeout"  # backend enforces spec.compute.idle_timeout_s
CAP_MAX_RUNTIME = "max-runtime"  # backend enforces spec.compute.max_runtime_s
# Packages can be installed into the running runtime's own Python (an
# isolated environment). The local backend lacks it: its Python is the user's.
CAP_LIVE_INSTALL = "live-install"
# The provider can report its current prices and what this tool's runtimes
# actually cost: the backend has `current_prices()` and `spend()` (see
# BillingBackend).
CAP_BILLING = "billing"
# The runtime's filesystem can be saved (`snapshot(handle)` returns an id)
# and a runtime created from it (RuntimeRequest.base_image). See SnapshotBackend.
CAP_SNAPSHOT = "snapshot"

# Where spec.compute.cache_volume is mounted, and what points into it. The
# variables are the standard ones libraries read, so notebooks need no change.
CACHE_MOUNT = "/nbgpu-cache"


def cache_env(mount: str = CACHE_MOUNT) -> dict[str, str]:
    return {
        "NBGPU_CACHE": mount,  # for the notebook's own files
        "HF_HOME": f"{mount}/huggingface",
        "TORCH_HOME": f"{mount}/torch",
        "TORCHINDUCTOR_CACHE_DIR": f"{mount}/torchinductor",
        "TRITON_CACHE_DIR": f"{mount}/triton",
        "UV_CACHE_DIR": f"{mount}/uv",
    }


# Prefix for temporary files made by atomic writes, locally and remotely.
# Sync never treats such files as project files.
TEMP_PREFIX = ".nbgpu-tmp-"

# Directory names never synced and left out of hash_files: version control,
# environments, caches, and marimo's per-notebook session state. (The idle
# watchdog keeps a stdlib-only copy, tested to match.)
EXCLUDED_PARTS: frozenset[str] = frozenset(
    {".git", ".venv", "__pycache__", "node_modules", ".nbgpu", "__marimo__"}
)

# `expect` value for write_file meaning "only if the file does not exist".
ABSENT = ""


class BackendError(RuntimeError):
    """A provider operation failed. The message is shown to the user."""


class UnsupportedError(BackendError):
    """The spec asks for something this backend cannot do (e.g. an unknown GPU)."""


@dataclass(frozen=True)
class RuntimeRequest:
    spec: NotebookSpec
    tags: Mapping[str, str]
    # Ports the runtime must be able to expose later with `expose`.
    ports: tuple[int, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    # A `snapshot()` id to start from instead of building the environment
    # (only for backends with CAP_SNAPSHOT). The backend boots it as is.
    base_image: str | None = None


@dataclass(frozen=True)
class RuntimeHandle:
    backend: str
    runtime_id: str
    remote_root: str  # absolute path of the project directory on the runtime
    created_at: float  # unix seconds
    tags: Mapping[str, str] = field(default_factory=dict)
    # Requested port -> port the process must actually listen on. Equal unless
    # the backend had to remap (the local backend picks free ports).
    port_map: Mapping[int, int] = field(default_factory=dict)

    def port_for(self, port: int) -> int:
        """The port a process in the runtime should listen on for `port`."""
        return self.port_map.get(port, port)


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


class RemoteProcess(Protocol):
    """A long-running process started with `spawn`."""

    def lines(self) -> Iterator[str]:
        """Merged stdout and stderr, line by line, without trailing newlines.
        Ends when the process exits."""
        ...

    def poll(self) -> int | None:
        """Exit code, or None while running."""
        ...

    def terminate(self) -> None:
        """Stop the process. No error if it already exited."""
        ...


class Backend(Protocol):
    name: str
    capabilities: frozenset[str]
    # Address servers in the runtime should bind to so `expose` can reach
    # them: "0.0.0.0" for remote runtimes, "127.0.0.1" for the local backend.
    bind_host: str

    def create(self, request: RuntimeRequest) -> RuntimeHandle:
        """Provision a runtime with the spec's environment installed and
        `remote_root` existing and empty. Raises UnsupportedError for a spec
        it cannot satisfy, before provisioning anything."""
        ...

    def get(self, runtime_id: str) -> RuntimeHandle | None:
        """The runtime if it exists and is running, else None."""
        ...

    def find(self, tags: Mapping[str, str]) -> list[RuntimeHandle]:
        """Running runtimes whose tags include every given key/value."""
        ...

    def exec(
        self,
        handle: RuntimeHandle,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        """Run to completion. `cwd` defaults to remote_root. A non-zero exit
        is returned, not raised; a timeout raises BackendError."""
        ...

    def spawn(
        self,
        handle: RuntimeHandle,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RemoteProcess:
        """Start in the background and return immediately."""
        ...

    def expose(self, handle: RuntimeHandle, port: int) -> str:
        """A base URL (no trailing slash) on which the laptop reaches `port`
        inside the runtime. `port` must have been in request.ports."""
        ...

    def read_file(self, handle: RuntimeHandle, path: str) -> bytes | None:
        """File contents, or None if it does not exist."""
        ...

    def write_file(
        self,
        handle: RuntimeHandle,
        path: str,
        data: bytes,
        *,
        expect: str | None = None,
    ) -> bool:
        """Create parent directories; replace atomically so readers never see
        a partial file. The temp file's name starts with TEMP_PREFIX.

        `expect` makes the write conditional (compare-and-swap), so a change
        made in the runtime since the caller last hashed it is never
        overwritten:
          None     -> write unconditionally
          ABSENT   -> write only if the file does not exist
          <sha256> -> write only if the file's current SHA-256 hex equals it
        The check and the replace happen in one step inside the runtime (one
        exec, or one local function call), never as a separate round trip.
        Returns True if written, False if the precondition failed."""
        ...

    def hash_files(
        self, handle: RuntimeHandle, root: str = ".", *, write_first: tuple[str, bytes] | None = None
    ) -> dict[str, str]:
        """Every regular file under `root`: POSIX path relative to `root` ->
        SHA-256 hex. Paths with a part in EXCLUDED_PARTS are left out.

        `write_first` = (absolute path outside `root`, bytes): written
        atomically before hashing, in the same round trip (the sync loop sends
        its manifest this way). Best effort: a failed write does not fail the
        hash, and the file keeps its old content."""
        ...

    def terminate(self, handle: RuntimeHandle) -> None:
        """Stop and release the runtime. No error if already gone."""
        ...


@dataclass(frozen=True)
class ProviderPrices:
    """A provider's current prices, in USD per hour."""

    gpu_hourly: Mapping[str, float]  # by GPU name as written in the header (e.g. "A100-80GB")
    cpu_core_hourly: float | None  # per core, at the rate the runtimes are billed
    memory_gib_hourly: float | None  # per GiB


@dataclass(frozen=True)
class SpendRow:
    """What this tool's runtimes cost in one interval, as billed."""

    interval_start: datetime  # UTC
    cost: float  # USD
    by_resource: Mapping[str, float]  # e.g. {"A10G": 1.2, "CPU": 0.1}; keys are the provider's


class BillingBackend(Backend, Protocol):
    """A Backend with CAP_BILLING."""

    def current_prices(self) -> ProviderPrices:
        """The provider's current prices for this account. Raises BackendError."""
        ...

    def spend(self, start: datetime, end: datetime, resolution: str) -> list[SpendRow]:
        """Billed cost of this tool's runtimes between `start` (inclusive) and
        `end` (exclusive), both UTC, per interval: resolution "h" or "d".
        Only complete intervals are reported. Raises BackendError."""
        ...


class SnapshotBackend(Backend, Protocol):
    """A Backend with CAP_SNAPSHOT."""

    def snapshot(self, handle: RuntimeHandle) -> str:
        """Save the runtime's filesystem (not mounted volumes) and return an id
        for RuntimeRequest.base_image. Raises BackendError."""
        ...
