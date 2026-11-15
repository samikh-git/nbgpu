"""The provider contract.

A backend creates a runtime (a machine or container with the notebook's
environment), runs commands in it, exposes one port to the laptop, moves
files in and out, and terminates it. Everything else (marimo, sync, cost)
is built on these operations, so a new provider only implements this.

Paths given to file operations are POSIX paths relative to the runtime's
`remote_root` unless they are absolute; absolute paths are used as-is.
`env` passed to exec/spawn is added to the runtime's environment, never a
replacement for it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from nbgpu.spec import NotebookSpec

# Optional features a backend may support. Absence means the feature is
# missing, never approximated.
CAP_GPU = "gpu"
CAP_VOLUMES = "volumes"
CAP_IDLE_TIMEOUT = "idle-timeout"  # backend enforces spec.compute.idle_timeout_s
CAP_MAX_RUNTIME = "max-runtime"  # backend enforces spec.compute.max_runtime_s


# Prefix for temporary files made by atomic writes, locally and remotely.
# Sync never treats such files as project files.
TEMP_PREFIX = ".nbgpu-tmp-"

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

    def hash_files(self, handle: RuntimeHandle, root: str = ".") -> dict[str, str]:
        """Every regular file under `root`: POSIX path relative to `root` ->
        SHA-256 hex. Paths whose parts include .git, .venv, __pycache__ or
        node_modules are left out."""
        ...

    def terminate(self, handle: RuntimeHandle) -> None:
        """Stop and release the runtime. No error if already gone."""
        ...
