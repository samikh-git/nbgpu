"""In-memory test doubles shared by the test suites.

`FakeBackend` implements the Backend protocol (src/nbgpu/backends/base.py)
without any processes or network: each runtime's files live in a dict keyed
by absolute POSIX path. Command execution (`exec`/`spawn`) is not simulated
and raises NotImplementedError.

`write_file` implements `expect` (compare-and-swap) as documented in the
protocol: None writes unconditionally, ABSENT only if the file is missing,
a SHA-256 hex only if the current content has that hash; returns whether it
wrote.

Test hooks (not part of the protocol, never counted):
- `remote_write(handle, path, data)`: simulate an edit made on the runtime,
  e.g. in the marimo browser UI.
- `remote_read(handle, path)`, `remote_delete(handle, path)`,
  `remote_files(handle)`: inspect or change the runtime's files directly.

Race hooks, to make "an edit landed mid-poll" deterministic. Each callback
runs once, inside the protocol call:
- `after_hash_files(cb)`: after hash_files has taken its hashes.
- `before_write(path, cb)`: in write_file of `path`, before `expect` is checked.
- `before_read(path, cb)`: in read_file of `path`, before content is returned.

Call counters (only protocol calls are counted):
- `reads` / `writes`: list of paths passed to read_file / write_file
  (every call, including writes refused by `expect`), normalised to be
  relative to the runtime's remote_root when inside it.
- `rejected_writes`: write_file calls refused because `expect` did not match.
- `hash_calls`: number of hash_files calls.
- `reset_counters()` clears them all.
"""

from __future__ import annotations

import hashlib
import itertools
import posixpath
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from nbgpu.backends.base import (
    ABSENT,
    BackendError,
    ExecResult,
    RemoteProcess,
    RuntimeHandle,
    RuntimeRequest,
)
from nbgpu.spec import ComputeSpec, NotebookSpec

HASH_EXCLUDED_PARTS = frozenset({".git", ".venv", "__pycache__", "node_modules"})


def fake_request(
    *,
    tags: Mapping[str, str] | None = None,
    ports: tuple[int, ...] = (2718,),
    notebook: Path = Path("/tmp/nbgpu-tests/notebook.py"),
    compute: ComputeSpec | None = None,
) -> RuntimeRequest:
    """A minimal RuntimeRequest for tests that do not care about the spec."""
    spec = NotebookSpec(
        path=notebook,
        requires_python=None,
        dependencies=(),
        compute=compute or ComputeSpec(provider="fake"),
    )
    return RuntimeRequest(spec=spec, tags=dict(tags or {}), ports=ports)


class FakeBackend:
    name = "fake"
    capabilities: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        root_prefix: str = "/runtimes",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._root_prefix = root_prefix.rstrip("/") or "/runtimes"
        self._clock = clock
        self._ids = itertools.count(1)
        self._handles: dict[str, RuntimeHandle] = {}
        self._ports: dict[str, tuple[int, ...]] = {}
        # runtime_id -> {absolute POSIX path -> bytes}
        self.files: dict[str, dict[str, bytes]] = {}
        self.requests: list[RuntimeRequest] = []
        self.terminated: list[str] = []
        self.reads: list[str] = []
        self.writes: list[str] = []
        self.hash_calls = 0
        self.rejected_writes: list[str] = []  # write_file calls whose `expect` failed
        self._after_hash: list[Callable[[], None]] = []
        self._before_write: dict[str, Callable[[], None]] = {}
        self._before_read: dict[str, Callable[[], None]] = {}

    # -- counters ---------------------------------------------------------

    def reset_counters(self) -> None:
        self.reads.clear()
        self.writes.clear()
        self.rejected_writes.clear()
        self.hash_calls = 0

    # -- lifecycle --------------------------------------------------------

    def create(self, request: RuntimeRequest) -> RuntimeHandle:
        runtime_id = f"fake-{next(self._ids)}"
        handle = RuntimeHandle(
            backend=self.name,
            runtime_id=runtime_id,
            remote_root=f"{self._root_prefix}/{runtime_id}/project",
            created_at=self._clock(),
            tags=dict(request.tags),
            port_map={p: p for p in request.ports},
        )
        self.requests.append(request)
        self._handles[runtime_id] = handle
        self._ports[runtime_id] = tuple(request.ports)
        self.files[runtime_id] = {}
        return handle

    def get(self, runtime_id: str) -> RuntimeHandle | None:
        return self._handles.get(runtime_id)

    def find(self, tags: Mapping[str, str]) -> list[RuntimeHandle]:
        return [
            h
            for h in self._handles.values()
            if all(h.tags.get(k) == v for k, v in tags.items())
        ]

    def terminate(self, handle: RuntimeHandle) -> None:
        if self._handles.pop(handle.runtime_id, None) is not None:
            self.terminated.append(handle.runtime_id)
        self.files.pop(handle.runtime_id, None)
        self._ports.pop(handle.runtime_id, None)

    # -- processes (not simulated) ---------------------------------------

    def exec(
        self,
        handle: RuntimeHandle,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        raise NotImplementedError("FakeBackend does not run commands")

    def spawn(
        self,
        handle: RuntimeHandle,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RemoteProcess:
        raise NotImplementedError("FakeBackend does not run commands")

    def expose(self, handle: RuntimeHandle, port: int) -> str:
        self._require(handle)
        if port not in self._ports[handle.runtime_id]:
            raise BackendError(f"port {port} was not requested for {handle.runtime_id}")
        return f"https://{handle.runtime_id}-{port}.fake.invalid"

    # -- files ------------------------------------------------------------

    def read_file(self, handle: RuntimeHandle, path: str) -> bytes | None:
        files = self._require(handle)
        abs_path = self._abs(handle, path)
        rel = self._rel(handle, abs_path)
        self.reads.append(rel)
        hook = self._before_read.pop(rel, None)
        if hook is not None:
            hook()
        return files.get(abs_path)

    def write_file(
        self,
        handle: RuntimeHandle,
        path: str,
        data: bytes,
        *,
        expect: str | None = None,
    ) -> bool:
        files = self._require(handle)
        abs_path = self._abs(handle, path)
        rel = self._rel(handle, abs_path)
        self.writes.append(rel)
        hook = self._before_write.pop(rel, None)
        if hook is not None:
            hook()
        # Check and replace are one step, as the protocol requires.
        current = files.get(abs_path)
        if expect is not None:
            if expect == ABSENT:
                ok = current is None
            else:
                ok = current is not None and hashlib.sha256(current).hexdigest() == expect
            if not ok:
                self.rejected_writes.append(rel)
                return False
        files[abs_path] = bytes(data)
        return True

    def hash_files(self, handle: RuntimeHandle, root: str = ".") -> dict[str, str]:
        files = self._require(handle)
        self.hash_calls += 1
        abs_root = self._abs(handle, root)
        prefix = abs_root.rstrip("/") + "/"
        out: dict[str, str] = {}
        for abs_path, data in files.items():
            if not abs_path.startswith(prefix):
                continue
            rel = abs_path[len(prefix):]
            if any(part in HASH_EXCLUDED_PARTS for part in rel.split("/")):
                continue
            out[rel] = hashlib.sha256(data).hexdigest()
        hooks, self._after_hash = self._after_hash, []
        for hook in hooks:
            hook()
        return out

    # -- race hooks (fire once, inside a protocol call) -------------------

    def after_hash_files(self, callback: Callable[[], None]) -> None:
        """Run `callback` once, inside the next hash_files call after the
        hashes are taken: anything it changes (locally or remotely) happens
        "during" the poll that is using those hashes."""
        self._after_hash.append(callback)

    def before_write(self, path: str, callback: Callable[[], None]) -> None:
        """Run `callback` once, inside the next write_file of `path` (relative
        to remote_root), before its `expect` precondition is checked."""
        self._before_write[path] = callback

    def before_read(self, path: str, callback: Callable[[], None]) -> None:
        """Run `callback` once, inside the next read_file of `path`, before
        the content is returned."""
        self._before_read[path] = callback

    # -- test hooks (not counted) -----------------------------------------

    def remote_write(self, handle: RuntimeHandle, path: str, data: bytes | str) -> None:
        """Simulate an edit on the runtime (e.g. saved from the marimo UI)."""
        files = self._require(handle)
        if isinstance(data, str):
            data = data.encode()
        files[self._abs(handle, path)] = bytes(data)

    def remote_read(self, handle: RuntimeHandle, path: str) -> bytes | None:
        return self._require(handle).get(self._abs(handle, path))

    def remote_delete(self, handle: RuntimeHandle, path: str) -> None:
        self._require(handle).pop(self._abs(handle, path), None)

    def remote_files(self, handle: RuntimeHandle) -> dict[str, bytes]:
        """All files under remote_root, keyed by path relative to it."""
        files = self._require(handle)
        prefix = handle.remote_root.rstrip("/") + "/"
        return {p[len(prefix):]: d for p, d in files.items() if p.startswith(prefix)}

    # -- helpers ----------------------------------------------------------

    def _require(self, handle: RuntimeHandle) -> dict[str, bytes]:
        files = self.files.get(handle.runtime_id)
        if files is None or handle.runtime_id not in self._handles:
            raise BackendError(f"runtime {handle.runtime_id} is not running")
        return files

    @staticmethod
    def _abs(handle: RuntimeHandle, path: str) -> str:
        if not path.startswith("/"):
            path = posixpath.join(handle.remote_root, path)
        return posixpath.normpath(path)

    @staticmethod
    def _rel(handle: RuntimeHandle, abs_path: str) -> str:
        prefix = handle.remote_root.rstrip("/") + "/"
        return abs_path[len(prefix):] if abs_path.startswith(prefix) else abs_path
