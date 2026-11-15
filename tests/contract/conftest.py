"""Fixtures for the backend contract suite.

Every test in tests/contract runs once per backend: `local` always, `modal`
only with NBGPU_MODAL_TESTS=1 (the `modal` mark is skipped otherwise by
tests/conftest.py). Modal runtimes are CPU-only and short-lived to keep the
cost minimal, and every runtime a test creates is terminated in teardown,
even when the test fails.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest

from nbgpu.backends.base import Backend, RuntimeHandle, RuntimeRequest
from nbgpu.spec import ComputeSpec, NotebookSpec

# A port every test runtime can expose.
EXPOSED_PORT = 8000

# Modal app used by the contract suite, kept apart from the user's "nbgpu" app.
MODAL_TEST_APP = "nbgpu-contract-tests"

NOTEBOOK_TEXT = """\
import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    x = 1
    return


if __name__ == "__main__":
    app.run()
"""


def make_spec(directory: Path, provider: str, gpu: str | None = None) -> NotebookSpec:
    """A minimal spec built directly (not via load_spec), CPU-only by default,
    with short timeouts so a leaked Modal sandbox dies on its own quickly."""
    directory.mkdir(parents=True, exist_ok=True)
    notebook = directory / "notebook.py"
    if not notebook.exists():
        notebook.write_text(NOTEBOOK_TEXT)
    return NotebookSpec(
        path=notebook.resolve(),
        requires_python=None,
        dependencies=(),
        compute=ComputeSpec(
            provider=provider,
            gpu=gpu,
            idle_timeout_s=10 * 60,
            max_runtime_s=30 * 60,
        ),
    )


def _make_backend(name: str, base_dir: Path) -> Backend:
    if name == "local":
        from nbgpu.backends.local import LocalBackend

        return LocalBackend(base_dir=base_dir)
    if name == "modal":
        from nbgpu.backends.modal import ModalBackend

        return ModalBackend(app_name=MODAL_TEST_APP)
    raise AssertionError(name)


@pytest.fixture(
    scope="module",
    params=["local", pytest.param("modal", marks=pytest.mark.modal)],
)
def backend(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Backend:
    return _make_backend(request.param, tmp_path_factory.mktemp(f"runtimes-{request.param}"))


@pytest.fixture(scope="module")
def spec_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("spec")


@pytest.fixture(scope="module")
def run_tag() -> str:
    """Unique per module run, so `find` never matches runtimes from other runs."""
    return uuid.uuid4().hex[:12]


class RuntimeFactory:
    """Creates runtimes and remembers them so teardown can terminate them."""

    def __init__(self, backend: Backend, spec_dir: Path, run_tag: str) -> None:
        self.backend = backend
        self.spec_dir = spec_dir
        self.run_tag = run_tag
        # Each factory tags its runtimes so cleanup never touches another
        # factory's (e.g. the module-shared runtime).
        self.factory_tag = uuid.uuid4().hex[:12]
        self.created: list[RuntimeHandle] = []

    def __call__(
        self,
        tags: Mapping[str, str] | None = None,
        ports: tuple[int, ...] = (EXPOSED_PORT,),
        gpu: str | None = None,
    ) -> RuntimeHandle:
        all_tags = {"nbgpu.test": self.run_tag, "nbgpu.test.factory": self.factory_tag, **(tags or {})}
        request = RuntimeRequest(
            spec=make_spec(self.spec_dir, self.backend.name, gpu=gpu),
            tags=all_tags,
            ports=ports,
        )
        handle = self.backend.create(request)
        self.created.append(handle)
        return handle

    def cleanup(self) -> None:
        errors: list[BaseException] = []
        for handle in self.created:
            try:
                self.backend.terminate(handle)
            except BaseException as exc:  # keep terminating the rest
                errors.append(exc)
        # Belt and braces: anything this factory created that is still running.
        try:
            for handle in self.backend.find({"nbgpu.test.factory": self.factory_tag}):
                self.backend.terminate(handle)
        except BaseException as exc:
            errors.append(exc)
        self.created.clear()
        if errors and not isinstance(errors[0], NotImplementedError):
            raise errors[0]


@pytest.fixture(scope="module")
def module_runtimes(backend: Backend, spec_dir: Path, run_tag: str) -> Iterator[RuntimeFactory]:
    factory = RuntimeFactory(backend, spec_dir, run_tag)
    try:
        yield factory
    finally:
        factory.cleanup()


@pytest.fixture
def runtimes(backend: Backend, spec_dir: Path, run_tag: str) -> Iterator[RuntimeFactory]:
    """Per-test runtimes, for tests that need a fresh or terminated runtime."""
    factory = RuntimeFactory(backend, spec_dir, run_tag)
    try:
        yield factory
    finally:
        factory.cleanup()


@pytest.fixture(scope="module")
def handle(module_runtimes: RuntimeFactory) -> RuntimeHandle:
    """One runtime shared by the tests in a module (saves Modal cost). Tests
    using it work in their own subdirectory so they do not see each other."""
    return module_runtimes(tags={"nbgpu.test.role": "shared"})


@pytest.fixture
def workdir() -> str:
    """A fresh relative directory name for this test inside the shared runtime."""
    return f"t-{uuid.uuid4().hex[:8]}"


def wait_for(predicate: Callable[[], object], timeout_s: float, interval_s: float = 0.2, what: str = "condition") -> object:
    deadline = time.monotonic() + timeout_s
    last: object = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval_s)
    raise AssertionError(f"timed out after {timeout_s}s waiting for {what} (last value: {last!r})")
