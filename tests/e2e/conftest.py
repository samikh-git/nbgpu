"""Fixtures for end-to-end tests: real marimo through the local backend.

Every runtime and marimo process a test starts is cleaned up in teardown,
even when the test (or the code under test) fails: sessions are force-stopped,
any runtime left in the test's LocalBackend is terminated, and as a last
resort any `marimo edit --watch` process with a token file that appeared
during the test is killed.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

READY_TIMEOUT_S = 120.0

NOTEBOOK_TEXT = '''\
# /// script
# [tool.nbgpu]
# provider = "local"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    x = 1
    print("hello-from-notebook")
    return


@app.cell
def _():
    y = 2
    return


if __name__ == "__main__":
    app.run()
'''

FAILING_NOTEBOOK_TEXT = '''\
# /// script
# [tool.nbgpu]
# provider = "local"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    for i in range(5):
        print(f"line-{i:04d}")
    raise ValueError("notebook-failed")
    return


if __name__ == "__main__":
    app.run()
'''

CHATTY_NOTEBOOK_TEXT = '''\
# /// script
# [tool.nbgpu]
# provider = "local"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    for i in range(1000):
        print(f"line-{i:04d}")
    return


if __name__ == "__main__":
    app.run()
'''


def wait_for(predicate: Callable[[], object], timeout_s: float, interval_s: float = 0.25, what: str = "condition") -> object:
    deadline = time.monotonic() + timeout_s
    last: object = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval_s)
    raise AssertionError(f"timed out after {timeout_s}s waiting for {what} (last value: {last!r})")


# --------------------------------------------------------------------------
# HTTP helpers


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


def http_get(url: str, *, follow_redirects: bool = False, headers: dict[str, str] | None = None, timeout_s: float = 10) -> HttpResponse:
    handlers: list[urllib.request.BaseHandler] = [urllib.request.HTTPCookieProcessor()]
    if not follow_redirects:
        handlers.append(_NoRedirect())
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with opener.open(request, timeout=timeout_s) as resp:
            return HttpResponse(resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read())


def server_answers(url: str) -> bool:
    try:
        return http_get(url + "/health", timeout_s=3).status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


# --------------------------------------------------------------------------
# Driving the marimo UI like the browser / an agent does


class MarimoUI:
    """Opens a notebook session on a running marimo server the way the
    browser does (websocket /ws with the access token), then drives it with
    `marimo pair`, whose code-mode edits are saved to the notebook file by
    the server. This is the real path by which UI edits reach disk."""

    def __init__(self, base_url: str, token: str, token_file: Path) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.token_file = token_file
        token_file.write_text(token)
        token_file.chmod(0o600)
        self.client_session_id = f"nbgpu-test-{uuid.uuid4().hex[:8]}"
        self.messages: list[dict] = []
        self._ws = None
        self._reader: threading.Thread | None = None

    def connect(self, timeout_s: float = 60) -> None:
        from websockets.sync.client import connect

        parsed = urllib.parse.urlsplit(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        query = urllib.parse.urlencode({"access_token": self.token, "session_id": self.client_session_id})
        ws_url = urllib.parse.urlunsplit((scheme, parsed.netloc, parsed.path.rstrip("/") + "/ws", query, ""))
        self._stack = contextlib.ExitStack()
        self._ws = self._stack.enter_context(connect(ws_url, open_timeout=timeout_s, max_size=None))

        def read() -> None:
            try:
                for raw in self._ws:  # type: ignore[union-attr]
                    try:
                        self.messages.append(json.loads(raw))
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass

        self._reader = threading.Thread(target=read, daemon=True)
        self._reader.start()
        wait_for(lambda: any(m.get("op") == "kernel-ready" for m in self.messages), timeout_s, what="kernel-ready")

    def close(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._stack.close()
            self._ws = None

    def _pair(self, *args: str, input_text: str | None = None, timeout_s: float = 60) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", "marimo", "pair", *args],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        assert proc.returncode == 0, f"marimo pair {args[:2]} failed: {proc.stdout}\n{proc.stderr}"
        return json.loads(proc.stdout)

    def pair_session_id(self, timeout_s: float = 30) -> str:
        """The stable session id `marimo pair` uses for our browser session."""

        def find() -> str | None:
            data = self._pair("notebook", "list", "--url", self.base_url, "--token-file", str(self.token_file))
            for notebook in data.get("notebooks", []):
                for s in notebook.get("sessions", []):
                    if s.get("id") == self.client_session_id:
                        return s["session_id"]
            return None

        return str(wait_for(find, timeout_s, interval_s=0.5, what="pair session"))

    def execute(self, code: str) -> dict:
        result = self._pair(
            "execute",
            "--url", self.base_url,
            "--token-file", str(self.token_file),
            "--session", self.pair_session_id(),
            "--code-file", "-",
            input_text=code,
        )
        assert result.get("success"), result
        return result

    def cell_codes(self) -> list[str]:
        code = (
            "import json\n"
            "import marimo._code_mode as cm\n"
            "async with cm.get_context() as ctx:\n"
            "    print('CODES=' + json.dumps([c.code for c in ctx.cells]))\n"
        )
        out = self.execute(code)["stdout"]
        line = next(line for line in out.splitlines() if line.startswith("CODES="))
        return json.loads(line[len("CODES="):])

    def create_cell(self, source: str) -> None:
        code = (
            "import marimo._code_mode as cm\n"
            "async with cm.get_context() as ctx:\n"
            f"    cid = ctx.create_cell({source!r})\n"
            "    ctx.run_cell(cid)\n"
        )
        self.execute(code)


# --------------------------------------------------------------------------
# process safety net


def _marimo_server_pids(roots: tuple[str, ...]) -> set[int]:
    """marimo servers (and nbgpu watchdogs) whose token / marker file lies
    under one of `roots`, so concurrent test runs never kill each other's."""
    import psutil

    pids: set[int] = set()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
        except Exception:
            continue
        joined = " ".join(cmd)
        if "marimo" in joined and " edit " in f" {joined} " and "--watch" in cmd and "--token-password-file" in cmd:
            target = cmd[cmd.index("--token-password-file") + 1] if cmd.index("--token-password-file") + 1 < len(cmd) else ""
        elif any(part.endswith("nbgpu-watchdog.py") for part in cmd) and "--marker" in cmd:
            target = cmd[cmd.index("--marker") + 1] if cmd.index("--marker") + 1 < len(cmd) else ""
        else:
            continue
        resolved = os.path.realpath(target)
        if any(resolved.startswith(root + os.sep) for root in roots):
            pids.add(proc.info["pid"])
    return pids


def marimo_pids_on_port(port: int) -> set[int]:
    import psutil

    pids: set[int] = set()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = proc.info.get("cmdline") or []
        if any("marimo" in part for part in cmd) and "edit" in cmd and str(port) in cmd:
            pids.add(proc.info["pid"])
    return pids


def pid_alive(pid: int) -> bool:
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.fixture(autouse=True)
def _kill_leaked_marimo(tmp_path: Path) -> Iterator[None]:
    roots = tuple(
        os.path.realpath(p) for p in (str(tmp_path), os.environ.get("NBGPU_STATE_DIR", "")) if p
    )
    before = _marimo_server_pids(roots)
    yield
    leaked = _marimo_server_pids(roots) - before
    for pid in leaked:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


# --------------------------------------------------------------------------
# project, backend, store, session


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "e2e-proj"\nversion = "0"\n')
    (root / "notebook.py").write_text(NOTEBOOK_TEXT)
    (root / "helper.py").write_text("VALUE = 1\n")
    (root / "data").mkdir()
    (root / "data" / "input.csv").write_text("a,b\n1,2\n")
    # Excluded from sync.
    (root / ".venv").mkdir()
    (root / ".venv" / "ignored.py").write_text("ignored = True\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "helper.cpython-312.pyc").write_bytes(b"\x00\x01")
    return root


@pytest.fixture
def notebook(project: Path) -> Path:
    return project / "notebook.py"


@pytest.fixture
def local_backend(tmp_path: Path) -> Iterator:
    from nbgpu.backends.local import LocalBackend

    backend = LocalBackend(base_dir=tmp_path / "runtimes")
    yield backend
    with contextlib.suppress(Exception):
        for handle in backend.find({}):
            with contextlib.suppress(Exception):
                backend.terminate(handle)


@pytest.fixture
def store(tmp_path: Path):
    from nbgpu.session import SessionStore

    return SessionStore(tmp_path / "state")


@pytest.fixture
def start_session(local_backend, store) -> Iterator[Callable]:
    """Starts sessions with the real lifecycle and force-stops them afterwards."""
    from nbgpu import lifecycle

    started: list = []

    def start(notebook: Path, **kwargs):
        kwargs.setdefault("ready_timeout_s", READY_TIMEOUT_S)
        session = lifecycle.start(notebook, backend=local_backend, store=store, **kwargs)
        started.append(session)
        return session

    yield start
    for session in started:
        with contextlib.suppress(Exception):
            if store.load(session.session_id) is not None:
                lifecycle.stop(session, backend=local_backend, store=store, force=True)
        with contextlib.suppress(Exception):
            handle = local_backend.get(session.runtime_id)
            if handle is not None:
                local_backend.terminate(handle)


@pytest.fixture
def session(start_session, notebook: Path):
    return start_session(notebook)


@pytest.fixture
def marimo_ui(session, tmp_path: Path) -> Iterator[MarimoUI]:
    ui = MarimoUI(session.url, session.token, tmp_path / "pair-token")
    ui.connect()
    yield ui
    ui.close()
