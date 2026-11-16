"""Talking to the live marimo kernel of a session (`nbgpu exec`).

marimo 0.25 lets an authenticated client run code in a session's kernel:
`POST /api/kernel/execute` with `{"code": ...}` and the header
`Marimo-Stable-Session-Id` (ids from `GET /api/sessions`) runs the code in
the scratchpad and streams server-sent events: `stdout` and `stderr`
({"data"}) as the code prints, then `done` ({"success", "output"}). Closing
the connection interrupts the kernel. nbgpu calls it from the laptop over
the session's URL with `Authorization: Bearer <token>`.

The code sent is nbgpu/kernel_runner.py's source plus one call; see that
module for what runs in the kernel and the result line it prints.

`marimo edit` has no kernel until a client connects (probed 2026-11-15):
ensure_kernel() opens marimo's websocket once (`/ws?session_id=...`),
waits for `kernel-ready` and disconnects. The session outlives that
connection, and a browser opened later resumes it as the editor. A client
that stayed connected would make that browser read-only, so nothing stays
connected. A kernel created this way has run no cells.

Errors are KernelError with a message fit for one line.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from nbgpu.kernel_runner import RESULT_MARK

KERNEL_READY_TIMEOUT_S = 120.0
REQUEST_TIMEOUT_S = 30.0  # connect and non-streaming requests; execute streams without a limit
STABLE_SESSION_HEADER = "Marimo-Stable-Session-Id"


class KernelError(Exception):
    """The kernel could not be reached or asked."""


RELOAD_WAIT_S = 15.0  # for marimo's file watcher to load a pushed notebook
RELOAD_POLL_S = 0.2


@dataclass
class ExecOutcome:
    ok: bool
    value: str | None  # repr() of the last expression, if any
    error: str | None  # one line, when not ok
    stdout: str  # with the result line removed
    stderr: str
    cells: list[dict[str, object]] = field(default_factory=list)
    pending: bool = False  # the kernel had not loaded the pushed notebook yet
    ran_notebook: bool = False  # no cell had run, so every cell ran first ("prime")
    notebook_error: str | None = None  # cells that failed in that first run


def runner_source() -> str:
    return importlib.resources.files("nbgpu").joinpath("kernel_runner.py").read_text(encoding="utf-8")


def build_code(params: dict[str, object]) -> str:
    """The scratchpad code for one call: the runner, then its entry point."""
    return f"{runner_source()}\n\nawait _nbgpu_main({json.dumps(json.dumps(params))})\n"


_RESULT_RE = re.compile(re.escape(RESULT_MARK) + r"([^\n]*)\n?")


def strip_results(text: str) -> tuple[str, dict[str, object] | None]:
    """`text` without the runner's result lines (which may follow output
    printed without a newline), and the last result that parses."""
    result: dict[str, object] | None = None
    for match in _RESULT_RE.finditer(text):
        try:
            parsed = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(parsed, dict):
            result = parsed
    return _RESULT_RE.sub("", text), result


def parse_result(stdout: str, stderr: str) -> ExecOutcome:
    """Split the runner's result line from the output. A missing result line
    means the runner itself did not finish (killed, interrupted, or a marimo
    without the APIs it needs)."""
    out, result = strip_results(stdout)
    if result is None:
        last = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), "")
        return ExecOutcome(False, None, last or "the kernel did not finish the request", out, stderr)
    cells = result.get("cells")
    return ExecOutcome(
        ok=bool(result.get("ok")),
        value=result.get("value") if isinstance(result.get("value"), str) else None,
        error=result.get("error") if isinstance(result.get("error"), str) else None,
        stdout=out,
        stderr=stderr,
        cells=[c for c in cells if isinstance(c, dict)] if isinstance(cells, list) else [],
        pending=result.get("pending") is True,
        ran_notebook=result.get("ran_notebook") is True,
        notebook_error=result.get("notebook_error") if isinstance(result.get("notebook_error"), str) else None,
    )


class KernelClient:
    """marimo's HTTP API for one session (base URL without the token)."""

    def __init__(self, url: str, token: str) -> None:
        parts = urllib.parse.urlsplit(url)
        self.base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
        self.token = token

    # -- plain requests ---------------------------------------------------

    def _open(self, path: str, *, data: bytes | None = None, headers: dict[str, str] | None = None,
              timeout: float | None = REQUEST_TIMEOUT_S):  # type: ignore[no-untyped-def]
        request = urllib.request.Request(
            self.base + path, data=data, method="POST" if data is not None else "GET",
            headers={"Authorization": f"Bearer {self.token}", **(headers or {})},
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace").strip()
            except OSError:
                pass
            if exc.code in (401, 403):
                raise KernelError("marimo refused the session's token") from None
            if exc.code == 404 and path.startswith("/api/kernel/execute"):
                raise KernelError(
                    "this marimo has no kernel execute API (`nbgpu exec` needs marimo 0.25 or newer); "
                    "raise marimo's version in the notebook's dependencies"
                ) from None
            raise KernelError(f"marimo answered {exc.code} to {path}: {detail[:200] or exc.reason}") from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise KernelError(f"could not reach marimo at {self.base}: {reason}") from None

    def sessions(self) -> dict[str, dict[str, object]]:
        """marimo's sessions: {connection id: {"session_id" (stable), "filename", "path"}}."""
        with self._open("/api/sessions") as resp:
            try:
                data = json.loads(resp.read().decode("utf-8"))
            except ValueError as exc:
                raise KernelError("marimo's session list was not JSON") from exc
        if not isinstance(data, dict):
            raise KernelError("marimo's session list has an unexpected shape")
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}

    def stable_session_id(self, notebook_suffix: str | None = None) -> str | None:
        """The kernel's stable session id, or None if there is no kernel.
        With several (not expected: nbgpu serves one notebook), the one whose
        path ends with `notebook_suffix`."""
        found = [s for s in self.sessions().values() if isinstance(s.get("session_id"), str)]
        if notebook_suffix and len(found) > 1:
            matching = [s for s in found if str(s.get("path") or "").endswith(notebook_suffix)]
            found = matching or found
        return str(found[0]["session_id"]) if found else None

    # -- the kernel -------------------------------------------------------

    def ensure_kernel(self, notebook_suffix: str | None = None) -> tuple[str, bool]:
        """(stable session id, created): connect once to create the kernel
        when there is none."""
        existing = self.stable_session_id(notebook_suffix)
        if existing is not None:
            return existing, False
        asyncio.run(self._connect_once())
        created = self.stable_session_id(notebook_suffix)
        if created is None:
            raise KernelError("marimo started a kernel but does not list it")
        return created, True

    async def _connect_once(self) -> None:
        import aiohttp

        parts = urllib.parse.urlsplit(self.base)
        scheme = "wss" if parts.scheme == "https" else "ws"
        query = urllib.parse.urlencode({"session_id": f"nbgpu-{secrets.token_hex(6)}", "access_token": self.token})
        url = f"{scheme}://{parts.netloc}{parts.path}/ws?{query}"
        timeout = aiohttp.ClientTimeout(total=KERNEL_READY_TIMEOUT_S)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as http, http.ws_connect(
                url, headers={"Authorization": f"Bearer {self.token}"}, max_msg_size=0
            ) as ws:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break
                    try:
                        op = json.loads(msg.data).get("op")
                    except (ValueError, AttributeError):
                        continue
                    if op == "kernel-ready":
                        return
                raise KernelError(f"marimo closed the connection before the kernel was ready ({ws.close_code})")
        except TimeoutError:
            raise KernelError(f"the kernel was not ready after {KERNEL_READY_TIMEOUT_S:g}s") from None
        except aiohttp.ClientError as exc:
            raise KernelError(f"could not connect to marimo's websocket: {exc}") from None

    def execute(self, session_id: str, code: str,
                on_event: Callable[[str, str], None] | None = None) -> tuple[str, str, bool]:
        """Run `code` in the scratchpad: (stdout, stderr, finished). Events
        are passed to `on_event(channel, text)` as they arrive. finished is
        False when the stream ended without marimo's `done` event."""
        with self._open(
            "/api/kernel/execute",
            data=json.dumps({"code": code}).encode("utf-8"),
            headers={"Content-Type": "application/json", STABLE_SESSION_HEADER: session_id,
                     "Accept": "text/event-stream"},
            timeout=None,
        ) as resp:
            stdout: list[str] = []
            stderr: list[str] = []
            for event, data in iter_sse(resp):
                try:
                    payload = json.loads(data)
                except ValueError:
                    continue
                if event in ("stdout", "stderr") and isinstance(payload, dict):
                    text = str(payload.get("data") or "")
                    (stdout if event == "stdout" else stderr).append(text)
                    if on_event is not None:
                        on_event(event, text)
                elif event == "done":
                    return "".join(stdout), "".join(stderr), True
            return "".join(stdout), "".join(stderr), False


def iter_sse(lines):  # type: ignore[no-untyped-def]
    """(event, data) pairs from a server-sent event stream of byte lines."""
    event = "message"
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" ") if line[5:6] == " " else line[5:])
    if data:
        yield event, "\n".join(data)


def run_in_kernel(client: KernelClient, session_id: str, params: dict[str, object],
                  on_event: Callable[[str, str], None] | None = None) -> ExecOutcome:
    """One request. With params["wait_for"], repeat it while the kernel has
    not loaded the pushed notebook (up to RELOAD_WAIT_S, read at call time)."""
    deadline = time.monotonic() + RELOAD_WAIT_S
    while True:
        stdout, stderr, finished = client.execute(session_id, build_code(params), on_event)
        outcome = parse_result(stdout, stderr)
        if not outcome.pending:
            break
        if time.monotonic() >= deadline:
            outcome.error = (
                f"the kernel did not load the pushed notebook within {RELOAD_WAIT_S:g}s (marimo's file "
                "watcher reloads it); check that the notebook parses, then run again"
            )
            return outcome
        time.sleep(RELOAD_POLL_S)
    if not finished and outcome.ok:
        outcome.ok = False
        outcome.error = "the connection to marimo ended before the kernel finished"
    return outcome
