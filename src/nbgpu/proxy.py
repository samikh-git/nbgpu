"""A localhost proxy in front of a remote marimo server.

marimo's frontend is ~250 static files (~5 MB, served uncompressed). Through
a provider tunnel they take seconds to load, and since every session gets a
new tunnel hostname the browser cache never helps. This proxy serves
`/assets/*` from the marimo installed on the laptop (same version as the
runtime's) and forwards everything else (the page, the API, websockets) to
the runtime. Served from a stable http://127.0.0.1:<port> origin, the assets
are cached across sessions and load in milliseconds.

Assets are served locally only once the page marimo sends (which carries its
version in `data-version`) matches the local marimo version; until then, and
on a mismatch, they are forwarded too. Authentication is untouched: marimo
still checks the token, so the proxy adds no access a local process or web
page would not have without it.

It also watches marimo's package requests (`/api/packages/add`,
`/api/packages/remove`, `/api/kernel/install_missing_packages`) and reports
what the user installed or removed to `on_packages(added, removed)`, so nbgpu
can record it in the notebook's header on the laptop.

The proxy runs in a background thread with its own event loop
(`LocalProxy.start()` / `.stop()`), inside the CLI process that runs the sync
loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import re
import socket
import threading
import urllib.parse
from collections.abc import Callable, Iterable
from pathlib import Path

DEFAULT_PORTS = range(2718, 2738)  # a stable port keeps the browser cache warm

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "transfer-encoding", "upgrade", "host", "content-length",
})
_VERSION_RE = re.compile(rb'data-version="([^"]+)"')
_ASSET_CACHE = "public, max-age=31536000, immutable"  # asset names are content-hashed


def local_marimo() -> tuple[str, Path] | None:
    """(version, static assets dir) of the marimo installed here, or None."""
    try:
        import marimo
    except ImportError:
        return None
    assets = Path(marimo.__file__).parent / "_static" / "assets"
    return (marimo.__version__, assets) if assets.is_dir() else None


def pick_port(ports: Iterable[int] = DEFAULT_PORTS, host: str = "127.0.0.1") -> int:
    """The first free port of `ports`, else any free port."""
    for port in ports:
        with socket.socket() as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _cookie_for_localhost(value: str) -> str:
    """Drop Domain (it names the tunnel host) and Secure (the proxy is plain
    http on localhost) from a Set-Cookie value."""
    parts = [p for p in value.split(";") if p.strip().split("=", 1)[0].strip().lower() not in ("domain", "secure")]
    return ";".join(parts)


def package_operation(method: str, path: str) -> str | None:
    """"add", "remove" or "install" for marimo's package requests, else None."""
    if method != "POST":
        return None
    if path.endswith("/api/packages/add"):
        return "add"
    if path.endswith("/api/packages/remove"):
        return "remove"
    if path.endswith("/install_missing_packages"):
        return "install"
    return None


def parse_package_operation(op: str, request_body: bytes, response_body: bytes) -> tuple[list[str], list[str]]:
    """(added, removed) requirement strings from a package request and its
    response; ([], []) when it failed or cannot be read."""
    try:
        req = json.loads(request_body or b"{}")
        resp = json.loads(response_body or b"{}")
    except ValueError:
        return [], []
    if not isinstance(req, dict) or not isinstance(resp, dict) or resp.get("success") is False:
        return [], []
    if op in ("add", "remove"):
        package = req.get("package")
        if not isinstance(package, str) or not package.strip() or req.get("group"):
            return [], []
        package = package.strip()
        if any(package.startswith(p) for p in ("/", ".", "git+", "http://", "https://", "file:")):
            return [], []  # local paths and URLs do not belong in a portable header
        return ([package], []) if op == "add" else ([], [package])
    if req.get("source") == "server":
        return [], []
    versions = req.get("versions")
    if not isinstance(versions, dict):
        return [], []
    added = []
    for name, version in versions.items():
        if not isinstance(name, str) or not name.strip():
            continue
        version = version.strip() if isinstance(version, str) else ""
        if not version:
            added.append(name.strip())
        elif version[0] in "=<>~!":
            added.append(name.strip() + version)
        else:
            added.append(f"{name.strip()}=={version}")
    return added, []


class LocalProxy:
    def __init__(
        self,
        remote_url: str,
        *,
        port: int | None = None,
        host: str = "127.0.0.1",
        on_packages: Callable[[list[str], list[str]], None] | None = None,
    ) -> None:
        self.remote = remote_url.rstrip("/")
        self._on_packages = on_packages
        parts = urllib.parse.urlsplit(self.remote)
        self._remote_origin = f"{parts.scheme}://{parts.netloc}"
        self._ws_base = ("wss" if parts.scheme == "https" else "ws") + "://" + parts.netloc
        self.host = host
        self.port = port or pick_port(host=host)
        local = local_marimo()
        self._local_version, self._assets = local if local else (None, None)
        self._remote_version: bytes | None = None
        self.assets_served = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._runner = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_with_token(self, token: str) -> str:
        return f"{self.url}/?access_token={urllib.parse.quote(token)}"

    @property
    def serving_assets(self) -> bool:
        return (
            self._assets is not None
            and self._remote_version is not None
            and self._remote_version.decode(errors="replace") == self._local_version
        )

    # -- lifecycle ------------------------------------------------------------

    def start(self, timeout_s: float = 10.0) -> None:
        self._thread = threading.Thread(target=self._run, name="nbgpu-proxy", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise RuntimeError("the local proxy did not start")
        if self._error is not None:
            raise RuntimeError(f"the local proxy could not start: {self._error}") from self._error

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        if self._runner is not None:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._runner.cleanup(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(5)
        self._loop = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001 - reported by start()
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(self._session.close())
            loop.close()

    async def _serve(self) -> None:
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

        self._session = ClientSession(
            auto_decompress=False,
            timeout=ClientTimeout(total=None, sock_connect=30),
            connector=TCPConnector(limit=32),
            cookie_jar=None,
        )
        app = web.Application(client_max_size=1024**3)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None, handle_signals=False)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()

    # -- handlers -------------------------------------------------------------

    async def _handle(self, request):  # type: ignore[no-untyped-def]
        from aiohttp import web

        if request.headers.get("upgrade", "").lower() == "websocket":
            return await self._websocket(request)
        if request.method in ("GET", "HEAD") and request.path.startswith("/assets/") and self.serving_assets:
            local = self._asset(request.path)
            if local is not None:
                self.assets_served += 1
                ctype = mimetypes.guess_type(local.name)[0] or "application/octet-stream"
                return web.FileResponse(local, headers={"Cache-Control": _ASSET_CACHE, "Content-Type": ctype})
        return await self._forward(request)

    def _asset(self, path: str) -> Path | None:
        assert self._assets is not None
        rel = urllib.parse.unquote(path[len("/assets/"):])
        if not rel or rel.startswith("/") or any(p in ("", ".", "..") for p in rel.split("/")):
            return None
        candidate = self._assets / rel
        return candidate if candidate.is_file() else None

    def _upstream_headers(self, request) -> dict[str, str]:  # type: ignore[no-untyped-def]
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        if "Origin" in headers:
            headers["Origin"] = self._remote_origin
        if "Referer" in headers:
            headers["Referer"] = self._remote_origin + urllib.parse.urlsplit(headers["Referer"]).path
        return headers

    async def _forward(self, request):  # type: ignore[no-untyped-def]
        from aiohttp import web

        body = await request.read() if request.body_exists else None
        async with self._session.request(
            request.method,
            self.remote + request.path_qs,
            headers=self._upstream_headers(request),
            data=body,
            allow_redirects=False,
        ) as upstream:
            response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
            for key, value in upstream.headers.items():
                lower = key.lower()
                if lower in _HOP_BY_HOP:
                    continue
                if lower == "set-cookie":
                    value = _cookie_for_localhost(value)
                elif lower == "location" and value.startswith(self._remote_origin):
                    value = value[len(self._remote_origin):] or "/"
                response.headers.add(key, value)
            package_op = package_operation(request.method, request.path)
            if package_op is not None and upstream.status == 200 and self._on_packages is not None:
                data = await upstream.read()
                added, removed = parse_package_operation(package_op, body or b"", data)
                if added or removed:
                    try:
                        self._on_packages(added, removed)
                    except Exception:  # noqa: BLE001 - recording must never break the UI
                        pass
                response.content_length = len(data)
                await response.prepare(request)
                await response.write(data)
                return response
            is_html = upstream.headers.get("content-type", "").startswith("text/html")
            if is_html and self._remote_version is None and "content-encoding" not in upstream.headers:
                data = await upstream.read()
                match = _VERSION_RE.search(data)
                if match:
                    self._remote_version = match.group(1)
                response.content_length = len(data)
                await response.prepare(request)
                await response.write(data)
                return response
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            return response

    async def _websocket(self, request):  # type: ignore[no-untyped-def]
        from aiohttp import WSMsgType, web

        headers = {
            k: v for k, v in self._upstream_headers(request).items()
            if not k.lower().startswith("sec-websocket")
        }
        protocols = [p.strip() for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if p.strip()]
        try:
            upstream = await self._session.ws_connect(
                self._ws_base + request.path_qs, headers=headers, protocols=protocols, max_msg_size=0,
                autoping=True, heartbeat=None,
            )
        except Exception:  # noqa: BLE001 - tell the browser, which retries
            return web.Response(status=502, text="nbgpu proxy: could not reach the runtime")
        client = web.WebSocketResponse(protocols=protocols, max_msg_size=0, autoping=True)
        await client.prepare(request)

        async def pipe(src, dst) -> None:  # type: ignore[no-untyped-def]
            async for msg in src:
                if msg.type == WSMsgType.TEXT:
                    await dst.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await dst.send_bytes(msg.data)
                else:
                    break

        tasks = [asyncio.ensure_future(pipe(client, upstream)), asyncio.ensure_future(pipe(upstream, client))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await upstream.close()
            await client.close()
        return client
