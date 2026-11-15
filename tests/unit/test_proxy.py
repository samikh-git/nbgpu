"""nbgpu.proxy against a small fake marimo upstream."""

from __future__ import annotations

import asyncio
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest
from aiohttp import WSMsgType, web

from nbgpu import proxy as proxy_mod
from nbgpu.proxy import LocalProxy, _cookie_for_localhost, local_marimo


class Upstream:
    """A fake marimo in a background thread: /, /assets/*, /redirect, /ws."""

    def __init__(self, version: str) -> None:
        self.version = version
        self.asset_hits = 0
        self.seen_origin: str | None = None
        self.loop = asyncio.new_event_loop()
        self.port = 0
        ready = threading.Event()

        async def index(request: web.Request) -> web.Response:
            resp = web.Response(
                text=f'<html><marimo-version data-version="{self.version}"></marimo-version></html>',
                content_type="text/html",
            )
            resp.headers["Set-Cookie"] = "session=abc; Domain=remote.example; Secure; HttpOnly; Path=/"
            return resp

        async def asset(request: web.Request) -> web.Response:
            self.asset_hits += 1
            return web.Response(text="upstream asset", content_type="application/javascript")

        async def redirect(request: web.Request) -> web.Response:
            raise web.HTTPSeeOther(f"http://127.0.0.1:{self.port}/")

        async def packages_add(request: web.Request) -> web.Response:
            body = await request.json()
            return web.json_response({"success": body.get("package") != "broken"})

        async def install_missing(request: web.Request) -> web.Response:
            return web.json_response({"success": True})

        async def echo(request: web.Request) -> web.WebSocketResponse:
            self.seen_origin = request.headers.get("Origin")
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await ws.send_str("echo:" + msg.data)
            return ws

        async def serve() -> None:
            app = web.Application()
            app.router.add_get("/", index)
            app.router.add_get("/assets/{name}", asset)
            app.router.add_get("/redirect", redirect)
            app.router.add_get("/ws", echo)
            app.router.add_post("/api/packages/add", packages_add)
            app.router.add_post("/api/packages/remove", install_missing)
            app.router.add_post("/api/kernel/install_missing_packages", install_missing)
            self.runner = web.AppRunner(app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, "127.0.0.1", 0)
            await site.start()
            self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
            ready.set()

        def run() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(serve())
            self.loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        ready.wait(5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)


def _get(url: str) -> tuple[int, dict[str, str], bytes]:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(url, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


@pytest.fixture
def asset_name() -> str:
    local = local_marimo()
    assert local is not None, "marimo is a dependency of nbgpu"
    return next(p.name for p in local[1].iterdir() if p.suffix == ".js")


@pytest.fixture
def pair(request: pytest.FixtureRequest) -> Iterator[tuple[Upstream, LocalProxy]]:
    version = getattr(request, "param", None) or local_marimo()[0]  # type: ignore[index]
    upstream = Upstream(version)
    events: list = []
    proxy = LocalProxy(upstream.url, port=proxy_mod.pick_port(range(0)), on_packages=lambda a, r: events.append((a, r)))
    proxy.events = events  # type: ignore[attr-defined]
    proxy.start()
    try:
        yield upstream, proxy
    finally:
        proxy.stop()
        upstream.close()


def test_assets_are_forwarded_until_the_page_shows_a_matching_version(pair, asset_name: str) -> None:
    upstream, proxy = pair
    status, _, body = _get(f"{proxy.url}/assets/{asset_name}")
    assert (status, body, upstream.asset_hits) == (200, b"upstream asset", 1)  # version not known yet

    status, headers, body = _get(proxy.url + "/")
    assert status == 200 and b"data-version" in body
    status, headers, body = _get(f"{proxy.url}/assets/{asset_name}")
    assert status == 200 and body != b"upstream asset"
    assert upstream.asset_hits == 1  # served from the local marimo
    assert "immutable" in headers["Cache-Control"]
    assert proxy.assets_served == 1


@pytest.mark.parametrize("pair", ["0.0.1-other"], indirect=True)
def test_a_version_mismatch_forwards_every_asset(pair, asset_name: str) -> None:
    upstream, proxy = pair
    _get(proxy.url + "/")
    _, _, body = _get(f"{proxy.url}/assets/{asset_name}")
    assert body == b"upstream asset" and proxy.assets_served == 0


def test_unsafe_asset_paths_are_never_served_locally(pair) -> None:
    upstream, proxy = pair
    _get(proxy.url + "/")
    for path in ("/assets/../index.html", "/assets/%2e%2e/index.html", "/assets/"):
        _get(proxy.url + path)
    assert proxy.assets_served == 0


def test_cookies_and_redirects_are_rewritten_for_localhost(pair) -> None:
    upstream, proxy = pair
    _, headers, _ = _get(proxy.url + "/")
    cookie = headers["Set-Cookie"]
    assert "session=abc" in cookie and "Domain" not in cookie and "Secure" not in cookie and "HttpOnly" in cookie
    status, headers, _ = _get(proxy.url + "/redirect")
    assert status == 303 and headers["Location"] == "/"


def test_websockets_are_relayed_with_the_remote_origin(pair) -> None:
    upstream, proxy = pair

    async def talk() -> str:
        import aiohttp

        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(proxy.url.replace("http", "ws") + "/ws", origin=proxy.url) as ws:
                await ws.send_str("hi")
                msg = await ws.receive(timeout=5)
                return msg.data

    assert asyncio.run(talk()) == "echo:hi"
    assert upstream.seen_origin == upstream.url


def test_cookie_rewrite_keeps_other_attributes() -> None:
    assert _cookie_for_localhost("a=b; Path=/; secure; domain=x.y; SameSite=Lax") == "a=b; Path=/; SameSite=Lax"


def test_pick_port_prefers_the_given_ports() -> None:
    port = proxy_mod.pick_port(range(0))  # none given: any free port
    assert port > 0
    assert proxy_mod.pick_port([port]) == port


def _post(url: str, body: dict) -> int:
    import json

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


def test_package_requests_are_reported(pair) -> None:
    upstream, proxy = pair
    _post(proxy.url + "/api/kernel/install_missing_packages", {"manager": "pip", "versions": {"emoji": "", "polars": "1.2.0"}})
    _post(proxy.url + "/api/packages/add", {"package": "httpx>=0.27"})
    _post(proxy.url + "/api/packages/add", {"package": "broken"})  # marimo reports failure: not recorded
    _post(proxy.url + "/api/packages/add", {"package": "git+https://example.com/x"})  # not portable
    _post(proxy.url + "/api/packages/remove", {"package": "httpx"})
    assert proxy.events == [(["emoji", "polars==1.2.0"], []), (["httpx>=0.27"], []), ([], ["httpx"])]
