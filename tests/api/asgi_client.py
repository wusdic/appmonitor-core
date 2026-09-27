"""A test client for the FastAPI app.

FastAPI's TestClient is used when it can be imported; starlette >= 1.x makes
it depend on the optional `httpx2` package, which this environment may not
ship, so a minimal in-process ASGI client with the same small surface
(get / post, .status_code / .json() / .text / .headers) is the fallback.
Neither variant runs the app's startup hook, so no Runtime is built behind
the test's back (the tests attach their own warmed Runtime).
"""
from __future__ import annotations

import asyncio
import json as _json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit


class _Response:
    def __init__(self, status: int, headers: List[Tuple[bytes, bytes]], body: bytes) -> None:
        self.status_code = status
        self.headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in headers}
        self.content = body

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return _json.loads(self.content)


class MiniASGIClient:
    def __init__(self, app: Any) -> None:
        self.app = app

    def request(self, method: str, url: str, json: Any = None,
                headers: Optional[Dict[str, str]] = None) -> _Response:
        parts = urlsplit(url)
        body = b""
        hdrs = [(b"host", b"testserver")]
        if json is not None:
            body = _json.dumps(json).encode()
            hdrs.append((b"content-type", b"application/json"))
        hdrs.append((b"content-length", str(len(body)).encode()))
        for k, v in (headers or {}).items():
            hdrs.append((k.lower().encode(), v.encode()))
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": method.upper(), "scheme": "http", "path": parts.path or "/",
                 "raw_path": (parts.path or "/").encode(), "root_path": "",
                 "query_string": parts.query.encode(), "headers": hdrs,
                 "client": ("testclient", 50000), "server": ("testserver", 80)}
        out: Dict[str, Any] = {"status": 500, "headers": [], "body": b""}
        sent = {"done": False}

        async def receive() -> Dict[str, Any]:
            if not sent["done"]:
                sent["done"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(3600)
            return {"type": "http.disconnect"}

        async def send(msg: Dict[str, Any]) -> None:
            if msg["type"] == "http.response.start":
                out["status"] = msg["status"]
                out["headers"] = list(msg.get("headers") or [])
            elif msg["type"] == "http.response.body":
                out["body"] += msg.get("body", b"")

        asyncio.run(self.app(scope, receive, send))
        return _Response(out["status"], out["headers"], out["body"])

    def get(self, url: str, **kw: Any) -> _Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json: Any = None, **kw: Any) -> _Response:
        return self.request("POST", url, json=json, **kw)


def make_client(app: Any) -> Any:
    try:
        from fastapi.testclient import TestClient
        return TestClient(app)
    except Exception:       # RuntimeError: starlette.testclient requires httpx2
        return MiniASGIClient(app)
