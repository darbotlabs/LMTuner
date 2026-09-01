# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""MCP 2.0 (2026-07-28) stateless Streamable HTTP + stdio server for LMTuner.

Spec: https://modelcontextprotocol.io/specification/2026-07-28
Transport: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http
Tasks: https://modelcontextprotocol.io/extensions/tasks/overview

Every POST is self-contained. No initialize handshake. No Mcp-Session-Id.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any
from urllib.parse import urlparse

from olive.protocols import MCP_PROTOCOL_VERSION, MCP_SPEC_URL, MCP_TASKS_URL
from olive.protocols.lmcli_tools import (
    LMCLI_TOOLS,
    TASK_STORE,
    build_lmcli_argv,
    normalize_tool_name,
    run_lmcli,
    tool_schemas,
)

logger = logging.getLogger(__name__)

HEADER_MISMATCH = -32020
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

META_PROTOCOL = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

ALLOWED_ORIGINS = {"http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost"}


def _olive_version() -> str:
    try:
        import olive

        return getattr(olive, "__version__", "unknown")
    except Exception:
        return "unknown"


def server_info() -> dict[str, str]:
    return {"name": "lmcli", "version": _olive_version()}


def _meta_from_params(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    meta = params.get("_meta") or {}
    return meta if isinstance(meta, dict) else {}


def client_supports_tasks(params: Any) -> bool:
    meta = _meta_from_params(params)
    caps = meta.get(META_CLIENT_CAPS) or {}
    if not isinstance(caps, dict):
        return False
    extensions = caps.get("extensions") or {}
    return isinstance(extensions, dict) and "io.modelcontextprotocol/tasks" in extensions


def decode_header_value(value: str | None) -> str:
    if value is None:
        return ""
    raw = value.strip()
    if raw.startswith("=?base64?") and raw.endswith("?="):
        import base64

        encoded = raw[len("=?base64?") : -2]
        return base64.b64decode(encoded).decode("utf-8")
    return raw


def header_get(headers: dict[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def jsonrpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    result.setdefault("_meta", {})
    if isinstance(result["_meta"], dict):
        result["_meta"].setdefault(META_SERVER_INFO, server_info())
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def discover_result() -> dict[str, Any]:
    return {
        "resultType": "complete",
        "supportedVersions": [MCP_PROTOCOL_VERSION],
        "capabilities": {
            "tools": {"listChanged": False},
            "extensions": {"io.modelcontextprotocol/tasks": {}},
        },
        "instructions": (
            "LMTuner MCP 2.0 server. Tools wrap the real `lmcli` CLI: "
            + ", ".join(LMCLI_TOOLS)
            + f". Spec: {MCP_SPEC_URL}. Tasks: {MCP_TASKS_URL}."
        ),
        "ttlMs": 3_600_000,
        "cacheScope": "public",
        "_meta": {META_SERVER_INFO: server_info()},
    }


def validate_http_headers(body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
    """Return a JSON-RPC error object if headers do not match the body, else None."""
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    meta = _meta_from_params(params)

    proto_header = header_get(headers, "MCP-Protocol-Version")
    proto_meta = meta.get(META_PROTOCOL)
    if not proto_header:
        return jsonrpc_error(
            req_id,
            HEADER_MISMATCH,
            "Header mismatch: MCP-Protocol-Version is required",
            {"name": "HeaderMismatch"},
        )
    if proto_meta and proto_header != proto_meta:
        return jsonrpc_error(
            req_id,
            HEADER_MISMATCH,
            f"Header mismatch: MCP-Protocol-Version header value '{proto_header}' does not match body value '{proto_meta}'",
            {"name": "HeaderMismatch"},
        )
    if proto_header != MCP_PROTOCOL_VERSION:
        return jsonrpc_error(
            req_id,
            INVALID_REQUEST,
            f"Unsupported protocol version: {proto_header}",
            {"name": "UnsupportedProtocolVersionError", "supported": [MCP_PROTOCOL_VERSION]},
        )

    method_header = header_get(headers, "Mcp-Method")
    if not method_header:
        return jsonrpc_error(
            req_id,
            HEADER_MISMATCH,
            "Header mismatch: Mcp-Method is required",
            {"name": "HeaderMismatch"},
        )
    if method and method_header != method:
        return jsonrpc_error(
            req_id,
            HEADER_MISMATCH,
            f"Header mismatch: Mcp-Method header value '{method_header}' does not match body value '{method}'",
            {"name": "HeaderMismatch"},
        )

    if method in {"tools/call", "resources/read", "prompts/get"}:
        name_header = header_get(headers, "Mcp-Name")
        expected = None
        if method == "tools/call":
            expected = params.get("name")
        elif method == "resources/read":
            expected = params.get("uri")
        else:
            expected = params.get("name")
        if not name_header:
            return jsonrpc_error(
                req_id,
                HEADER_MISMATCH,
                "Header mismatch: Mcp-Name is required",
                {"name": "HeaderMismatch"},
            )
        decoded = decode_header_value(name_header)
        if expected is not None and decoded != str(expected):
            return jsonrpc_error(
                req_id,
                HEADER_MISMATCH,
                f"Header mismatch: Mcp-Name header value '{decoded}' does not match body value '{expected}'",
                {"name": "HeaderMismatch"},
            )
    return None


def handle_rpc(body: dict[str, Any], *, transport: str = "stdio") -> dict[str, Any]:
    """Handle a single JSON-RPC request. Returns a JSON-RPC response dict."""
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") if isinstance(body.get("params"), dict) else {}

    if body.get("jsonrpc") != "2.0" or not method:
        return jsonrpc_error(req_id, INVALID_REQUEST, "Invalid JSON-RPC request")

    meta = _meta_from_params(params)
    proto = meta.get(META_PROTOCOL)
    if proto and proto != MCP_PROTOCOL_VERSION:
        return jsonrpc_error(
            req_id,
            INVALID_REQUEST,
            f"Unsupported protocol version: {proto}",
            {"name": "UnsupportedProtocolVersionError", "supported": [MCP_PROTOCOL_VERSION]},
        )

    if method == "initialize":
        # 2026-07-28 removed the handshake. Reject so legacy clients can fall back.
        return jsonrpc_error(
            req_id,
            METHOD_NOT_FOUND,
            "initialize is not used in MCP 2026-07-28; call server/discover instead",
        )

    if method == "server/discover":
        return jsonrpc_result(req_id, discover_result())

    if method == "tools/list":
        return jsonrpc_result(
            req_id,
            {
                "resultType": "complete",
                "tools": tool_schemas(),
                "ttlMs": 300_000,
                "cacheScope": "public",
            },
        )

    if method == "tools/call":
        return _handle_tools_call(req_id, params)

    if method == "tasks/get":
        task_id = params.get("taskId") or params.get("task_id")
        task = TASK_STORE.get(str(task_id)) if task_id else None
        if task is None:
            return jsonrpc_error(req_id, INVALID_PARAMS, f"Unknown taskId: {task_id}")
        snap = TASK_STORE.snapshot(task)
        snap["resultType"] = "complete"
        return jsonrpc_result(req_id, snap)

    if method == "tasks/cancel":
        task_id = params.get("taskId") or params.get("task_id")
        task = TASK_STORE.cancel(str(task_id)) if task_id else None
        if task is None:
            return jsonrpc_error(req_id, INVALID_PARAMS, f"Unknown taskId: {task_id}")
        return jsonrpc_result(req_id, {"resultType": "complete"})

    if method == "tasks/update":
        return jsonrpc_result(req_id, {"resultType": "complete"})

    if method == "ping":
        return jsonrpc_result(req_id, {"resultType": "complete"})

    return jsonrpc_error(req_id, METHOD_NOT_FOUND, f"Method not found: {method}")


def _handle_tools_call(req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name = normalize_tool_name(str(params.get("name") or ""))
    arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
    if name not in LMCLI_TOOLS:
        return jsonrpc_error(req_id, INVALID_PARAMS, f"Unknown tool: {params.get('name')}")

    if client_supports_tasks(params):
        task = TASK_STORE.create(name, arguments)
        return jsonrpc_result(
            req_id,
            {
                "resultType": "task",
                "taskId": task.task_id,
                "status": task.status,
                "pollIntervalMs": task.poll_interval_ms,
                "ttlMs": task.ttl_ms,
            },
        )

    # Dry-run path for tests: arguments["_dry_run"] returns argv without executing.
    if arguments.get("_dry_run"):
        argv = build_lmcli_argv(name, {k: v for k, v in arguments.items() if k != "_dry_run"})
        result = {"ok": True, "tool": name, "argv": argv, "dry_run": True}
    else:
        result = run_lmcli(name, arguments)

    return jsonrpc_result(
        req_id,
        {
            "resultType": "complete",
            "content": [{"type": "text", "text": json.dumps(result, default=str)[:8000]}],
            "structuredContent": result,
            "isError": not result.get("ok", False),
        },
    )


class McpHttpResult:
    """HTTP response produced by the /mcp endpoint."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes, sse_events: list[str] | None = None):
        self.status = status
        self.headers = headers
        self.body = body
        self.sse_events = sse_events or []


def handle_http_request(
    method: str,
    headers: dict[str, str],
    body_bytes: bytes,
    *,
    origin: str | None = None,
) -> McpHttpResult:
    """Handle one Streamable HTTP request against the single /mcp endpoint."""
    json_headers = {"content-type": "application/json"}
    if method in {"GET", "DELETE"}:
        return McpHttpResult(405, {"allow": "POST"}, b"", None)
    if method != "POST":
        return McpHttpResult(405, {"allow": "POST"}, b"", None)

    if origin:
        parsed = urlparse(origin)
        origin_base = f"{parsed.scheme}://{parsed.hostname}" if parsed.scheme and parsed.hostname else origin
        if parsed.port and parsed.port not in (80, 443):
            origin_base = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        allowed = origin_base in ALLOWED_ORIGINS or (parsed.hostname in {"127.0.0.1", "localhost"})
        if not allowed:
            payload = jsonrpc_error(None, INVALID_REQUEST, "Forbidden origin")
            return McpHttpResult(403, json_headers, json.dumps(payload).encode("utf-8"))

    if header_get(headers, "Mcp-Session-Id"):
        # 2026-07-28: ignore session ids; do not mint or echo them.
        logger.debug("Ignoring Mcp-Session-Id on stateless 2026-07-28 transport")

    content_type = (header_get(headers, "Content-Type") or "").split(";")[0].strip().lower()
    if content_type and content_type != "application/json":
        return McpHttpResult(415, json_headers, b'{"error":"Content-Type must be application/json"}')

    try:
        body = json.loads(body_bytes.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        payload = jsonrpc_error(None, PARSE_ERROR, "Parse error")
        return McpHttpResult(400, json_headers, json.dumps(payload).encode("utf-8"))

    if isinstance(body, list):
        payload = jsonrpc_error(None, INVALID_REQUEST, "Batch JSON-RPC is not supported")
        return McpHttpResult(400, json_headers, json.dumps(payload).encode("utf-8"))
    if not isinstance(body, dict):
        payload = jsonrpc_error(None, INVALID_REQUEST, "Invalid JSON-RPC request")
        return McpHttpResult(400, json_headers, json.dumps(payload).encode("utf-8"))

    # Notifications have no id — 202 with empty body if accepted.
    is_notification = "id" not in body
    mismatch = validate_http_headers(body, headers)
    if mismatch:
        status = 404 if mismatch["error"]["code"] == METHOD_NOT_FOUND else 400
        return McpHttpResult(status, json_headers, json.dumps(mismatch).encode("utf-8"))

    if is_notification:
        return McpHttpResult(202, {}, b"")

    rpc = handle_rpc(body, transport="http")
    status = 200
    if "error" in rpc:
        code = rpc["error"].get("code")
        if code == METHOD_NOT_FOUND:
            status = 404
        elif code in {HEADER_MISMATCH, INVALID_REQUEST, PARSE_ERROR, INVALID_PARAMS}:
            status = 400
        else:
            status = 400

    accept = header_get(headers, "Accept") or ""
    wants_sse = "text/event-stream" in accept
    wants_json = "application/json" in accept
    if accept and not wants_sse and not wants_json:
        return McpHttpResult(406, json_headers, b'{"error":"Accept must include application/json or text/event-stream"}')

    if wants_sse and not wants_json:
        event = f"data: {json.dumps(rpc)}\r\n\r\n"
        return McpHttpResult(
            200,
            {"content-type": "text/event-stream", "x-accel-buffering": "no", "cache-control": "no-cache"},
            event.encode("utf-8"),
            [event],
        )

    return McpHttpResult(status, json_headers, json.dumps(rpc).encode("utf-8"))


async def mcp_asgi_app(scope, receive, send):
    """ASGI 3.0 application exposing POST /mcp."""
    if scope["type"] != "http":
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b""})
        return

    path = scope.get("path") or "/"
    method = scope.get("method", "GET").upper()
    header_map = {k.decode("latin1"): v.decode("latin1") for k, v in scope.get("headers") or []}
    origin = header_map.get("origin")

    body = b""
    while True:
        event = await receive()
        if event["type"] == "http.request":
            body += event.get("body") or b""
            if not event.get("more_body"):
                break
        elif event["type"] == "http.disconnect":
            return

    if not path.rstrip("/").endswith("/mcp") and path not in {"/mcp", "/mcp/"}:
        await send({"type": "http.response.start", "status": 404, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"Not Found"})
        return

    result = handle_http_request(method, header_map, body, origin=origin)
    headers = [(k.encode("latin1"), v.encode("latin1")) for k, v in result.headers.items()]
    await send({"type": "http.response.start", "status": result.status, "headers": headers})
    await send({"type": "http.response.body", "body": result.body})


def serve_stdio() -> None:
    """JSON-RPC stdio loop (newline-delimited, Content-Length also accepted)."""
    stdin = sys.stdin
    while True:
        line = stdin.readline()
        if not line:
            break
        raw = line
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError:
                continue
            # consume remaining headers
            while True:
                hdr = stdin.readline()
                if hdr in ("\n", "\r\n", ""):
                    break
            raw = stdin.read(length)
        raw = raw.strip()
        if not raw:
            continue
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(jsonrpc_error(None, PARSE_ERROR, "Parse error")) + "\n")
            sys.stdout.flush()
            continue
        if isinstance(body, dict) and "id" not in body:
            continue
        response = handle_rpc(body if isinstance(body, dict) else {}, transport="stdio")
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


def serve_http(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve /mcp with an HTTP/2-capable ASGI server (Hypercorn / Granian / Daphne)."""
    try:
        import asyncio

        from hypercorn.asyncio import serve
        from hypercorn.config import Config

        config = Config()
        config.bind = [f"{host}:{port}"]
        config.alpn_protocols = ["h2", "http/1.1"]
        logger.info("LMTuner MCP 2.0 Streamable HTTP on http://%s:%s/mcp (Hypercorn HTTP/2)", host, port)
        asyncio.run(serve(mcp_asgi_app, config))
        return
    except ImportError:
        logger.warning("hypercorn not installed; trying granian/daphne. Do not use uvicorn for this endpoint.")

    try:
        from granian import Granian

        logger.info("LMTuner MCP 2.0 on http://%s:%s/mcp (Granian)", host, port)
        Granian("olive.protocols.mcp:mcp_asgi_app", address=host, port=port, interface="asgi").serve()
        return
    except ImportError:
        pass

    raise SystemExit(
        "MCP Streamable HTTP requires an HTTP/2-capable ASGI server. "
        "Install hypercorn (`pip install hypercorn`) and retry. Uvicorn is not HTTP/2 capable."
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LMTuner MCP 2.0 (2026-07-28) server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.transport == "http":
        serve_http(args.host, args.port)
    else:
        serve_stdio()


if __name__ == "__main__":
    main()
