# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Zed Agent Client Protocol (ACP) stdio + Streamable HTTP/WebSocket server.

Spec: https://agentclientprotocol.com
RFD: https://agentclientprotocol.com/rfds/streamable-http-websocket-transport
Python SDK: https://agentclientprotocol.github.io/python-sdk/web-transport/

Prefer the official `agent-client-protocol` SDK when installed. The built-in
ASGI adapter implements the RFD: initialize -> 200 + Acp-Connection-Id;
other POST -> 202; GET SSE connection-scoped and session-scoped; DELETE ends
the connection. HTTP/2 is required for Streamable HTTP (Hypercorn, not uvicorn).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from typing import Any

from olive.protocols.lmcli_tools import LMCLI_TOOLS, build_lmcli_argv, normalize_tool_name, run_lmcli

logger = logging.getLogger(__name__)

ACP_PROTOCOL_VERSION = 1


def _olive_version() -> str:
    try:
        import olive

        return getattr(olive, "__version__", "unknown")
    except Exception:
        return "unknown"


def header_get(headers: dict[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _extract_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or item.get("text/plain") or ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "".join(parts)
    if isinstance(prompt, dict):
        return str(prompt.get("text") or "")
    return str(prompt or "")


def _parse_tool_from_text(text: str) -> tuple[str, dict[str, Any]] | None:
    """Map a user prompt onto an lmcli tool when possible."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict) and payload.get("tool"):
                return normalize_tool_name(str(payload["tool"])), payload.get("arguments") or {}
        except json.JSONDecodeError:
            pass
    tokens = stripped.split()
    if tokens and tokens[0] in {"lmcli", "olive"}:
        tokens = tokens[1:]
    if tokens and normalize_tool_name(tokens[0]) in LMCLI_TOOLS:
        name = normalize_tool_name(tokens[0])
        extra = tokens[1:]
        arguments: dict[str, Any] = {"extra_args": extra, "_dry_run": "--dry_run" in extra}
        return name, arguments
    return None


class AcpSession:
    def __init__(self, session_id: str, cwd: str | None = None) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.cancelled = False
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False


class AcpConnection:
    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id
        self.sessions: dict[str, AcpSession] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False


class AcpAgent:
    """JSON-RPC ACP agent that can drive lmcli tools from session/prompt."""

    def __init__(self) -> None:
        self.connections: dict[str, AcpConnection] = {}

    def new_connection(self) -> AcpConnection:
        conn = AcpConnection(uuid.uuid4().hex)
        self.connections[conn.connection_id] = conn
        return conn

    def get_connection(self, connection_id: str | None) -> AcpConnection | None:
        if not connection_id:
            return None
        return self.connections.get(connection_id)

    def close_connection(self, connection_id: str) -> None:
        conn = self.connections.pop(connection_id, None)
        if conn:
            conn.closed = True
            for session in conn.sessions.values():
                session.closed = True

    def handle_rpc(self, body: dict[str, Any], *, connection: AcpConnection | None = None) -> dict[str, Any]:
        req_id = body.get("id")
        method = body.get("method")
        params = body.get("params") if isinstance(body.get("params"), dict) else {}
        if body.get("jsonrpc") not in {None, "2.0"}:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid Request"}}

        if method == "initialize":
            if connection is None:
                connection = self.new_connection()
            result = {
                "protocolVersion": params.get("protocolVersion") or ACP_PROTOCOL_VERSION,
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {"image": False, "audio": False, "embeddedContext": False},
                },
                "agentInfo": {"name": "lmcli", "title": "LMTuner ACP agent", "version": _olive_version()},
                "connectionId": connection.connection_id,
            }
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        if connection is None:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": "Missing Acp-Connection-Id"}}

        if method == "session/new":
            session = AcpSession(uuid.uuid4().hex, cwd=params.get("cwd"))
            connection.sessions[session.session_id] = session
            return {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": session.session_id}}

        if method == "session/load":
            session_id = str(params.get("sessionId") or "")
            session = connection.sessions.get(session_id)
            if session is None:
                session = AcpSession(session_id, cwd=params.get("cwd"))
                connection.sessions[session_id] = session
            return {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": session.session_id}}

        if method == "session/prompt":
            return self._prompt(req_id, connection, params)

        if method == "session/cancel":
            session_id = str(params.get("sessionId") or "")
            session = connection.sessions.get(session_id)
            if session:
                session.cancelled = True
            return {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": session_id}}

        if method == "session/close":
            session_id = str(params.get("sessionId") or "")
            session = connection.sessions.pop(session_id, None)
            if session:
                session.closed = True
            return {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": session_id}}

        if method == "authenticate":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if req_id is None:
            return {}
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def _prompt(self, req_id: Any, connection: AcpConnection, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        session = connection.sessions.get(session_id)
        if session is None:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown sessionId: {session_id}"}}
        session.cancelled = False
        text = _extract_prompt_text(params.get("prompt"))
        updates: list[dict[str, Any]] = [
            _agent_update(session_id, "agent_thought_chunk", f"LMTuner received prompt ({len(text)} chars)."),
        ]
        parsed = _parse_tool_from_text(text)
        stop_reason = "end_turn"
        if parsed:
            tool, arguments = parsed
            argv = build_lmcli_argv(tool, {k: v for k, v in arguments.items() if k != "_dry_run"})
            updates.append(_tool_call_update(session_id, tool, argv))
            if arguments.get("_dry_run") or params.get("dryRun"):
                result = {"ok": True, "tool": tool, "argv": argv, "dry_run": True}
            else:
                result = run_lmcli(tool, arguments)
            updates.append(
                _agent_update(
                    session_id,
                    "agent_message_chunk",
                    json.dumps(result, default=str)[:4000],
                )
            )
            if session.cancelled:
                stop_reason = "cancelled"
        else:
            help_text = (
                "LMTuner ACP agent. Send a prompt like `lmcli optimize --model_name_or_path MODEL --dry_run` "
                f"or JSON `{{\"tool\": \"optimize\", \"arguments\": {{...}}}}`. Tools: {', '.join(LMCLI_TOOLS)}."
            )
            updates.append(_agent_update(session_id, "agent_message_chunk", help_text))
        session.last_updates = updates  # type: ignore[attr-defined]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"stopReason": stop_reason, "sessionId": session_id},
            "_acp_updates": updates,
        }


def _agent_update(session_id: str, kind: str, text: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": kind,
                "content": {"type": "text", "text": text},
            },
        },
    }


def _tool_call_update(session_id: str, tool: str, argv: list[str]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": uuid.uuid4().hex,
                "title": f"lmcli {tool}",
                "kind": "execute",
                "status": "completed",
                "rawInput": {"argv": argv},
            },
        },
    }


AGENT = AcpAgent()


class AcpHttpResult:
    def __init__(self, status: int, headers: dict[str, str], body: bytes, sse: list[str] | None = None):
        self.status = status
        self.headers = headers
        self.body = body
        self.sse = sse or []


def handle_http(
    method: str,
    headers: dict[str, str],
    body_bytes: bytes,
    *,
    upgrade: str | None = None,
) -> AcpHttpResult:
    json_headers = {"content-type": "application/json"}
    conn_id = header_get(headers, "Acp-Connection-Id")
    session_id = header_get(headers, "Acp-Session-Id")

    if method == "DELETE":
        if not conn_id:
            return AcpHttpResult(400, json_headers, b'{"error":"Acp-Connection-Id required"}')
        AGENT.close_connection(conn_id)
        return AcpHttpResult(202, {}, b"")

    if method == "GET":
        if (upgrade or "").lower() == "websocket":
            new_id = conn_id or AGENT.new_connection().connection_id
            return AcpHttpResult(101, {"Acp-Connection-Id": new_id, "Upgrade": "websocket"}, b"")
        accept = header_get(headers, "Accept") or ""
        if "text/event-stream" not in accept:
            return AcpHttpResult(406, json_headers, b'{"error":"Accept must include text/event-stream"}')
        if not conn_id:
            return AcpHttpResult(400, json_headers, b'{"error":"Acp-Connection-Id required"}')
        conn = AGENT.get_connection(conn_id)
        if conn is None:
            return AcpHttpResult(404, json_headers, b'{"error":"Unknown Acp-Connection-Id"}')
        if session_id and session_id not in conn.sessions:
            return AcpHttpResult(404, json_headers, b'{"error":"Unknown Acp-Session-Id"}')
        # Caller holds the GET stream; the ASGI app yields queued events.
        return AcpHttpResult(
            200,
            {
                "content-type": "text/event-stream",
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
                "Acp-Connection-Id": conn_id,
                **({"Acp-Session-Id": session_id} if session_id else {}),
            },
            b": connected\r\n\r\n",
        )

    if method != "POST":
        return AcpHttpResult(405, {"allow": "GET, POST, DELETE"}, b"")

    content_type = (header_get(headers, "Content-Type") or "").split(";")[0].strip().lower()
    if content_type and content_type != "application/json":
        return AcpHttpResult(415, json_headers, b'{"error":"Content-Type must be application/json"}')

    try:
        body = json.loads(body_bytes.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return AcpHttpResult(400, json_headers, b'{"error":"Parse error"}')

    if isinstance(body, list):
        return AcpHttpResult(501, json_headers, b'{"error":"Batch JSON-RPC is not supported"}')
    if not isinstance(body, dict):
        return AcpHttpResult(400, json_headers, b'{"error":"Invalid JSON-RPC request"}')

    rpc_method = body.get("method")
    if rpc_method == "initialize" and not conn_id:
        conn = AGENT.new_connection()
        rpc = AGENT.handle_rpc(body, connection=conn)
        payload = json.dumps({k: v for k, v in rpc.items() if not str(k).startswith("_")}).encode("utf-8")
        return AcpHttpResult(
            200,
            {
                "content-type": "application/json",
                "Acp-Connection-Id": conn.connection_id,
                "Set-Cookie": f"acp_conn={conn.connection_id}; Path=/acp; HttpOnly",
            },
            payload,
        )

    if not conn_id:
        return AcpHttpResult(400, json_headers, b'{"error":"Acp-Connection-Id required"}')
    conn = AGENT.get_connection(conn_id)
    if conn is None:
        return AcpHttpResult(404, json_headers, b'{"error":"Unknown Acp-Connection-Id"}')

    session_methods = {"session/prompt", "session/cancel", "session/close"}
    if rpc_method in session_methods and not session_id:
        return AcpHttpResult(400, json_headers, b'{"error":"Acp-Session-Id required"}')

    rpc = AGENT.handle_rpc(body, connection=conn)
    updates = rpc.pop("_acp_updates", None) if isinstance(rpc, dict) else None
    session = None
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    sid = session_id or params.get("sessionId")
    if sid:
        session = conn.sessions.get(str(sid))
    if updates:
        target_q = session.queue if session is not None else conn.queue
        for update in updates:
            try:
                target_q.put_nowait(update)
            except Exception:
                pass
        # JSON-RPC result for session/new goes on the connection-scoped stream;
        # session-scoped POSTs go on the session stream.
        result_msg = {k: v for k, v in rpc.items() if not str(k).startswith("_")}
        try:
            (session.queue if session is not None and rpc_method != "session/new" else conn.queue).put_nowait(result_msg)
        except Exception:
            pass
    elif rpc and body.get("id") is not None:
        try:
            conn.queue.put_nowait({k: v for k, v in rpc.items() if not str(k).startswith("_")})
        except Exception:
            pass

    return AcpHttpResult(202, {"Acp-Connection-Id": conn.connection_id}, b"")


def pop_queued_events(connection_id: str, session_id: str | None = None) -> list[dict[str, Any]]:
    conn = AGENT.get_connection(connection_id)
    if conn is None:
        return []
    queue = conn.sessions[session_id].queue if session_id and session_id in conn.sessions else conn.queue
    events = []
    while True:
        try:
            events.append(queue.get_nowait())
        except Exception:
            break
    return events


def encode_sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\r\n\r\n".encode()


async def acp_asgi_app(scope, receive, send):
    """ASGI 3.0 application for POST/GET/DELETE /acp and WebSocket upgrade."""
    if scope["type"] == "websocket":
        await _ws_handler(scope, receive, send)
        return
    if scope["type"] != "http":
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b""})
        return

    path = scope.get("path") or "/"
    if not path.rstrip("/").endswith("/acp"):
        await send({"type": "http.response.start", "status": 404, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"Not Found"})
        return

    method = scope.get("method", "GET").upper()
    header_map = {k.decode("latin1"): v.decode("latin1") for k, v in scope.get("headers") or []}
    upgrade = header_map.get("upgrade")

    body = b""
    while True:
        event = await receive()
        if event["type"] == "http.request":
            body += event.get("body") or b""
            if not event.get("more_body"):
                break
        elif event["type"] == "http.disconnect":
            return

    result = handle_http(method, header_map, body, upgrade=upgrade)
    headers = [(k.encode("latin1"), str(v).encode("latin1")) for k, v in result.headers.items()]

    if method == "GET" and result.status == 200 and "text/event-stream" in result.headers.get("content-type", ""):
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": result.body, "more_body": True})
        conn_id = header_get(header_map, "Acp-Connection-Id") or result.headers.get("Acp-Connection-Id")
        session_id = header_get(header_map, "Acp-Session-Id")
        try:
            while True:
                events = pop_queued_events(conn_id or "", session_id)
                if events:
                    chunk = b"".join(encode_sse(e) for e in events)
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                else:
                    await send({"type": "http.response.body", "body": b": keepalive\r\n\r\n", "more_body": True})
                await asyncio.sleep(0.25)
                conn = AGENT.get_connection(conn_id or "")
                if conn is None or conn.closed:
                    break
                if session_id and (session_id not in conn.sessions or conn.sessions[session_id].closed):
                    break
        except asyncio.CancelledError:
            pass
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return

    await send({"type": "http.response.start", "status": result.status, "headers": headers})
    await send({"type": "http.response.body", "body": result.body})


async def _ws_handler(scope, receive, send):
    await send({"type": "websocket.accept", "headers": [(b"acp-connection-id", AGENT.new_connection().connection_id.encode())]})
    conn = list(AGENT.connections.values())[-1]
    while True:
        event = await receive()
        if event["type"] == "websocket.disconnect":
            AGENT.close_connection(conn.connection_id)
            return
        if event["type"] == "websocket.receive":
            text = event.get("text")
            if not text:
                continue
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                continue
            rpc = AGENT.handle_rpc(body, connection=conn)
            updates = rpc.pop("_acp_updates", None) if isinstance(rpc, dict) else None
            if updates:
                for update in updates:
                    await send({"type": "websocket.send", "text": json.dumps(update)})
            if rpc:
                await send(
                    {
                        "type": "websocket.send",
                        "text": json.dumps({k: v for k, v in rpc.items() if not str(k).startswith("_")}),
                    }
                )


def serve_stdio() -> None:
    conn = AGENT.new_connection()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        raw = line.strip()
        if not raw:
            continue
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rpc = AGENT.handle_rpc(body, connection=conn)
        updates = rpc.pop("_acp_updates", None) if isinstance(rpc, dict) else None
        if updates:
            for update in updates:
                sys.stdout.write(json.dumps(update) + "\n")
        if rpc and body.get("id") is not None:
            sys.stdout.write(json.dumps({k: v for k, v in rpc.items() if not str(k).startswith("_")}) + "\n")
        sys.stdout.flush()


def serve_http(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve /acp with Hypercorn (HTTP/2). Uvicorn is not HTTP/2 capable."""
    app = acp_asgi_app
    try:
        from acp.http.asgi import create_asgi_app

        class SdkAgent:
            def __init__(self, _conn=None) -> None:
                self._inner = AGENT

        app = create_asgi_app(lambda conn: SdkAgent(conn))
        logger.info("Using official agent-client-protocol ASGI adapter")
    except ImportError:
        logger.info("Official ACP SDK not installed; using built-in Streamable HTTP/WS adapter")

    try:
        import asyncio as _asyncio

        from hypercorn.asyncio import serve
        from hypercorn.config import Config

        config = Config()
        config.bind = [f"{host}:{port}"]
        config.alpn_protocols = ["h2", "http/1.1"]
        logger.info("LMTuner ACP Streamable HTTP/WS on http://%s:%s/acp (Hypercorn HTTP/2)", host, port)
        _asyncio.run(serve(app, config))
        return
    except ImportError:
        pass
    raise SystemExit(
        "ACP Streamable HTTP requires an HTTP/2-capable ASGI server. "
        "Install hypercorn (`pip install hypercorn`) and retry. Do not use uvicorn for this transport. "
        "See https://agentclientprotocol.github.io/python-sdk/web-transport/"
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LMTuner ACP (Agent Client Protocol) server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.transport == "http":
        serve_http(args.host, args.port)
    else:
        serve_stdio()


if __name__ == "__main__":
    main()
