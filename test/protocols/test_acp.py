# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

from olive.cli.launcher import get_cli_parser
from olive.protocols.acp import AGENT, AcpAgent, handle_http


def test_initialize_returns_connection_id_header():
    AGENT.connections.clear()
    result = handle_http(
        "POST",
        {"Content-Type": "application/json"},
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}}).encode(),
    )
    assert result.status == 200
    assert "Acp-Connection-Id" in result.headers
    body = json.loads(result.body)
    assert body["result"]["connectionId"] == result.headers["Acp-Connection-Id"]
    assert body["result"]["agentInfo"]["name"] == "lmcli"


def test_session_new_prompt_cancel_close_http_status_codes():
    AGENT.connections.clear()
    init = handle_http(
        "POST",
        {"Content-Type": "application/json"},
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).encode(),
    )
    conn = init.headers["Acp-Connection-Id"]
    headers = {"Content-Type": "application/json", "Acp-Connection-Id": conn}

    created = handle_http(
        "POST",
        headers,
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"cwd": "."}}).encode(),
    )
    assert created.status == 202

    from olive.protocols.acp import pop_queued_events

    events = pop_queued_events(conn)
    session_id = None
    for event in events:
        if event.get("id") == 2:
            session_id = event["result"]["sessionId"]
    assert session_id

    sess_headers = {**headers, "Acp-Session-Id": session_id}
    prompt = handle_http(
        "POST",
        sess_headers,
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "lmcli optimize --model_name_or_path foo --dry_run"}],
                },
            }
        ).encode(),
    )
    assert prompt.status == 202
    sess_events = pop_queued_events(conn, session_id)
    assert any(e.get("method") == "session/update" for e in sess_events)
    assert any(e.get("id") == 3 and e.get("result", {}).get("stopReason") == "end_turn" for e in sess_events)

    cancel = handle_http(
        "POST",
        sess_headers,
        json.dumps({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": session_id}}).encode(),
    )
    assert cancel.status == 202

    close = handle_http(
        "POST",
        sess_headers,
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "session/close", "params": {"sessionId": session_id}}).encode(),
    )
    assert close.status == 202

    sse = handle_http("GET", {"Acp-Connection-Id": conn, "Accept": "text/event-stream"}, b"")
    assert sse.status == 200
    assert "text/event-stream" in sse.headers["content-type"]

    unknown_sess = handle_http(
        "GET",
        {"Acp-Connection-Id": conn, "Acp-Session-Id": "nope", "Accept": "text/event-stream"},
        b"",
    )
    assert unknown_sess.status == 404

    ended = handle_http("DELETE", {"Acp-Connection-Id": conn}, b"")
    assert ended.status == 202


def test_stdio_agent_prompt_dry_run():
    agent = AcpAgent()
    conn = agent.new_connection()
    init = agent.handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, connection=conn)
    assert init["result"]["connectionId"]
    new = agent.handle_rpc({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {}}, connection=conn)
    sid = new["result"]["sessionId"]
    prompt = agent.handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [{"type": "text", "text": '{"tool":"optimize","arguments":{"model_name_or_path":"x","_dry_run":true}}'}],
            },
        },
        connection=conn,
    )
    assert prompt["result"]["stopReason"] == "end_turn"
    assert any("optimize" in json.dumps(u) for u in prompt["_acp_updates"])


def test_cli_registers_acp():
    parser = get_cli_parser()
    args = parser.parse_args(["acp", "--transport", "http", "--host", "127.0.0.1"])
    assert args.func.__name__ == "AcpCommand"
