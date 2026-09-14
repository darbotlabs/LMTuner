# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

from olive.cli.launcher import get_cli_parser
from olive.protocols.lmcli_tools import build_lmcli_argv, normalize_tool_name, tool_schemas
from olive.protocols.mcp import (
    HEADER_MISMATCH,
    MCP_PROTOCOL_VERSION,
    handle_http_request,
    handle_rpc,
)


def _meta(caps=None):
    return {
        "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "0"},
        "io.modelcontextprotocol/clientCapabilities": caps or {},
    }


def test_tool_schemas_cover_lmcli_suite():
    names = {t["name"] for t in tool_schemas()}
    assert names == {
        "optimize",
        "auto-opt",
        "finetune",
        "diffusion-lora",
        "capture-onnx-graph",
        "run",
        "benchmark",
    }


def test_build_lmcli_argv_optimize():
    argv = build_lmcli_argv("optimize", {"model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "precision": "int4"})
    assert "-m" in argv
    assert "olive" in argv
    assert "optimize" in argv
    assert "--model_name_or_path" in argv
    assert "Qwen/Qwen2.5-0.5B-Instruct" in argv
    assert "--precision" in argv


def test_normalize_tool_name():
    assert normalize_tool_name("capture_onnx_graph") == "capture-onnx-graph"
    assert normalize_tool_name("auto_opt") == "auto-opt"


def test_server_discover():
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "server/discover",
        "params": {"_meta": _meta()},
    }
    resp = handle_rpc(body)
    assert "result" in resp
    assert MCP_PROTOCOL_VERSION in resp["result"]["supportedVersions"]
    assert "tools" in resp["result"]["capabilities"]
    assert "io.modelcontextprotocol/tasks" in resp["result"]["capabilities"]["extensions"]
    assert resp["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "lmcli"


def test_initialize_rejected():
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"_meta": _meta()}}
    resp = handle_rpc(body)
    assert resp["error"]["code"] == -32601


def test_tools_list_and_dry_run_call():
    listed = handle_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": _meta()}})
    assert listed["result"]["resultType"] == "complete"
    assert any(t["name"] == "optimize" for t in listed["result"]["tools"])

    call = handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "optimize",
                "arguments": {"model_name_or_path": "foo", "_dry_run": True},
                "_meta": _meta(),
            },
        }
    )
    assert call["result"]["resultType"] == "complete"
    assert call["result"]["structuredContent"]["dry_run"] is True
    assert "optimize" in call["result"]["structuredContent"]["argv"]


def test_tools_call_with_tasks_returns_create_task_result():
    caps = {"extensions": {"io.modelcontextprotocol/tasks": {}}}
    call = handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "benchmark",
                "arguments": {"model_name_or_path": "foo", "_dry_run": True},
                "_meta": _meta(caps),
            },
        }
    )
    assert call["result"]["resultType"] == "task"
    assert call["result"]["taskId"]
    got = handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tasks/get",
            "params": {"taskId": call["result"]["taskId"], "_meta": _meta(caps)},
        }
    )
    assert got["result"]["taskId"] == call["result"]["taskId"]
    assert got["result"]["status"] in {"working", "completed", "failed", "cancelled"}


def test_http_header_mismatch_and_version():
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "server/discover",
        "params": {"_meta": _meta()},
    }
    raw = json.dumps(body).encode()
    mismatch = handle_http_request(
        "POST",
        {
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": "tools/list",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        raw,
    )
    assert mismatch.status == 400
    payload = json.loads(mismatch.body)
    assert payload["error"]["code"] == HEADER_MISMATCH

    bad_ver = handle_http_request(
        "POST",
        {
            "MCP-Protocol-Version": "2025-11-25",
            "Mcp-Method": "server/discover",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": {
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": "2025-11-25",
                        "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "0"},
                        "io.modelcontextprotocol/clientCapabilities": {},
                    }
                },
            }
        ).encode(),
    )
    assert bad_ver.status == 400
    err = json.loads(bad_ver.body)["error"]
    assert err["data"]["name"] == "UnsupportedProtocolVersionError"

    ok = handle_http_request(
        "POST",
        {
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": "server/discover",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        json.dumps(body).encode(),
    )
    assert ok.status == 200
    assert json.loads(ok.body)["result"]["resultType"] == "complete"

    get = handle_http_request("GET", {}, b"")
    assert get.status == 405
    delete = handle_http_request("DELETE", {"Mcp-Session-Id": "should-be-ignored"}, b"")
    assert delete.status == 405


def test_tools_call_requires_mcp_name_header():
    body = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "optimize",
            "arguments": {"model_name_or_path": "foo", "_dry_run": True},
            "_meta": _meta(),
        },
    }
    missing = handle_http_request(
        "POST",
        {
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": "tools/call",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json.dumps(body).encode(),
    )
    assert missing.status == 400
    ok = handle_http_request(
        "POST",
        {
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": "tools/call",
            "Mcp-Name": "optimize",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json.dumps(body).encode(),
    )
    assert ok.status == 200


def test_cli_registers_mcp():
    parser = get_cli_parser()
    args = parser.parse_args(["mcp", "--transport", "http", "--port", "9001"])
    assert args.func.__name__ == "McpCommand"
    assert args.port == 9001
