# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.cli.launcher import get_cli_parser
from olive.protocols.activity import handle_activity, invoke_activity, make_activity, typing_activity


def test_activity_schema_fields():
    activity = make_activity("message", text="hello", conversation_id="c1")
    assert activity["type"] == "message"
    assert activity["channelId"] == "lmcli"
    assert activity["from"]["id"] == "lmcli"
    assert activity["recipient"]["id"] == "user"
    assert activity["conversation"]["id"] == "c1"
    assert activity["text"] == "hello"
    assert "attachments" not in activity or isinstance(activity.get("attachments"), list)


def test_typing_and_invoke_roundtrip():
    typing = typing_activity("c1")
    assert typing["type"] == "typing"
    inbound = invoke_activity("optimize", {"model_name_or_path": "foo", "_dry_run": True}, "c1")
    assert inbound["type"] == "invoke"
    assert inbound["name"] == "optimize"
    replies = handle_activity(inbound, execute=True)
    types = [r["type"] for r in replies]
    assert "typing" in types
    assert "message" in types
    assert "event" in types


def test_cli_copilot_activity_emit():
    parser = get_cli_parser()
    args = parser.parse_args(["copilot", "activity", "emit", "--type", "typing"])
    assert args.copilot_cmd == "activity"
    assert args.activity_cmd == "emit"
    assert args.activity_type == "typing"


def test_cli_copilot_existing_info_still_works():
    parser = get_cli_parser()
    args = parser.parse_args(["copilot", "--info"])
    assert args.info is True
