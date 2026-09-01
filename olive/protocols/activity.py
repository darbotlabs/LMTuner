# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Microsoft 365 Agents SDK Activity Protocol helpers for LMTuner copilot.

Spec: https://learn.microsoft.com/en-us/microsoft-365/agents-sdk/activity-protocol
Schema: https://github.com/microsoft/Agents/blob/main/specs/activity/protocol-activity.md

This is the Microsoft Activity JSON (type, from, recipient, conversation, channelId,
text, attachments, invoke/event/typing) — not IBM Agent Commerce Protocol and not A2A.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from olive.protocols.lmcli_tools import LMCLI_TOOLS, normalize_tool_name, run_lmcli

logger = logging.getLogger(__name__)

CHANNEL_ID = "lmcli"
AGENT_ACCOUNT = {"id": "lmcli", "name": "LMTuner", "role": "bot"}
USER_ACCOUNT = {"id": "user", "name": "User", "role": "user"}

ACTIVITY_TYPES = ("message", "typing", "invoke", "event", "conversationUpdate")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def make_activity(
    type_: str,
    *,
    text: str | None = None,
    conversation_id: str | None = None,
    from_account: dict[str, Any] | None = None,
    recipient: dict[str, Any] | None = None,
    name: str | None = None,
    value: Any = None,
    attachments: list | None = None,
    channel_id: str = CHANNEL_ID,
    reply_to_id: str | None = None,
) -> dict[str, Any]:
    if type_ not in ACTIVITY_TYPES:
        raise ValueError(f"Unsupported activity type: {type_}")
    activity: dict[str, Any] = {
        "type": type_,
        "id": str(uuid.uuid4()),
        "timestamp": utc_now(),
        "channelId": channel_id,
        "from": from_account or AGENT_ACCOUNT,
        "recipient": recipient or USER_ACCOUNT,
        "conversation": {"id": conversation_id or str(uuid.uuid4())},
    }
    if text is not None:
        activity["text"] = text
    if attachments:
        activity["attachments"] = attachments
    if name is not None:
        activity["name"] = name
    if value is not None:
        activity["value"] = value
    if reply_to_id:
        activity["replyToId"] = reply_to_id
    return activity


def typing_activity(conversation_id: str, reply_to_id: str | None = None) -> dict[str, Any]:
    return make_activity("typing", conversation_id=conversation_id, reply_to_id=reply_to_id)


def message_activity(text: str, conversation_id: str, reply_to_id: str | None = None) -> dict[str, Any]:
    return make_activity("message", text=text, conversation_id=conversation_id, reply_to_id=reply_to_id)


def invoke_activity(name: str, value: Any, conversation_id: str | None = None) -> dict[str, Any]:
    return make_activity("invoke", name=name, value=value, conversation_id=conversation_id, from_account=USER_ACCOUNT, recipient=AGENT_ACCOUNT)


def handle_activity(activity: dict[str, Any], *, execute: bool = True) -> list[dict[str, Any]]:
    """Turn an inbound Activity into outbound activities (typing + message / invoke result)."""
    if not isinstance(activity, dict) or not activity.get("type"):
        raise ValueError("Activity must be a JSON object with a type field")
    conversation_id = str((activity.get("conversation") or {}).get("id") or uuid.uuid4())
    inbound_id = activity.get("id")
    act_type = activity.get("type")
    outbound = [typing_activity(conversation_id, reply_to_id=inbound_id)]

    if act_type == "typing":
        return []

    if act_type == "invoke":
        name = normalize_tool_name(str(activity.get("name") or ""))
        value = activity.get("value") if isinstance(activity.get("value"), dict) else {}
        if name not in LMCLI_TOOLS:
            outbound.append(
                message_activity(
                    f"Unknown invoke '{activity.get('name')}'. Supported: {', '.join(LMCLI_TOOLS)}",
                    conversation_id,
                    reply_to_id=inbound_id,
                )
            )
            return outbound
        if value.get("_dry_run") or not execute:
            from olive.protocols.lmcli_tools import build_lmcli_argv

            result = {"ok": True, "tool": name, "argv": build_lmcli_argv(name, value), "dry_run": True}
        else:
            result = run_lmcli(name, value)
        outbound.append(
            make_activity(
                "event",
                name=f"{name}.complete",
                value=result,
                conversation_id=conversation_id,
                reply_to_id=inbound_id,
            )
        )
        outbound.append(
            message_activity(
                json.dumps(result, default=str)[:4000],
                conversation_id,
                reply_to_id=inbound_id,
            )
        )
        return outbound

    if act_type in {"message", "event"}:
        text = str(activity.get("text") or "")
        if act_type == "event" and activity.get("name"):
            name = normalize_tool_name(str(activity["name"]))
            value = activity.get("value") if isinstance(activity.get("value"), dict) else {}
            fake_invoke = invoke_activity(name, value, conversation_id)
            fake_invoke["id"] = inbound_id
            return handle_activity(fake_invoke, execute=execute)
        lower = text.strip()
        tool = None
        for candidate in LMCLI_TOOLS:
            if lower.startswith((candidate, f"lmcli {candidate}")):
                tool = candidate
                break
        if tool:
            fake = invoke_activity(tool, {"extra_args": text.split()[1:], "_dry_run": True}, conversation_id)
            fake["id"] = inbound_id
            return handle_activity(fake, execute=False)
        outbound.append(
            message_activity(
                "LMTuner copilot activity endpoint. Send a message activity or an invoke "
                f"named one of: {', '.join(LMCLI_TOOLS)}.",
                conversation_id,
                reply_to_id=inbound_id,
            )
        )
        return outbound

    if act_type == "conversationUpdate":
        outbound.append(message_activity("LMTuner copilot is ready.", conversation_id, reply_to_id=inbound_id))
        return outbound

    outbound.append(message_activity(f"Unhandled activity type: {act_type}", conversation_id, reply_to_id=inbound_id))
    return outbound


def serve_activity(host: str = "127.0.0.1", port: int = 3978) -> None:
    """HTTP server accepting Activity JSON on POST /activity and POST /api/messages."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in {"", "/"}:
                self._send(
                    200,
                    {
                        "protocol": "Microsoft 365 Agents SDK Activity Protocol",
                        "spec": "https://learn.microsoft.com/en-us/microsoft-365/agents-sdk/activity-protocol",
                        "post": ["/activity", "/api/messages"],
                    },
                )
                return
            self._send(404, {"error": "Not Found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path.rstrip("/") not in {"/activity", "/api/messages"}:
                self._send(404, {"error": "Not Found"})
                return
            origin = self.headers.get("Origin")
            if origin:
                host_name = urlparse(origin).hostname
                if host_name not in {"127.0.0.1", "localhost"}:
                    self._send(403, {"error": "Forbidden origin"})
                    return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                activity = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "Invalid JSON"})
                return
            try:
                replies = handle_activity(activity)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, replies)

    httpd = ThreadingHTTPServer((host, port), Handler)
    logger.info("LMTuner Activity Protocol on http://%s:%s/activity", host, port)
    httpd.serve_forever()
