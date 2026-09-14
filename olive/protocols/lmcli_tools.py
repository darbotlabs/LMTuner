# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Real lmcli tool wrappers shared by MCP, ACP, and Activity Protocol servers."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

LMCLI_TOOLS = (
    "optimize",
    "auto-opt",
    "finetune",
    "diffusion-lora",
    "capture-onnx-graph",
    "run",
    "benchmark",
)

_FLAG_KEYS = {
    "model_name_or_path": "--model_name_or_path",
    "output_path": "--output_path",
    "precision": "--precision",
    "provider": "--provider",
    "device": "--device",
    "data_name": "--data_name",
    "data_dir": "--data_dir",
    "config": "--config",
    "model_variant": "--model_variant",
    "lora_r": "--lora_r",
    "max_train_steps": "--max_train_steps",
    "learning_rate": "--learning_rate",
    "train_batch_size": "--train_batch_size",
    "mixed_precision": "--mixed_precision",
    "instance_prompt": "--instance_prompt",
    "class_prompt": "--class_prompt",
    "class_data_dir": "--class_data_dir",
    "algorithm": "--algorithm",
    "task": "--task",
    "use_ort_genai": "--use_ort_genai",
}

_BOOL_FLAGS = {
    "dreambooth": "--dreambooth",
    "merge_lora": "--merge_lora",
    "with_prior_preservation": "--with_prior_preservation",
    "use_dora": "--use_dora",
    "use_rslora": "--use_rslora",
    "dry_run": "--dry_run",
    "disable_telemetry": "--disable_telemetry",
}


def tool_schemas() -> list[dict[str, Any]]:
    """JSON Schema tool definitions for MCP tools/list."""
    extra = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Additional lmcli flags forwarded verbatim, e.g. ['--precision', 'int4'].",
    }
    common_model = {
        "type": "string",
        "description": "Hugging Face model id or local path.",
        "x-mcp-header": "Model",
    }
    output = {"type": "string", "description": "Output directory."}

    def obj(properties: dict, required: list | None = None) -> dict:
        schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
        if required:
            schema["required"] = required
        return schema

    return [
        {
            "name": "optimize",
            "title": "LMTuner optimize",
            "description": "Run `lmcli optimize` — comprehensive pass scheduling for a model.",
            "inputSchema": obj(
                {
                    "model_name_or_path": common_model,
                    "output_path": output,
                    "precision": {"type": "string", "description": "int4, int8, fp16, fp32, ..."},
                    "provider": {"type": "string"},
                    "device": {"type": "string", "description": "cpu, gpu, or npu"},
                    "dry_run": {"type": "boolean"},
                    "extra_args": extra,
                },
                ["model_name_or_path"],
            ),
        },
        {
            "name": "auto-opt",
            "title": "LMTuner auto-opt",
            "description": "Run `lmcli auto-opt` — automatic optimizer.",
            "inputSchema": obj(
                {
                    "model_name_or_path": common_model,
                    "output_path": output,
                    "precision": {"type": "string"},
                    "device": {"type": "string"},
                    "extra_args": extra,
                },
                ["model_name_or_path"],
            ),
        },
        {
            "name": "finetune",
            "title": "LMTuner finetune",
            "description": "Run `lmcli finetune` — PEFT fine-tune a text model.",
            "inputSchema": obj(
                {
                    "model_name_or_path": common_model,
                    "data_name": {"type": "string"},
                    "output_path": output,
                    "extra_args": extra,
                },
                ["model_name_or_path", "data_name"],
            ),
        },
        {
            "name": "diffusion-lora",
            "title": "LMTuner diffusion-lora",
            "description": "Run `lmcli diffusion-lora` — train LoRA for SD/SDXL/SD3/Flux/Sana.",
            "inputSchema": obj(
                {
                    "model_name_or_path": common_model,
                    "data_dir": {"type": "string"},
                    "data_name": {"type": "string"},
                    "output_path": output,
                    "model_variant": {
                        "type": "string",
                        "description": "auto|sd|sdxl|sd3|flux|sana",
                    },
                    "lora_r": {"type": "integer"},
                    "use_dora": {"type": "boolean"},
                    "use_rslora": {"type": "boolean"},
                    "init_lora_weights": {"type": "string"},
                    "target_modules": {"type": "string"},
                    "dreambooth": {"type": "boolean"},
                    "instance_prompt": {"type": "string"},
                    "extra_args": extra,
                },
                ["model_name_or_path"],
            ),
        },
        {
            "name": "capture-onnx-graph",
            "title": "LMTuner capture-onnx-graph",
            "description": "Run `lmcli capture-onnx-graph` — export HF/PyTorch to ONNX.",
            "inputSchema": obj(
                {
                    "model_name_or_path": common_model,
                    "output_path": output,
                    "precision": {"type": "string"},
                    "extra_args": extra,
                },
                ["model_name_or_path"],
            ),
        },
        {
            "name": "run",
            "title": "LMTuner run",
            "description": "Run `lmcli run` — execute an Olive/LMTuner workflow config.",
            "inputSchema": obj(
                {
                    "config": {"type": "string", "description": "Path to workflow JSON/YAML."},
                    "extra_args": extra,
                },
                ["config"],
            ),
        },
        {
            "name": "benchmark",
            "title": "LMTuner benchmark",
            "description": "Run `lmcli benchmark` — evaluate with lm-eval.",
            "inputSchema": obj(
                {
                    "model_name_or_path": common_model,
                    "output_path": output,
                    "device": {"type": "string"},
                    "extra_args": extra,
                },
                ["model_name_or_path"],
            ),
        },
    ]


def normalize_tool_name(name: str) -> str:
    mapping = {
        "auto_opt": "auto-opt",
        "diffusion_lora": "diffusion-lora",
        "capture_onnx_graph": "capture-onnx-graph",
        "capture-onnx": "capture-onnx-graph",
    }
    return mapping.get(name, name)


def build_lmcli_argv(tool_name: str, arguments: dict[str, Any] | None = None) -> list[str]:
    """Build `python -m olive <command> ...` argv wrapping the real CLI."""
    name = normalize_tool_name(tool_name)
    if name not in LMCLI_TOOLS:
        raise ValueError(f"Unknown lmcli tool: {tool_name}")
    arguments = dict(arguments or {})
    extra = list(arguments.pop("extra_args", None) or [])
    argv = [sys.executable, "-m", "olive", name]
    if name == "run" and "config" in arguments:
        config = arguments.pop("config")
        argv.append(str(config))
    for key, value in list(arguments.items()):
        if value is None:
            continue
        if key in _BOOL_FLAGS:
            if value:
                argv.append(_BOOL_FLAGS[key])
            continue
        flag = _FLAG_KEYS.get(key, f"--{key}")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    argv.extend(str(a) for a in extra)
    return argv


def run_lmcli(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout: float | None = None,
    on_line: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute an lmcli tool as a subprocess. Returns a structured result dict."""
    argv = build_lmcli_argv(tool_name, arguments)
    logger.info("Running lmcli: %s", " ".join(argv))
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip("\n")
            lines.append(text)
            if on_line:
                on_line(text)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {
            "ok": False,
            "tool": normalize_tool_name(tool_name),
            "argv": argv,
            "returncode": None,
            "timed_out": True,
            "output": "\n".join(lines[-200:]),
        }
    output = "\n".join(lines)
    return {
        "ok": proc.returncode == 0,
        "tool": normalize_tool_name(tool_name),
        "argv": argv,
        "returncode": proc.returncode,
        "output": output[-8000:],
    }


@dataclass
class TaskRecord:
    task_id: str
    status: str = "working"
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    poll_interval_ms: int = 2000
    ttl_ms: int = 3_600_000
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    result: Any = None
    error: Any = None
    status_message: str = ""
    logs: list[str] = field(default_factory=list)
    cancelled: bool = False


class TaskStore:
    """In-memory MCP Tasks store for long-running lmcli jobs."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def create(self, tool: str, arguments: dict[str, Any]) -> TaskRecord:
        task = TaskRecord(task_id=uuid.uuid4().hex, tool=tool, arguments=arguments)
        with self._lock:
            self._tasks[task.task_id] = task
        thread = threading.Thread(target=self._run, args=(task.task_id,), daemon=True)
        thread.start()
        return task

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.cancelled = True
            if task.status == "working":
                task.status = "cancelled"
                task.status_message = "Cancellation requested."
            return task

    def snapshot(self, task: TaskRecord) -> dict[str, Any]:
        body: dict[str, Any] = {
            "taskId": task.task_id,
            "status": task.status,
            "pollIntervalMs": task.poll_interval_ms,
            "ttlMs": task.ttl_ms,
            "statusMessage": task.status_message,
            "createdAt": task.created_at,
        }
        if task.status == "completed":
            body["result"] = task.result
        if task.status == "failed":
            body["error"] = task.error
        return body

    def _run(self, task_id: str) -> None:
        task = self.get(task_id)
        if task is None:
            return

        def on_line(line: str) -> None:
            rec = self.get(task_id)
            if rec is None:
                return
            rec.logs.append(line)
            rec.status_message = line[:500]
            if rec.cancelled:
                raise RuntimeError("cancelled")

        try:
            if task.arguments.get("_dry_run"):
                argv = build_lmcli_argv(task.tool, {k: v for k, v in task.arguments.items() if k != "_dry_run"})
                rec = self.get(task_id)
                if rec is None:
                    return
                rec.status = "completed"
                rec.result = {
                    "resultType": "complete",
                    "content": [{"type": "text", "text": str(argv)}],
                    "structuredContent": {"ok": True, "tool": rec.tool, "argv": argv, "dry_run": True},
                    "isError": False,
                }
                rec.status_message = "Dry-run completed."
                return
            result = run_lmcli(task.tool, task.arguments, on_line=on_line)
            rec = self.get(task_id)
            if rec is None:
                return
            if rec.cancelled:
                rec.status = "cancelled"
                rec.status_message = "Cancelled."
                return
            if result.get("ok"):
                rec.status = "completed"
                rec.result = {
                    "resultType": "complete",
                    "content": [{"type": "text", "text": json.dumps(result, default=str)[:8000]}],
                    "structuredContent": result,
                    "isError": False,
                }
                rec.status_message = "Completed."
            else:
                rec.status = "failed"
                rec.error = {"code": -32000, "message": result.get("output") or "lmcli failed"}
                rec.status_message = "Failed."
        except Exception as exc:
            rec = self.get(task_id)
            if rec is None:
                return
            if rec.cancelled:
                rec.status = "cancelled"
                return
            rec.status = "failed"
            rec.error = {"code": -32000, "message": str(exc)}


TASK_STORE = TaskStore()
