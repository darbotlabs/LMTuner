# LMTuner spec

Easy-start guide for this tree after merging current Microsoft Olive and vendoring OptiGuide + Foundry Local.

## What LMTuner is

**LMTuner** (Language Model Tuner) is DarbotLabs' fork of [Microsoft Olive](https://github.com/microsoft/Olive). Given a model and target hardware, it composes optimization techniques (quantization, graph capture, fine-tuning, diffusion LoRA, packaging) and writes efficient ONNX (or related) artifacts for cloud or edge inference.

- Python package name: **`lmcli`** (not `olive-ai`)
- CLI: **`lmcli`** (the `olive` console script still points at the same launcher)
- Python import package remains `olive` (upstream module layout)
- Copilot helper: `lmcli copilot`

Upstream Olive documentation still applies for passes, workflows, and hardware: https://microsoft.github.io/Olive/

## Relation to Olive

This branch merges **microsoft/Olive `main`** into LMTuner and then layers LMTuner identity on top.

| Keep from LMTuner | Take from Olive |
|-------------------|-----------------|
| Branding (README, LICENSE, logo) | Later Olive engine, CLI (`init`, `generate-model-package`, telemetry, MCP, skills) |
| Package `lmcli` + `lmcli`/`olive` entry points | Single diffusion-lora / SD LoRA / IO-config / ONNX conversion implementation |
| `olive/cli/copilot.py` | `olive/assets/io_configs`, `olive/passes/diffusers/lora.py`, `olive/cli/diffusion_lora.py` |

Hypothesis verified at merge time: LMTuner commit `79925777` overlapped Olive's later diffusion-lora / IO-config / conversion work. The merge prefers **Olive's files** for those overlapping paths so there is **one** diffusion-lora stack (`olive/cli/diffusion_lora.py` + `olive/passes/diffusers/lora.py`), not two.

Olive is **not** nested as a second copy under `harness/`. The merge **is** how Olive lives in this repo.

## Harness layout

```
LMTuner/
  olive/                      # Olive engine (forked; import olive, run lmcli)
  olive/cli/launcher.py       # LMTuner CLI (lmcli) + Copilot
  olive/cli/diffusion_lora.py # diffusion-lora (Olive implementation)
  olive/passes/diffusers/     # SDLoRA pass
  olive/assets/io_configs/    # IO configs (defaults.yaml, diffusers.yaml, tasks.yaml)
  docs/LMTUNER_SPEC.md        # this file
  harness/README.md           # thin pointers
  harness/optiguide/          # full microsoft/OptiGuide tree (git subtree)
  harness/foundry-local/      # full microsoft/Foundry-Local tree (git subtree)
  mcp/                        # Olive MCP server (upstream)
  skills/olive/               # Olive agent skills (upstream)
```

Attribution:

- `LICENSE` at repo root (LMTuner fork notice + MIT)
- `harness/optiguide/LICENSE`
- `harness/foundry-local/LICENSE`

## Install

Python **>= 3.10**. Use a venv or conda env.

From this clone:

```bash
python -m venv .venv
```

Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

```bash
pip install -U pip
pip install -e ".[auto-opt]"
pip install transformers onnxruntime-genai
```

Extras live in `olive/olive_config.json`. Common ones:

| Extra | Use |
|-------|-----|
| `auto-opt` | Automatic optimizer / Optimum |
| `diffusers` | Diffusion LoRA (`accelerate`, `peft`, `diffusers`) |
| `lora` | PEFT LoRA / finetune helpers |
| `gpu` | GPU extras |
| `finetune` | Fine-tuning |

PyPI name when published: `pip install lmcli[auto-opt]` (same CLI).

Windows + Hugging Face: if you see `HF_HUB_DISABLE_SYMLINKS_WARNING`, enable Developer Mode or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

Telemetry: distributions of the merged Olive engine may collect usage data. Disable per command with `--disable_telemetry`. See `docs/Privacy.md`.

## First run: optimize a small model

Optimize [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) to INT4:

```bash
lmcli optimize \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --precision int4 \
    --output_path models/qwen
```

PowerShell:

```powershell
lmcli optimize `
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct `
    --precision int4 `
    --output_path models/qwen
```

This acquires the HF model, GPTQ-quantizes to int4, captures the ONNX graph, and writes artifacts under `models/qwen`.

Interactive config wizard (generates an Olive/LMTuner command):

```bash
lmcli init --output_path ./olive-output
```

Dry-run / inspect without executing:

```bash
lmcli optimize --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --precision int4 --output_path models/qwen --dry_run
```

Chat against ONNX Runtime GenAI using [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py).

## Diffusion LoRA (SD, SDXL, Flux)

One path: `lmcli diffusion-lora` -> `olive/cli/diffusion_lora.py` -> `olive/passes/diffusers/lora.py` (`SDLoRA`). Feature write-up: `docs/source/features/sd-lora.md`.

Needs the diffusers extra:

```bash
pip install -e ".[diffusers]"
```

```bash
# Local image folder
lmcli diffusion-lora -m runwayml/stable-diffusion-v1-5 -d ./train_images

# Hugging Face dataset
lmcli diffusion-lora -m runwayml/stable-diffusion-v1-5 --data_name linoyts/Tuxemon --caption_column prompt

# SDXL
lmcli diffusion-lora -m stabilityai/stable-diffusion-xl-base-1.0 -d ./train_images

# Flux (higher LoRA rank)
lmcli diffusion-lora -m black-forest-labs/FLUX.1-dev -d ./train_images -r 32
```

Useful flags: `-o/--output_path` (default `diffusion-lora-adapter`), `--model_variant auto|sd|sdxl|sd3|flux|sana` (not `sd15`), `-r/--lora_r`, `--max_train_steps`, `--mixed_precision bf16`.

Opt-in PEFT flags (defaults remain current Olive LoRA — gaussian init, attention-projection target modules, DoRA/RSLoRA off, `trust_remote_code=False`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--use_dora` | off | PEFT DoRA |
| `--use_rslora` | off | PEFT RSLoRA |
| `--init_lora_weights gaussian or pissa` | `gaussian` | Set `pissa` for PiSSA init |
| `--target_modules all-linear` | auto (attn projections) | Target every linear layer |
| `--trust_remote_code` | off | Forwarded to `from_pretrained` |

Install extras with `pip install -e ".[diffusers]"` (the SDLoRA pass extra_dependencies key is `diffusers`, not `sd-lora`).




## Agent protocols (ACP, MCP 2.0, Activity)

LMTuner exposes the real tuner CLI (`optimize`, `auto-opt`, `finetune`, `diffusion-lora`, `capture-onnx-graph`, `run`, `benchmark`) over three protocols. Spec URLs are the source of truth — this tree does not invent dialects.

### ACP — Zed Agent Client Protocol

- Spec: https://agentclientprotocol.com
- Streamable HTTP/WS RFD: https://agentclientprotocol.com/rfds/streamable-http-websocket-transport
- Python SDK: https://agentclientprotocol.github.io/python-sdk/web-transport/

```bash
# stdio JSON-RPC (initialize, session/new, session/prompt, session/cancel, session/close)
lmcli acp

# Streamable HTTP + WebSocket on /acp (HTTP/2 via Hypercorn — not uvicorn)
pip install -e ".[acp]"
lmcli acp --transport http --host 127.0.0.1 --port 8000
```

Wire rules: `POST /acp` `initialize` returns **200** with `Acp-Connection-Id`; other POSTs return **202**; `GET /acp` opens connection-scoped SSE (`Acp-Connection-Id`) or session-scoped SSE (`Acp-Connection-Id` + `Acp-Session-Id`); `DELETE /acp` ends the connection. Prefer the official SDK extra when present; otherwise LMTuner's built-in ASGI adapter implements the RFD.

### MCP 2.0 — 2026-07-28 stateless Streamable HTTP

- Spec: https://modelcontextprotocol.io/specification/2026-07-28
- Transport: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http
- Blog: https://blog.modelcontextprotocol.io/posts/2026-07-28/
- Tasks extension: https://modelcontextprotocol.io/extensions/tasks/overview

```bash
lmcli mcp                          # stdio
pip install -e ".[mcp]"
lmcli mcp --transport http --host 127.0.0.1 --port 8765
```

Single endpoint `POST /mcp`. Every request is self-contained. Headers: `MCP-Protocol-Version: 2026-07-28`, `Mcp-Method`, and `Mcp-Name` for `tools/call`. Body `_meta` carries `io.modelcontextprotocol/protocolVersion`, `clientInfo`, and `clientCapabilities`. **No** initialize handshake and **no** `Mcp-Session-Id` (ignored if sent). Optional `server/discover`. Long-running tools return MCP Tasks (`resultType: "task"`) when the client advertises `io.modelcontextprotocol/tasks`.

The older `mcp/` tree is upstream Olive's FastMCP stdio helper; `lmcli mcp` is the 2026-07-28 implementation.

### Activity Protocol — Microsoft 365 Agents SDK

- Spec: https://learn.microsoft.com/en-us/microsoft-365/agents-sdk/activity-protocol
- Schema: https://github.com/microsoft/Agents/blob/main/specs/activity/protocol-activity.md

This is **not** IBM Agent Commerce Protocol and not A2A.

```bash
lmcli copilot activity serve --host 127.0.0.1 --port 3978
lmcli copilot activity emit --type typing
lmcli copilot activity emit --type invoke --name optimize --value-json "{\"model_name_or_path\":\"Qwen/Qwen2.5-0.5B-Instruct\",\"_dry_run\":true}"
```

POST Activity JSON to `/activity` or `/api/messages`. Types: `message`, `typing`, `invoke`, `event`, `conversationUpdate`. Invoke names map to tuner operations (`optimize`, `finetune`, `diffusion-lora`, ...). Existing `lmcli copilot --info|--suggest|--best-practices|--example` helpers are unchanged.

## Foundry Local inference

Tree: `harness/foundry-local/`. Product README: `harness/foundry-local/README.md`. Samples: `harness/foundry-local/samples/`. Official docs: https://learn.microsoft.com/azure/foundry-local/

Install the runtime/SDK (do not reimplement it):

```bash
pip install foundry-local-sdk
```

Python chat with catalog model `qwen2.5-0.5b` (same family as the optimize quickstart):

```python
from foundry_local_sdk import Configuration, FoundryLocalManager

config = Configuration(app_name="lmtuner_foundry")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

model = manager.catalog.get_model("qwen2.5-0.5b")
model.download()
model.load()
client = model.get_chat_client()
response = client.complete_chat([{"role": "user", "content": "What is the golden ratio?"}])
print(response.choices[0].message.content)
model.unload()
```

Runnable sample: `harness/foundry-local/samples/python/native-chat-completions/`.

Optional CLI (public preview asset from Foundry-Local releases):

```bash
foundry run qwen2.5-0.5b
foundry model list
```

Reference: https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-cli

This Foundry snapshot includes Bring Your Own Local Model (BYOM) APIs so an LMTuner-produced ONNX under `models/qwen` can be loaded in-app; see `harness/foundry-local/` and Microsoft Learn. Optional in-process web server samples: `harness/foundry-local/samples/python/web-server/`.

## OptiGuide

Tree: `harness/optiguide/`. Top README: `harness/optiguide/README.md`.

| Subtree | What it is |
|---------|------------|
| `harness/optiguide/what-if/` | OptiGuide what-if analysis for supply-chain optimization (paper + code + notebook) |
| `harness/optiguide/milp-evolve/` | MILP-Evolve / foundation models for mixed-integer linear programming |
| `harness/optiguide/optimind/` | OptiMind: teaching LLMs to think like optimization experts |

What-if local path (from their README):

```bash
pip install optiguide
```

Then open `harness/optiguide/what-if/notebook/optiguide_example.ipynb`. That project also expects Gurobi and an `OAI_CONFIG_LIST` for the LLM. Details: `harness/optiguide/what-if/README.md`.

LMTuner does not wrap OptiGuide's solvers; use their notebooks and packages as-is when you need decision-intelligence / what-if guidance alongside model optimization.

## lmcli command map

Launcher: `olive/cli/launcher.py`. Usage: `lmcli <command> ...` or `python -m olive <command> ...`.

| Command | Purpose |
|---------|---------|
| `init` | Interactive wizard to generate an optimization command |
| `run` | Run an Olive/LMTuner workflow config |
| `run-pass` | Run a single pass |
| `auto-opt` | Automatic optimizer |
| `optimize` | Comprehensive pass scheduling (quickstart) |
| `capture-onnx-graph` | Export HF/PyTorch model to ONNX |
| `diffusion-lora` | Train LoRA for SD / SDXL / SD3 / Flux / Sana |
| `acp` | Zed Agent Client Protocol (stdio or Streamable HTTP/WS on `/acp`) |
| `mcp` | MCP 2.0 2026-07-28 (stdio or stateless Streamable HTTP on `/mcp`) |
| `finetune` | PEFT fine-tune |
| `generate-adapter` | ONNX model with adapters as inputs |
| `convert-adapters` | Convert adapters |
| `quantize` | Quantize |
| `tune-session-params` | ONNX Runtime session parameter tuning |
| `generate-cost-model` | Cost model |
| `configure-qualcomm-sdk` | Qualcomm SDK helper |
| `shared-cache` | Shared cache operations |
| `extract-adapters` | Extract adapters |
| `generate-model-package` | Package a model |
| `benchmark` | Evaluate with lm-eval |
| `copilot` | GitHub Copilot helpers (`--info`, `--suggest`, `--best-practices`, `--example`) plus `activity serve` / `activity emit` |

```bash
lmcli --help
lmcli optimize --help
lmcli copilot --info
lmcli copilot --suggest qwen
lmcli copilot --example optimize
lmcli acp --help
lmcli mcp --help
lmcli copilot activity --help
```

## End-to-end workflow

1. **Install** this clone: `pip install -e ".[auto-opt]"` plus `transformers` and `onnxruntime-genai`.
2. **Optimize** a small chat model: `lmcli optimize --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --precision int4 --output_path models/qwen`.
3. **(Optional) adapters / diffusion**: `pip install -e ".[diffusers]"` then `lmcli diffusion-lora ...` or `lmcli finetune ...`.
4. **Infer locally**: ONNX Runtime GenAI `model-chat.py` on `models/qwen`, **or** Foundry Local SDK/CLI (`qwen2.5-0.5b` catalog, or BYOM with the ONNX you just built). Start from `harness/foundry-local/samples/python/native-chat-completions/`.
5. **Guidance / what-if** (operations research, not model-graph passes): `harness/optiguide/what-if/notebook/optiguide_example.ipynb`.
6. **Package / MCP / skills** (from merged Olive): `lmcli generate-model-package`, `mcp/README.md`, `skills/olive/SKILL.md`.

Need a pass listed? `docs/source/` is the Olive Sphinx doc set; Copilot examples: `lmcli copilot --example finetune`.
