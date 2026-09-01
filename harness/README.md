# LMTuner harness

This directory vendors Microsoft [OptiGuide](https://github.com/microsoft/OptiGuide) and [Foundry Local](https://github.com/microsoft/Foundry-Local) as first-class trees so a clone of LMTuner has them without submodule init.

LMTuner itself **is** the Olive fork (package `lmcli`). Olive is not nested here; the merge on this branch is how Olive lives in the harness.

| Path | Upstream | Role |
|------|----------|------|
| *(repo root)* | [microsoft/Olive](https://github.com/microsoft/Olive) fork | Optimize, quantize, convert, and train adapters with `lmcli` |
| [optiguide/](optiguide/) | [microsoft/OptiGuide](https://github.com/microsoft/OptiGuide) | GenAI guidance for optimization / decision intelligence (what-if, MILP-Evolve, OptiMind) |
| [foundry-local/](foundry-local/) | [microsoft/Foundry-Local](https://github.com/microsoft/Foundry-Local) | On-device run/serve of optimized models (SDK + optional CLI/server) |

Each subtree keeps its own `LICENSE` and attribution. Do not reimplement those products here; follow their READMEs.

## Typical flow

1. Optimize with LMTuner: `lmcli optimize --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --precision int4 --output_path models/qwen`
2. Ask OptiGuide for what-if / decision-intelligence guidance: see [optiguide/README.md](optiguide/README.md) and [optiguide/what-if/README.md](optiguide/what-if/README.md)
3. Run or serve locally with Foundry Local: see [foundry-local/README.md](foundry-local/README.md)

Full walkthrough: [../docs/LMTUNER_SPEC.md](../docs/LMTUNER_SPEC.md)
