# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GPU-free unit tests for opt-in PEFT LoRA flags. Defaults must stay Olive LoRA."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from olive.constants import DiffusersModelVariant
from olive.passes.diffusers.lora import (
    ALL_LINEAR,
    SDLoRA,
    peft_lora_kwargs,
    pretrained_kwargs,
    resolve_target_modules,
)


def _cfg(**kwargs):
    defaults = {
        "r": 16,
        "alpha": None,
        "lora_dropout": 0.0,
        "init_lora_weights": "gaussian",
        "target_modules": None,
        "use_dora": False,
        "use_rslora": False,
        "trust_remote_code": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_resolve_target_modules_all_linear():
    assert resolve_target_modules(None, ["to_k"]) == ["to_k"]
    assert resolve_target_modules(["all-linear"], ["to_k"]) == ALL_LINEAR
    assert resolve_target_modules("all-linear", ["to_k"]) == ALL_LINEAR
    assert resolve_target_modules(["to_q", "to_k"], ["to_k"]) == ["to_q", "to_k"]


def test_pretrained_kwargs_default_false():
    assert pretrained_kwargs(_cfg()) == {"trust_remote_code": False}
    assert pretrained_kwargs(_cfg(trust_remote_code=True)) == {"trust_remote_code": True}


def test_build_peft_lora_config_defaults_match_olive():
    kwargs = peft_lora_kwargs(_cfg(), ["to_k", "to_q", "to_v", "to_out.0"])
    assert kwargs["r"] == 16
    assert kwargs["lora_alpha"] == 16
    assert kwargs["init_lora_weights"] == "gaussian"
    assert kwargs["target_modules"] == ["to_k", "to_q", "to_v", "to_out.0"]
    assert "use_dora" not in kwargs
    assert "use_rslora" not in kwargs


def test_build_peft_lora_config_opt_in_dora_rslora_pissa_all_linear():
    kwargs = peft_lora_kwargs(
        _cfg(use_dora=True, use_rslora=True, init_lora_weights="pissa", target_modules=["all-linear"], alpha=32),
        ["to_k"],
    )
    assert kwargs["use_dora"] is True
    assert kwargs["use_rslora"] is True
    assert kwargs["init_lora_weights"] == "pissa"
    assert kwargs["target_modules"] == "all-linear"
    assert kwargs["lora_alpha"] == 32


def test_pass_config_defaults_include_opt_in_flags():
    cfg = SDLoRA._default_config(MagicMock())
    assert cfg["use_dora"].default_value is False
    assert cfg["use_rslora"].default_value is False
    assert cfg["init_lora_weights"].default_value == "gaussian"
    assert cfg["trust_remote_code"].default_value is False
    assert "(not sd15)" in cfg["model_variant"].description
    assert "auto|sd|sdxl|sd3|flux|sana" in cfg["model_variant"].description
    assert DiffusersModelVariant.SD.value == "sd"
    assert DiffusersModelVariant.SANA.value == "sana"


def test_diffusion_lora_cli_flags_present():
    from olive.cli.launcher import get_cli_parser

    parser = get_cli_parser()
    args = parser.parse_args(
        [
            "diffusion-lora",
            "-m",
            "runwayml/stable-diffusion-v1-5",
            "-d",
            "./imgs",
            "--use_dora",
            "--use_rslora",
            "--init_lora_weights",
            "pissa",
            "--target_modules",
            "all-linear",
            "--model_variant",
            "sd",
        ]
    )
    assert args.use_dora is True
    assert args.use_rslora is True
    assert args.init_lora_weights == "pissa"
    assert args.target_modules == "all-linear"
    assert args.model_variant == "sd"
    assert args.trust_remote_code is False
