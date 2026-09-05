"""Model config factory.

Builds the (image_encoder + translate_MLP + PEFT) model for any combination
of {H-optimus-0, UNI} × {lora, adalora, hra, bone}.

This file IS THE FIX for the broken `_target_: histomil2.*` references in
`hydra_zen/Configs/Models/*.yaml` — we point `builds()` directly at the
real `peka.Hydra_helper.model_part_helpers.model_config` function.
"""
from typing import Dict

import peka  # noqa: F401 — ensures sys.path is set up
from hydra_zen import builds

from peka.Hydra_helper.model_part_helpers import model_config


# (timm/HuggingFace name, encoder output dim)
ENCODER_TABLE: Dict[str, tuple] = {
    "H-optimus-0": ("hf-hub:bioptimus/H-optimus-0", 1536),
    "UNI":         ("hf-hub:MahmoodLab/UNI",        1024),
}

PEFT_METHODS = ("lora", "adalora", "hra", "bone")


def _peft_kwargs(peft: str) -> dict:
    """Return PEFT-method-specific extra kwargs.

    These extra params are read by `model_config` via `translate_additional_params`
    (see peka/Hydra_helper/model_part_helpers.py:36-70).
    """
    base = {"mid_dim": 512}  # translate MLP hidden dim
    if peft == "lora":
        return {**base, "peft_method": "lora"}
    if peft == "adalora":
        return {
            **base,
            "peft_method": "adalora",
            "target_r": 128,
            "init_r": 256,
            "beta1": 0.85,
            "beta2": 0.85,
            "total_step": 10000,
        }
    if peft == "hra":
        return {**base, "peft_method": "hra", "apply_GS": True}
    if peft == "bone":
        return {**base, "peft_method": "bone", "init_weights": True}
    raise ValueError(f"Unknown PEFT method: {peft}. Expected one of {PEFT_METHODS}")


def build_model_config(
    encoder: str,
    peft: str,
    lora_r: int = 256,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    pre_trained_ckpt_path: str = None,
):
    """Build a hydra-zen config for the PEKA student model.

    Args:
        encoder: one of {"H-optimus-0", "UNI"}
        peft: one of {"lora", "adalora", "hra", "bone"}
        lora_r: low-rank dimension (paper default: 256)
        lora_alpha: scale factor (paper default: 32)
        lora_dropout: dropout rate (paper default: 0.1)
        pre_trained_ckpt_path: optional checkpoint to warm-start from
    """
    if encoder not in ENCODER_TABLE:
        raise ValueError(f"Unknown encoder '{encoder}'. Expected one of {list(ENCODER_TABLE)}")
    if peft not in PEFT_METHODS:
        raise ValueError(f"Unknown PEFT '{peft}'. Expected one of {PEFT_METHODS}")

    encoder_name, encoder_dim = ENCODER_TABLE[encoder]
    return builds(
        model_config,
        encoder_name=encoder_name,
        encoder_output_dim=encoder_dim,
        translate_module_name="MLP",
        translate_additional_params=_peft_kwargs(peft),
        target_dim="???",  # filled at instantiate-time = scLLM embedding dim (1536)
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        pre_trained_ckpt_path=pre_trained_ckpt_path,
        populate_full_signature=True,
    )
