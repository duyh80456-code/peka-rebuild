"""PL model (KD_LoRA) hyperparameter config.

This is a plain dict-like object (not a builds() config) because
`pl_KD_LoRA` is constructed via `peka.Hydra_helper.pl_model_helpers.create_pl_model`
which expects a config-like object with attribute access.

Paper defaults (Section 4 / hyperparameter table):
    temperature τ = 2.0
    α = 0.5 (loss weight: α·KD + (1-α)·CE)
    classifier_hidden_dim = 512
    num_classes = 100 (k-means clusters)
"""
from dataclasses import dataclass


@dataclass
class PekaKDConfig:
    num_classes: int = 100
    classifier_hidden_dim: int = 512
    input_dim: int = 1536  # scFoundation embedding dim
    temperature: float = 2.0
    alpha: float = 0.5
    lora_save_path: str = ""  # set per-experiment by training script


def build_pl_kd_config(
    num_classes: int = 100,
    classifier_hidden_dim: int = 512,
    input_dim: int = 1536,
    temperature: float = 2.0,
    alpha: float = 0.5,
    lora_save_path: str = "",
) -> PekaKDConfig:
    return PekaKDConfig(
        num_classes=num_classes,
        classifier_hidden_dim=classifier_hidden_dim,
        input_dim=input_dim,
        temperature=temperature,
        alpha=alpha,
        lora_save_path=lora_save_path,
    )
