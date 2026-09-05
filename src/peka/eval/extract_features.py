"""Extract per-spot PEKA features from a trained Phase 2 checkpoint.

Re-creates the model architecture, loads the Lightning checkpoint (stripping
the `model.` prefix and dropping the classifier weights), and runs forward
across all breast adata files. Output: one `.npy` per adata file at
`<output_dir>/<HEST_breast_adata_i>.npy`.
"""
from pathlib import Path
from typing import List, Optional

import peka  # noqa: F401
import torch
from hydra_zen import instantiate

from peka.DownstreamTasks_helper.inference import inference_from_folder

from peka import logger
from peka.paths import (
    BREAST_DATASET_DIR,
    DEFAULT_SCLLM,
    DEFAULT_SCLLM_CKPT,
)


def _strip_model_prefix(state_dict: dict) -> dict:
    """Remove the `model.` prefix Lightning wraps weights with."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model."):]] = v
        else:
            out[k] = v
    return out


def _drop_classifier_weights(state_dict: dict) -> dict:
    """The Phase 2 checkpoint also contains the frozen classifier — drop it.

    We only want the encoder + translate_model for feature extraction.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}


def extract_peka_features(
    model_config,
    checkpoint_path: Path,
    output_dir: Path,
    target_dim: int = 1536,
    dataset_dir: Optional[Path] = None,
    scllm: str = DEFAULT_SCLLM,
    scllm_ckpt: str = DEFAULT_SCLLM_CKPT,
    device: str = "cuda",
    include_slides: Optional[List[str]] = None,
) -> Path:
    """Extract translate-MLP features from a trained PEKA model.

    Args:
        model_config: same hydra-zen config used in Phase 2 (rebuilds same arch)
        checkpoint_path: Lightning .ckpt from Phase 2
        output_dir: where to save per-adata `.npy` features
        target_dim: scLLM embedding dim (1536)
        dataset_dir: breast dataset folder (default: DATA/breast/breast_in_hest)
        scllm: teacher name
        scllm_ckpt: teacher checkpoint name
        device: 'cuda' or 'cpu'
    """
    if dataset_dir is None:
        dataset_dir = BREAST_DATASET_DIR
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (slow)")
        device = "cpu"

    # Rebuild model
    model = instantiate(model_config, target_dim=target_dim)

    # Load checkpoint
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _drop_classifier_weights(_strip_model_prefix(state_dict))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model = model.to(device).eval()

    inference_from_folder(
        model=model,
        dataset_save_folder=str(dataset_dir),
        scLLM_emb_name=scllm,
        scLLM_emb_ckpt=scllm_ckpt,
        output_dir=str(output_dir),
        adata_prefix="HEST_breast_adata_",
        img_prefix="patch_224_0.5_",
        include_slides=include_slides,
        device=device,
    )
    logger.info(f"Extracted features → {output_dir}")
    return output_dir
