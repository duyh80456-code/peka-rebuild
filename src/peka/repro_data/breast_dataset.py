"""Breast dataset factory.

Wraps `peka.Data.dataset_helper.BatchLocalityDataset` and ensures the
`dataset_config.csv` is in place before any peka helper tries to read it.

Note: `BatchLocalityDataset.adata_prefix` is hardcoded to `"HEST_breast_adata_"`
(see peka/Data/dataset_helper.py:30) which matches the prefix written by
`peka.Data.hest1k_helper.gene_name_alignment` — fine for breast-only.
"""
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd

import peka  # noqa: F401
from hydra_zen import instantiate
from torch.utils.data import DataLoader

from peka import logger
from peka.paths import (
    BREAST_DATA_DIR,
    BREAST_DATASET_DIR,
    BREAST_DATASET_NAME,
    HEST1K_STORAGE_PATH,
    SUPPORT_DIR,
)


def ensure_breast_dataset_csv() -> Path:
    """Materialize <DATA>/breast/dataset_config.csv with absolute paths filled in.

    `peka.Hydra_helper.dataset_part_helpers.dataset_generator` reads
    `<data_root>/<tissue>/dataset_config.csv` and asserts the requested task is
    listed there. The shipped support CSV has empty `dataset_storage_folder`
    and `hest_loc` (workspace-relative), so we fill them in based on
    `peka.paths` before writing.
    """
    BREAST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = BREAST_DATA_DIR / "dataset_config.csv"
    source = SUPPORT_DIR / "peka_breast_datasets.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}. The repo should ship this file.")

    df = pd.read_csv(source)
    df["dataset_storage_folder"] = str(BREAST_DATA_DIR)
    df["hest_loc"] = str(HEST1K_STORAGE_PATH)
    df.to_csv(target, index=False)
    logger.info(f"Wrote {target} (storage={BREAST_DATA_DIR}, hest={HEST1K_STORAGE_PATH})")
    return target


def build_breast_loaders(
    dataset_config,
    data_root: Path = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """Instantiate the breast train/val DataLoaders + return embedding dim.

    Args:
        dataset_config: hydra-zen config from `build_breast_in_hest_config()`
        data_root: directory containing `<tissue>/dataset_config.csv` (default: DATA/)

    Returns:
        (train_loader, val_loader, embedding_dim)
    """
    if data_root is None:
        from peka.paths import DATA_ROOT
        data_root = DATA_ROOT
    ensure_breast_dataset_csv()
    train_loader, val_loader, embedding_dim = instantiate(
        dataset_config,
        data_root=str(data_root),
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
                f"Embedding dim: {embedding_dim}")
    return train_loader, val_loader, embedding_dim


def get_breast_aligned_adata_dir() -> Path:
    """Return the folder where gene-name-aligned breast adata files live."""
    return BREAST_DATASET_DIR / "aligned_adata"


def get_breast_paired_seq_dir(scllm: str = "scFoundation",
                              ckpt: str = "default_model") -> Path:
    """Return the per-scLLM `paired_seq/` folder (filtered adata + cluster labels)."""
    return BREAST_DATASET_DIR / "scLLM_embed" / scllm / ckpt / "paired_seq"


def get_breast_embedding_dir(scllm: str = "scFoundation",
                             ckpt: str = "default_model") -> Path:
    """Return the per-scLLM `embeddings/` folder (.npy files of teacher embeddings)."""
    return BREAST_DATASET_DIR / "scLLM_embed" / scllm / ckpt / "embeddings"
