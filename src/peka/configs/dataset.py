"""Dataset config factory for breast_in_hest."""
import peka  # noqa: F401 — ensures sys.path is set up
from hydra_zen import builds

from peka.Hydra_helper.dataset_part_helpers import dataset_generator

from peka.paths import (
    BREAST_DATASET_NAME,
    DEFAULT_SCLLM,
    DEFAULT_SCLLM_CKPT,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PIXEL_SIZE,
    DEFAULT_IMG_PREFIX,
)


def build_breast_in_hest_config(
    label_name: str = "gen_clustered_label_100",
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    random_sample_barcode: bool = False,
    train_slides=None,
    val_slides=None,
):
    """Build a hydra-zen config for the breast_in_hest dataloader.

    Targets `peka.Hydra_helper.dataset_part_helpers.dataset_generator`, which
    returns `(train_loader, val_loader, embedding_dim)` when instantiated.

    Args:
        label_name: column name in adata.obs for cluster labels (k-means pseudo-labels)
        batch_size: per-loader batch size
        num_workers: dataloader workers (0 for smoke tests so tracebacks surface)
        val_ratio: fraction held out for validation
        split_seed: deterministic split seed
        random_sample_barcode: if True, randomly sample barcodes (paper trains
            with False for deterministic ordering across the breast samples)
    """
    return builds(
        dataset_generator,
        # data_root is filled at instantiate-time (see scripts/30_train_phase2_kd.py)
        data_root="???",
        tissue_name="breast",
        task=BREAST_DATASET_NAME,
        random_sample_barcode=random_sample_barcode,
        batch_switch_interval=5,
        scLLM_emb_name=DEFAULT_SCLLM,
        scLLM_emb_prefix="",
        patch_size=DEFAULT_PATCH_SIZE,
        pixel_size=DEFAULT_PIXEL_SIZE,
        img_prefix=DEFAULT_IMG_PREFIX,
        scLLM_emb_ckpt=DEFAULT_SCLLM_CKPT,
        batch_size=batch_size,
        num_workers=num_workers,
        additional_dataloader_para={
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        },
        label_name=label_name,
        split_dataset=True,
        shuffle=False,
        val_ratio=val_ratio,
        split_seed=split_seed,
        train_slides=train_slides,
        val_slides=val_slides,
        populate_full_signature=True,
    )
