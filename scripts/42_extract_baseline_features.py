#!/usr/bin/env python
"""Extract frozen image-encoder features — the control PEKA has to beat.

`50_eval_gene_regression_kfold.py --feature_type image_encoder` reads
`DATA/breast/breast_in_hest/patches_embed/<encoder>/`, which nothing in the
numbered pipeline ever writes. Without it there is no baseline measured under
the same protocol as PEKA, and a PEKA score cannot be interpreted at all.

This delegates to `peka.Data.hest1k_helper.extract_img_vectors`, which is the
function the original repo's `Exp_helper/5_patch_feature_embeder.py` calls.
Going through it rather than reimplementing matters in three ways that an
earlier version of this script got wrong:

  * it normalises. ToTensor() then Normalize(mean, std) — the frozen backbone
    is compared against PEKA features that are extracted from raw uint8, so
    at minimum the baseline must be fed what its own pipeline feeds it.
  * it embeds every patch in the h5, not the filter_flag subset. Script 50
    applies `image_mask` to image_encoder features and expects the full set.
  * it names files `patch_224_0.5_<idx>.npy`. Script 50 overrides embed_prefix
    to img_prefix for this feature type, so adata-style names are never found.

Batch size is 1 inside that helper, which also sidesteps the OOM a batched
version hit on a 15 GiB T4.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: F401,E402
import timm  # noqa: E402
import torch  # noqa: E402

from peka import logger  # noqa: E402
from peka.configs.model import ENCODER_TABLE  # noqa: E402
from peka.paths import (  # noqa: E402
    BREAST_DATASET_DIR, DEFAULT_PATCH_SIZE, DEFAULT_PIXEL_SIZE,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default="H-optimus-0", choices=list(ENCODER_TABLE))
    p.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    p.add_argument("--pixel_size", type=float, default=DEFAULT_PIXEL_SIZE)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU visible. This walks 30k patches through a 1.1B ViT-g; on "
            "CPU that is tens of hours."
        )

    encoder_name, num_features = ENCODER_TABLE[args.encoder]
    # Same construction as peka.Hydra_helper.model_part_helpers.model_config,
    # so the baseline differs from PEKA only by the adapter and translate MLP.
    model = timm.create_model(
        encoder_name, pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    model.to("cuda").eval()

    from peka.Data.hest1k_helper import extract_img_vectors

    out = BREAST_DATASET_DIR / "patches_embed" / args.encoder
    logger.info(f"Frozen {args.encoder} features → {out}")

    extract_img_vectors(
        subdataset_folder=str(BREAST_DATASET_DIR),
        # "hf-hub:bioptimus/H-optimus-0" -> "H-optimus-0", which is the folder
        # name script 50 looks under (image_backbone=--encoder).
        model_name=encoder_name.split(":")[-1],
        model_instance=model,
        num_features=num_features,
        patch_size=args.patch_size,
        pixel_size=args.pixel_size,
    )
    logger.info(f"Done. Baseline features at {out}")


if __name__ == "__main__":
    main()
