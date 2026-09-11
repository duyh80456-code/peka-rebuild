#!/usr/bin/env python
"""Extract frozen image-encoder features — the control PEKA has to beat.

`50_eval_gene_regression_kfold.py --feature_type image_encoder` reads
`DATA/breast/breast_in_hest/patches_embed/<encoder>/`, which nothing in the
pipeline ever writes. Without it there is no baseline measured under the same
slide-holdout protocol as PEKA, and a PEKA score cannot be interpreted at all.

This is the untouched backbone: no PEFT adapter, no translate MLP, no
checkpoint. Output is the encoder's own representation.
"""
import argparse
import os
import sys
from pathlib import Path

# Phai dat TRUOC khi torch khoi tao CUDA. Lan chay dau OOM o batch 130/152 voi
# 1.43 GiB "reserved but unallocated" -- do la phan manh, khong phai thieu that.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: F401,E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from peka import logger  # noqa: E402
from peka.DownstreamTasks_helper.inference import inference_from_folder  # noqa: E402
from peka.configs.model import ENCODER_TABLE  # noqa: E402
from peka.paths import (  # noqa: E402
    BREAST_DATASET_DIR, DEFAULT_SCLLM, DEFAULT_SCLLM_CKPT,
)


class _RawEncoder(nn.Module):
    """inference_from_folder calls model(img); hand back the encoder output."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, img, gene_expression_input_batch=None):
        with torch.cuda.amp.autocast():
            return self.encoder(img)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", default="H-optimus-0", choices=list(ENCODER_TABLE))
    p.add_argument("--scllm", default=DEFAULT_SCLLM)
    p.add_argument("--scllm_ckpt", default=DEFAULT_SCLLM_CKPT)
    p.add_argument("--output_dir", default=None)
    # ViT-g fp32 (~4.4 GiB) cong ban sao fp16 ma autocast giu lai, cong
    # activation cua batch 32 -> vua khit 14.5 GiB cua T4 roi tran. 8 la muc
    # chay duoc; con so trich ra khong doi, chi cham hon.
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU visible. This walks 30k patches through a 1.1B ViT-g; on "
            "CPU that is tens of hours."
        )

    encoder_name, _ = ENCODER_TABLE[args.encoder]
    # Same construction as peka.Hydra_helper.model_part_helpers.model_config,
    # so the baseline differs from PEKA only by the adapter and translate MLP.
    encoder = timm.create_model(
        encoder_name, pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    model = _RawEncoder(encoder).cuda().eval()

    out = Path(args.output_dir) if args.output_dir else (
        BREAST_DATASET_DIR / "patches_embed" / args.encoder
    )
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Frozen {args.encoder} features → {out}")

    inference_from_folder(
        model=model,
        dataset_save_folder=str(BREAST_DATASET_DIR),
        scLLM_emb_name=args.scllm,
        scLLM_emb_ckpt=args.scllm_ckpt,
        output_dir=str(out),
        adata_prefix="HEST_breast_adata_",
        img_prefix="patch_224_0.5_",
        batch_size=args.batch_size,
        device="cuda",
    )
    logger.info(f"Done. Baseline features at {out}")


if __name__ == "__main__":
    main()
