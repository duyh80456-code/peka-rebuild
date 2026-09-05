#!/usr/bin/env python
"""Extract PEKA features from a trained Phase 2 checkpoint.

Output: per-adata `.npy` files at
  DATA/breast/breast_in_hest/peka_embed/{encoder}_{peft}/{scllm}/{ckpt}/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.configs import build_model_config, ENCODER_TABLE, PEFT_METHODS  # noqa: E402
from peka.eval import extract_peka_features  # noqa: E402
from peka.paths import (  # noqa: E402
    BREAST_DATASET_DIR, DEFAULT_SCLLM, DEFAULT_SCLLM_CKPT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PEKA features")
    parser.add_argument("--encoder", choices=list(ENCODER_TABLE), required=True)
    parser.add_argument("--peft", choices=list(PEFT_METHODS), required=True)
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint .ckpt")
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_dim", type=int, default=1536,
                        help="scLLM embedding dim — translate_model output dim")
    parser.add_argument("--scllm", default=DEFAULT_SCLLM)
    parser.add_argument("--scllm_ckpt", default=DEFAULT_SCLLM_CKPT)
    parser.add_argument("--output_dir", default=None,
                        help="Default: DATA/breast/breast_in_hest/peka_embed/{encoder}_{peft}/...")
    parser.add_argument("--split_manifest", default=None,
                        help="Slide split JSON. Required with --fold.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Extract slides for this leakage-free outer fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    include_slides = None
    if args.fold is not None:
        from peka.splits import (
            file_sha256, get_fold, load_or_create_manifest,
            validate_provenance, write_provenance, manifest_digest,
        )
        manifest = load_or_create_manifest(args.split_manifest, n_splits=args.folds)
        split = get_fold(manifest, args.fold)
        include_slides = split["train"] + split["val"] + split["test"]
        if not args.run_id:
            parser.error("--run_id is required with --fold")
        validate_provenance(
            Path(args.ckpt).parent / "provenance.json",
            manifest, args.fold, run_id=args.run_id,
            encoder=args.encoder, peft=args.peft,
            checkpoint=str(Path(args.ckpt).resolve()),
            checkpoint_sha256=file_sha256(Path(args.ckpt)),
        )
    elif args.split_manifest is not None:
        parser.error("--split_manifest requires --fold")

    if args.output_dir is None:
        tag = f"{args.encoder}_{args.peft}"
        if args.fold is not None:
            tag += f"_fold_{args.fold}_{args.run_id}"
        out = BREAST_DATASET_DIR / "peka_embed" / tag / args.scllm / args.scllm_ckpt
    else:
        out = Path(args.output_dir)
    if args.fold is not None and out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Fold feature directory is not empty: {out}. Use a new --run_id."
        )
    out.mkdir(parents=True, exist_ok=True)

    cfg = build_model_config(
        encoder=args.encoder, peft=args.peft,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    extract_peka_features(
        model_config=cfg,
        checkpoint_path=Path(args.ckpt),
        output_dir=out,
        target_dim=args.target_dim,
        scllm=args.scllm,
        scllm_ckpt=args.scllm_ckpt,
        include_slides=include_slides,
    )
    if args.fold is not None:
        feature_hashes = {
            path.name: file_sha256(path)
            for path in sorted(out.glob("HEST_breast_adata_*.npy"))
        }
        if set(feature_hashes) != {f"{slide}.npy" for slide in include_slides}:
            raise ValueError("Extracted feature inventory does not match the fold manifest")
        write_provenance(
            out / "provenance.json",
            manifest_digest=manifest_digest(manifest), fold=args.fold,
            train=split["train"], val=split["val"], test=split["test"],
            run_id=args.run_id, encoder=args.encoder, peft=args.peft,
            checkpoint=str(Path(args.ckpt).resolve()),
            checkpoint_sha256=file_sha256(Path(args.ckpt)),
            feature_hashes=feature_hashes,
        )
    logger.info(f"Features at {out}")


if __name__ == "__main__":
    main()
