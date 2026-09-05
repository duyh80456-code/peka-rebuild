#!/usr/bin/env python
"""Compute the top-50 highly variable genes across breast adata files.

Wraps `peka.data.hvg.compute_top_hvg` over the aligned_adata folder.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.repro_data import compute_top_hvg  # noqa: E402
from peka.paths import BREAST_DATASET_DIR, SUPPORT_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-50 HVG selection")
    parser.add_argument("--input_dir",
                        default=str(BREAST_DATASET_DIR / "aligned_adata"),
                        help="Directory with HEST_breast_adata_*.h5ad")
    parser.add_argument("--output_json",
                        default=str(SUPPORT_DIR / "top_50_genes_breast.json"))
    parser.add_argument("--n_top", type=int, default=50)
    parser.add_argument("--batch_key", default="sample_id")
    parser.add_argument("--split_manifest", default=None,
                        help="Slide split JSON. Required with --fold.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Select HVGs using only this fold's training slides")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    include_slides = None
    if args.fold is not None:
        from peka.splits import (
            file_sha256, get_fold, load_or_create_manifest,
            manifest_digest, write_provenance,
        )
        manifest = load_or_create_manifest(args.split_manifest, n_splits=args.folds)
        split = get_fold(manifest, args.fold)
        include_slides = split["train"]
        if not args.run_id:
            parser.error("--run_id is required with --fold")
        if args.output_json == str(SUPPORT_DIR / "top_50_genes_breast.json"):
            args.output_json = str(
                BREAST_DATASET_DIR / "splits"
                / f"fold_{args.fold}_{args.run_id}_top_50_genes.json"
            )
    elif args.split_manifest is not None:
        parser.error("--split_manifest requires --fold")

    if args.fold is not None and Path(args.output_json).exists():
        raise FileExistsError(
            f"Fold HVG file already exists: {args.output_json}. Use a new --run_id."
        )
    compute_top_hvg(
        input_dir=Path(args.input_dir),
        output_json=Path(args.output_json),
        batch_key=args.batch_key,
        n_top_hvg=args.n_top,
        include_slides=include_slides,
    )
    if args.fold is not None:
        write_provenance(
            Path(args.output_json).with_suffix(".provenance.json"),
            manifest_digest=manifest_digest(manifest), fold=args.fold,
            train=split["train"], val=split["val"], test=split["test"],
            run_id=args.run_id,
            hvg_sha256=file_sha256(Path(args.output_json)),
        )


if __name__ == "__main__":
    main()
