#!/usr/bin/env python
"""Smoke test: instantiate the breast dataloader and fetch a few batches.

Verifies the entire data pipeline without GPU. Run after step 13.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.configs import build_breast_in_hest_config  # noqa: E402
from peka.repro_data import build_breast_loaders  # noqa: E402
from peka.paths import DEFAULT_N_CLUSTERS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test breast dataloader")
    parser.add_argument("--num_batches", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Use 0 so failures show real tracebacks")
    parser.add_argument("--label_name", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    train_slides = val_slides = None
    if args.fold is not None:
        from peka.splits import get_fold, load_or_create_manifest, run_fold_label_name
        manifest = load_or_create_manifest(args.split_manifest, n_splits=args.folds)
        split = get_fold(manifest, args.fold)
        train_slides, val_slides = split["train"], split["val"]
        if not args.run_id:
            parser.error("--run_id is required with --fold")
        args.label_name = args.label_name or run_fold_label_name(
            DEFAULT_N_CLUSTERS, args.fold, args.run_id, manifest,
        )
    else:
        if args.split_manifest is not None:
            parser.error("--split_manifest requires --fold")
        args.label_name = args.label_name or f"gen_clustered_label_{DEFAULT_N_CLUSTERS}"

    cfg = build_breast_in_hest_config(
        label_name=args.label_name,
        batch_size=4,
        num_workers=args.num_workers,
        train_slides=train_slides,
        val_slides=val_slides,
    )
    train_loader, val_loader, embedding_dim = build_breast_loaders(cfg)

    print(f"Embedding dim:        {embedding_dim}")
    print(f"Train dataset size:   {len(train_loader.dataset)}")
    print(f"Val dataset size:     {len(val_loader.dataset)}")

    failures = 0
    for split_name, loader in (("train", train_loader), ("val", val_loader)):
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            img, emb, label = batch
            print(f"\n{split_name} batch {i}:")
            print(f"  img:   {tuple(img.shape)} {img.dtype}")
            print(f"  emb:   {tuple(emb.shape)} {emb.dtype}")
            print(f"  label: {tuple(label.shape)} {label.dtype} "
                  f"min={label.min().item()} max={label.max().item()}")
            assert img.ndim == 4 and img.shape[1] == 3, f"img shape wrong: {img.shape}"
            assert img.shape[2] == 224 and img.shape[3] == 224, f"patch size wrong: {img.shape}"
            assert emb.ndim == 2 and emb.shape[1] == embedding_dim, f"emb shape wrong: {emb.shape}"
            assert label.ndim == 1, f"label shape wrong: {label.shape}"
            assert label.min().item() >= 0, "negative label"
    if failures:
        return 1
    print("\nSmoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
