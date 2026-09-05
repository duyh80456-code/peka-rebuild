#!/usr/bin/env python
"""Phase 1: train the MLP teacher classifier on scFoundation embeddings.

The MLP predicts k-means cluster pseudo-labels from frozen scLLM embeddings.
This MLP is then frozen and reused as the distillation teacher in Phase 2.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
import torch  # noqa: E402
from peka import logger  # noqa: E402
from peka.configs import build_breast_in_hest_config  # noqa: E402
from peka.repro_data import build_breast_loaders  # noqa: E402
from peka.train import train_teacher_mlp  # noqa: E402
from peka.paths import (  # noqa: E402
    BREAST_DATASET_DIR, DEFAULT_N_CLUSTERS, PRETRAINED_ROOT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: MLP teacher training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_path",
                        default=None)
    parser.add_argument("--label_name",
                        default=None)
    parser.add_argument("--split_manifest", default=None,
                        help="Slide split JSON. Required with --fold.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train on one leakage-free outer fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    train_slides = val_slides = None
    if args.fold is not None:
        from peka.splits import (
            get_fold, load_or_create_manifest, run_fold_label_name,
            file_sha256, label_hashes, manifest_digest,
            validate_provenance, write_provenance,
        )
        manifest = load_or_create_manifest(args.split_manifest, n_splits=args.folds)
        split = get_fold(manifest, args.fold)
        train_slides, val_slides = split["train"], split["val"]
        if not args.run_id:
            parser.error("--run_id is required with --fold")
        if args.label_name is None:
            args.label_name = run_fold_label_name(
                args.num_classes, args.fold, args.run_id, manifest,
            )
        if args.output_path is None:
            args.output_path = str(
                PRETRAINED_ROOT / f"fold_{args.fold}_{args.run_id}" / "teacher_mlp.pt"
            )
        validate_provenance(
            BREAST_DATASET_DIR / "splits"
            / f"fold_{args.fold}_{args.run_id}_kmeans.provenance.json",
            manifest, args.fold, n_clusters=args.num_classes,
            label_name=args.label_name, run_id=args.run_id,
            scllm="scFoundation", scllm_ckpt="default_model",
            label_hashes=label_hashes(
                BREAST_DATASET_DIR, "scFoundation", "default_model",
                train_slides + val_slides, args.label_name,
            ),
        )
        logger.info(f"Fold {args.fold} train slides: {train_slides}")
        logger.info(f"Fold {args.fold} val slides: {val_slides}")
        logger.info(f"Fold {args.fold} held-out test slides: {split['test']}")
    else:
        if args.split_manifest is not None:
            parser.error("--split_manifest requires --fold")
        args.label_name = args.label_name or f"gen_clustered_label_{DEFAULT_N_CLUSTERS}"
        args.output_path = args.output_path or str(PRETRAINED_ROOT / "teacher_mlp.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Phase 1: device={device}, epochs={args.epochs}, lr={args.lr}")
    if args.fold is not None and Path(args.output_path).exists():
        raise FileExistsError(
            f"Fold teacher already exists: {args.output_path}. Use a new --run_id."
        )

    cfg = build_breast_in_hest_config(
        label_name=args.label_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_slides=train_slides,
        val_slides=val_slides,
    )
    train_loader, val_loader, embedding_dim = build_breast_loaders(cfg)

    train_teacher_mlp(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        save_path=Path(args.output_path),
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    if args.fold is not None:
        write_provenance(
            Path(args.output_path).with_suffix(".provenance.json"),
            manifest_digest=manifest_digest(manifest), fold=args.fold,
            train=split["train"], val=split["val"], test=split["test"],
            run_id=args.run_id, num_classes=args.num_classes,
            teacher_sha256=file_sha256(Path(args.output_path)),
        )


if __name__ == "__main__":
    main()
