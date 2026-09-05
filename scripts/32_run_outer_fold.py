#!/usr/bin/env python
"""Run one leakage-free two-phase outer fold end to end."""
import argparse
import shlex
import subprocess
import sys
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peka.paths import BREAST_DATASET_DIR, PRETRAINED_ROOT  # noqa: E402
from peka.splits import default_split_manifest, load_or_create_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one held-out slide fold")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--encoder", choices=["H-optimus-0", "UNI"], default="H-optimus-0")
    parser.add_argument("--peft", choices=["bone", "lora", "adalora", "hra"], default="bone")
    parser.add_argument("--phase1_epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--with_logger", choices=["wandb", "csv"], default="wandb")
    parser.add_argument("--use_gpu_kmeans", action="store_true")
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_id", default=None,
                        help="Unique artifact namespace; defaults to a timestamp")
    args = parser.parse_args()

    manifest_path = Path(args.split_manifest) if args.split_manifest else default_split_manifest(args.folds)
    if not args.dry_run:
        load_or_create_manifest(manifest_path, n_splits=args.folds)
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    teacher = PRETRAINED_ROOT / f"fold_{args.fold}_{run_id}" / "teacher_mlp.pt"
    exp_name = f"{args.encoder}_{args.peft}_breast_in_hest_phase1_fold_{args.fold}_{run_id}"
    checkpoint = PRETRAINED_ROOT / exp_name / "last.ckpt"
    feature_dir = (
        BREAST_DATASET_DIR / "peka_embed"
        / f"{args.encoder}_{args.peft}_fold_{args.fold}_{run_id}"
        / "scFoundation" / "default_model"
    )
    gene_list = BREAST_DATASET_DIR / "splits" / f"fold_{args.fold}_{run_id}_top_50_genes.json"
    if not args.dry_run:
        existing = [
            path for path in
            (teacher.parent, checkpoint.parent, feature_dir, gene_list)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"Run ID {run_id} already has artifacts: {existing}. "
                "Choose a new --run_id; existing fold artifacts are never reused."
            )

    split_common = ["--fold", str(args.fold), "--folds", str(args.folds),
                    "--split_manifest", str(manifest_path)]
    artifact_common = [*split_common, "--run_id", run_id]
    commands = [
        [sys.executable, str(ROOT / "scripts/13_kmeans_cluster_labels.py"),
         *split_common, "--run_id", run_id]
        + (["--use_gpu"] if args.use_gpu_kmeans else []),
        [sys.executable, str(ROOT / "scripts/14_smoke_test_loader.py"),
         *split_common, "--run_id", run_id,
         "--num_batches", "1", "--num_workers", "0"],
        [sys.executable, str(ROOT / "scripts/20_train_phase1_mlp.py"), *artifact_common,
         "--epochs", str(args.phase1_epochs), "--output_path", str(teacher)],
        [sys.executable, str(ROOT / "scripts/30_train_phase2_kd.py"), *artifact_common,
         "--encoder", args.encoder, "--peft", args.peft, "--use-phase1",
         "--phase1_ckpt", str(teacher), "--epochs", str(args.epochs),
         "--batch_size", str(args.batch_size), "--with_logger", args.with_logger],
        [sys.executable, str(ROOT / "scripts/40_extract_peka_features.py"), *artifact_common,
         "--encoder", args.encoder, "--peft", args.peft, "--ckpt", str(checkpoint),
         "--output_dir", str(feature_dir)],
        [sys.executable, str(ROOT / "scripts/41_compute_hvg_top50.py"), *artifact_common,
         "--output_json", str(gene_list)],
        [sys.executable, str(ROOT / "scripts/50_eval_gene_regression_kfold.py"), *artifact_common,
         "--encoder", args.encoder, "--peft", args.peft,
         "--feature_dir", str(feature_dir), "--gene_list_json", str(gene_list)],
    ]
    if args.limit_train_batches is not None:
        commands[3] += ["--limit_train_batches", str(args.limit_train_batches)]
    if args.limit_val_batches is not None:
        commands[3] += ["--limit_val_batches", str(args.limit_val_batches)]

    for command in commands:
        print("$", shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
