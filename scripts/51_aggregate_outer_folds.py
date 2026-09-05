#!/usr/bin/env python
"""Aggregate per-gene metrics from independently trained outer folds."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peka import logger  # noqa: E402
from peka.paths import OUTPUT_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate leakage-free outer-fold results")
    parser.add_argument("--encoder_short", choices=["H0", "UNI"], default="H0")
    parser.add_argument("--peft", choices=["bone", "lora", "adalora", "hra"], default="bone")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_root", default=str(OUTPUT_ROOT / "eval"))
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    base = (
        Path(args.output_root) / args.peft / args.encoder_short
        / "peka_gene_level_raw_grouped_regression_scFoundation"
    )
    frames = []
    expected_manifest_digest = None
    for fold in range(args.folds):
        path = base / f"fold_{fold}" / args.run_id / "gene_regression_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing outer-fold result: {path}")
        frame = pd.read_csv(path)
        if set(frame["run_id"]) != {args.run_id}:
            raise ValueError(f"Mixed run IDs in {path}")
        if set(frame["fold"]) != {fold}:
            raise ValueError(f"Wrong fold provenance in {path}")
        digests = set(frame["manifest_digest"])
        if len(digests) != 1:
            raise ValueError(f"Mixed manifest digests inside {path}")
        digest = next(iter(digests))
        if expected_manifest_digest is None:
            expected_manifest_digest = digest
        elif digest != expected_manifest_digest:
            raise ValueError("Cannot aggregate folds from different split manifests")
        frame["fold"] = fold
        frames.append(frame)

    all_results = pd.concat(frames, ignore_index=True)
    metric_columns = [
        column for column in
        ("mse", "pearson_correlation", "cosine_similarity", "kl_divergence")
        if column in all_results.columns
    ]
    per_gene = all_results.groupby("gene")[metric_columns].agg(["mean", "std", "count"])
    per_gene.columns = ["_".join(column) for column in per_gene.columns]
    per_gene = per_gene.reset_index()

    base.mkdir(parents=True, exist_ok=True)
    all_path = base / f"{args.run_id}_all_outer_folds.csv"
    summary_path = base / f"{args.run_id}_outer_fold_summary_by_gene.csv"
    all_results.to_csv(all_path, index=False)
    per_gene.to_csv(summary_path, index=False)
    logger.info(f"Saved all fold results to {all_path}")
    logger.info(f"Saved per-gene summary to {summary_path}")
    if "pearson_correlation" in all_results:
        logger.info(
            f"Macro mean Pearson across fold-gene pairs: "
            f"{all_results['pearson_correlation'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
