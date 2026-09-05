#!/usr/bin/env python
"""Evaluate PEKA features on top-50 HVG gene expression regression.

Slide-grouped 5-fold cross-validation: 256-dim PCA → Ridge regression.
No slide contributes spots to both probe training and testing in a fold.

This prevents leakage in the PCA/Ridge probe only. For a fully held-out result,
the PEKA checkpoint and HVG list must also be trained without the test slides.

Run per (encoder, peft) experiment after extracting features.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.configs import ENCODER_TABLE, PEFT_METHODS  # noqa: E402
from peka.eval import evaluate_kfold  # noqa: E402
from peka.paths import (  # noqa: E402
    BREAST_DATASET_NAME, DEFAULT_SCLLM, OUTPUT_ROOT, SUPPORT_DIR,
)


# Map from --encoder (full name) → image_encoder_name (short tag used by
# peka.DownstreamTasks_helper.gene_expression_prediction.get_dataset_paths).
_ENCODER_SHORT = {"H-optimus-0": "H0", "UNI": "UNI"}


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold gene regression on PEKA features")
    parser.add_argument("--encoder", choices=list(ENCODER_TABLE), required=True)
    parser.add_argument("--peft", choices=list(PEFT_METHODS), default="bone")
    parser.add_argument("--feature_type", choices=["peka", "image_encoder", "scLLM"], default="peka")
    parser.add_argument("--gene_list_json",
                        default=str(SUPPORT_DIR / "top_50_genes_breast.json"))
    parser.add_argument("--output_root",
                        default=str(OUTPUT_ROOT / "eval"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of GroupKFold splits by slide (default: 5)")
    parser.add_argument("--split_manifest", default=None,
                        help="Slide split JSON. Required with --fold.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Evaluate the fixed held-out test slides for this fold")
    parser.add_argument("--feature_dir", default=None,
                        help="Fold-specific PEKA feature directory")
    parser.add_argument("--probe_only", action="store_true",
                        help="Allow grouped probing of a checkpoint trained on the full cohort")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--use_binned", action="store_true")
    parser.add_argument("--mask_zero_values", action="store_true")
    parser.add_argument("--not_use_scllm_subfolder", action="store_true")
    args = parser.parse_args()

    image_encoder_short = _ENCODER_SHORT[args.encoder]
    output_root = Path(args.output_root) / args.peft

    feature_dir = Path(args.feature_dir) if args.feature_dir else None
    if args.fold is not None and args.gene_list_json == str(SUPPORT_DIR / "top_50_genes_breast.json"):
        from peka.paths import BREAST_DATASET_DIR
        args.gene_list_json = str(
            BREAST_DATASET_DIR / "splits"
            / f"fold_{args.fold}_{args.run_id}_top_50_genes.json"
        )
    if args.fold is not None and feature_dir is None and args.feature_type == "peka":
        from peka.paths import BREAST_DATASET_DIR
        feature_dir = (
            BREAST_DATASET_DIR / "peka_embed"
            / f"{args.encoder}_{args.peft}_fold_{args.fold}_{args.run_id}"
            / DEFAULT_SCLLM / "default_model"
        )
    if args.split_manifest is not None and args.fold is None:
        parser.error("--split_manifest requires --fold")
    if args.fold is None and not args.probe_only:
        parser.error(
            "Leakage-free evaluation requires --fold. Pass --probe_only only when "
            "you intentionally want a grouped probe of a full-cohort checkpoint."
        )
    if args.fold is not None:
        if args.feature_type != "peka":
            parser.error("Leakage-free fold mode currently supports --feature_type peka only")
        if not args.run_id:
            parser.error("--run_id is required with --fold")
        from peka.splits import (
            file_sha256, load_or_create_manifest, validate_provenance,
        )
        manifest = load_or_create_manifest(
            Path(args.split_manifest) if args.split_manifest else None,
            n_splits=args.folds,
        )
        feature_provenance = validate_provenance(
            feature_dir / "provenance.json", manifest, args.fold,
            run_id=args.run_id, encoder=args.encoder, peft=args.peft,
        )
        actual_feature_hashes = {
            path.name: file_sha256(path)
            for path in sorted(feature_dir.glob("HEST_breast_adata_*.npy"))
        }
        if actual_feature_hashes != feature_provenance.get("feature_hashes"):
            raise ValueError("Fold feature files do not match their provenance hashes")
        validate_provenance(
            Path(args.gene_list_json).with_suffix(".provenance.json"),
            manifest, args.fold, run_id=args.run_id,
            hvg_sha256=file_sha256(Path(args.gene_list_json)),
        )

    df = evaluate_kfold(
        tissue_type="breast",
        dataset_name=BREAST_DATASET_NAME,
        embedder_name=DEFAULT_SCLLM,
        image_encoder_name=image_encoder_short,
        image_backbone=args.encoder,
        feature_type=args.feature_type,
        gene_list_json=Path(args.gene_list_json),
        output_root=output_root,
        epochs=args.epochs,
        use_binned=args.use_binned,
        mask_zero_values=args.mask_zero_values,
        not_use_scllm_subfolder=args.not_use_scllm_subfolder,
        n_splits=args.folds,
        split_manifest=Path(args.split_manifest) if args.split_manifest else None,
        fold=args.fold,
        feature_dir=feature_dir,
        run_id=args.run_id,
    )
    logger.info(f"Evaluation complete — {len(df)} genes evaluated, results at {output_root}")
    if "pearson_correlation" in df.columns:
        logger.info(f"Mean Pearson across genes: {df['pearson_correlation'].mean():.4f}")


if __name__ == "__main__":
    main()
