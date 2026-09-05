#!/usr/bin/env python
"""Generate k-means cluster pseudo-labels on scFoundation embeddings.

Writes `gen_clustered_label_<n>` column into the paired_seq/*.h5ad files.
These labels are the structure-alignment targets in the paper (Phase 1
trains MLP on them; Phase 2 student gets CE loss against them).

Falls back to sklearn KMeans if FAISS-GPU not available.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.paths import (  # noqa: E402
    BREAST_DATASET_DIR, DEFAULT_SCLLM, DEFAULT_SCLLM_CKPT, DEFAULT_N_CLUSTERS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="k-means clustering on scLLM embeddings")
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS,
                        help=f"Number of clusters (paper: {DEFAULT_N_CLUSTERS})")
    parser.add_argument("--scllm", default=DEFAULT_SCLLM)
    parser.add_argument("--ckpt", default=DEFAULT_SCLLM_CKPT)
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use FAISS-GPU if available")
    parser.add_argument("--split_manifest", default=None,
                        help="Slide split JSON. Required with --fold.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fit only on this fold's training slides")
    parser.add_argument("--folds", type=int, default=5,
                        help="Create this many folds when manifest is absent")
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    if args.fold is not None:
        from peka.repro_data.fold_kmeans import fit_fold_kmeans
        from peka.splits import (
            get_fold, load_or_create_manifest, run_fold_label_name,
            manifest_digest, write_provenance,
        )

        manifest = load_or_create_manifest(
            args.split_manifest, n_splits=args.folds,
        )
        split = get_fold(manifest, args.fold)
        if not args.run_id:
            parser.error("--run_id is required with --fold")
        label_name = run_fold_label_name(
            args.n_clusters, args.fold, args.run_id, manifest,
        )
        written_labels = fit_fold_kmeans(
            dataset_dir=BREAST_DATASET_DIR,
            scllm=args.scllm,
            ckpt=args.ckpt,
            train_slides=split["train"],
            label_slides=split["train"] + split["val"],
            n_clusters=args.n_clusters,
            label_name=label_name,
            use_gpu=args.use_gpu,
        )
        label_hashes = {
            slide_id: hashlib.sha256(
                json.dumps(labels, separators=(",", ":")).encode()
            ).hexdigest()
            for slide_id, labels in written_labels.items()
        }
        write_provenance(
            BREAST_DATASET_DIR / "splits"
            / f"fold_{args.fold}_{args.run_id}_kmeans.provenance.json",
            manifest_digest=manifest_digest(manifest), fold=args.fold,
            train=split["train"], val=split["val"], test=split["test"],
            n_clusters=args.n_clusters, label_name=label_name,
            run_id=args.run_id, scllm=args.scllm, scllm_ckpt=args.ckpt,
            label_hashes=label_hashes,
        )
        logger.info(f"Fold {args.fold} labels written to obs['{label_name}']")
        return

    if args.split_manifest is not None:
        parser.error("--split_manifest requires --fold")

    # Reuse the existing implementation from peka/Exp_helper/7_*.py.
    # Import it as a module by file path (filename starts with a digit).
    import importlib.util
    helper_path = Path(peka.paths.PKG_DIR) / "Exp_helper" / \
        "7_generate_cluster_labels_for_KD.py"
    spec = importlib.util.spec_from_file_location("_kmeans_helper", helper_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    logger.info(f"Running k-means (k={args.n_clusters}, use_gpu={args.use_gpu})")
    mod.process_dataset(
        dataset_folder=str(BREAST_DATASET_DIR),
        scLLM_emb_name=args.scllm,
        n_clusters=args.n_clusters,
        scLLM_emb_ckpt=args.ckpt,
        use_gpu=args.use_gpu,
    )
    logger.info(f"Cluster labels written to obs['gen_clustered_label_{args.n_clusters}']")


if __name__ == "__main__":
    main()
