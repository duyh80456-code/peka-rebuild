"""5-fold cross-validated gene expression regression.

Per paper: 256-dim PCA → Ridge regression on top-50 HVGs.
Metric: Pearson correlation coefficient (PCC) per gene, then averaged.

This wraps `peka.DownstreamTasks_helper.{train_and_val_exp, gene_expression_prediction}`.
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

import peka  # noqa: F401

from peka import logger
from peka.paths import WORKSPACE


def evaluate_kfold(
    tissue_type: str,
    dataset_name: str,
    embedder_name: str,
    image_encoder_name: str,
    image_backbone: str,
    feature_type: str,
    gene_list_json: Path,
    output_root: Path,
    epochs: int = 200,
    use_binned: bool = False,
    mask_zero_values: bool = False,
    not_use_scllm_subfolder: bool = False,
    n_splits: int = 5,
    split_manifest: Path = None,
    fold: int = None,
    feature_dir: Path = None,
    run_id: str = None,
) -> pd.DataFrame:
    """Run slide-grouped gene-expression regression and return results.

    Args:
        tissue_type: e.g. 'breast'
        dataset_name: e.g. 'breast_in_hest'
        embedder_name: 'scFoundation'
        image_encoder_name: short tag, e.g. 'H0' or 'UNI'
        image_backbone: full name, e.g. 'H-optimus-0' or 'UNI'
        feature_type: one of {'peka', 'image_encoder', 'scLLM'}
        gene_list_json: path to {"genes": [...]} JSON from HVG step
        output_root: where to write CSV + plots
        epochs: regressor training epochs
        use_binned: use binned expression values
        mask_zero_values: mask zeros during training
        not_use_scllm_subfolder: pass-through flag to `get_dataset_paths`
        n_splits: maximum number of GroupKFold splits; reduced to the number
            of available slides when necessary

    Returns:
        DataFrame with one row per gene (Pearson, MSE, etc.)
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    # Lazy import — peka.DownstreamTasks_helper has heavy deps
    from peka.DownstreamTasks_helper.train_and_val_exp import (
        train_and_val_step_KFold, train_and_val_step_holdout,
        plot_gene_correlations,
    )
    from peka.DownstreamTasks_helper.gene_expression_prediction import (
        load_data, get_dataset_paths,
    )

    project_root = str(WORKSPACE)
    paths = get_dataset_paths(
        project_root, tissue_type, dataset_name, embedder_name,
        feature_type=feature_type,
        image_encoder_name=image_encoder_name,
        image_backbone=image_backbone,
        use_scLLM_name_as_subfolder=not not_use_scllm_subfolder,
        embed_path_override=str(feature_dir) if feature_dir else None,
    )

    split = None
    include_slides = None
    if fold is not None:
        from peka.splits import get_fold, load_or_create_manifest
        manifest = load_or_create_manifest(split_manifest, n_splits=n_splits)
        split = get_fold(manifest, fold)
        include_slides = split["train"] + split["val"] + split["test"]

    with open(gene_list_json) as f:
        genes = json.load(f)["genes"]
    logger.info(f"Loaded {len(genes)} genes from {gene_list_json}")

    data_type = "binned" if use_binned else "raw"
    output_dir = Path(output_root) / image_encoder_name / \
        f"{feature_type}_gene_level_{data_type}_grouped_regression_{embedder_name}"
    if fold is not None:
        output_dir = output_dir / f"fold_{fold}" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ckpt").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    embeddings_dict, labels_dict, groups_dict = load_data(
        paths, genes,
        feature_type=feature_type,
        img_prefix="patch_224_0.5_",
        embed_prefix="HEST_breast_adata_",
        use_binned=use_binned,
        mask_zero_values=mask_zero_values,
        return_groups=True,
        include_slides=include_slides,
    )

    missing = set(genes) - set(labels_dict.keys())
    if missing:
        logger.warning(f"{len(missing)} genes missing from data; proceeding with the rest")
        genes = [g for g in genes if g in labels_dict]

    config = {
        "project_root": project_root,
        "tissue_type": tissue_type,
        "dataset_name": dataset_name,
        "embedder_name": embedder_name,
        "gene_list_json": str(gene_list_json),
        "output_root": str(output_root),
        "use_binned": use_binned,
        "feature_type": feature_type,
        "with_independent_test_set": False,
        "image_encoder_name": image_encoder_name,
        "image_backbone": image_backbone,
        "mask_zero_values": mask_zero_values,
    }

    results: List[Dict] = []
    for gene in genes:
        embeddings = embeddings_dict[gene]
        labels = labels_dict[gene]
        groups = groups_dict[gene]
        if embeddings.shape[0] < 321:  # paper threshold
            logger.warning(f"Skipping {gene}: only {embeddings.shape[0]} points")
            continue
        if split is None:
            result = train_and_val_step_KFold(
                embeddings, labels, groups, str(output_dir), gene, config,
                Ksplit=n_splits, epochs=epochs,
            )
        else:
            result = train_and_val_step_holdout(
                embeddings, labels, groups,
                split["train"] + split["val"], split["test"],
                str(output_dir), gene, config, fold=fold, epochs=epochs,
            )
        results.append(result)

    df = pd.DataFrame(results)
    if fold is not None:
        df["run_id"] = run_id
        from peka.splits import manifest_digest
        df["manifest_digest"] = manifest_digest(manifest)
    csv_path = output_dir / "gene_regression_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results → {csv_path}")

    if not df.empty:
        plot_gene_correlations(df, str(output_dir))
    else:
        logger.warning("No genes were eligible for evaluation; skipping plot")
    return df
