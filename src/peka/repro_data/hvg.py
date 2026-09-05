"""Highly variable gene selection.

Wraps `scripts/2_downstream_gene_pred/step1_process_hvg.py:process_anndata_folder`,
re-implementing the small bit of logic to avoid sourcing that script.
"""
import json
import os
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

from peka import logger


def compute_top_hvg(
    input_dir: os.PathLike,
    output_json: os.PathLike,
    batch_key: str = "sample_id",
    n_top_hvg: int = 50,
    include_slides: Optional[List[str]] = None,
) -> List[str]:
    """Concatenate all `*.h5ad` in input_dir, compute top-N HVGs, save to JSON.

    Uses scanpy's batch-corrected HVG (Seurat flavor). Falls back to plain
    variance ranking if scanpy fails.

    Args:
        input_dir: folder containing `HEST_breast_adata_*.h5ad`
        output_json: where to write `{"genes": [...]}`
        batch_key: obs column to use as batch (auto-filled with file stem if missing)
        n_top_hvg: number of HVGs to select (paper: 50)

    Returns:
        list of HVG gene symbols
    """
    input_dir = Path(input_dir)
    h5ad_files = sorted(input_dir.glob("*.h5ad"))
    if include_slides is not None:
        include_slides = set(include_slides)
        h5ad_files = [path for path in h5ad_files if path.stem in include_slides]
        found = {path.stem for path in h5ad_files}
        missing = include_slides - found
        if missing:
            raise ValueError(f"Requested HVG slides not found: {sorted(missing)}")
    if not h5ad_files:
        raise ValueError(f"No .h5ad files in {input_dir}")
    logger.info(f"Found {len(h5ad_files)} adata files")

    adatas = []
    for f in h5ad_files:
        a = sc.read_h5ad(f)
        a.var_names_make_unique()
        if batch_key not in a.obs:
            a.obs[batch_key] = f.stem
        a.obs[batch_key] = a.obs[batch_key].astype("category")
        adatas.append(a)

    combined = ad.concat(adatas, join="outer", merge="first", fill_value=0)
    combined.var_names_make_unique()
    if issparse(combined.X):
        combined.X = combined.X.toarray()

    sc.pp.normalize_total(combined)
    sc.pp.log1p(combined)

    try:
        sc.pp.highly_variable_genes(
            combined,
            n_top_genes=n_top_hvg,
            batch_key=batch_key,
            flavor="seurat",
            subset=False,
        )
        hvg = combined.var_names[combined.var.highly_variable].tolist()
    except Exception as e:
        logger.warning(f"scanpy HVG failed ({e}); falling back to variance ranking")
        gene_vars = np.var(combined.X, axis=0)
        top_idx = np.argsort(gene_vars)[-n_top_hvg:]
        hvg = combined.var_names[top_idx].tolist()

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"genes": hvg}, indent=2))
    logger.info(f"Saved {len(hvg)} HVGs to {output_json}")
    return hvg
