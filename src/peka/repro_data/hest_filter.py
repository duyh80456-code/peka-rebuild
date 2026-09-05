"""Filtered HEST1K download — fetch ONLY the breast samples we need.

The full MahmoodLab/hest dataset is ~1 TB. With organ=Breast filter, this is
reduced to ~125 samples (~30-100 GB depending on platform mix).

Pipeline:
  1. Download HEST_v1_1_0.csv (the ~500 KB index file).
  2. Filter rows: organ=Breast, species=Homo sapiens.
  3. Use huggingface_hub.snapshot_download with allow_patterns derived from
     the filtered IDs to fetch only those samples.

HEST per-sample file layout (verified from peka.External_models.HEST source —
HESTData._read_st):
  st/{ID}.h5ad           required — adata with spatial transcriptomics
  wsis/{ID}.tif          required — full-resolution H&E image (BIG: 1-10 GB each)
  metadata/{ID}.json     required — pixel size + QC metadata
  tissue_seg/{ID}_*      optional — tissue mask (used by HEST patcher)
  cellvit_seg/{ID}_*     optional — cell segmentation
  xenium_seg/{ID}_*      optional — Xenium-only cell/nucleus boundaries

Top-level files we ALSO need (or that are tiny):
  HEST_v1_*.csv          the index
  human_gene_db.parquet  gene name DB used by HEST patcher

Notes on speed:
  - snapshot_download first calls list_repo_files (10-60s on a 10k+ file repo).
  - Then it computes the matched-files set by walking allow_patterns through
    fnmatch — no progress bar in this phase.
  - Set HF_HUB_ENABLE_HF_TRANSFER=1 (or pass enable_hf_transfer=True) for
    parallel multi-threaded downloads. Requires `pip install hf_transfer`.
  - To skip the huge WSI .tif files (only train/eval phase 0 doesn't need them
    once patches are extracted), pass include_wsis=False — but you DO need WSIs
    for the patch extraction step (11_build_breast_dataset).
"""
import os
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from peka import logger


def _local_index_path(hest_storage_path: Path) -> Path:
    return Path(hest_storage_path) / "HEST_v1_1_0.csv"


def _maybe_enable_hf_transfer() -> bool:
    """Enable hf_transfer if installed (multi-threaded HF download). Returns True if active."""
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        return True
    except ImportError:
        return False


def download_hest_index(hest_storage_path: Path, hf_token: Optional[str] = None) -> Path:
    """Download just the HEST index CSV (~500 KB)."""
    from huggingface_hub import snapshot_download
    Path(hest_storage_path).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        local_dir=str(hest_storage_path),
        allow_patterns=["HEST_v1_*.csv"],
        token=hf_token,
    )
    csv = _local_index_path(hest_storage_path)
    if not csv.exists():
        raise FileNotFoundError(f"Expected {csv} to exist after snapshot_download")
    logger.info(f"Downloaded HEST index: {csv}")
    return csv


def get_breast_ids(
    csv_path: Path,
    organ: str = "Breast",
    species: str = "Homo sapiens",
    platforms: Optional[Sequence[str]] = None,
) -> List[str]:
    """Filter the HEST index CSV to breast samples and return their IDs.

    Args:
        csv_path: path to HEST_v1_*.csv
        organ: tissue/organ filter (default: Breast)
        species: species filter (default: Homo sapiens)
        platforms: optional list of st_technology values to include
                   (e.g. ["Visium"] for visium-only; default: all platforms)
    """
    df = pd.read_csv(csv_path)
    mask = (df["organ"] == organ) & (df["species"] == species)
    if platforms is not None:
        mask &= df["st_technology"].isin(list(platforms))
    sub = df[mask]
    ids = sub["id"].astype(str).tolist()
    logger.info(f"Filter organ={organ}, species={species}, platforms={platforms}: "
                f"{len(ids)} samples (out of {len(df)} total)")
    return ids


def build_id_patterns(
    ids: Sequence[str],
    include_wsis: bool = True,
    include_tissue_seg: bool = True,
    include_cellvit_seg: bool = False,
    include_xenium_seg: bool = False,
) -> List[str]:
    """Build huggingface_hub allow_patterns for a set of HEST IDs.

    PEKA paper essentials (always included): st/{id}.h5ad, metadata/{id}.json.
    Plus wsis/{id}.tif when include_wsis=True (needed for patch extraction).

    Args:
        ids: list of HEST IDs
        include_wsis: include .tif WSIs (~95% of bytes; required by step 11)
        include_tissue_seg: include tissue_seg/{id}_* (small; helps HEST patcher
            skip background regions). Default True. Safe to disable.
        include_cellvit_seg: include cellvit_seg/{id}_* (cell-level segmentation;
            NOT used by PEKA pipeline). Default False.
        include_xenium_seg: include xenium_seg/{id}_* (Xenium cell/nucleus
            segmentation; NOT used by PEKA pipeline). Default False.
    """
    patterns: List[str] = []
    for sid in ids:
        # Paper essentials.
        patterns.append(f"st/{sid}.h5ad")
        patterns.append(f"metadata/{sid}.json")
        if include_wsis:
            patterns.append(f"wsis/{sid}.tif")
        # Optional auxiliaries.
        if include_tissue_seg:
            patterns.append(f"tissue_seg/{sid}_*")
        if include_cellvit_seg:
            patterns.append(f"cellvit_seg/{sid}_*")
        if include_xenium_seg:
            patterns.append(f"xenium_seg/{sid}_*")
    return patterns


_GLOB_FOLDERS = ("tissue_seg", "cellvit_seg", "xenium_seg")
_EXACT_FOLDERS_TEMPLATE = {
    "st": "{id}.h5ad",
    "metadata": "{id}.json",
    "wsis": "{id}.tif",
}


def list_files_to_download(
    ids: Sequence[str],
    include_wsis: bool = True,
    include_tissue_seg: bool = True,
    include_cellvit_seg: bool = False,
    include_xenium_seg: bool = False,
    hf_token: Optional[str] = None,
) -> List[str]:
    """Diagnostic: list files in the HEST repo that match our patterns.

    Useful to verify allow_patterns work BEFORE committing to a multi-hour download.

    Implementation note: fnmatch over (N_files × M_patterns) is O(NM) and very slow
    for 10k+ files × 750+ patterns (~5 min CPU). We instead build a set of
    (folder, filename_prefix) tuples and do O(N) bucket lookup.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    logger.info("Calling HF list_repo_files (10k+ files, can take 30-90s)...")
    all_files = api.list_repo_files(repo_id="MahmoodLab/hest", repo_type="dataset", token=hf_token)
    logger.info(f"  Got {len(all_files)} files in repo. Filtering to {len(ids)} sample IDs...")

    ids_set = set(ids)
    # Pre-build expected exact paths.
    exact: set = set()
    for sid in ids:
        exact.add(f"st/{sid}.h5ad")
        exact.add(f"metadata/{sid}.json")
        if include_wsis:
            exact.add(f"wsis/{sid}.tif")

    # Which prefix-glob folders are enabled.
    glob_folders: set = set()
    if include_tissue_seg:
        glob_folders.add("tissue_seg")
    if include_cellvit_seg:
        glob_folders.add("cellvit_seg")
    if include_xenium_seg:
        glob_folders.add("xenium_seg")

    matched: List[str] = []
    for f in all_files:
        # Exact match (fast).
        if f in exact:
            matched.append(f)
            continue
        if not glob_folders:
            continue
        # Glob folders: file path looks like "tissue_seg/{ID}_<rest>"
        slash = f.find("/")
        if slash < 0:
            continue
        folder, fname = f[:slash], f[slash + 1:]
        if folder not in glob_folders:
            continue
        # Extract the leading ID — split on '_' or '.'.
        underscore = fname.find("_")
        dot = fname.find(".")
        cut = min(x for x in (underscore, dot) if x >= 0) if (underscore >= 0 or dot >= 0) else len(fname)
        sid = fname[:cut]
        if sid in ids_set:
            matched.append(f)
    return matched


def download_filtered_hest(
    hest_storage_path: Path,
    hf_token: Optional[str] = None,
    organ: str = "Breast",
    species: str = "Homo sapiens",
    platforms: Optional[Sequence[str]] = None,
    extra_patterns: Optional[Sequence[str]] = None,
    include_wsis: bool = True,
    include_tissue_seg: bool = True,
    include_cellvit_seg: bool = False,
    include_xenium_seg: bool = False,
    max_workers: int = 8,
) -> List[str]:
    """End-to-end: fetch index, filter to breast, snapshot_download those samples.

    Args:
        hest_storage_path: where to download
        hf_token: HuggingFace token
        organ, species, platforms: filter args
        extra_patterns: extra allow_patterns to append
        include_wsis: include .tif WSIs (~95% of bytes; required for patch extraction)
        max_workers: parallel download workers (default 8)

    Returns:
        list of IDs that were targeted for download
    """
    from huggingface_hub import snapshot_download

    hf_transfer_active = _maybe_enable_hf_transfer()
    if hf_transfer_active:
        logger.info("hf_transfer enabled — multi-threaded HF downloads")
    else:
        logger.info("hf_transfer NOT installed (single-threaded). "
                    "Install: `pip install hf_transfer` for ~5-10x speedup.")

    hest_storage_path = Path(hest_storage_path)
    hest_storage_path.mkdir(parents=True, exist_ok=True)

    # 1. Index
    csv = _local_index_path(hest_storage_path)
    if not csv.exists():
        csv = download_hest_index(hest_storage_path, hf_token=hf_token)

    # 2. Filter
    ids = get_breast_ids(csv, organ=organ, species=species, platforms=platforms)
    if not ids:
        raise RuntimeError("No samples matched the filter — check organ/species/platforms")

    # 3. Build patterns
    patterns = build_id_patterns(
        ids,
        include_wsis=include_wsis,
        include_tissue_seg=include_tissue_seg,
        include_cellvit_seg=include_cellvit_seg,
        include_xenium_seg=include_xenium_seg,
    )
    patterns += [
        "HEST_v1_*.csv",
        "human_gene_db.parquet",
    ]
    if extra_patterns:
        patterns += list(extra_patterns)

    # 4. Pre-flight: list and report what will actually be downloaded
    logger.info(f"Pre-flight: listing matched files for {len(ids)} samples...")
    matched = list_files_to_download(
        ids,
        include_wsis=include_wsis,
        include_tissue_seg=include_tissue_seg,
        include_cellvit_seg=include_cellvit_seg,
        include_xenium_seg=include_xenium_seg,
        hf_token=hf_token,
    )
    logger.info(f"Pre-flight: {len(matched)} files match the patterns "
                f"(out of {len(patterns)} patterns built).")
    if not matched:
        raise RuntimeError("Pre-flight matched 0 files. allow_patterns may not match the repo layout.")

    # 5. Download
    logger.info(f"Starting snapshot_download → {hest_storage_path} (max_workers={max_workers})")
    logger.info("First-time progress is silent for ~30-90s while HF lists 10k+ repo files. "
                "Then per-file progress bars appear. WSI .tif files are 1-10 GB each.")
    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        local_dir=str(hest_storage_path),
        allow_patterns=patterns,
        token=hf_token,
        max_workers=max_workers,
    )
    logger.info(f"Filtered HEST download complete. {len(ids)} samples at {hest_storage_path}")
    return ids
