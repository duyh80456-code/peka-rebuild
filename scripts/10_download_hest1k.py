#!/usr/bin/env python
"""Filtered HEST1K download — fetch only files PEKA paper actually needs.

The full HEST1K is ~1 TB. PEKA paper needs only:
  - st/{ID}.h5ad        gene expression (small, ~10s of MB each)
  - wsis/{ID}.tif       H&E whole-slide image (BIG: 1-10 GB each)
  - metadata/{ID}.json  pixel size + QC info (tiny)

The other folders (cellvit_seg, xenium_seg, tissue_seg) are NOT used by PEKA.

By default we download:
  - Essentials: st/, wsis/, metadata/
  - tissue_seg/ (small, helps HEST patcher skip background)

Use --minimal to skip tissue_seg too (only the 3 paper essentials).

Examples:
  # Default — paper essentials + tissue_seg (~100-500 GB for breast)
  python scripts/10_download_hest1k.py --max-workers 16

  # Truly minimal — just paper essentials (~100-500 GB; same WSI bytes, just fewer aux files)
  python scripts/10_download_hest1k.py --minimal --max-workers 16

  # Dry-run to see what will be fetched
  python scripts/10_download_hest1k.py --dry-run --minimal

Notes on speed:
  - First 30-90s is silent (HF lists 10k+ repo files).
  - WSIs (~1-10 GB each) dominate download time. Install hf_transfer for ~5-10x speedup:
       pip install hf_transfer
  - Resume works automatically (HF skips files already on disk).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.env import get_env, hf_login  # noqa: E402
from peka.paths import HEST1K_STORAGE_PATH  # noqa: E402
from peka.repro_data.hest_filter import (  # noqa: E402
    download_filtered_hest, download_hest_index, get_breast_ids,
    list_files_to_download,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filtered HEST1K download (PEKA paper essentials)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--storage", default=str(HEST1K_STORAGE_PATH),
                        help=f"Where to download (default: {HEST1K_STORAGE_PATH})")
    parser.add_argument("--organ", default="Breast",
                        help="Organ filter (default: Breast)")
    parser.add_argument("--species", default="Homo sapiens",
                        help="Species filter (default: Homo sapiens)")
    parser.add_argument("--platform", default=None, action="append",
                        help="Platform filter (st_technology). Pass multiple times. Default: Visium "
                             "(paper Section 4.1: 'Visium ST data of Homo Sapiens with breast cancer, "
                             "n=30,414 pairs'). Pass --all-platforms to include Xenium + ST too.")
    parser.add_argument("--all-platforms", action="store_true",
                        help="Disable platform filter — include Visium + Xenium + Spatial Transcriptomics. "
                             "Note: paper uses Visium only.")
    parser.add_argument("--no-wsis", action="store_true",
                        help="Skip the .tif WSI files. WARNING: step 11 needs them for patch extraction.")
    parser.add_argument("--minimal", action="store_true",
                        help="Skip cellvit_seg, xenium_seg (NOT used by PEKA). "
                             "tissue_seg IS still included — HEST loader needs it.")
    parser.add_argument("--no-tissue-seg", action="store_true",
                        help="Skip tissue_seg too. WARNING: HEST.load_hest crashes without it. "
                             "Only use if you've patched HEST or won't call load_hest.")
    parser.add_argument("--include-cellvit-seg", action="store_true",
                        help="Include cellvit_seg/ (cell segmentation; NOT used by PEKA).")
    parser.add_argument("--include-xenium-seg", action="store_true",
                        help="Include xenium_seg/ (Xenium-specific seg; NOT used by PEKA).")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel download workers (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List matched files without downloading.")
    parser.add_argument("--full", action="store_true",
                        help="Disable filter — download FULL HEST1K (~1 TB). NOT RECOMMENDED.")
    args = parser.parse_args()

    # tissue_seg is REQUIRED by HEST loader (we keep it unless --no-tissue-seg explicit).
    # --minimal only skips cellvit/xenium_seg (which PEKA doesn't use).
    include_tissue_seg = not args.no_tissue_seg
    include_cellvit_seg = args.include_cellvit_seg
    include_xenium_seg = args.include_xenium_seg
    include_wsis = not args.no_wsis

    # Platform default: Visium (paper config). Override with --platform or --all-platforms.
    if args.all_platforms:
        platforms = None  # no filter
    elif args.platform:
        platforms = args.platform
    else:
        platforms = ["Visium"]
    args.platform = platforms

    storage = Path(args.storage).expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    hf_login(required=False)

    if args.full:
        logger.warning("Downloading FULL HEST1K (~1 TB). Consider Ctrl-C now.")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="MahmoodLab/hest",
            repo_type="dataset",
            local_dir=str(storage),
            token=get_env("HF_TOKEN"),
            max_workers=args.max_workers,
        )
        logger.info("Full HEST1K download complete")
        return

    if args.dry_run:
        csv = storage / "HEST_v1_1_0.csv"
        if not csv.exists():
            csv = download_hest_index(storage, hf_token=get_env("HF_TOKEN"))
        ids = get_breast_ids(csv, organ=args.organ, species=args.species, platforms=args.platform)
        logger.info(f"Dry-run: pre-flight listing files for {len(ids)} samples...")
        matched = list_files_to_download(
            ids,
            include_wsis=include_wsis,
            include_tissue_seg=include_tissue_seg,
            include_cellvit_seg=include_cellvit_seg,
            include_xenium_seg=include_xenium_seg,
            hf_token=get_env("HF_TOKEN"),
        )
        logger.info(f"Dry-run: {len(matched)} files would be downloaded.")
        from collections import Counter
        counter = Counter(p.split("/")[0] for p in matched)
        for folder, n in counter.most_common():
            logger.info(f"  {folder}/: {n} files")
        logger.info("Sample paths:")
        for p in matched[:10]:
            logger.info(f"  {p}")
        return

    ids = download_filtered_hest(
        hest_storage_path=storage,
        hf_token=get_env("HF_TOKEN"),
        organ=args.organ,
        species=args.species,
        platforms=args.platform,
        include_wsis=include_wsis,
        include_tissue_seg=include_tissue_seg,
        include_cellvit_seg=include_cellvit_seg,
        include_xenium_seg=include_xenium_seg,
        max_workers=args.max_workers,
    )
    logger.info(f"Filtered HEST1K ready at {storage} ({len(ids)} samples)")
    os.environ["HEST1K_STORAGE_PATH"] = str(storage)


if __name__ == "__main__":
    main()
