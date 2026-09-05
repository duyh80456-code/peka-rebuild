#!/usr/bin/env python
"""Build the breast_in_hest dataset:
  1. Filter HEST1K to (organ=Breast, species=Homo sapiens, all platforms)
  2. Extract 224x224 patches at 0.5 μm/pixel
  3. Align gene names to standard HGNC symbols
  4. Write per-sample h5ad files to DATA/breast/breast_in_hest/aligned_adata/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.env import require_env  # noqa: E402
from peka.paths import BREAST_DATA_DIR, BREAST_DATASET_NAME, DATA_ROOT  # noqa: E402
from peka.repro_data.breast_dataset import ensure_breast_dataset_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build breast_in_hest dataset")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extraction even if patches exist")
    args = parser.parse_args()

    hest_loc = require_env("HEST1K_STORAGE_PATH")
    BREAST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Place our breast-only dataset_config.csv at DATA/breast/dataset_config.csv
    ensure_breast_dataset_csv()

    from peka.Data.database_helper import create_hest1k_sub_database_instance
    from peka.Data.hest1k_helper import (
        construct_sub_dataset_index, load_subdataset,
        extract_patches_from_hest, gene_name_alignment,
    )

    csv_path = BREAST_DATA_DIR / "dataset_config.csv"
    data_para = create_hest1k_sub_database_instance(
        csv_file_path=str(csv_path),
        dataset_name=BREAST_DATASET_NAME,
        data_root=str(BREAST_DATA_DIR),
        hest_storage_path=hest_loc,
        copy_flag=True,
    )

    logger.info(f"Step 1/3: building dataset index for {BREAST_DATASET_NAME}")
    construct_sub_dataset_index(data_para=data_para, with_explore=True)

    logger.info("Step 2/3: loading subdataset + extracting patches")
    hest_data, len_hest, subdataset_folder = load_subdataset(
        hest_loc=hest_loc,
        datasets_folder=str(BREAST_DATA_DIR),
        subdataset_name=BREAST_DATASET_NAME,
    )
    extract_patches_from_hest(
        hest_data=hest_data,
        len_hest_data=len_hest,
        subdataset_folder=subdataset_folder,
        patch_size=data_para.patch_size,
        pixel_size=data_para.pixel_size,
        force_process=args.force,
    )

    logger.info("Step 3/3: aligning gene names")
    gene_name_alignment(hest_data=hest_data, subdataset_folder=subdataset_folder)
    logger.info(f"Done. Dataset at {subdataset_folder}")


if __name__ == "__main__":
    main()
