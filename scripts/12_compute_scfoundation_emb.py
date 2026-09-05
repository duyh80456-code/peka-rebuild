#!/usr/bin/env python
"""Compute scFoundation embeddings for all aligned breast adata files.

Output: DATA/breast/breast_in_hest/scLLM_embed/scFoundation/<ckpt>/{paired_seq,embeddings}/
  - paired_seq/HEST_breast_adata_*.h5ad   filtered adata with QC flags
  - embeddings/HEST_breast_adata_*.npy    (n_passed_qc, 1536) embedding tensor

Pre-requisites (auto-checked):
  1. scFoundation vocab → auto-copied from external/scFoundation/
  2. scFoundation .ckpt → must be downloaded MANUALLY from biomap-research
     (see https://github.com/biomap-research/scFoundation#inference)
     SharePoint link in scFoundation/model/README.md.

     Place at:
       <data_root>/Pretrained/scFoundation/<ckpt_name>.ckpt
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Avoid CUDA OOM fragmentation on small GPUs (RTX A4000 16 GB).
# Must be set BEFORE `import torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.env import require_env  # noqa: E402
from peka.paths import (  # noqa: E402
    BREAST_DATA_DIR, BREAST_DATASET_NAME, DEFAULT_SCLLM, WORKSPACE, EXTERNAL_DIR,
)


SCFOUNDATION_HF_REPO = "genbio-ai/scFoundation"
SCFOUNDATION_HF_FILE = "models.ckpt"  # ~1.4 GB

SCFOUNDATION_DOWNLOAD_HINT = """
scFoundation checkpoint download failed.

Tried HuggingFace mirror: {repo} (file: {fn}).

Alternative — manual download from biomap-research SharePoint:
  https://hopebio2020-my.sharepoint.com/:f:/g/personal/dongsheng_biomap_com/Eh22AX78_AVDv6k6v4TZDikBXt33gaWXaz27U9b1SldgbA
(see external/scFoundation/model/README.md)

Then place the file at:
  {ckpt_path}

After that, re-run this script.
"""


def _download_scfoundation_ckpt(target: Path) -> None:
    """Download scFoundation checkpoint from HuggingFace mirror.

    Uses genbio-ai/scFoundation which hosts the official checkpoint as `models.ckpt`.
    Downloads to a temp location then renames to {target}.
    """
    from huggingface_hub import hf_hub_download
    logger.info(f"Downloading scFoundation checkpoint from {SCFOUNDATION_HF_REPO} "
                f"({SCFOUNDATION_HF_FILE}, ~1.4 GB)...")
    src = hf_hub_download(
        repo_id=SCFOUNDATION_HF_REPO,
        filename=SCFOUNDATION_HF_FILE,
        repo_type="model",
        local_dir=str(target.parent),
        token=os.getenv("HF_TOKEN"),
    )
    src_path = Path(src)
    if src_path != target:
        src_path.rename(target)
    logger.info(f"Saved scFoundation checkpoint → {target}")


def setup_scfoundation_assets(data_root: Path, ckpt_name: str, auto_download: bool = True) -> None:
    """Ensure vocab + ckpt are in place; auto-download ckpt from HF if missing."""
    pretrained_dir = data_root / "Pretrained" / "scFoundation"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vocab — copy from vendored external/scFoundation if missing.
    vocab_target = pretrained_dir / "OS_scRNA_gene_index.19264.tsv"
    if not vocab_target.exists():
        vocab_source = EXTERNAL_DIR / "scFoundation" / "OS_scRNA_gene_index.19264.tsv"
        if not vocab_source.exists():
            raise FileNotFoundError(
                f"scFoundation vocab not found at {vocab_source}. "
                f"Re-clone external/scFoundation."
            )
        shutil.copy2(vocab_source, vocab_target)
        logger.info(f"Copied scFoundation vocab → {vocab_target}")
    else:
        logger.info(f"scFoundation vocab OK at {vocab_target}")

    # 2. Checkpoint — try auto-download from HF mirror, fall back to manual.
    ckpt_path = pretrained_dir / f"{ckpt_name}.ckpt"
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / 1e6
        logger.info(f"scFoundation checkpoint OK at {ckpt_path} ({size_mb:.1f} MB)")
        return

    if auto_download:
        try:
            _download_scfoundation_ckpt(ckpt_path)
            return
        except Exception as e:
            logger.error(f"HF download failed: {type(e).__name__}: {e}")

    msg = SCFOUNDATION_DOWNLOAD_HINT.format(
        repo=SCFOUNDATION_HF_REPO, fn=SCFOUNDATION_HF_FILE, ckpt_path=ckpt_path,
    )
    logger.error(msg)
    raise FileNotFoundError(f"Missing scFoundation checkpoint: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute scFoundation embeddings on breast adata")
    parser.add_argument("--ckpt_name", default="default_model",
                        help="scFoundation checkpoint variant (default: default_model)")
    parser.add_argument("--no-auto-download", action="store_true",
                        help="Skip auto-downloading scFoundation ckpt from HF mirror.")
    args = parser.parse_args()

    require_env("HF_TOKEN")

    # Auto-setup assets: vocab + ckpt (auto-download from HF mirror by default).
    setup_scfoundation_assets(
        BREAST_DATA_DIR, args.ckpt_name,
        auto_download=not args.no_auto_download,
    )

    os.environ.setdefault("PROJECT_ROOT", str(WORKSPACE))
    os.environ.setdefault("TISSUE_TYPE", "breast")
    os.environ.setdefault("DATASET_NAME", BREAST_DATASET_NAME)
    os.environ.setdefault("SCLLM_EMBEDDER_NAME", DEFAULT_SCLLM)

    from peka.Model.LLM.utils import get_scLLM_embedder

    embedder = get_scLLM_embedder(
        data_root=str(BREAST_DATA_DIR),
        dataset_name=BREAST_DATASET_NAME,
        scLLM_embedder_name=DEFAULT_SCLLM,
        ckpt_name=args.ckpt_name,
    )
    logger.info(f"Computing {DEFAULT_SCLLM} embeddings for {BREAST_DATASET_NAME}")
    embedder.run()
    embedder.valid_check()
    logger.info("scFoundation embeddings done")


if __name__ == "__main__":
    main()
