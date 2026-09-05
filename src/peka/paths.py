"""Single source of truth for all paths.

Derived from `__file__`, NOT `os.getcwd()` — so scripts work regardless of cwd.

Layout (workspace-relative):
    PEKA/                         (workspace root)
    ├── src/peka/paths.py         (this file)
    ├── external/HEST/
    ├── external/scFoundation/
    ├── scripts/
    ├── support/
    ├── DATA/                     (gitignored)
    ├── OUTPUT/                   (gitignored)
    └── Pretrained/               (gitignored)
"""
import os
from pathlib import Path

# This file: <WORKSPACE>/src/peka/paths.py — go up 3 levels to reach <WORKSPACE>.
_THIS = Path(__file__).resolve()
WORKSPACE: Path = Path(os.environ.get("WORKSPACE", _THIS.parents[2]))

PKG_DIR: Path = WORKSPACE / "src" / "peka"
EXTERNAL_DIR: Path = WORKSPACE / "external"
SCRIPTS_DIR: Path = WORKSPACE / "scripts"
SUPPORT_DIR: Path = WORKSPACE / "support"

DATA_ROOT: Path = WORKSPACE / "DATA"
OUTPUT_ROOT: Path = WORKSPACE / "OUTPUT"
PRETRAINED_ROOT: Path = WORKSPACE / "Pretrained"

# HEST1K storage (1 TB unfiltered; ~30-100 GB filtered to breast).
# Override via .env: HEST1K_STORAGE_PATH=/path/to/storage
HEST1K_STORAGE_PATH: Path = Path(
    os.environ.get("HEST1K_STORAGE_PATH", str(DATA_ROOT / "HEST1K"))
)

# Breast-only subpaths.
BREAST_DATA_DIR: Path = DATA_ROOT / "breast"
BREAST_DATASET_NAME: str = "breast_in_hest"
BREAST_DATASET_DIR: Path = BREAST_DATA_DIR / BREAST_DATASET_NAME

# Defaults from the paper.
DEFAULT_SCLLM: str = "scFoundation"
DEFAULT_SCLLM_CKPT: str = "default_model"
DEFAULT_N_CLUSTERS: int = 100
DEFAULT_PATCH_SIZE: int = 224
DEFAULT_PIXEL_SIZE: float = 0.5
DEFAULT_IMG_PREFIX: str = "patch_{patch_size}_{pixel_size}"


def ensure_dirs() -> None:
    """Create DATA/, OUTPUT/, Pretrained/, breast/, HEST1K/ if missing."""
    for d in (DATA_ROOT, OUTPUT_ROOT, PRETRAINED_ROOT, BREAST_DATA_DIR,
              HEST1K_STORAGE_PATH):
        d.mkdir(parents=True, exist_ok=True)
