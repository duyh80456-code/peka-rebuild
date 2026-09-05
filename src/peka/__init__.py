"""peka — PEKA paper reproduction package (breast-cancer-only).

Provides the full reproduction pipeline for arxiv 2504.07061:
  - peka.Data, peka.Model, peka.Trainer, peka.Hydra_helper, ...
        Inlined from the original PEKA codebase (HEST loading, KD trainer,
        PEFT models, downstream evaluation).
  - peka.configs        Pure-Python hydra-zen config builders.
  - peka.train          Phase 1 (MLP teacher) + Phase 2 (PEFT KD) orchestration.
  - peka.eval           Feature extraction + 5-fold ridge regression.
  - peka.repro_data     Breast dataset factory + HVG selection.
  - peka.utils          Logging + seeding.
"""
import logging
import sys
from pathlib import Path

from peka.paths import WORKSPACE, EXTERNAL_DIR

# Make vendored external models (HEST + scFoundation) importable regardless of cwd.
_HEST_SRC = EXTERNAL_DIR / "HEST" / "src"
_SCFOUNDATION = EXTERNAL_DIR / "scFoundation"
for _p in (EXTERNAL_DIR, _HEST_SRC, _SCFOUNDATION):
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# Load .env from workspace root if present.
try:
    from dotenv import load_dotenv
    _env_file = WORKSPACE / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass

# Package logger.
logger = logging.getLogger("peka")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Convenience: WORKSPACE_DIR alias kept for compatibility with old peka.WORKSPACE_DIR
WORKSPACE_DIR = str(WORKSPACE)
REPO_DIR = str(Path(__file__).parent)

__version__ = "0.2.0"
__all__ = ["logger", "WORKSPACE", "WORKSPACE_DIR", "REPO_DIR", "EXTERNAL_DIR"]
