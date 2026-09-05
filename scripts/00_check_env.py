#!/usr/bin/env python
"""Validate environment for peka before running anything else.

Run: python scripts/00_check_env.py
"""
import sys
from pathlib import Path

# Make peka importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
from peka import logger  # noqa: E402
from peka.env import get_env, hf_login, wandb_setup  # noqa: E402
from peka.paths import (  # noqa: E402
    DATA_ROOT, OUTPUT_ROOT, PRETRAINED_ROOT, PKG_DIR, EXTERNAL_DIR,
    BREAST_DATA_DIR, SUPPORT_DIR, ensure_dirs,
)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> int:
    print(f"  ✗ {msg}")
    return 1


def main() -> int:
    print("=" * 60)
    print("peka environment check")
    print("=" * 60)

    failures = 0

    # 1. Required env vars
    print("\n[1/5] Environment variables")
    for name in ("HEST1K_STORAGE_PATH",):
        if get_env(name):
            _ok(f"{name} = {get_env(name)}")
        else:
            failures += _fail(f"{name} is not set (required)")
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "HF_TOKEN"):
        if get_env(name):
            _ok(f"{name} is set")
        else:
            print(f"  ! {name} is not set (warning — some steps will fail)")

    # 2. Paths
    print("\n[2/5] Paths")
    ensure_dirs()
    for p in (PKG_DIR, EXTERNAL_DIR, SUPPORT_DIR,
              DATA_ROOT, OUTPUT_ROOT, PRETRAINED_ROOT, BREAST_DATA_DIR):
        if p.exists():
            _ok(str(p))
        else:
            failures += _fail(f"{p} does not exist")

    # 3. peka support files
    print("\n[3/5] Support files")
    csv = SUPPORT_DIR / "peka_breast_datasets.csv"
    if csv.exists():
        _ok(str(csv))
    else:
        failures += _fail(f"{csv} missing — should ship with the repo")

    # 4. Imports
    print("\n[4/5] Imports")
    try:
        import torch
        _ok(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        failures += _fail(f"torch import: {e}")

    try:
        import peka  # noqa: F401
        _ok("peka (sibling package)")
    except ImportError as e:
        failures += _fail(f"peka import: {e}")

    try:
        import hest  # noqa: F401
        _ok("hest (vendored)")
    except ImportError as e:
        print(f"  ! hest import: {e} (needed for Phase 0 steps 11-12)")

    try:
        import scFoundation  # noqa: F401
        _ok("scFoundation (vendored)")
    except ImportError as e:
        print(f"  ! scFoundation import: {e} (needed for Phase 0 step 12)")

    try:
        import pytorch_lightning as pl  # noqa: F401
        _ok(f"pytorch_lightning {pl.__version__}")
    except ImportError as e:
        failures += _fail(f"pytorch_lightning import: {e}")

    try:
        import peft  # noqa: F401
        from peft import LoraConfig, AdaLoraConfig, HRAConfig, BoneConfig  # noqa: F401
        _ok("peft (LoRA + AdaLoRA + HRA + BONE)")
    except ImportError as e:
        failures += _fail(f"peft import: {e}")

    try:
        from hydra_zen import builds, instantiate  # noqa: F401
        _ok("hydra_zen")
    except ImportError as e:
        failures += _fail(f"hydra_zen import: {e}")

    # 5. Optional credentials
    print("\n[5/5] Credentials (optional)")
    if get_env("HF_TOKEN"):
        try:
            hf_login(required=False)
            _ok("HuggingFace authenticated")
        except Exception as e:
            print(f"  ! HF login failed: {e}")
    if wandb_setup():
        _ok("WANDB_API_KEY set")

    print()
    if failures:
        print(f"FAILED ({failures} error(s)). Fix above before proceeding.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
