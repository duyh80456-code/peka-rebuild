#!/usr/bin/env python
"""Create the deterministic outer slide-fold manifest used by all later stages."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peka import logger  # noqa: E402
from peka.splits import default_split_manifest, load_or_create_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-free slide folds")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else default_split_manifest(args.folds)
    manifest = load_or_create_manifest(
        output, n_splits=args.folds, seed=args.seed,
    )
    logger.info(f"Split manifest: {output}")
    for split in manifest["folds"]:
        logger.info(
            f"fold={split['fold']} train={split['train']} "
            f"val={split['val']} test={split['test']}"
        )


if __name__ == "__main__":
    main()
