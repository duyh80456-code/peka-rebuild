"""Deterministic slide-level splits shared by training and evaluation."""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import KFold

from peka.paths import BREAST_DATASET_DIR, DEFAULT_SCLLM, DEFAULT_SCLLM_CKPT


def default_split_manifest(n_splits: int = 5) -> Path:
    return BREAST_DATASET_DIR / "splits" / f"slide_{n_splits}fold.json"


def paired_seq_dir(
    scllm: str = DEFAULT_SCLLM,
    ckpt: str = DEFAULT_SCLLM_CKPT,
) -> Path:
    return BREAST_DATASET_DIR / "scLLM_embed" / scllm / ckpt / "paired_seq"


def discover_slide_ids(folder: Path) -> List[str]:
    slide_ids = sorted(path.stem for path in Path(folder).glob("HEST_breast_adata_*.h5ad"))
    if len(slide_ids) < 2:
        raise ValueError(f"Need at least 2 slides in {folder}, found {slide_ids}")
    return slide_ids


def create_slide_manifest(
    output_path: Path,
    slide_ids: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    slide_ids = sorted(set(slide_ids))
    if n_splits < 2 or n_splits > len(slide_ids):
        raise ValueError(
            f"n_splits must be in [2, {len(slide_ids)}], got {n_splits}"
        )

    slides = np.asarray(slide_ids, dtype=object)
    outer = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (development_idx, test_idx) in enumerate(outer.split(slides)):
        development = slides[development_idx].tolist()
        test = slides[test_idx].tolist()

        rng = np.random.default_rng(seed + fold_idx)
        shuffled = list(development)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * 0.2))
        val = sorted(shuffled[:val_count])
        train = sorted(shuffled[val_count:])
        if not train:
            raise ValueError(f"Fold {fold_idx} has no training slides")

        assert not (set(train) & set(val))
        assert not (set(train) & set(test))
        assert not (set(val) & set(test))
        folds.append({"fold": fold_idx, "train": train, "val": val, "test": sorted(test)})

    manifest = {
        "version": 1,
        "unit": "slide",
        "seed": seed,
        "n_splits": n_splits,
        "slides": slide_ids,
        "folds": folds,
    }
    validate_manifest(manifest)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_or_create_manifest(
    manifest_path: Optional[Path] = None,
    n_splits: int = 5,
    seed: Optional[int] = None,
    source_dir: Optional[Path] = None,
) -> Dict:
    path = Path(manifest_path) if manifest_path else default_split_manifest(n_splits)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("version") != 1:
            raise ValueError(
                f"Split manifest {path} uses obsolete version "
                f"{manifest.get('version')}; remove it and regenerate slide folds."
            )
        if manifest.get("n_splits") != n_splits:
            raise ValueError(
                f"Existing manifest {path} uses n_splits={manifest.get('n_splits')} "
                f"but {n_splits} was requested. Use a different path or remove it "
                "intentionally."
            )
        if seed is not None and manifest.get("seed") != seed:
            raise ValueError(
                f"Existing manifest {path} uses seed={manifest.get('seed')} but "
                f"seed={seed} was requested. Use a different path or remove it intentionally."
            )
        current_slides = discover_slide_ids(source_dir or paired_seq_dir())
        if manifest.get("slides") != current_slides:
            raise ValueError(
                f"Split manifest slides differ from current dataset. Remove {path} "
                "and regenerate it intentionally."
            )
        validate_manifest(manifest)
        return manifest
    return create_slide_manifest(
        path,
        discover_slide_ids(source_dir or paired_seq_dir()),
        n_splits=n_splits,
        seed=42 if seed is None else seed,
    )


def get_fold(manifest: Dict, fold: int) -> Dict[str, List[str]]:
    validate_manifest(manifest)
    folds = manifest.get("folds", [])
    matching = [split for split in folds if split.get("fold") == fold]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one definition for fold {fold}")
    split = matching[0]
    train, val, test = map(set, (split["train"], split["val"], split["test"]))
    if train & val or train & test or val & test:
        raise ValueError(f"Overlap detected in fold {fold}: {split}")
    return split


def fold_label_name(n_clusters: int, fold: int) -> str:
    return f"gen_clustered_label_{n_clusters}_fold_{fold}"


def run_fold_label_name(n_clusters: int, fold: int, run_id: str, manifest: Dict) -> str:
    safe_run_id = "".join(char if char.isalnum() else "_" for char in run_id)
    return (
        f"gen_clustered_label_{n_clusters}_fold_{fold}_"
        f"{manifest_digest(manifest)[:12]}_{safe_run_id}"
    )


def validate_manifest(manifest: Dict) -> None:
    slides = manifest.get("slides", [])
    folds = manifest.get("folds", [])
    if len(slides) != len(set(slides)):
        raise ValueError("Split manifest contains duplicate slide IDs")
    if len(folds) != manifest.get("n_splits"):
        raise ValueError("Split manifest fold count does not match n_splits")
    expected_folds = set(range(len(folds)))
    actual_folds = {split.get("fold") for split in folds}
    if actual_folds != expected_folds:
        raise ValueError(f"Invalid fold IDs: {sorted(actual_folds)}")

    slide_set = set(slides)
    outer_test = []
    for split in folds:
        train, val, test = split.get("train", []), split.get("val", []), split.get("test", [])
        if any(len(values) != len(set(values)) for values in (train, val, test)):
            raise ValueError(f"Duplicate slide inside fold {split.get('fold')}")
        train_set, val_set, test_set = map(set, (train, val, test))
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError(f"Overlap detected in fold {split.get('fold')}")
        if train_set | val_set | test_set != slide_set:
            raise ValueError(f"Fold {split.get('fold')} does not partition all slides")
        outer_test.extend(test)
    if sorted(outer_test) != sorted(slides):
        raise ValueError("Every slide must appear exactly once across outer test folds")


def manifest_digest(manifest: Dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_hashes(dataset_dir: Path, scllm: str, ckpt: str,
                 slide_ids: List[str], label_name: str) -> Dict[str, str]:
    import anndata

    folder = Path(dataset_dir) / "scLLM_embed" / scllm / ckpt / "paired_seq"
    hashes = {}
    for slide_id in sorted(slide_ids):
        adata = anndata.read_h5ad(folder / f"{slide_id}.h5ad")
        if label_name not in adata.obs:
            raise ValueError(f"Missing label {label_name} in {slide_id}")
        values = adata.obs[label_name].to_numpy().tolist()
        payload = json.dumps(values, separators=(",", ":")).encode()
        hashes[slide_id] = hashlib.sha256(payload).hexdigest()
    return hashes


def write_provenance(path: Path, **values) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    return path


def read_provenance(path: Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact provenance: {path}")
    return json.loads(path.read_text())


def validate_provenance(path: Path, manifest: Dict, fold: int, **expected) -> Dict:
    provenance = read_provenance(path)
    required = {
        "manifest_digest": manifest_digest(manifest),
        "fold": fold,
        "train": get_fold(manifest, fold)["train"],
        "val": get_fold(manifest, fold)["val"],
        "test": get_fold(manifest, fold)["test"],
        **expected,
    }
    mismatches = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in required.items() if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Artifact provenance mismatch at {path}: {mismatches}")
    return provenance
