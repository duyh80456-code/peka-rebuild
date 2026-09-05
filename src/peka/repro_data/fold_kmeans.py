"""Fit K-means on training slides and write fold-specific pseudo-labels."""
from pathlib import Path
from typing import Dict, List, Tuple

import anndata
import numpy as np
from sklearn.cluster import KMeans

from peka import logger


def _load_slide(
    paired_seq_dir: Path,
    embedding_dir: Path,
    slide_id: str,
) -> Tuple[anndata.AnnData, np.ndarray, np.ndarray]:
    adata_path = paired_seq_dir / f"{slide_id}.h5ad"
    embedding_path = embedding_dir / f"{slide_id}.npy"
    if not adata_path.exists() or not embedding_path.exists():
        raise FileNotFoundError(f"Missing paired data for {slide_id}")
    adata = anndata.read_h5ad(adata_path)
    passed_qc = ~adata.obs["filter_flag"].to_numpy()
    embeddings = np.load(embedding_path)
    if passed_qc.sum() != len(embeddings):
        raise ValueError(
            f"{slide_id}: {passed_qc.sum()} QC-passed spots but "
            f"{len(embeddings)} embeddings"
        )
    return adata, passed_qc, embeddings


def fit_fold_kmeans(
    dataset_dir: Path,
    scllm: str,
    ckpt: str,
    train_slides: List[str],
    label_slides: List[str],
    n_clusters: int,
    label_name: str,
    use_gpu: bool = False,
) -> None:
    paired_seq_dir = Path(dataset_dir) / "scLLM_embed" / scllm / ckpt / "paired_seq"
    embedding_dir = Path(dataset_dir) / "scLLM_embed" / scllm / ckpt / "embeddings"
    if set(train_slides) - set(label_slides):
        raise ValueError("label_slides must include every training slide")

    cache: Dict[str, Tuple[anndata.AnnData, np.ndarray, np.ndarray]] = {}
    written_labels = {}
    for slide_id in sorted(set(label_slides)):
        cache[slide_id] = _load_slide(paired_seq_dir, embedding_dir, slide_id)

    train_embeddings = np.concatenate(
        [cache[slide_id][2] for slide_id in train_slides], axis=0
    ).astype(np.float32)
    logger.info(
        f"Fitting fold K-means on {len(train_embeddings)} spots from "
        f"{len(train_slides)} training slides"
    )

    predictor = None
    faiss_model = None
    if use_gpu:
        try:
            import faiss
            faiss_model = faiss.Kmeans(
                train_embeddings.shape[1], n_clusters, niter=300,
                verbose=True, gpu=True, seed=42,
            )
            faiss_model.train(train_embeddings)
        except ImportError:
            logger.warning("FAISS unavailable; falling back to sklearn KMeans")
    if faiss_model is None:
        predictor = KMeans(n_clusters=n_clusters, random_state=42)
        predictor.fit(train_embeddings)

    for slide_id in sorted(set(label_slides)):
        adata, passed_qc, embeddings = cache[slide_id]
        if faiss_model is not None:
            _, labels = faiss_model.index.search(embeddings.astype(np.float32), 1)
            labels = labels[:, 0]
        else:
            labels = predictor.predict(embeddings)
        full_labels = np.full(len(adata), -1, dtype=np.int64)
        full_labels[passed_qc] = labels
        adata.obs[label_name] = full_labels
        adata.write_h5ad(paired_seq_dir / f"{slide_id}.h5ad")
        written_labels[slide_id] = full_labels.tolist()
        logger.info(f"Wrote {label_name} for {slide_id}")
    return written_labels
