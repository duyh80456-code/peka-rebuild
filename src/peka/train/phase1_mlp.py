"""Phase 1: MLP teacher classifier on scFoundation embeddings.

Thin wrapper around `peka.Trainer.KD_LoRA.pl_KD_LoRA.train_phase1` (KD_LoRA.py:50-151).
The MLP is `1536 → hidden → hidden → num_classes`, trained on (emb, cluster_label) pairs.
The same MLP is later frozen and used as the teacher in Phase 2.
"""
from pathlib import Path

import peka  # noqa: F401
import torch

from peka.Model.base import MLPClassifier
from peka.Trainer.KD_LoRA import pl_KD_LoRA

from peka import logger


def train_teacher_mlp(
    train_loader,
    val_loader,
    input_dim: int = 1536,
    hidden_dim: int = 512,
    num_classes: int = 100,
    save_path: Path = None,
    epochs: int = 20,
    lr: float = 1e-4,
    device: str = "cuda",
) -> MLPClassifier:
    """Train the MLP teacher and save weights.

    Args:
        train_loader, val_loader: yield `(img, emb, label)` tuples — only `emb`
            and `label` are used in Phase 1
        input_dim: scFoundation embedding dim (1536)
        hidden_dim: MLP hidden dim
        num_classes: number of k-means clusters (100)
        save_path: where to save state_dict
        epochs: training epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
    """
    if save_path is None:
        from peka.paths import PRETRAINED_ROOT
        PRETRAINED_ROOT.mkdir(parents=True, exist_ok=True)
        save_path = PRETRAINED_ROOT / "teacher_mlp.pt"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Phase 1: training MLP {input_dim}→{hidden_dim}→{hidden_dim}→{num_classes}")
    mlp = pl_KD_LoRA.train_phase1(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        classifier_hidden_dim=hidden_dim,
        num_classes=num_classes,
        save_path=str(save_path),
        device=device,
        num_epochs=epochs,
        learning_rate=lr,
    )
    logger.info(f"Phase 1 done. Teacher MLP saved to {save_path}")
    return mlp
