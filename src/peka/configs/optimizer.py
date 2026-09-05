"""Optimizer + scheduler + loss + metrics factory.

Targets ``peka.Hydra_helper.opt_sch_part_helpers.opt_sch_config`` which returns
``(optimizer_list, scheduler_list, metrics_factory, loss_instance)``.

Note on metrics: PEKA Phase 2 trains the student to align with scLLM embeddings
(continuous feature space), so we use **regression metrics** (MSE +
CosineSimilarity), not classification ones — even though cluster pseudo-labels
ARE used in the CE loss term. The trainer monitors ``val_CosineSimilarity``.
"""
import peka  # noqa: F401
from hydra_zen import builds

from peka.Hydra_helper.opt_sch_part_helpers import opt_sch_config
from peka.Trainer.metrics import metrics_para_dict


def build_optimizer_config(
    lr: float = 1e-4,
    step_size: int = 10,
    gamma: float = 0.1,
):
    """Build optimizer/scheduler config matching the paper.

    Paper hyperparameters (§4): Adam lr=1e-4, 50 epochs.
    StepLR is added for stability (paper doesn't specify a scheduler explicitly).

    Loss: CrossEntropyLoss (the KD_LoRA "hard-loss" term — structure alignment
    against k-means cluster pseudo-labels).
    """
    return builds(
        opt_sch_config,
        # Regression metrics: MSE + CosineSimilarity (alias "CosSim" in metrics_dict).
        metrics_names=["MSE", "CosSim"],
        metrics_paras=metrics_para_dict,
        loss_name="CrossEntropyLoss",
        loss_paras={},
        optimizer_name_list=["Adam"],
        optimizer_paras_list=[{"lr": lr}],
        scheduler_name_list=["StepLR"],
        scheduler_paras_list=[{"step_size": step_size, "gamma": gamma}],
        # n_classes=None → MetricsFactory uses regression task (matches MSE/CosSim).
        n_classes=None,
        populate_full_signature=True,
    )
