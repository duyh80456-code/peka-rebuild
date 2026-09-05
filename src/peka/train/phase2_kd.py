"""KD training stage (paper-aligned joint, or two-phase with frozen teacher).

Default: joint training (paper §4) — adapter + translate MLP + classifier MLP
all trainable, no separate Phase 1 needed.

Pass `joint=False` to fall back to the original two-phase recipe (frozen
teacher loaded from Phase 1 ckpt).
"""
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

from pathlib import Path
from typing import Optional

import peka  # noqa: F401
from hydra_zen import instantiate

from peka.Hydra_helper.pl_model_helpers import create_pl_model

from peka import logger
from peka.train.kd_module import PekaKDLoRA, PekaKDLoRAJoint


def run_phase2(
    train_loader,
    val_loader,
    model_config,
    optimizer_config,
    trainer_config,
    pl_model_config,
    output_dir: Path,
    target_dim: int,
    exp_name: str,
    wandb_api_key: Optional[str],
    wandb_entity: Optional[str],
    model_name: str,
    ckpt_folder: Path,
    teacher_classifier=None,
    joint: bool = True,
    max_epochs: int = 50,
    additional_pl_paras: Optional[dict] = None,
):
    """Run KD training and return the fitted model.

    Args:
        train_loader, val_loader: from `build_breast_loaders`
        model_config, optimizer_config, trainer_config: hydra-zen configs
        pl_model_config: PekaKDConfig (dataclass) with KD hyperparams
        output_dir: experiment output root
        target_dim: scLLM embedding dim (1536)
        exp_name: W&B / checkpoint name
        wandb_api_key, wandb_entity: W&B credentials (None disables W&B)
        model_name: short tag like 'H-optimus-0_bone'
        ckpt_folder: directory for Lightning checkpoints
        teacher_classifier: optional Phase 1 MLP. Used only when joint=False.
            If joint=True and provided, used as warm-start for the trainable classifier.
        joint: True (default) → paper-aligned joint training (no Phase 1 required).
               False → original two-phase recipe (Phase 1 ckpt required).
        max_epochs: training epochs (paper: 50)
        additional_pl_paras: extra kwargs for pl.Trainer
    """
    output_dir = Path(output_dir)
    ckpt_folder = Path(ckpt_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)

    if not joint and teacher_classifier is None:
        raise ValueError("Two-phase mode (joint=False) requires teacher_classifier from Phase 1.")

    # Instantiate model + optimizer/scheduler/loss.
    model = instantiate(model_config, target_dim=target_dim)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone+translate trainable params: {trainable / 1e6:.2f} M "
                f"(paper: ~5% of 1.1B = ~55 M)")

    optimizer_list, scheduler_list, metrics_factory, loss_instance = instantiate(optimizer_config)

    pl_model_config.lora_save_path = str(output_dir / "phase2" / "lora")

    # Build the KD lightning module — joint or two-phase.
    if joint:
        logger.info("Joint training (paper-aligned): adapter + translate + classifier all trainable")
        kd_model = PekaKDLoRAJoint(
            model_instance=model,
            loss_instance=loss_instance(),
            metrics_factory=metrics_factory,
            optimizer_instance_list=optimizer_list,
            scheduler_instance_list=scheduler_list,
            num_classes=pl_model_config.num_classes,
            classifier_hidden_dim=pl_model_config.classifier_hidden_dim,
            input_dim=pl_model_config.input_dim,
            temperature=pl_model_config.temperature,
            alpha=pl_model_config.alpha,
            lora_save_path=pl_model_config.lora_save_path,
        )
        # Optional warm-start.
        if teacher_classifier is not None:
            kd_model.setup_teacher_model(teacher_classifier)
            logger.info("Joint mode: warm-started classifier from teacher_classifier ckpt")
    else:
        logger.info("Two-phase training (Phase 1 + Phase 2): classifier frozen as teacher")
        kd_model = create_pl_model(
            model_instance=model,
            optimizer_instance_list=optimizer_list,
            scheduler_instance_list=scheduler_list,
            metrics_factory=metrics_factory,
            loss_instance=loss_instance,
            pl_model_config=pl_model_config,
            model_type="kd_lora",
        )
        kd_model.setup_teacher_model(teacher_classifier)

    trainer = instantiate(
        trainer_config,
        entity=wandb_entity or "",
        exp_name=exp_name,
        task_type="classification",
        class_nb=pl_model_config.num_classes,
        model_name=model_name,
        ckpt_folder=str(ckpt_folder),
        max_epochs=max_epochs,
        trainer_output_dir=str(output_dir),
        additional_pl_paras=additional_pl_paras or {},
        wandb_api_key=wandb_api_key or "",
    )

    logger.info(f"Starting trainer.fit (exp={exp_name}, epochs={max_epochs}, joint={joint})")
    trainer.fit(kd_model, train_loader, val_loader)
    logger.info(f"Training done. Outputs at {output_dir}")
    return kd_model, trainer
