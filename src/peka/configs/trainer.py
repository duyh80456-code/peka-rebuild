"""PyTorch Lightning Trainer config factory."""
import peka  # noqa: F401
from hydra_zen import builds

from peka.Hydra_helper.trainer_part_helpers import trainer_config


def build_trainer_config(
    project: str = "PEKA_repro_breast",
    max_epochs: int = 50,
    clip_grad: float = 1.0,
    save_ckpt: bool = True,
    with_logger: str = "wandb",
    ckpt_format: str = "{epoch:02d}-{val_CosineSimilarity:.4f}",
    ckpt_para: dict = None,
):
    """Build pl.Trainer config matching the paper (50 epochs, Adam, etc).

    `entity`, `exp_name`, `model_name`, `ckpt_folder`, `trainer_output_dir`,
    `wandb_api_key`, `class_nb`, `task_type`, `additional_pl_paras`
    are filled at instantiate-time by the training script (they depend on
    runtime args / env vars).
    """
    if ckpt_para is None:
        ckpt_para = {"save_top_k": 1, "mode": "max", "monitor": "val_CosineSimilarity"}

    return builds(
        trainer_config,
        project=project,
        entity="???",
        exp_name="???",
        task_type="classification",
        class_nb="???",
        model_name="???",
        ckpt_folder="???",
        clip_grad=clip_grad,
        max_epochs=max_epochs,
        trainer_output_dir="???",
        additional_pl_paras={},
        with_logger=with_logger,
        wandb_api_key="???",
        save_ckpt=save_ckpt,
        ckpt_format=ckpt_format,
        ckpt_para=ckpt_para,
        populate_full_signature=True,
    )
