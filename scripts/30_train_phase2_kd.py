#!/usr/bin/env python
"""KD training — paper-aligned joint training (default) OR two-phase (legacy).

Paper config (from Section 4 hyperparameters):
  - Adam, lr=1e-4
  - 50 epochs
  - LoRA r=256, α=32, dropout=0.1
  - λ1 = λ2 = 0.5  (alpha=0.5 in our code: alpha·KL + (1-alpha)·CE)
  - ~5% of backbone parameters trainable

By default, this script does JOINT training (paper §4):
    "the adapter parameters and MLP weights are updated while keeping the
     backbone frozen"
i.e. adapter (PEFT) + translate_MLP + classifier_MLP all train together.
NO Phase 1 needed.

Pass `--use-phase1` to fall back to the original two-phase recipe (loads
Pretrained/teacher_mlp.pt as a frozen teacher).

Outputs:
  - OUTPUT/{exp_name}_{timestamp}/configs/*.yaml         frozen config snapshot
  - OUTPUT/{exp_name}_{timestamp}/phase2/lora/            PEFT adapter weights
  - OUTPUT/{exp_name}_{timestamp}/phase2/lora/translate_model.pth
  - Pretrained/{exp_name}/*.ckpt                          Lightning checkpoint
"""
import argparse
import datetime
import multiprocessing
import os
import sys
from pathlib import Path

# Mitigate CUDA OOM fragmentation on small GPUs (must be set before `import torch`).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import peka  # noqa: E402
import torch  # noqa: E402

torch.set_float32_matmul_precision("high")
from peka import logger  # noqa: E402
from peka.configs import (  # noqa: E402
    build_breast_in_hest_config, build_model_config, build_optimizer_config,
    build_trainer_config, build_pl_kd_config, save_yaml_snapshot,
    ENCODER_TABLE, PEFT_METHODS,
)
from peka.repro_data import build_breast_loaders  # noqa: E402
from peka.env import get_env  # noqa: E402
from peka.train import run_phase2  # noqa: E402
from peka.train.kd_module import PekaKDLoRA  # noqa: E402
from peka.utils import set_global_seed  # noqa: E402
from peka.paths import (  # noqa: E402
    BREAST_DATASET_DIR, DEFAULT_N_CLUSTERS, OUTPUT_ROOT, PRETRAINED_ROOT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="KD training (paper-aligned joint by default)")
    parser.add_argument("--encoder", choices=list(ENCODER_TABLE), default="H-optimus-0")
    parser.add_argument("--peft", choices=list(PEFT_METHODS), default="bone")
    parser.add_argument("--use-phase1", action="store_true",
                        help="Two-phase training: load Pretrained/teacher_mlp.pt as frozen teacher. "
                             "Default: joint training (paper-aligned).")
    parser.add_argument("--phase1_ckpt", default=None,
                        help="Phase 1 ckpt path. Used if --use-phase1 (frozen teacher) "
                             "or as warm-start in joint mode if file exists.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-step batch size on GPU. Default 16 (RTX 4090 24 GB w/ bf16, "
                             "ViT-G/14 activations). Try 24 if OOM headroom allows.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2,
                        help="Gradient accumulation steps. Effective batch = batch_size × this. "
                             "Paper uses effective batch=32 → default 16 × 2 = 32.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed",
                        choices=["32-true", "16-mixed", "bf16-mixed", "32"],
                        help="Lightning precision. bf16-mixed: native on Ampere+/Ada (RTX 30xx/40xx), "
                             "wide dynamic range, no GradScaler needed — recommended.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Loss weight: alpha·KL + (1-alpha)·CE. Paper: 0.5 (= λ1=λ2=0.5).")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--exp_name", default=None,
                        help="Default: {encoder}_{peft}_breast_in_hest")
    parser.add_argument("--resume", default=None,
                        help="Lightning .ckpt to resume from (e.g. last.ckpt). "
                             "Restores optimizer, scheduler and epoch state.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_logger", choices=["wandb", "csv"], default="wandb")
    parser.add_argument("--limit_train_batches", type=int, default=None,
                        help="Lightning dry-run flag")
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--split_manifest", default=None,
                        help="Slide split JSON. Required with --fold.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train on one leakage-free outer fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run_id", default=None,
                        help="Unique outer-run ID used to isolate checkpoints")
    args = parser.parse_args()

    set_global_seed(args.seed)

    train_slides = val_slides = None
    if args.fold is not None:
        from peka.splits import (
            file_sha256, get_fold, load_or_create_manifest, run_fold_label_name,
            label_hashes, manifest_digest, validate_provenance, write_provenance,
        )
        manifest = load_or_create_manifest(args.split_manifest, n_splits=args.folds)
        split = get_fold(manifest, args.fold)
        train_slides, val_slides = split["train"], split["val"]
        if not args.run_id:
            parser.error("--run_id is required with --fold to isolate fold artifacts")
        label_name = run_fold_label_name(
            args.num_classes, args.fold, args.run_id, manifest,
        )
        if args.phase1_ckpt is None:
            args.phase1_ckpt = str(
                PRETRAINED_ROOT / f"fold_{args.fold}_{args.run_id}" / "teacher_mlp.pt"
            )
        logger.info(f"Fold {args.fold} train slides: {train_slides}")
        logger.info(f"Fold {args.fold} val slides: {val_slides}")
        logger.info(f"Fold {args.fold} held-out test slides: {split['test']}")
        phase1_path = Path(args.phase1_ckpt)
        if phase1_path.exists():
            validate_provenance(
                phase1_path.with_suffix(".provenance.json"),
                manifest, args.fold, run_id=args.run_id,
                num_classes=args.num_classes,
                teacher_sha256=file_sha256(phase1_path),
            )
        elif args.use_phase1:
            raise FileNotFoundError(f"Missing required Phase 1 checkpoint: {phase1_path}")
        validate_provenance(
            BREAST_DATASET_DIR / "splits"
            / f"fold_{args.fold}_{args.run_id}_kmeans.provenance.json",
            manifest, args.fold, n_clusters=args.num_classes,
            label_name=label_name, run_id=args.run_id,
            scllm="scFoundation", scllm_ckpt="default_model",
            label_hashes=label_hashes(
                BREAST_DATASET_DIR, "scFoundation", "default_model",
                train_slides + val_slides, label_name,
            ),
        )
    else:
        if args.split_manifest is not None:
            parser.error("--split_manifest requires --fold")
        label_name = f"gen_clustered_label_{args.num_classes}"
        args.phase1_ckpt = args.phase1_ckpt or str(PRETRAINED_ROOT / "teacher_mlp.pt")

    if args.exp_name is None:
        suffix = "_phase1" if args.use_phase1 else "_joint"
        args.exp_name = f"{args.encoder}_{args.peft}_breast_in_hest{suffix}"
    if args.fold is not None:
        args.exp_name += f"_fold_{args.fold}_{args.run_id}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{args.exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fold is not None:
        checkpoint_dir = PRETRAINED_ROOT / args.exp_name
        if checkpoint_dir.exists():
            raise FileExistsError(
                f"Fold checkpoint directory already exists: {checkpoint_dir}. "
                "Use a new --run_id; fold artifacts are never reused."
            )
        phase1_path = Path(args.phase1_ckpt)
        write_provenance(
            checkpoint_dir / "provenance.json",
            manifest_digest=manifest_digest(manifest),
            fold=args.fold,
            train=split["train"], val=split["val"], test=split["test"],
            run_id=args.run_id, encoder=args.encoder, peft=args.peft,
            phase1_ckpt=str(Path(args.phase1_ckpt).resolve()),
        )
    logger.info(f"Experiment: {args.exp_name}  (mode={'two-phase' if args.use_phase1 else 'joint'})")
    logger.info(f"Output dir: {output_dir}")

    # Build configs.
    dataset_cfg = build_breast_in_hest_config(
        label_name=label_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_slides=train_slides,
        val_slides=val_slides,
    )
    model_cfg = build_model_config(
        encoder=args.encoder, peft=args.peft,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    opt_cfg = build_optimizer_config(lr=args.lr)

    trainer_cfg = build_trainer_config(
        max_epochs=args.epochs, with_logger=args.with_logger,
        ckpt_format="{epoch:02d}-{val_kd_loss:.4f}",
        ckpt_para={"save_top_k": 1, "mode": "min", "monitor": "val_kd_loss",
                   "save_last": True},
    )

    pl_cfg = build_pl_kd_config(
        num_classes=args.num_classes,
        classifier_hidden_dim=512,
        input_dim=ENCODER_TABLE[args.encoder][1],
        temperature=args.temperature,
        alpha=args.alpha,
    )

    save_yaml_snapshot(output_dir, {
        "dataset": dataset_cfg,
        "model": model_cfg,
        "optimizer": opt_cfg,
        "trainer": trainer_cfg,
        "pl_model": pl_cfg,
        "args": vars(args),
    })

    train_loader, val_loader, embedding_dim = build_breast_loaders(dataset_cfg)
    pl_cfg.input_dim = embedding_dim

    # Teacher: required if --use-phase1, optional warm-start if joint.
    teacher = None
    phase1_path = Path(args.phase1_ckpt)
    if args.use_phase1:
        if not phase1_path.exists():
            raise FileNotFoundError(
                f"--use-phase1 set but no Phase 1 ckpt at {phase1_path}. "
                f"Run `python scripts/20_train_phase1_mlp.py` first."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        teacher = PekaKDLoRA.load_phase1_model(
            checkpoint_path=str(phase1_path),
            input_dim=embedding_dim,
            classifier_hidden_dim=pl_cfg.classifier_hidden_dim,
            num_classes=args.num_classes,
            device=device,
        )
    elif phase1_path.exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Phase 1 ckpt found at {phase1_path} — using as warm-start "
                    f"for joint training's classifier.")
        teacher = PekaKDLoRA.load_phase1_model(
            checkpoint_path=str(phase1_path),
            input_dim=embedding_dim,
            classifier_hidden_dim=pl_cfg.classifier_hidden_dim,
            num_classes=args.num_classes,
            device=device,
        )

    additional = {
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "precision": args.precision,
    }
    if args.limit_train_batches is not None:
        additional["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        additional["limit_val_batches"] = args.limit_val_batches

    run_phase2(
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_classifier=teacher,
        joint=not args.use_phase1,
        model_config=model_cfg,
        optimizer_config=opt_cfg,
        trainer_config=trainer_cfg,
        pl_model_config=pl_cfg,
        output_dir=output_dir,
        target_dim=embedding_dim,
        exp_name=args.exp_name,
        wandb_api_key=get_env("WANDB_API_KEY"),
        wandb_entity=get_env("WANDB_ENTITY"),
        model_name=f"{args.encoder}_{args.peft}_MLP",
        ckpt_folder=PRETRAINED_ROOT,
        max_epochs=args.epochs,
        additional_pl_paras=additional,
        resume_checkpoint=args.resume,
    )
    if args.fold is not None:
        checkpoint_path = PRETRAINED_ROOT / args.exp_name / "last.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Expected final checkpoint at {checkpoint_path}")
        write_provenance(
            checkpoint_path.parent / "provenance.json",
            manifest_digest=manifest_digest(manifest), fold=args.fold,
            train=split["train"], val=split["val"], test=split["test"],
            run_id=args.run_id, encoder=args.encoder, peft=args.peft,
            phase1_ckpt=str(phase1_path.resolve()) if phase1_path.exists() else None,
            phase1_sha256=file_sha256(phase1_path) if phase1_path.exists() else None,
            checkpoint=str(checkpoint_path.resolve()),
            checkpoint_sha256=file_sha256(checkpoint_path),
        )
    logger.info(f"Done. Outputs at {output_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
