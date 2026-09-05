# AGENTS.md

## Scope

- This is a breast-only PEKA reproduction. Keep the hardcoded `HEST_breast_adata_` prefix unless intentionally generalizing all producers and consumers.
- `src/peka/{Data,Model,Trainer,Hydra_helper,DownstreamTasks_helper,Exp_helper}` is inlined legacy code; preserve its capitalized package names and imports. New orchestration lives in lowercase packages.
- Treat the numbered files in `scripts/` as entrypoints. Some legacy `Exp_helper` files still derive paths from `cwd`; do not run them directly without auditing them.

## Environment And Paths

- Use the Python 3.9 conda env: `conda activate hest && pip install -e .`.
- Copy `.env.example` to root `.env`. `HEST1K_STORAGE_PATH` is required by the environment check/build; `HF_TOKEN` is needed for HEST, scFoundation, and gated encoder weights; W&B needs `WANDB_API_KEY` and `WANDB_ENTITY`.
- Run `python scripts/00_check_env.py` for focused setup/import verification. There is no repository test, lint, formatter, typecheck, or CI command.
- Canonical paths come from `src/peka/paths.py`, not `cwd`; root `.env` is loaded by `peka.__init__`, which also adds vendored HEST and scFoundation paths to `sys.path`. Each numbered Python script adds `src/` itself.
- `WORKSPACE` overrides path discovery. `HEST1K_STORAGE_PATH` only relocates raw HEST data; processed data still goes under `<WORKSPACE>/DATA`.
- Do not assume all generated artifacts are ignored: `.gitignore` currently ignores `DATA/*`, but not `OUTPUT/` or `Pretrained/`. Never stage large data, checkpoints, or credentials accidentally.

## Pipeline

- Preserve preprocessing order: `00 -> 10 -> 11 -> 12 -> 15`. For leakage-free two-phase evaluation, run `32` per fold (or `33` for all folds), then aggregate with `51`; `32` owns fold-specific steps `13 -> 14 -> 20 -> 30 -> 40 -> 41 -> 50`.
- Paper-sized download: `python scripts/10_download_hest1k.py --minimal --max-workers 16`. It defaults to Visium breast. Despite its name, `--minimal` retains `tissue_seg`, which step 11's HEST loader requires; do not add `--no-tissue-seg`. Use `--dry-run --minimal` before changing filters.
- Step 12 auto-downloads the scFoundation checkpoint to `DATA/breast/Pretrained/scFoundation/default_model.ckpt`; `--no-auto-download` requires placing it there manually.
- `python scripts/14_smoke_test_loader.py` must pass before training. It is the cheapest end-to-end data verification and checks `(image, embedding, cluster_label)` shapes.
- Phase 2 is joint training by default; Phase 1 is not required. `scripts/20_train_phase1_mlp.py` creates `Pretrained/teacher_mlp.pt`; if that file exists, default Phase 2 silently warm-starts from it. Use `--use-phase1` only for the legacy frozen-teacher recipe.
- Focused GPU verification:
  `python scripts/30_train_phase2_kd.py --encoder H-optimus-0 --peft bone --epochs 1 --limit_train_batches 5 --limit_val_batches 2 --with_logger csv`
- Full sweep: `EXTRA_FLAGS="--epochs 50" bash scripts/31_train_all_combinations.sh`; it runs sequentially and stops on the first failure.
- Phase 2 defaults to per-step batch 16 with two-step gradient accumulation (effective 32) and `bf16-mixed`; do not describe the CLI default as batch 32.

## Configuration And Artifacts

- Runtime configs are hydra-zen builders in `src/peka/configs/`; there are no source YAML configs. Training writes YAML snapshots under `OUTPUT/` only. Add or change hyperparameters in builders and numbered script arguments, not legacy YAML references.
- Supported model matrix is defined by `ENCODER_TABLE` and `PEFT_METHODS` in `src/peka/configs/model.py`: `{H-optimus-0, UNI} x {bone, lora, adalora, hra}`.
- Default joint experiment/checkpoint names include `_joint`; checkpoints land in `Pretrained/<exp_name>/`, while adapters and snapshots land in timestamped `OUTPUT/<exp_name>_<timestamp>/`.
- Feature extraction must rebuild the same encoder, PEFT method, rank, alpha, and dropout used for training; pass the exact Lightning `.ckpt` to script 40.
- Evaluation path wiring is currently inconsistent: script 40 defaults to `peka_embed/<encoder>_<peft>/...`, but script 50 searches `peka_embed/H0/...` or `peka_embed/UNI/...`. Until fixed, pass script 40 `--output_dir` matching script 50's expected short encoder folder, or verify paths before an expensive evaluation run.
- Leakage-free evaluation requires the shared manifest from script 15 and independently trained fold checkpoints. Use one unique `RUN_ID` across scripts 33 and 51; provenance sidecars bind K-means labels, teacher, checkpoint, features, and HVGs to that run/fold. Script 50 requires `--fold` unless `--probe_only` is explicitly requested; never present `--probe_only` results as held-out generalization.

## Verification

- For data/config changes, run `python scripts/00_check_env.py`, then the cheapest affected numbered stage; use `python scripts/14_smoke_test_loader.py` for loader changes.
- For Phase 2 changes, run the one-epoch limited-batch command above with CSV logging. Full data preparation, training, and regression are large/expensive and are not routine verification.
