# PEKA — paper reproduction (breast-only)

Clean reproduction of [PEKA (arxiv 2504.07061)](https://arxiv.org/abs/2504.07061) — *"Teaching pathology foundation models to accurately predict gene expression with parameter efficient knowledge transfer"* — focused on breast cancer.

## What's special

- **Filtered HEST1K download** — paper config (Visium-only breast) = ~13 GB instead of ~1 TB.
- **Pure-Python hydra-zen configs** — no broken YAML.
- **Robust paths** — derived from `__file__`, never `os.getcwd()`. Scripts run from any directory.
- **End-to-end pipeline** — 12 numbered scripts: download → preprocess → train → evaluate.
- **All 4 PEFT methods × 2 encoders** = 8 experiments. BONE is paper's main method.

## Paper config (Section 4.1)

> "Visium ST data of Homo Sapiens with breast cancer (n=30,414 pairs of image tiles and gene expression data)"

= **8 Visium breast samples** in HEST1K (~30,543 raw spots → ~30,414 after QC).

## Layout

```
PEKA/
├── README.md, CLAUDE.md         documentation
├── pyproject.toml               package: peka @ src/peka
├── .env.example
├── src/peka/                    the package
│   ├── __init__.py, paths.py, env.py
│   ├── configs/                 pure-Python hydra-zen builders (no broken YAML)
│   │   ├── dataset.py, model.py (build_model_config — 8 combos)
│   │   ├── optimizer.py, trainer.py, pl_model.py, snapshot.py
│   ├── repro_data/              breast factory + HEST filter + HVG
│   │   ├── breast_dataset.py, hest_filter.py, hvg.py
│   ├── train/                   phase1 MLP + phase2 KD orchestration
│   ├── eval/                    feature extraction + ridge_kfold
│   ├── utils/                   seed, logging
│   └── Data/, Model/, Trainer/, Hydra_helper/, DownstreamTasks_helper/, Exp_helper/
│       inlined from original PEKA codebase (preserves `from peka.Trainer.KD_LoRA import ...`)
├── external/HEST/, scFoundation/   vendored (loaded via sys.path)
├── scripts/                     12 numbered pipeline scripts (00 → 50)
├── support/                     HEST_v1_1_0.csv, peka_breast_datasets.csv, top_50_genes_breast.json
└── DATA/, OUTPUT/, Pretrained/  gitignored
```

## Setup

```bash
conda activate hest
pip install -e .
pip install hf_transfer        # ~5-10x faster HF downloads
cp .env.example .env
$EDITOR .env                   # fill in WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN, HEST1K_STORAGE_PATH
python scripts/00_check_env.py # validate
```

## Architecture (paper recap)

**Phase 1**: MLP `1536 → 512 → 512 → 100` trained on frozen scFoundation embeddings to predict k-means cluster labels (k=100). This MLP becomes the frozen teacher.

**Phase 2**: Image encoder (H-optimus-0 / UNI) + PEFT (BONE/LoRA/AdaLoRA/HRA) + translate MLP `enc_dim → 1536`. Loss:

```
L_total = α · KL(student/T || teacher/T) · T²       ← KD term
        + (1-α) · CrossEntropy(student, cluster_label)  ← structure alignment
```

Paper hyperparameters: α=0.5, T=2.0, lr=1e-4, Adam, 50 epochs, batch=32, LoRA r=256 α=32 dropout=0.1.

**Evaluation**: 5-fold CV with 256-PCA → Ridge regression on top-50 HVGs. Metric: Pearson PCC.

## Pipeline

| # | Script | Purpose | Time |
|---|---|---|---|
| 00 | `00_check_env.py` | Validate env + imports | <1 min |
| 10 | `10_download_hest1k.py` | **Filtered Visium-breast download (~13 GB)** | hours |
| 11 | `11_build_breast_dataset.py` | Filter → patches → align genes | ~30 min |
| 12 | `12_compute_scfoundation_emb.py` | scFoundation embeddings (1536-dim) | ~2-4 GPU-h |
| 15 | `15_create_slide_folds.py` | Freeze shared train/val/test slide folds | <1 min |
| 13 | `13_kmeans_cluster_labels.py` | k-means k=100 pseudo-labels | ~10 min |
| 14 | `14_smoke_test_loader.py` | **Smoke test — must pass before training** | <1 min |
| 20 | `20_train_phase1_mlp.py` | Train MLP teacher | ~10 min |
| 30 | `30_train_phase2_kd.py` | **MAIN training** (PEFT student) | ~12 GPU-h |
| 31 | `31_train_all_combinations.sh` | Sweep 8 (encoder × PEFT) experiments | ~96 GPU-h |
| 32 | `32_run_outer_fold.py` | Leakage-free two-phase train + eval for one fold | GPU-hours |
| 33 | `33_run_all_outer_folds.sh` | Run independently trained outer folds | GPU-days |
| 40 | `40_extract_peka_features.py` | Extract per-spot features | ~30 min |
| 41 | `41_compute_hvg_top50.py` | Top-50 HVG selection | <5 min |
| 50 | `50_eval_gene_regression_kfold.py` | 5-fold PCA+Ridge → Pearson | ~30 min |
| 51 | `51_aggregate_outer_folds.py` | Aggregate independently trained folds | <1 min |

### Quickstart (paper config: Visium-only, BONE + H-optimus-0)

```bash
python scripts/00_check_env.py
python scripts/10_download_hest1k.py --minimal --max-workers 16   # ~13 GB
python scripts/11_build_breast_dataset.py
python scripts/12_compute_scfoundation_emb.py
python scripts/15_create_slide_folds.py --folds 5
python scripts/32_run_outer_fold.py \
    --fold 0 --folds 5 \
    --encoder H-optimus-0 --peft bone \
    --use_gpu_kmeans

# Repeat folds 1-4, or run all folds sequentially:
RUN_ID=paper_bone FOLDS=5 \
    EXTRA_FLAGS="--encoder H-optimus-0 --peft bone --use_gpu_kmeans" \
    bash scripts/33_run_all_outer_folds.sh
python scripts/51_aggregate_outer_folds.py \
    --encoder_short H0 --peft bone --folds 5 --run_id paper_bone
```

Each outer fold fits K-means, Phase 1, Phase 2, HVG selection, and Ridge without
its held-out test slides. Do not use a full-cohort checkpoint for held-out claims;
script 50 permits that diagnostic only with the explicit `--probe_only` flag.

### Download options

```bash
# Default — paper config: Visium-only, paper essentials only (~13 GB)
python scripts/10_download_hest1k.py --minimal --max-workers 16

# Visium + Xenium (extra modality, ~50-100 GB)
python scripts/10_download_hest1k.py --minimal --platform Visium --platform Xenium --max-workers 16

# All 125 breast samples (Visium + Xenium + ST, ~100-500 GB)
python scripts/10_download_hest1k.py --minimal --all-platforms --max-workers 16

# Verify before download
python scripts/10_download_hest1k.py --dry-run --minimal

# Full HEST1K (~1 TB; NOT recommended)
python scripts/10_download_hest1k.py --full
```

### Full sweep (paper-scale, all 8 PEFT × encoder combos)

```bash
EXTRA_FLAGS="--epochs 50" bash scripts/31_train_all_combinations.sh
```

### GPU dry-run (~10 min)

Smoke-test Phase 2 training without committing 12 hours:

```bash
python scripts/30_train_phase2_kd.py \
    --encoder H-optimus-0 --peft bone \
    --epochs 1 --limit_train_batches 5 --limit_val_batches 2 \
    --with_logger csv
```

## Configuration

All configs are pure-Python hydra-zen `builds()` calls under `src/peka/configs/`. Targeting functions in `peka.Hydra_helper.*`. NO YAML to maintain. The original codebase's broken `_target_: histomil2.*` YAMLs are gone.

## Outputs

```
PEKA/
├── DATA/HEST1K/                 raw HEST1K (filtered to 8 Visium breast)
│   ├── HEST_v1_1_0.csv
│   ├── st/{ID}.h5ad             gene expression
│   ├── wsis/{ID}.tif            H&E whole-slide images
│   └── metadata/{ID}.json
├── DATA/breast/breast_in_hest/
│   ├── breast_in_hest.csv       filtered HEST1K index for breast
│   ├── patches/                 224x224 patches extracted from WSIs
│   ├── aligned_adata/           gene-name-aligned adata
│   └── scLLM_embed/scFoundation/default_model/
│       ├── paired_seq/          filtered adata + cluster labels
│       └── embeddings/          (.npy of teacher embeddings)
├── Pretrained/
│   ├── teacher_mlp.pt           Phase 1 MLP teacher
│   └── {encoder}_{peft}_breast_in_hest/*.ckpt   Phase 2 Lightning checkpoints
└── OUTPUT/
    ├── {exp_name}_{timestamp}/
    │   ├── configs/*.yaml       frozen run config snapshot
    │   └── phase2/lora/         PEFT adapters + translate_model.pth
    └── eval/{peft}/{encoder}/   per-experiment K-fold results CSV + plots
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'peka'`**: run `pip install -e .` from workspace root, or set `PYTHONPATH=$PWD/src`.

**Download seems "stuck"**: `snapshot_download` is silent during HF API listing (~30-90s) and during huge .tif download. Monitor with `watch -n 30 'du -sh DATA/HEST1K/ && find DATA/HEST1K/ -type f | wc -l'`.

**`No valid samples found`**: phase 0 incomplete. Re-run scripts 11→12→13. Verify `gen_clustered_label_100` exists in `paired_seq/*.h5ad` obs columns.

**FAISS-GPU not installed**: `13_kmeans_cluster_labels.py` falls back to sklearn KMeans automatically.

**W&B fails / offline**: pass `--with_logger csv` to `30_train_phase2_kd.py`.

## Tiếng Việt — Quickstart

Pipeline reproduce paper PEKA chỉ dataset breast cancer Visium (paper config). Tổng download chỉ ~13 GB (paper essentials only).

```bash
conda activate hest
pip install -e .
pip install hf_transfer
cp .env.example .env  # rồi điền WANDB/HF/HEST1K_STORAGE_PATH
python scripts/00_check_env.py        # validate
python scripts/10_download_hest1k.py --minimal --max-workers 16  # ~13 GB Visium-only
python scripts/11_build_breast_dataset.py
python scripts/12_compute_scfoundation_emb.py
python scripts/13_kmeans_cluster_labels.py --use_gpu
python scripts/14_smoke_test_loader.py  # PHẢI PASS
python scripts/20_train_phase1_mlp.py
python scripts/30_train_phase2_kd.py --encoder H-optimus-0 --peft bone
# ... extract + eval
```

Sweep cả 8 combos: `bash scripts/31_train_all_combinations.sh`
# peka-rebuild
