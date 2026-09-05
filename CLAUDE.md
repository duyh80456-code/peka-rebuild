# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PEKA reproduction codebase — focused on **breast cancer only**. Reproduces arxiv 2504.07061: parameter-efficient knowledge transfer that distills scFoundation knowledge into a pathology image encoder via PEFT (BONE / LoRA / AdaLoRA / HRA), built around a filtered HEST1K download.

## Layout (src layout)

```
PEKA/                              workspace root
├── pyproject.toml                 package: peka @ src/peka
├── src/peka/                      the package (importable as `peka`)
│   ├── __init__.py                logger + sys.path bootstrap (adds external/{HEST,scFoundation})
│   ├── paths.py                   single source of truth — all paths from __file__, not cwd
│   ├── env.py                     dotenv + HF/W&B helpers
│   ├── configs/                   pure-Python hydra-zen builders (no YAML)
│   ├── repro_data/                breast dataset factory + HEST filter + HVG
│   ├── train/                     phase1_mlp + phase2_kd orchestration
│   ├── eval/                      feature extraction + ridge_kfold
│   ├── utils/                     seed, logging
│   ├── Data/, Model/, Trainer/, Hydra_helper/, DownstreamTasks_helper/, Exp_helper/
│       inlined from original PEKA codebase (capitalized — preserves imports like
│       `from peka.Trainer.KD_LoRA import pl_KD_LoRA` and `from peka.Data.dataset_helper import ...`)
├── external/HEST/                 vendored MahmoodLab/HEST loader (loaded via sys.path)
├── external/scFoundation/         vendored scFoundation foundation model (loaded via sys.path)
├── scripts/                       12 numbered pipeline scripts (00 → 50)
├── support/                       HEST_v1_1_0.csv, peka_breast_datasets.csv, top_50_genes_breast.json
├── DATA/, OUTPUT/, Pretrained/    gitignored
└── README.md, .env.example, CLAUDE.md
```

## Environment

- Conda env: `hest` (Python 3.9). Install with: `conda activate hest && pip install -e .` (uses src layout).
- Required env vars (loaded from workspace `.env`, template in `.env.example`): `WANDB_API_KEY`, `WANDB_ENTITY`, `HF_TOKEN`, `HEST1K_STORAGE_PATH`. Optional: `WORKSPACE`.
- No test suite, linter, or build step. "Running" = executing numbered scripts in `scripts/` in order.

## Path resolution

Unlike the original codebase, paths are derived from `__file__` (`src/peka/paths.py:_THIS.parents[2]`), NOT from `os.getcwd()`. **Scripts work from any directory.**

Scripts add `<workspace>/src` to `sys.path` themselves (line near top: `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))`), so `pip install -e .` is convenient but not required.

## Pipeline (the mental model)

Strict numerical order. Outputs of step N feed step N+1.

| Phase | Scripts | What |
|---|---|---|
| 0 — data | `00_check_env`, `10_download_hest1k`, `11_build_breast_dataset`, `12_compute_scfoundation_emb`, `13_kmeans_cluster_labels`, `14_smoke_test_loader` | Filtered breast-only HEST1K download → image patches + aligned adata → scFoundation embeddings → k-means cluster pseudo-labels. **`14_smoke_test_loader.py` must pass before training.** |
| 1 — train | `20_train_phase1_mlp`, `30_train_phase2_kd`, `31_train_all_combinations.sh` | Phase 1: MLP teacher on cluster labels (CPU OK). Phase 2: PEFT student with KD + structure loss. The sweep script runs all 4 PEFT × 2 encoders = 8 experiments. |
| 2 — eval | `40_extract_peka_features`, `41_compute_hvg_top50`, `50_eval_gene_regression_kfold` | Extract per-spot features → top-50 HVG → 5-fold PCA(256)+Ridge → Pearson PCC. |

## Filtered HEST1K download

`scripts/10_download_hest1k.py` downloads ONLY the breast samples (~125 IDs, ~30-100 GB) instead of the full 1 TB.

Mechanism (in `peka.repro_data.hest_filter`):
1. Download `HEST_v1_1_0.csv` only (~500 KB)
2. Filter rows: `organ=Breast, species=Homo sapiens, [optional st_technology in platforms]`
3. `huggingface_hub.snapshot_download(allow_patterns=[f"st/{id}.h5ad", f"wsis/{id}.tif", f"metadata/{id}.json", ...])`

Use `--platform Visium` to filter further (smallest, fastest path); `--full` to disable filtering (~1 TB).

## Phase 2 internals — KD + structure-alignment loss

`peka.Trainer.KD_LoRA.pl_KD_LoRA.distillation_loss` (KD_LoRA.py:188-205):

```python
soft_loss = F.kl_div(student_logits/T, teacher_logits/T) * T**2     # KD
hard_loss = self.loss_fn(student_logits, cluster_labels)            # structure alignment (CE)
total = α · soft_loss + (1-α) · hard_loss
```

`cluster_labels` come from k-means on scFoundation embeddings (script 13 → adata.obs["gen_clustered_label_100"]). The dataloader returns `(img, emb, label)` triples.

Paper hyperparameters (defaults in `30_train_phase2_kd.py`): T=2.0, α=0.5, lr=1e-4, Adam, 50 epochs, batch=32, LoRA r=256 α=32 dropout=0.1.

## Configs (no YAML)

All configs are pure-Python hydra-zen `builds()` calls under `src/peka/configs/`. They target functions in `peka.Hydra_helper.*`. There are NO YAML files to maintain. The original codebase's broken `_target_: histomil2.*` YAML references are gone.

To change a hyperparameter: pass kwargs to `build_*_config()` in `scripts/30_train_phase2_kd.py`. The 8 (encoder × PEFT) combinations live in `peka.configs.model.ENCODER_TABLE × PEFT_METHODS`.

## Conventions worth knowing

- **Numeric prefixes are load-bearing.** Scripts 00 → 50 must run in order.
- **Capitalized inline modules.** `Data/`, `Model/`, `Trainer/`, etc. inside `src/peka/` keep the original PEKA naming so existing internal imports (`from peka.Trainer.KD_LoRA import ...`) keep working.
- **Lowercase new modules.** `configs/`, `train/`, `eval/`, `repro_data/`, `utils/` are new packages introduced in this rewrite. `repro_data/` (not `data/`) avoids collision with the inlined Pascal-case `Data/`.
- **HEST adata prefix is hardcoded** to `HEST_breast_adata_` (`peka/Data/dataset_helper.py:30`, `peka/DownstreamTasks_helper/inference.py:15`). Fine for breast-only; would need parameterization for other tissues.
- **Outputs land outside the repo tree** (`OUTPUT/`, `Pretrained/`, `DATA/` are gitignored).
- **W&B is the source of truth.** Pass `--with_logger csv` to disable.
- **scFoundation imported at top level** via `from scFoundation.model.pretrainmodels.select_model import select_model` (works because `external/scFoundation` is on sys.path via `peka.__init__`).
