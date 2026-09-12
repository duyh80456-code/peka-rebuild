#!/usr/bin/env bash
# Chay TRAINING cua chinh RunningStone/PEKA, khong dung code cua repo nay.
#
# Repo goc khong tu chay duoc. Moi file config YAML deu co
#     _target_: histomil2.Hydra_helper....
# ma trong repo KHONG co package ten histomil2 -- chi co hai file csv trung
# ten. Nen `instantiate()` se nem ModuleNotFoundError ngay dong dau.
# Cong them vai cho nua. Script nay va cham dung nhung cho do, in ro tung cai,
# va khong sua gi khac. Xem phan "VA CHAM" khi chay de kiem lai.
#
# Training chi can h5py/anndata/timm/peft/lightning. KHONG can submodule HEST
# hay scFoundation (da kiem: KD_LoRA.py, dataset_helper.py, dataset_part_helpers.py
# khong import cai nao), nen clone khong kem --recursive.
set -euo pipefail

ROOT="${ROOT:-/kaggle/tmp/official}"
DATA_SRC="${DATA_SRC:-/kaggle/working/peka-rebuild/DATA}"
CKPT_DST="${CKPT_DST:-/kaggle/tmp/official/Pretrained}"
OUT_DST="${OUT_DST:-/kaggle/tmp/official/OUTPUT}"
DATASET_NAME="${DATASET_NAME:-breast_in_hest}"
MODEL_CONFIG="${MODEL_CONFIG:-Models/H-optimus-0_Bone_MLP.yaml}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

REPO="$ROOT/PEKA"
# kd_lora_train_with_cluster.py tinh root = dirname^3 cua os.getcwd(), nen NO
# PHAI duoc chay tu $ROOT/PEKA/scripts/1_train_with_2_encoders/. Ten thu muc
# "PEKA" cung la bat buoc: code_root = root + "/PEKA/".
mkdir -p "$ROOT" "$CKPT_DST" "$OUT_DST"
if [ ! -d "$REPO/.git" ]; then
    git clone --depth 1 https://github.com/RunningStone/PEKA.git "$REPO"
fi

CFG="$REPO/hydra_zen/Configs"
DS_YAML="$CFG/Datasets/${DATASET_NAME}_scFoundation_with_clustered100_label.yaml"

echo "=== VA CHAM vao repo goc (chi nhung dong duoi day) ==="

# 1. histomil2 -> peka. Khong co package nao ten histomil2 trong repo.
n=$(grep -rl "histomil2\." "$CFG" "$REPO/peka" 2>/dev/null | tee /dev/stderr | wc -l)
grep -rl "histomil2\." "$CFG" "$REPO/peka" 2>/dev/null | xargs -r sed -i 's/histomil2\./peka./g'
echo "  [1] histomil2. -> peka.   ($n file)"

# 2. Config dataset cua ho ten task la breast_visium_26k. DATA cua chung ta
#    nam o breast/$DATASET_NAME, va dataset_config.csv ghi ten do.
cp "$CFG/Datasets/breast_visium_26k_scFoundation_with_clustered100_label.yaml" "$DS_YAML"
sed -i "s|^task: .*|task: ${DATASET_NAME}|" "$DS_YAML"
echo "  [2] task: breast_visium_26k -> ${DATASET_NAME}"

# 3. batch_size 32 lam T4 16 GiB tran khi backward qua ViT-g.
sed -i "s|^batch_size: .*|batch_size: ${BATCH_SIZE}|" "$DS_YAML"
sed -i "s|^num_workers: .*|num_workers: ${NUM_WORKERS}|" "$DS_YAML"
echo "  [3] batch_size 32 -> ${BATCH_SIZE}, num_workers -> ${NUM_WORKERS}"

# 4. Trainers/kd_lora.yaml de max_epochs: 10, trong khi paper ghi 50.
TR="$CFG/Trainers/kd_lora.yaml"
sed -i "s|^max_epochs: .*|max_epochs: ${EPOCHS}|" "$TR"
echo "  [4] max_epochs 10 -> ${EPOCHS}   (config cua ho la 10, paper noi 50)"

# 5. with_logger: wandb -- khong co API key o day. trainer_part_helpers chi
#    xu ly dung chuoi "wandb", gia tri khac thi Lightning dung logger mac dinh.
sed -i "s|^with_logger: .*|with_logger: none|" "$TR"
echo "  [5] with_logger wandb -> none"

# 6. DATA/ trong repo goc chi co readme.md.
rm -rf "$REPO/DATA"; ln -s "$DATA_SRC" "$REPO/DATA"
ln -sfn "$CKPT_DST" "$ROOT/Pretrained"
ln -sfn "$OUT_DST"  "$ROOT/OUTPUT"
echo "  [6] DATA -> $DATA_SRC ; Pretrained, OUTPUT -> /kaggle/tmp"

{ echo "HF_TOKEN=${HF_TOKEN:-}"; echo "WANDB_API_KEY="; echo "WANDB_ENTITY="; } > "$REPO/.env"
echo "  [7] .env"
echo "======================================================="

test -f "$REPO/DATA/breast/dataset_config.csv" || {
    echo "!! thieu DATA/breast/dataset_config.csv -- get_preprocess_status() se nem loi"; exit 1; }
ls -1 "$REPO/DATA/breast/$DATASET_NAME" || exit 1

cd "$REPO/scripts/1_train_with_2_encoders"
python3 kd_lora_train_with_cluster.py \
    --dataset_config "Datasets/$(basename "$DS_YAML")" \
    --model_config   "$MODEL_CONFIG" \
    --optimizer_config "Optimizers/kd_lora.yaml" \
    --trainer_config   "Trainers/kd_lora.yaml" \
    --phase1_epochs 20 \
    --phase1_lr 1e-4 \
    --phase1_hidden_dim 512 \
    --exp_name "official_${DATASET_NAME}_bone"
