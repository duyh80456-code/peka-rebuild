#!/usr/bin/env bash
# Danh gia feature cua chung ta bang CHINH code cua RunningStone/PEKA.
#
# Script 52 chep lai giao thuc cua ho vao repo nay. Cai nay thi khong chep gi
# ca -- no keo repo goc ve, tro vao DATA cua chung ta, roi chay step3 cua ho.
# Khong con cho nao de tranh cai ve "ban da hieu dung y ho chua".
#
# Ba khac biet so voi script 50 cua repo nay, deu la mac dinh cua ho:
#   KFold(shuffle=True, random_state=2025)   spot tron ngau nhien, khong theo slide
#   --mask_zero_values                       bo moi spot co gen = 0
#   top_50_genes_Visium_...Breast_Cancer.json  danh sach gene ho ship san
#
# step3 chi can scanpy/sklearn/torch/dotenv. Khong can submodule HEST hay
# scFoundation, nen clone khong kem --recursive cho nhanh.
set -euo pipefail

ROOT="${ROOT:-/kaggle/tmp/official}"
DATA_SRC="${DATA_SRC:-/kaggle/working/peka-rebuild/DATA}"
OUT="${OUT:-/kaggle/working/OUTPUT/official_eval}"
FEATURE_TYPE="${FEATURE_TYPE:-peka}"
# De MASK_ZERO=0 neu muon tach rieng anh huong cua viec loc zero.
MASK_ZERO="${MASK_ZERO:-1}"
DATASET_NAME="${DATASET_NAME:-breast_in_hest}"

REPO="$ROOT/PEKA"
# setup_paths() cua ho ghep cung "$project_root/PEKA", va get_dataset_paths()
# ghep cung "$project_root/PEKA/DATA" -- nen ten thu muc PEKA la bat buoc.
mkdir -p "$ROOT"
if [ ! -d "$REPO/.git" ]; then
    git clone --depth 1 https://github.com/RunningStone/PEKA.git "$REPO"
fi

# DATA/ trong repo goc chi co readme.md, thay bang symlink sang du lieu that.
rm -rf "$REPO/DATA"
ln -s "$DATA_SRC" "$REPO/DATA"

# load_dotenv() doc file nay; thieu thi no im lang bo qua, nhung tao cho chac.
: > "$REPO/.env"
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN=$HF_TOKEN" >> "$REPO/.env"

STEP3="$REPO/scripts/2_downstream_gene_pred/step3_task_gene_expr_reg_KFold.py"
GENES="$REPO/scripts/2_downstream_gene_pred/top_50_genes_Visium_Homo_sapien_Breast_Cancer.json"

echo "=== INPUT cua step3 ==="
echo "  project_root : $ROOT"
echo "  DATA         : $DATA_SRC -> $REPO/DATA"
ls -1 "$REPO/DATA/breast/$DATASET_NAME" 2>/dev/null || {
    echo "  !! khong thay $REPO/DATA/breast/$DATASET_NAME"; exit 1; }
echo "  feature_type : $FEATURE_TYPE   mask_zero=$MASK_ZERO"
echo "  gene list    : $(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['genes']))" "$GENES") gene (cua ho)"
echo "======================="

ARGS=(
    --project_root "$ROOT"
    --tissue_type breast
    --dataset_name "$DATASET_NAME"
    --embedder_name scFoundation
    --image_encoder_name H0
    --image_backbone H-optimus-0
    --gene_list_json "$GENES"
    --output_root "$OUT"
    --feature_type "$FEATURE_TYPE"
    --epochs 300
)
# Driver cua ho de WITH_INDEPENDENT_TEST_SET=false, nen khong truyen co do.
[ "$MASK_ZERO" = "1" ] && ARGS+=(--mask_zero_values)

python3 "$STEP3" "${ARGS[@]}"

CSV=$(find "$OUT" -name gene_regression_results.csv -newermt '-1 hour' | head -1)
if [ -n "$CSV" ]; then
    python3 - "$CSV" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("pearson_correlation")]
v = sorted((float(r["pearson_correlation"]), r.get("gene", "?")) for r in rows)
print("=" * 64)
print("  CODE GOC, GIAO THUC GOC")
print("  PCC trung binh tren %d gene : %.4f" % (len(v), sum(x for x, _ in v) / len(v)))
print("  H-optimus-0 dong bang       : 0.624   (paper)")
print("  PEKA trong paper            : 0.698")
print("-" * 64)
print("  te nhat :", ", ".join("%s %.3f" % (g, x) for x, g in v[:5]))
print("  tot nhat:", ", ".join("%s %.3f" % (g, x) for x, g in v[-5:]))
print("  CSV     :", sys.argv[1])
print("=" * 64)
PY
else
    echo "!! khong tim thay gene_regression_results.csv trong $OUT"
fi
