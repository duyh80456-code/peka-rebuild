#!/usr/bin/env python
"""Re-score existing features under the ORIGINAL PEKA evaluation protocol.

RunningStone/PEKA @ bf9c6f5 splits folds like this
(peka/DownstreamTasks_helper/train_and_val_exp.py:267):

    k_fold = KFold(n_splits=Ksplit, shuffle=True, random_state=2025)

A plain shuffled KFold over spots. No groups, no slide holdout -- spots that
are physical neighbours on one slide land on both sides of the split, and
neighbouring spots in spatial transcriptomics are strongly correlated. This
repo evaluates with GroupKFold over slides instead, which is the honest
protocol but is NOT the one behind the paper's 0.624 / 0.698.

So the two numbers were never comparable. This script measures our own
features under the paper's split, which is the only way to say whether the
reproduction actually reproduces.

The regressor is untouched: PCA(256) + Ridge(alpha=1.0), identical in both
repos. Features and labels come from this repo's own loader, so the sole
difference from `50_eval_gene_regression_kfold.py` is the fold assignment.

Nothing here is a defence of the protocol. Quote it as "paper's protocol",
never as a result.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402

import peka  # noqa: F401,E402
from peka import logger  # noqa: E402
from peka.paths import (  # noqa: E402
    DEFAULT_SCLLM, OUTPUT_ROOT, SUPPORT_DIR, WORKSPACE,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default="H-optimus-0")
    p.add_argument("--peft", default="bone")
    p.add_argument("--feature_type", default="peka",
                   choices=["peka", "image_encoder", "scLLM"])
    p.add_argument("--n_splits", type=int, default=5)
    # Bang goc mac dinh True: cat them 20% khoi train lam val. val khong duoc
    # dung de chon model (train_regressor tra ve ngay), nen tac dung duy nhat
    # la train it di 20%. Giu dung mac dinh de so sanh trung thuc.
    p.add_argument("--with_independent_test_set", type=int, default=1)
    p.add_argument("--min_spots", type=int, default=321)
    p.add_argument("--feature_dir", default=None)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    from peka.DownstreamTasks_helper.gene_expression_prediction import (
        get_dataset_paths, load_data,
    )
    from peka.DownstreamTasks_helper.train_and_val_exp import GeneRegressor

    paths = get_dataset_paths(
        str(WORKSPACE), "breast", "breast_in_hest", DEFAULT_SCLLM,
        feature_type=args.feature_type,
        image_encoder_name="H0",
        image_backbone=args.encoder,
        use_scLLM_name_as_subfolder=True,
        embed_path_override=args.feature_dir,
    )

    gene_list_json = SUPPORT_DIR / "top_50_genes_breast.json"
    with open(gene_list_json) as f:
        genes = json.load(f)["genes"]
    logger.info(f"Loaded {len(genes)} genes from {gene_list_json}")

    embeddings_dict, labels_dict, _ = load_data(
        paths, genes,
        feature_type=args.feature_type,
        img_prefix="patch_224_0.5_",
        embed_prefix="HEST_breast_adata_",
        use_binned=False,
        mask_zero_values=False,
        return_groups=True,
    )
    genes = [g for g in genes if g in labels_dict]

    out = Path(args.output_dir) if args.output_dir else (
        OUTPUT_ROOT / "eval" / args.peft / "official_protocol")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for gene in genes:
        X = np.asarray(embeddings_dict[gene])
        y = np.asarray(labels_dict[gene])
        if X.shape[0] < args.min_spots:
            logger.warning(f"Skipping {gene}: only {X.shape[0]} points")
            continue

        # Dung nguyen van cach chia cua repo goc, ke ca random_state=2025.
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=2025)
        per_fold = []
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if args.with_independent_test_set:
                X_tr, _, y_tr, _ = train_test_split(
                    X_tr, y_tr, test_size=0.2, random_state=2025)
            reg = GeneRegressor(X.shape[1])
            reg.fit(X_tr, y_tr)
            pred = reg.inference(X_te)
            if np.std(y_te) == 0 or np.std(pred) == 0:
                continue          # pearsonr khong xac dinh tren dau vao hang so
            per_fold.append(pearsonr(y_te, pred)[0])

        if not per_fold:
            logger.warning(f"Skipping {gene}: every fold was degenerate")
            continue
        rows.append({
            "gene": gene,
            "n_spots": int(X.shape[0]),
            "pearson_correlation": float(np.mean(per_fold)),
            "pearson_std": float(np.std(per_fold)),
            "folds": len(per_fold),
        })
        logger.info("  %-10s %6.3f  (n=%d)"
                    % (gene, rows[-1]["pearson_correlation"], X.shape[0]))

    df = pd.DataFrame(rows)
    csv = out / "gene_regression_results.csv"
    df.to_csv(csv, index=False)

    if df.empty:
        logger.error("No gene was eligible; nothing to report")
        return
    mean = df["pearson_correlation"].mean()
    logger.info(f"Saved results → {csv}")
    print("=" * 64)
    print("  GIAO THUC CUA PAPER: KFold(shuffle=True, random_state=2025)")
    print("  spot lan nhau giua train/test -- KHONG phai slide-holdout")
    print("-" * 64)
    print("  PCC trung binh tren %d gene : %.4f" % (len(df), mean))
    print("  H-optimus-0 dong bang       : 0.624   (paper)")
    print("  PEKA trong paper            : 0.698")
    print("=" * 64)
    logger.info(f"Mean Pearson across genes: {mean:.4f}")


if __name__ == "__main__":
    main()
