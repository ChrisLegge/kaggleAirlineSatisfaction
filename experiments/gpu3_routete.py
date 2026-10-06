"""XGB GPU on raw columns + nested fold-safe, heavily smoothed route (Flight Distance) target encodings.
Train rows of outer fold f get TE from an inner 5-fold split of the fold-train; val/test rows get TE from the
whole fold-train. Test pred = mean of the 5 fold models.
Usage: python -u experiments/gpu3_routete.py --name gpu3_xgb_routete --folds all --m 200
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import common  # noqa: E402
from features import RATINGS, BASE_CATS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu3_routete_dev")
ap.add_argument("--folds", default="0")
ap.add_argument("--m", type=float, default=200)
ap.add_argument("--depth", type=int, default=7)
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--te_only", type=int, default=0, help="drop raw FD/Age numerics")
ap.add_argument("--notes", default="")
args = ap.parse_args()
print(vars(args), flush=True)

train, test, _ = common.load_raw()
y = train[common.TARGET].values.astype(np.float64)
n_tr = len(train)
df = pd.concat([train.drop(columns=[common.TARGET, "id"]), test.drop(columns=["id"])], ignore_index=True)
for c in BASE_CATS:
    df[c] = pd.factorize(df[c])[0]
S = pd.DataFrame(index=df.index)
fd = df["Flight Distance"].astype(str)
ageb = (df["Age"] // 5).astype(str)
keys = {
    "fd": fd,
    "fd_cls": fd + "_" + df["Class"].astype(str),
    "fd_tt": fd + "_" + df["Type of Travel"].astype(str),
    "fd_ct": fd + "_" + df["Customer Type"].astype(str),
    "fd_cls_tt": fd + "_" + df["Class"].astype(str) + df["Type of Travel"].astype(str),
    "fd_ageb": fd + "_" + ageb,
    "fd_ob": fd + "_" + df["Online boarding"].astype(str),
    "fd_wifi": fd + "_" + df["Inflight wifi service"].astype(str),
    "age_ctc": df["Age"].astype(str) + "_" + df["Class"].astype(str) + df["Type of Travel"].astype(str)
               + df["Customer Type"].astype(str),
}
K = {k: pd.factorize(v)[0] for k, v in keys.items()}


def te(code, src, dst, prior):
    n = np.bincount(code[src], minlength=code.max() + 1).astype(np.float64)
    s = np.bincount(code[src], weights=y[src], minlength=code.max() + 1)
    enc = (s + args.m * prior) / (n + args.m)
    return enc[code[dst]], n[code[dst]]


def te_block(tr, dst_sets):
    """returns TE frame for tr (nested inner folds) and for each dst set (from all of tr)."""
    prior = y[tr].mean()
    inner = np.random.default_rng(0).integers(0, 5, len(tr))
    out_tr = np.zeros((len(tr), 2 * len(K)), np.float32)
    outs = [np.zeros((len(d), 2 * len(K)), np.float32) for d in dst_sets]
    for j, (k, code) in enumerate(K.items()):
        for i in range(5):
            a, b = tr[inner != i], np.where(inner == i)[0]
            e, n = te(code, a, tr[b], prior)
            out_tr[b, 2 * j], out_tr[b, 2 * j + 1] = e, n * 1.25
        for o, d in zip(outs, dst_sets):
            e, n = te(code, tr, d, prior)
            o[:, 2 * j], o[:, 2 * j + 1] = e, n
    return out_tr, outs


base = df.drop(columns=["Flight Distance", "Age"]) if args.te_only else df
B = base.values.astype(np.float32)
names = list(base.columns) + [f"{k}_{s}" for k in K for s in ("te", "n")]
params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda",
              learning_rate=args.lr, max_depth=args.depth, min_child_weight=10, subsample=0.8,
              colsample_bytree=0.6, reg_lambda=3, max_bin=256)
folds = common.fold_ids()
folds_run = list(range(5)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
oof = np.zeros(n_tr)
tp = np.zeros(len(df) - n_tr)
te_idx = np.arange(n_tr, len(df))
for f in folds_run:
    t0 = time.time()
    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
    Ttr, (Tva, Tte) = te_block(tr, [va, te_idx])
    dtr = xgb.QuantileDMatrix(np.hstack([B[tr], Ttr]), y[tr], feature_names=names)
    dva = xgb.DMatrix(np.hstack([B[va], Tva]), y[va], feature_names=names)
    bst = xgb.train(params, dtr, 20000, evals=[(dva, "va")], early_stopping_rounds=300, verbose_eval=False)
    rng = (0, bst.best_iteration + 1)
    oof[va] = bst.predict(dva, iteration_range=rng)
    tp += bst.predict(xgb.DMatrix(np.hstack([B[te_idx], Tte]), feature_names=names), iteration_range=rng) / len(folds_run)
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va], oof[va]):.5f} it {bst.best_iteration} "
          f"({time.time() - t0:.0f}s)", flush=True)
if len(folds_run) == 5:
    common.save_preds(args.name, oof, tp, args.notes or f"XGB raw + nested route TE {vars(args)}")
