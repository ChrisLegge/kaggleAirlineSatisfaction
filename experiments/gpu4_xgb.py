"""XGBoost GPU (QuantileDMatrix) on v2 features + auxA, K-fold where each common fold is split in --ksplit parts.
Usage: python -u experiments/gpu4_xgb.py --name gpu4_xgb_v2aux_k10 --ksplit 2 --split_seed 31
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
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.chdir(ROOT)
import common  # noqa: E402
from features_v2 import build_v2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu4_xgb_dev")
ap.add_argument("--folds", default="all")
ap.add_argument("--ksplit", type=int, default=2)
ap.add_argument("--split_seed", type=int, default=31)
ap.add_argument("--leaves", type=int, default=127)
ap.add_argument("--mcw", type=float, default=20)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--cbt", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--notes", default="")
args = ap.parse_args()
print(vars(args), flush=True)

X, Xt, y, cats = build_v2()
a_tr = pd.read_parquet(os.path.join(common.PREDS, "auxA_train.parquet"))
a_te = pd.read_parquet(os.path.join(common.PREDS, "auxA_test.parquet"))
ac = [c for c in a_tr.columns if c not in X.columns and c not in ("id", common.TARGET)]
X = pd.concat([X.reset_index(drop=True), a_tr[ac].reset_index(drop=True)], axis=1)
Xt = pd.concat([Xt.reset_index(drop=True), a_te[ac].reset_index(drop=True)], axis=1)
X.columns = [c.replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]
Xt.columns = X.columns
for c in cats:
    X[c] = X[c].astype("category")
    Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)
print("X shape", X.shape, flush=True)

folds5 = common.fold_ids()
rng = np.random.default_rng(args.split_seed)
folds = folds5 * args.ksplit
for f in range(common.N_FOLDS):
    for cls in (0, 1):
        idx = rng.permutation(np.where((folds5 == f) & (y == cls))[0])
        folds[idx] += np.arange(len(idx)) % args.ksplit
NF = common.N_FOLDS * args.ksplit
folds_run = list(range(NF)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]

params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda",
              learning_rate=args.lr, max_depth=0, grow_policy="lossguide", max_leaves=args.leaves,
              min_child_weight=args.mcw, subsample=0.8, colsample_bytree=args.cbt, reg_lambda=2.0,
              max_bin=256, max_cat_to_onehot=1, nthread=4, seed=args.seed)
oof = np.zeros(len(X))
tp = np.zeros(len(Xt))
dte = xgb.DMatrix(Xt, enable_categorical=True)
for f in folds_run:
    t0 = time.time()
    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
    dtr = xgb.QuantileDMatrix(X.iloc[tr], y[tr], enable_categorical=True, max_bin=256)
    dva = xgb.QuantileDMatrix(X.iloc[va], y[va], enable_categorical=True, ref=dtr)
    bst = xgb.train(params, dtr, 20000, evals=[(dva, "va")], early_stopping_rounds=300, verbose_eval=False)
    r = (0, bst.best_iteration + 1)
    oof[va] = bst.predict(dva, iteration_range=r)
    tp += bst.predict(dte, iteration_range=r) / len(folds_run)
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va], oof[va]):.5f} iters {bst.best_iteration} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del bst, dtr, dva
if len(folds_run) == NF:
    common.save_preds(args.name, oof, tp, args.notes or f"XGB GPU v2+auxA {NF}-fold {vars(args)}")
