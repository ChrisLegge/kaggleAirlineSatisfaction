"""XGBoost GPU tuning with QuantileDMatrix. Saves only when all folds are run."""
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
from features import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu_xgb_dev")
ap.add_argument("--folds", default="0")
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--mcw", type=float, default=5)
ap.add_argument("--gamma", type=float, default=0)
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--sub", type=float, default=0.8)
ap.add_argument("--cbt", type=float, default=0.5)
ap.add_argument("--cbn", type=float, default=1.0)
ap.add_argument("--lam", type=float, default=2.0)
ap.add_argument("--alpha", type=float, default=0.0)
ap.add_argument("--max_bin", type=int, default=256)
ap.add_argument("--grow", default="depthwise")
ap.add_argument("--leaves", type=int, default=0)
ap.add_argument("--cat_ratings", type=int, default=0)
ap.add_argument("--od", type=int, default=300)
ap.add_argument("--notes", default="")
ap.add_argument("--v2", type=int, default=0)
ap.add_argument("--ref", default="lgbm")
args = ap.parse_args()
print(vars(args), flush=True)

if args.v2:
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from features_v2 import build_v2
    X, X_test, y, cats = build_v2()
    X, X_test = X.copy(), X_test.copy()
    X.columns = [c.replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]
    X_test.columns = X.columns
else:
    X, X_test, y, ids, cats, X_orig, y_orig = build(cat_ratings=bool(args.cat_ratings))
    del X_orig, y_orig
print("X shape", X.shape, flush=True)
for c in cats:
    X[c] = X[c].astype("category")
    X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)
folds = common.fold_ids()
folds_run = list(range(common.N_FOLDS)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]

params = dict(
    objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda",
    learning_rate=args.lr, max_depth=args.depth, min_child_weight=args.mcw, gamma=args.gamma,
    subsample=args.sub, colsample_bytree=args.cbt, colsample_bynode=args.cbn,
    reg_lambda=args.lam, reg_alpha=args.alpha, max_bin=args.max_bin, grow_policy=args.grow,
    max_leaves=args.leaves, max_cat_to_onehot=1, nthread=4, seed=common.SEED,
)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
for f in folds_run:
    t0 = time.time()
    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
    dtr = xgb.QuantileDMatrix(X.iloc[tr], y[tr], enable_categorical=True, max_bin=args.max_bin)
    dva = xgb.QuantileDMatrix(X.iloc[va], y[va], enable_categorical=True, ref=dtr)
    bst = xgb.train(params, dtr, 20000, evals=[(dva, "va")], early_stopping_rounds=args.od, verbose_eval=1000)
    rng = (0, bst.best_iteration + 1)
    oof[va] = bst.predict(dva, iteration_range=rng)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    test_pred += bst.predict(dte, iteration_range=rng) / len(folds_run)
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va], oof[va]):.5f} iters {bst.best_iteration} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del bst, dtr, dva, dte

if len(folds_run) == common.N_FOLDS:
    common.save_preds(args.name, oof, test_pred, args.notes or f"XGB GPU {vars(args)}")
    lg = np.load(os.path.join(common.PREDS, f"oof_{args.ref}.npy"))
    print(f"corr with {args.ref}:", np.corrcoef(oof, lg)[0, 1], flush=True)
