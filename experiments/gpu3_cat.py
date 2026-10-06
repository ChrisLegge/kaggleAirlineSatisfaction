"""CatBoost GPU tuning. Usage:
python experiments/cat_gpu.py --name gpu_cat_x --folds 0 --depth 8 --lr 0.06 ...
Saves only when all folds are run.
"""
import argparse
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import common  # noqa: E402
from features import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu_cat_dev")
ap.add_argument("--folds", default="0")
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.06)
ap.add_argument("--l2", type=float, default=3)
ap.add_argument("--border", type=int, default=128)
ap.add_argument("--ohe", type=int, default=2)
ap.add_argument("--grow", default="SymmetricTree")
ap.add_argument("--min_leaf", type=int, default=1)
ap.add_argument("--rs", type=float, default=1.0, help="random_strength")
ap.add_argument("--bag_temp", type=float, default=1.0)
ap.add_argument("--cat_ratings", type=int, default=1)
ap.add_argument("--max_ctr", type=int, default=None, help="max_ctr_complexity")
ap.add_argument("--iters", type=int, default=20000)
ap.add_argument("--od", type=int, default=300)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--metric", default="AUC")
ap.add_argument("--v2", type=int, default=0)
ap.add_argument("--fd_cat", type=int, default=0)
ap.add_argument("--num_cat", type=int, default=0)
ap.add_argument("--ref", default="lgbm", help="oof name for correlation")
ap.add_argument("--mp", type=int, default=25, help="metric_period")
ap.add_argument("--notes", default="")
ap.add_argument("--k10", type=int, default=0)
ap.add_argument("--aux", type=int, default=0)
args = ap.parse_args()
print(vars(args), flush=True)

if args.v2:
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from features_v2 import build_v2
    from features import RATINGS
    X, X_test, y, cats = build_v2()
    X, X_test = X.copy(), X_test.copy()
    cats = list(cats)
    NUMC = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    for df in (X, X_test):
        for c in cats:
            df[c] = df[c].astype(str)
        if args.cat_ratings:
            for c in RATINGS:
                df[c + "_cat"] = df[c].astype(str)
        if args.fd_cat:
            df["fd_cat"] = df["Flight Distance"].astype(str)
        if args.num_cat:
            for c in NUMC:
                df[c + "_ncat"] = df[c].fillna(-1).astype(int).astype(str)
    if args.cat_ratings:
        cats += [c + "_cat" for c in RATINGS]
    if args.fd_cat:
        cats.append("fd_cat")
    if args.num_cat:
        cats += [c + "_ncat" for c in NUMC]
else:
    X, X_test, y, ids, cats, X_orig, y_orig = build(cat_ratings=bool(args.cat_ratings))
    del X_orig, y_orig
if args.aux:
    import pandas as pd
    a_tr = pd.read_parquet(os.path.join(common.PREDS, "auxA_train.parquet"))
    a_te = pd.read_parquet(os.path.join(common.PREDS, "auxA_test.parquet"))
    ac = [c for c in a_tr.columns if c not in X.columns and c not in ("id", common.TARGET)]
    X = pd.concat([X.reset_index(drop=True), a_tr[ac].reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), a_te[ac].reset_index(drop=True)], axis=1)
print("X shape", X.shape, "cats", len(cats), flush=True)
folds = common.fold_ids()
NF = common.N_FOLDS
if args.k10:
    rng = np.random.default_rng(123)
    f10 = folds * 2
    for f in range(common.N_FOLDS):
        for cls in (0, 1):
            idx = np.where((folds == f) & (y == cls))[0]
            f10[rng.permutation(idx)[: len(idx) // 2]] += 1
    folds, NF = f10, 10
folds_run = list(range(NF)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]

params = dict(
    iterations=args.iters, learning_rate=args.lr, depth=args.depth, l2_leaf_reg=args.l2,
    border_count=args.border, one_hot_max_size=args.ohe, grow_policy=args.grow,
    random_strength=args.rs, bagging_temperature=args.bag_temp,
    eval_metric=args.metric, metric_period=args.mp, task_type="GPU", devices="0", gpu_ram_part=0.8,
    random_seed=args.seed, od_type="Iter", od_wait=args.od, verbose=1000,
    thread_count=4, used_ram_limit="4gb",
    train_dir=os.path.join(ROOT, "experiments", "catboost_info_gpu"),
)
if args.grow != "SymmetricTree":
    params["min_data_in_leaf"] = args.min_leaf
if args.max_ctr is not None:
    params["max_ctr_complexity"] = args.max_ctr

test_pool = Pool(X_test, cat_features=cats)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
for f in folds_run:
    t0 = time.time()
    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
    ptr = Pool(X.iloc[tr], y[tr], cat_features=cats)
    pva = Pool(X.iloc[va], y[va], cat_features=cats)
    m = CatBoostClassifier(**params)
    m.fit(ptr, eval_set=pva, use_best_model=True)
    oof[va] = m.predict_proba(pva)[:, 1]
    test_pred += m.predict_proba(test_pool)[:, 1] / len(folds_run)
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va], oof[va]):.5f} iters {m.get_best_iteration()} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del m, ptr, pva

if len(folds_run) == NF:
    common.save_preds(args.name, oof, test_pred, args.notes or f"CatBoost GPU {vars(args)}")
    lg = np.load(os.path.join(common.PREDS, f"oof_{args.ref}.npy"))
    print(f"corr with {args.ref}:", np.corrcoef(oof, lg)[0, 1], flush=True)
