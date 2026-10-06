"""10-fold LightGBM (each common fold split in 2, stratified; same split recipe as gpu3_realmlp --k10).
Usage: python -u experiments/cpu4_lgbm_k10.py --name cpu4_lgbm_t34_k10 [--aux 1] [--variant t34|var]
Per-fold checkpoints in preds/_ck_<name>/ so it can resume.
"""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.chdir(ROOT)
import lightgbm as lgb
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import common
from features_v2 import build_v2

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--aux", type=int, default=0)
ap.add_argument("--variant", default="t34")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

T34 = dict(num_leaves=67, min_child_samples=100, min_sum_hessian_in_leaf=0.0078, bagging_fraction=0.899,
           feature_fraction=0.541, lambda_l1=16.14, lambda_l2=0.0014, cat_smooth=2.59, cat_l2=15.19,
           min_data_per_group=11, max_bin=1023, path_smooth=0.008)
P = dict(T34)
lr = 0.02
if args.variant == "var":
    P.update(num_leaves=255, feature_fraction=0.3, min_child_samples=60, extra_trees=True)
    lr = 0.03

X, Xt, y, cats = build_v2()
if args.aux:
    a_tr = pd.read_parquet(os.path.join(common.PREDS, "auxA_train.parquet"))
    a_te = pd.read_parquet(os.path.join(common.PREDS, "auxA_test.parquet"))
    assert len(a_tr) == len(X) and len(a_te) == len(Xt)
    drop = set(X.columns) | {"id", common.TARGET}
    a_tr = a_tr[[c for c in a_tr.columns if c not in drop]].reset_index(drop=True)
    a_te = a_te[a_tr.columns].reset_index(drop=True)
    print("aux cols", len(a_tr.columns), flush=True)
    X = pd.concat([X.reset_index(drop=True), a_tr], axis=1)
    Xt = pd.concat([Xt.reset_index(drop=True), a_te], axis=1)
for c in cats:
    X[c] = X[c].astype("category")
    Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)

folds5 = common.fold_ids()
rng = np.random.default_rng(123)
f10 = folds5 * 2
for f in range(common.N_FOLDS):
    for cls in (0, 1):
        idx = np.where((folds5 == f) & (y == cls))[0]
        f10[rng.permutation(idx)[: len(idx) // 2]] += 1

base = dict(objective="binary", metric="auc", verbosity=-1, num_threads=6, device_type="cpu",
            seed=args.seed, bagging_seed=args.seed, feature_fraction_seed=args.seed,
            data_random_seed=args.seed, bagging_freq=1, learning_rate=lr)
base.update(P)
dsk = {k: base[k] for k in ("max_bin", "min_data_per_group", "cat_smooth", "cat_l2")}
ck = os.path.join(common.PREDS, f"_ck_{args.name}"); os.makedirs(ck, exist_ok=True)
oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
for f in range(10):
    p = os.path.join(ck, f"f{f}.npz")
    tr, va = np.where(f10 != f)[0], np.where(f10 == f)[0]
    if os.path.exists(p):
        d = np.load(p); oof[va] = d["o"]; tp += d["t"] / 10; continue
    t0 = time.time()
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cats, params=dsk)
    dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr, categorical_feature=cats)
    m = lgb.train(base, dtr, num_boost_round=20000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    o = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    t = m.predict(Xt, num_iteration=m.best_iteration)
    np.savez(p, o=o, t=t); oof[va] = o; tp += t / 10
    print(f"[{args.name}] fold {f}: {roc_auc_score(y[va], o):.6f} iters {m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    del dtr, dva, m
common.save_preds(args.name, oof, tp, f"LGBM k10 variant={args.variant} aux={args.aux} {P} lr {lr}")
