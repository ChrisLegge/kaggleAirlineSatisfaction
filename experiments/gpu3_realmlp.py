"""RealMLP (pytabkit) on v2 features with categorical twins of numerics/ratings.
Usage: python -u experiments/realmlp_gpu.py --name gpu_v2_realmlp --folds 0 --n_ens 8 [--mode full|raw] [--aux 1]
Saves only when all folds are run.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.chdir(ROOT)
import common  # noqa: E402
from features import RATINGS, BASE_CATS  # noqa: E402
from features_v2 import build_v2  # noqa: E402

torch.set_num_threads(4)
from pytabkit import RealMLP_TD_Classifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu_v2_realmlp_dev")
ap.add_argument("--folds", default="0")
ap.add_argument("--mode", default="full", choices=["full", "raw"])
ap.add_argument("--n_ens", type=int, default=8)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=0.053)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--aux", type=int, default=0)
ap.add_argument("--twins", type=int, default=1)
ap.add_argument("--aux_name", default="auxA")
ap.add_argument("--ref", default="feat_c3")
ap.add_argument("--notes", default="")
ap.add_argument("--k10", type=int, default=0, help="10-fold: split each common fold in 2 (stratified)")
args = ap.parse_args()
print(vars(args), flush=True)

X, Xt, y, cats = build_v2()
n_tr = len(X)
df = pd.concat([X, Xt], ignore_index=True)
del X, Xt
NUMRAW = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
if args.mode == "raw":
    keep = NUMRAW + RATINGS + BASE_CATS + [c for c in df.columns if c.startswith("fd_")]
    df = df[keep].copy()
    cat_cols = list(BASE_CATS)
else:
    cat_cols = list(cats)
    if "orig_pred" in df:
        p = df["orig_pred"].clip(1e-5, 1 - 1e-5)
        df["orig_pred"] = np.log(p / (1 - p))
if args.aux:
    aux_tr = pd.read_parquet(os.path.join(common.PREDS, f"{args.aux_name}_train.parquet"))
    aux_te = pd.read_parquet(os.path.join(common.PREDS, f"{args.aux_name}_test.parquet"))
    assert len(aux_tr) == n_tr
    aux = pd.concat([aux_tr, aux_te], ignore_index=True)
    aux = aux[[c for c in aux.columns if c not in df.columns and c not in ("id", common.TARGET)]]
    print("aux cols", list(aux.columns), flush=True)
    df = pd.concat([df, aux.reset_index(drop=True)], axis=1)
for c in cat_cols:
    df[c] = df[c].astype(str)
if args.twins:
    for c in NUMRAW + RATINGS:
        df[c + "_cat_"] = df[c].astype(int).astype(str)
        cat_cols.append(c + "_cat_")
num_cols = [c for c in df.columns if c not in cat_cols]
df[num_cols] = df[num_cols].astype(np.float32)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].astype("category")
print("shape", df.shape, "cats", len(cat_cols), flush=True)
X, Xt = df.iloc[:n_tr].reset_index(drop=True), df.iloc[n_tr:].reset_index(drop=True)
del df

params = dict(
    device="cuda", random_state=args.seed, n_threads=4, verbosity=0,
    n_ens=args.n_ens, n_epochs=args.epochs, batch_size=args.bs, use_early_stopping=False,
    lr=args.lr, wd=0.0236, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18,
    hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="expm4t",
    plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
    ls_eps=0.01, ls_eps_sched="sqrt_cos", use_ls=True, add_front_scale=False,
    bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
    val_metric_name="1-auc_ovr", predict_batch_size=8192,
)

folds = common.fold_ids()
NF = common.N_FOLDS
if args.k10:
    rng = np.random.default_rng(123)
    if args.k10 == 1:
        f10 = folds * 2
        for f in range(common.N_FOLDS):
            for cls in (0, 1):
                idx = np.where((folds == f) & (y == cls))[0]
                f10[rng.permutation(idx)[: len(idx) // 2]] += 1
        folds, NF = f10, 10
    else:  # split each common fold into args.k10 stratified parts
        m_ = args.k10
        fk = folds * m_
        for f in range(common.N_FOLDS):
            for cls in (0, 1):
                idx = rng.permutation(np.where((folds == f) & (y == cls))[0])
                fk[idx] += np.arange(len(idx)) % m_
        folds, NF = fk, common.N_FOLDS * m_
folds_run = list(range(NF)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
oof = np.zeros(n_tr)
test_pred = np.zeros(len(Xt))
for f in folds_run:
    t0 = time.time()
    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
    m = RealMLP_TD_Classifier(**params)
    m.fit(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_col_names=cat_cols)
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    test_pred += m.predict_proba(Xt)[:, 1] / len(folds_run)
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va], oof[va]):.5f} ({time.time() - t0:.0f}s) "
          f"maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    del m
    torch.cuda.empty_cache()

if len(folds_run) == NF:
    common.save_preds(args.name, oof, test_pred, args.notes or f"RealMLP {vars(args)}")
    lg = np.load(os.path.join(common.PREDS, f"oof_{args.ref}.npy"))
    print(f"corr with {args.ref}:", np.corrcoef(oof, lg)[0, 1], flush=True)
