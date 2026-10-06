"""GPU brute-force kNN classifier (torch). Features standardised then scaled by |coef| of a fold-train logistic
regression (supervised metric), optional extra weight on exact-route match (Flight Distance equality bonus).
Prediction = smoothed label mean of k nearest fold-train rows (Gaussian distance weights). Test: full-train neighbours.
Usage: python -u experiments/gpu3_knn.py --name gpu3_knn --folds all --k 200
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.chdir(ROOT)
import common  # noqa: E402
from features import RATINGS, BASE_CATS  # noqa: E402
from features_v2 import build_v2  # noqa: E402

torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu3_knn_dev")
ap.add_argument("--folds", default="0")
ap.add_argument("--k", type=int, default=200)
ap.add_argument("--extra", type=int, default=0, help="add orig_pred/fd_o_te to the metric")
ap.add_argument("--onehot", type=int, default=1, help="one-hot ratings in metric")
ap.add_argument("--fd_w", type=float, default=0.0, help="distance bonus for exact FD match")
ap.add_argument("--bw", type=float, default=1.0, help="gaussian bandwidth multiplier (relative to median kth dist)")
ap.add_argument("--notes", default="")
args = ap.parse_args()
print(vars(args), flush=True)

X, Xt, y, _ = build_v2()
NUM = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
cols = NUM + RATINGS + BASE_CATS + (["fd_o_te", "orig_pred"] if args.extra else [])
df = pd.concat([X[cols], Xt[cols]], ignore_index=True)
n_tr = len(X)
del X, Xt
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
fd_raw = df["Flight Distance"].values.astype(np.float32)
for c in ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
    df[c] = np.log1p(df[c])
for c in BASE_CATS:
    df[c] = pd.factorize(df[c])[0]
if args.extra:
    p = df["orig_pred"].clip(1e-5, 1 - 1e-5)
    df["orig_pred"] = np.log(p / (1 - p))
parts = [df.drop(columns=RATINGS) if args.onehot else df]
if args.onehot:
    parts.append(pd.get_dummies(df[RATINGS].astype(int).astype(str)).astype(np.float32))
F = pd.concat(parts, axis=1).astype(np.float32)
F = ((F - F.mean()) / (F.std() + 1e-6)).values.astype(np.float32)
print("metric dims", F.shape[1], flush=True)
dev = "cuda"
yt = torch.tensor(y, dtype=torch.float32, device=dev)
fd_t = torch.tensor(fd_raw, device=dev)


def predict(tr_idx, q_idx, w):
    W = torch.tensor(F[tr_idx] * w, device=dev)
    sq = (W * W).sum(1)
    ytr = yt[tr_idx]
    fdtr = fd_t[tr_idx]
    out = np.zeros(len(q_idx), dtype=np.float32)
    prior = float(y[tr_idx].mean())
    bs = 384
    med = None
    for s in range(0, len(q_idx), bs):
        qi = q_idx[s:s + bs]
        Q = torch.tensor(F[qi] * w, device=dev)
        d = (Q * Q).sum(1, keepdim=True) + sq[None, :] - 2 * Q @ W.T
        if args.fd_w > 0:
            d = d - args.fd_w * (fd_t[qi][:, None] == fdtr[None, :]).float()
        dk, ik = torch.topk(d, args.k + 1, dim=1, largest=False)
        # drop exact self-match only matters for test-in-train (not here), keep k
        dk, ik = dk[:, :args.k], ik[:, :args.k]
        if med is None:
            med = dk[:, -1].clamp_min(1e-6).median()
        wt = torch.exp(-dk.clamp_min(0) / (args.bw * med))
        lab = ytr[ik]
        p = ((wt * lab).sum(1) + 2 * prior) / (wt.sum(1) + 2)
        out[s:s + bs] = p.cpu().numpy()
    return out


folds = common.fold_ids()
folds_run = list(range(5)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
oof = np.zeros(n_tr)


def metric_w(tr):
    sub = tr[np.random.default_rng(0).choice(len(tr), 150000, replace=False)]
    lr = LogisticRegression(C=0.1, max_iter=500).fit(F[sub], y[sub])
    return np.abs(lr.coef_[0]).astype(np.float32)


for f in folds_run:
    t0 = time.time()
    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
    w = metric_w(tr)
    oof[va] = predict(tr, va, w)
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va], oof[va]):.5f} ({time.time() - t0:.0f}s)", flush=True)
if len(folds_run) == 5:
    tr = np.arange(n_tr)
    tp = predict(tr, np.arange(n_tr, len(F)), metric_w(tr))
    common.save_preds(args.name, np.clip(oof, 1e-6, 1 - 1e-6), np.clip(tp, 1e-6, 1 - 1e-6),
                      args.notes or f"GPU kNN supervised-metric {vars(args)}")
