"""Sparse one-hot GLM / factorisation-machine style members (very different inductive bias from trees/MLPs).
Fields: exact FD, exact Age, binned delays, 14 ratings, 4 cats, FD/Age buckets; pairwise one-hot crosses of
the low-cardinality fields (+ FD x Class/Travel/Customer). Model: L2 logistic regression on the sparse one-hots.
Usage: python -u experiments/gpu3_glm.py --name gpu3_glm_pair --folds all --C 0.05 [--triples 1]
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import common  # noqa: E402
from features import RATINGS, BASE_CATS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu3_glm_dev")
ap.add_argument("--folds", default="0")
ap.add_argument("--C", type=float, default=0.05)
ap.add_argument("--pairs", type=int, default=1)
ap.add_argument("--triples", type=int, default=0)
ap.add_argument("--min_count", type=int, default=5)
ap.add_argument("--notes", default="")
args = ap.parse_args()
print(vars(args), flush=True)

train, test, _ = common.load_raw()
y = train[common.TARGET].values
n_tr = len(train)
df = pd.concat([train.drop(columns=[common.TARGET]), test], ignore_index=True)
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
dbins = [-1, 0, 5, 15, 30, 60, 120, 1e9]
F = pd.DataFrame(index=df.index)
for c in RATINGS + BASE_CATS:
    F[c] = df[c].astype(str)
F["fd"] = df["Flight Distance"].astype(str)
F["age"] = df["Age"].astype(str)
F["fd_b"] = pd.cut(np.log1p(df["Flight Distance"]), 24, labels=False).astype(str)
F["age_b"] = (df["Age"] // 7).astype(str)
F["dep_b"] = pd.cut(df["Departure Delay in Minutes"], dbins, labels=False).astype(str)
F["arr_b"] = pd.cut(df["Arrival Delay in Minutes"], dbins, labels=False).astype(str)
F["dd_b"] = np.sign(df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]).astype(int).astype(str)
low = RATINGS + BASE_CATS + ["fd_b", "age_b", "dep_b", "arr_b"]
fields = list(F.columns)
if args.pairs:
    for a, b in itertools.combinations(low, 2):
        F[a + "|" + b] = F[a] + "_" + F[b]
    for c in ["Class", "Type of Travel", "Customer Type"]:
        F["fd|" + c] = F["fd"] + "_" + F[c]
        F["age|" + c] = F["age"] + "_" + F[c]
if args.triples:
    tri = ["Class", "Type of Travel", "Customer Type", "Inflight wifi service", "Online boarding",
           "Inflight entertainment", "Seat comfort", "Leg room service", "age_b", "fd_b"]
    for a, b, c in itertools.combinations(tri, 3):
        F[a + "|" + b + "|" + c] = F[a] + "_" + F[b] + "_" + F[c]
print("fields", F.shape[1], flush=True)

rows, cols_, off = [], [], 0
n = len(F)
cnt_mask_src = F.iloc[:n_tr]
for c in F.columns:
    codes, uniq = pd.factorize(F[c])
    vc = np.bincount(codes[:n_tr], minlength=len(uniq))
    keep = vc >= args.min_count  # rare levels -> shared "other" level
    remap = np.where(keep, np.cumsum(keep) - 1, keep.sum())
    cols_.append(remap[codes] + off)
    off += keep.sum() + 1
    F.drop(columns=c, inplace=True)
nf = len(cols_)
Xall = sp.csr_matrix((np.ones(n * nf, dtype=np.float32),
                      np.column_stack(cols_).ravel(), np.arange(0, n * nf + 1, nf)), shape=(n, off))
del cols_, F
print("X", Xall.shape, "nnz/row", nf, flush=True)
X, Xt = Xall[:n_tr], Xall[n_tr:]

folds = common.fold_ids()
folds_run = list(range(5)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
oof = np.zeros(n_tr)
tp = np.zeros(Xt.shape[0])
for f in folds_run:
    t0 = time.time()
    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
    m = LogisticRegression(C=args.C, max_iter=400, tol=1e-5).fit(X[tr], y[tr])
    oof[va] = m.decision_function(X[va])
    tp += m.decision_function(Xt) / len(folds_run)
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va], oof[va]):.5f} iters {m.n_iter_} ({time.time()-t0:.0f}s)",
          flush=True)
if len(folds_run) == 5:
    sig = lambda z: 1 / (1 + np.exp(-z))
    common.save_preds(args.name, sig(oof), sig(tp), args.notes or f"sparse one-hot GLM {vars(args)}")
