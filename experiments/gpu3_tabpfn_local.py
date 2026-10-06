"""Local-context TabPFN v2 (tabpfn 2.0.9, isolated in experiments/gpu3/tabpfn_lib with its own sklearn 1.6.1).
Rows (train+test, label-free) are clustered with MiniBatchKMeans; each cluster's queries are predicted by TabPFN
whose context = fold-train rows of that cluster plus nearest clusters until --ctx rows (random cap at --ctx).
OOF: fold-train context. Test: full-train context (one pass).
Run: PYTHONPATH=experiments/gpu3/tabpfn_lib python -u experiments/gpu3_tabpfn_local.py --name gpu3_tabpfn_local --folds all
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.chdir(ROOT)
import common  # noqa: E402
from features import RATINGS, BASE_CATS  # noqa: E402
from features_v2 import build_v2  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402

torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu3_tabpfn_local_dev")
ap.add_argument("--folds", default="0")
ap.add_argument("--k", type=int, default=150)
ap.add_argument("--ctx", type=int, default=8000)
ap.add_argument("--n_est", type=int, default=2)
ap.add_argument("--max_clusters", type=int, default=0, help="dev: only first N clusters")
ap.add_argument("--extra", type=int, default=1, help="add fd_o_te, fd_n, orig_pred")
ap.add_argument("--test", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--notes", default="")
args = ap.parse_args()
print(vars(args), flush=True)

X, Xt, y, _ = build_v2()
NUM = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
cols = NUM + RATINGS + BASE_CATS + (["fd_o_te", "fd_n", "orig_pred"] if args.extra else [])
df = pd.concat([X[cols], Xt[cols]], ignore_index=True)
n_tr = len(X)
del X, Xt
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
for c in BASE_CATS:
    df[c] = pd.factorize(df[c])[0]
if args.extra:
    p = df["orig_pred"].clip(1e-5, 1 - 1e-5)
    df["orig_pred"] = np.log(p / (1 - p))
A = df.values.astype(np.float32)
cat_idx = [cols.index(c) for c in BASE_CATS]

# clustering representation (label-free)
Z = df.copy()
for c in ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
    Z[c] = np.log1p(Z[c])
Z = (Z - Z.mean()) / Z.std()
w = {c: 2.0 for c in BASE_CATS}
if args.extra:
    w["orig_pred"] = 3.0
for c, v in w.items():
    Z[c] *= v
t0 = time.time()
km = MiniBatchKMeans(args.k, random_state=0, batch_size=8192, n_init=3).fit(Z.values)
lab = km.labels_
cent = km.cluster_centers_
cd = ((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
order = np.argsort(cd, axis=1)
print(f"kmeans {time.time() - t0:.0f}s, cluster sizes min/med/max", np.bincount(lab).min(),
      int(np.median(np.bincount(lab))), np.bincount(lab).max(), flush=True)
lab_tr, lab_te = lab[:n_tr], lab[n_tr:]
rng = np.random.default_rng(args.seed)


def make_model():
    return TabPFNClassifier(device="cuda", n_estimators=args.n_est, random_state=args.seed,
                            categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                            inference_precision="autocast", fit_mode="low_memory", memory_saving_mode=True)


def run(ctx_mask, q_idx, q_lab):
    """ctx_mask: bool over train rows usable as context; q_idx: global row indices of queries."""
    out = np.zeros(len(q_idx))
    clusters = np.unique(q_lab)
    if args.max_clusters:
        clusters = clusters[:args.max_clusters]
    for i, c in enumerate(clusters):
        qi = np.where(q_lab == c)[0]
        parts, tot = [], 0
        for c2 in order[c]:
            r = np.where((lab_tr == c2) & ctx_mask)[0]
            need = args.ctx - tot
            if len(r) > need:
                r = rng.choice(r, need, replace=False)
            parts.append(r)
            tot += len(r)
            if tot >= args.ctx:
                break
        ctx = np.concatenate(parts)
        m = make_model().fit(A[ctx], y[ctx])
        for s in range(0, len(qi), 4096):
            b = qi[s:s + 4096]
            out[b] = m.predict_proba(A[q_idx[b]])[:, 1]
        del m
        if i % 25 == 0:
            print(f"   cluster {i}/{len(clusters)} ctx {len(ctx)} q {len(qi)} t {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    return out


folds = common.fold_ids()
folds_run = list(range(5)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
oof = np.zeros(n_tr)
REF = np.load(os.path.join(common.PREDS, 'oof_feat_c3.npy'))
for f in folds_run:
    t0 = time.time()
    va = np.where(folds == f)[0]
    oof[va] = run(folds != f, va, lab_tr[va])
    sel = va if not args.max_clusters else va[np.isin(lab_tr[va], np.unique(lab_tr[va])[:args.max_clusters])]
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[sel], oof[sel]):.5f} ref_feat_c3 {roc_auc_score(y[sel], REF[sel]):.5f} ({time.time() - t0:.0f}s)", flush=True)
if len(folds_run) == 5:
    np.save(os.path.join(ROOT, "experiments", "gpu3", f"{args.name}_oof_tmp.npy"), oof)
    tp = np.zeros(len(lab_te))
    if args.test:
        t0 = time.time()
        tp = run(np.ones(n_tr, bool), np.arange(n_tr, n_tr + len(lab_te)), lab_te)
        print(f"test done {time.time() - t0:.0f}s", flush=True)
    common.save_preds(args.name, np.clip(oof, 1e-6, 1 - 1e-6), np.clip(tp, 1e-6, 1 - 1e-6),
                      args.notes or f"local-context TabPFN v2 (kmeans clusters) {vars(args)}")
