"""Diversity check vs public members + our members.
Usage: python experiments/gpu_div.py name1 name2 ...
For each name: AUC, max Pearson corr with any oof_pub_* member (and which), corr with feat_c3,
and the gain from adding it to a ridge-logistic stack of all pub members + feat_c3 (5-fold CV of the stacker).
"""
import glob
import os
import sys

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402

y = common.get_y()
folds = common.fold_ids()
pub = sorted(glob.glob(os.path.join(common.PREDS, "oof_pub_*.npy")))
P = {os.path.basename(p)[4:-4]: np.load(p) for p in pub}
P["feat_c3"] = np.load(os.path.join(common.PREDS, "oof_feat_c3.npy"))


def L(o):
    return logit(np.clip(o, 1e-6, 1 - 1e-6))


base_names = list(P)
B = np.column_stack([L(P[n]) for n in base_names]).astype(np.float32)


def stack_auc(M):
    o = np.zeros(len(y))
    for f in range(common.N_FOLDS):
        tr, va = folds != f, folds == f
        lr = LogisticRegression(C=0.05, max_iter=500).fit(M[tr], y[tr])
        o[va] = lr.decision_function(M[va])
    return roc_auc_score(y, o)


base_auc = stack_auc(B)
print(f"stack of {len(base_names)} pub+feat_c3: {base_auc:.6f}")
for n in sys.argv[1:]:
    o = np.load(os.path.join(common.PREDS, f"oof_{n}.npy"))
    cors = {k: np.corrcoef(L(o), L(v))[0, 1] for k, v in P.items()}
    k = max(cors, key=cors.get)
    a = stack_auc(np.column_stack([B, L(o)]))
    print(f"{n:24s} auc {roc_auc_score(y, o):.6f} corr_feat_c3 {cors['feat_c3']:.4f} "
          f"max_pub_corr {cors[k]:.4f} ({k}) stack+ {a:.6f} (gain {a - base_auc:+.6f})", flush=True)
