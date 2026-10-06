"""Shared helpers for stacker experiments (honest fold-wise CV on common.fold_ids()).

Caches the logit matrix of every eligible member into scratch_aux/stack_cache_{oof,test}.npy (float32) + names.
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
from common import PREDS, fold_ids, get_y  # noqa: E402

EXCLUDE = {"lgbm_orig", "feat_baseline"}
CACHE = os.path.join(ROOT, "scratch_aux")


def eligible():
    names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(PREDS, "oof_*.npy")))
    return [n for n in names if n not in EXCLUDE and not n.startswith(("naji_", "blend_"))]


def L(m):
    return logit(np.clip(m, 1e-6, 1 - 1e-6)).astype(np.float32)


def load_all(refresh=False):
    names = eligible()
    nf = os.path.join(CACHE, "stack_cache_names.txt")
    if not refresh and os.path.exists(nf) and open(nf).read().split() == names:
        return names, np.load(os.path.join(CACHE, "stack_cache_oof.npy")), np.load(os.path.join(CACHE, "stack_cache_test.npy"))
    oof = np.column_stack([L(np.load(os.path.join(PREDS, f"oof_{n}.npy"))) for n in names])
    test = np.column_stack([L(np.load(os.path.join(PREDS, f"test_{n}.npy"))) for n in names])
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, "stack_cache_oof.npy"), oof)
    np.save(os.path.join(CACHE, "stack_cache_test.npy"), test)
    open(nf, "w").write("\n".join(names))
    return names, oof, test


def mean_fold_auc(y, p, folds):
    return float(np.mean([roc_auc_score(y[folds == f], p[folds == f]) for f in np.unique(folds)]))


def cv_lr(X, y, folds, C=1.0, standardise=True, return_pred=False):
    """Honest CV: standardisation stats + LR fit on 4 folds, scored on the 5th."""
    pred = np.zeros(len(y))
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        Xtr, Xva = X[tr], X[va]
        if standardise:
            mu, sd = Xtr.mean(0), Xtr.std(0)
            Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        pred[va] = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y[tr]).decision_function(Xva)
    s = mean_fold_auc(y, pred, folds)
    return (s, pred) if return_pred else s


def dedup(names, X, thr=0.9995, score=None):
    """Drop members whose |corr| with an already-kept (better-scoring) member exceeds thr."""
    order = np.argsort(-np.asarray(score)) if score is not None else np.arange(len(names))
    C = np.corrcoef(X[::7].T)
    keep = []
    for j in order:
        if all(C[j, k] <= thr for k in keep):
            keep.append(j)
    keep = sorted(keep)
    return [names[j] for j in keep], keep
