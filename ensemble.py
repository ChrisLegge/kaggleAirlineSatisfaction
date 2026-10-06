"""Ensembling of saved OOF predictions.

Two methods, both evaluated honestly with fold-wise CV (weights fit on 4 folds, scored on the 5th):
  * hill   - greedy forward selection with replacement (Caruana) on rank-normalised preds
  * stack  - logistic regression on logit-transformed preds

Usage:
  python ensemble.py                      # use every preds/oof_*.npy
  python ensemble.py lgbm xgb cat         # only these
  python ensemble.py --out final_v1.csv   # also write a submission from the best method
"""
import glob
import os
import sys

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import PREDS, fold_ids, get_y, write_submission

EXCLUDE = {"lgbm_orig"}  # known-bad runs kept only for the record


def rank(p):
    return rankdata(p) / len(p)


def load(names):
    oof = np.column_stack([np.load(os.path.join(PREDS, f"oof_{n}.npy")) for n in names])
    test = np.column_stack([np.load(os.path.join(PREDS, f"test_{n}.npy")) for n in names])
    return oof, test


def hill_climb(R, y, n_iter=40):
    """Greedy ensemble selection with replacement. Returns weight vector."""
    counts = np.zeros(R.shape[1])
    cur = np.zeros(len(y))
    best = -1
    for _ in range(n_iter):
        scores = [roc_auc_score(y, (cur * counts.sum() + R[:, j]) / (counts.sum() + 1)) for j in range(R.shape[1])]
        j = int(np.argmax(scores))
        if scores[j] <= best + 1e-7:
            break
        best = scores[j]
        cur = (cur * counts.sum() + R[:, j]) / (counts.sum() + 1)
        counts[j] += 1
    return counts / counts.sum()


def stack_fit(L, y):
    return LogisticRegression(C=1.0, max_iter=1000).fit(L, y)


def cv_eval(oof, y, folds):
    """Honest CV of both ensemble methods: fit on 4 folds, predict the held-out fold."""
    R = np.column_stack([rank(c) for c in oof.T])
    L = logit(np.clip(oof, 1e-6, 1 - 1e-6))
    pred_h = np.zeros(len(y))
    pred_s = np.zeros(len(y))
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        pred_h[va] = R[va] @ hill_climb(R[tr], y[tr])
        pred_s[va] = stack_fit(L[tr], y[tr]).decision_function(L[va])
    # AUC per fold then mean, since scores across folds are on different scales
    auc = lambda p: np.mean([roc_auc_score(y[folds == f], p[folds == f]) for f in np.unique(folds)])
    return auc(pred_h), auc(pred_s)


def main():
    args = [a for a in sys.argv[1:]]
    out = None
    if "--out" in args:
        out = args[args.index("--out") + 1]
        args = [a for a in args if a not in ("--out", out)]
    names = args or sorted(
        os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(PREDS, "oof_*.npy"))
        if os.path.basename(p)[4:-4] not in EXCLUDE
    )
    y, folds = get_y(), fold_ids()
    oof, test = load(names)
    fold_mean = lambda p: np.mean([roc_auc_score(y[folds == f], p[folds == f]) for f in range(5)])

    print(f"{'model':28s} {'CV AUC':>9s} {'mean-fold':>9s}")
    for n, c in zip(names, oof.T):
        print(f"{n:28s} {roc_auc_score(y, c):9.6f} {fold_mean(c):9.6f}")

    h, s = cv_eval(oof, y, folds)
    best_single = max(fold_mean(c) for c in oof.T)
    print(f"\nBest single (mean-fold):   {best_single:.6f}")
    print(f"Hill-climb  (honest CV):   {h:.6f}")
    print(f"LogReg stack (honest CV):  {s:.6f}")

    R = np.column_stack([rank(c) for c in oof.T])
    w = hill_climb(R, y)
    print("\nHill-climb weights on all data:")
    for n, wi in sorted(zip(names, w), key=lambda t: -t[1]):
        if wi > 0:
            print(f"  {n:28s} {wi:.3f}")

    if out:
        if s > h:
            L = logit(np.clip(oof, 1e-6, 1 - 1e-6))
            Lt = logit(np.clip(test, 1e-6, 1 - 1e-6))
            pred = stack_fit(L, y).predict_proba(Lt)[:, 1]
            method = "stack"
        else:
            pred = np.column_stack([rank(c) for c in test.T]) @ w
            method = "hill"
        print(f"\nWrote {write_submission(pred, out)} using {method}")


if __name__ == "__main__":
    main()
