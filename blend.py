"""Blend saved OOF predictions with rank-averaged weights tuned on CV, then write a submission.

Usage: python blend.py lgbm xgb cat
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from features import DATA, TARGET

names = sys.argv[1:] or ["lgbm", "xgb", "cat"]
y = pd.read_csv(f"{DATA}/train.csv", usecols=[TARGET])[TARGET].astype(int).values
ids = pd.read_csv(f"{DATA}/test.csv", usecols=["id"])["id"].values

# AUC only cares about ordering, so blend normalised ranks
rank = lambda p: rankdata(p) / len(p)
oofs = np.column_stack([rank(np.load(f"preds/oof_{n}.npy")) for n in names])
tests = np.column_stack([rank(np.load(f"preds/test_{n}.npy")) for n in names])

for n, col in zip(names, oofs.T):
    print(f"{n:12s} CV AUC {roc_auc_score(y, col):.5f}")
print("Correlations:\n", pd.DataFrame(oofs, columns=names).corr().round(4))

eq = roc_auc_score(y, oofs.mean(axis=1))
print(f"\nEqual-weight blend CV AUC: {eq:.5f}")

res = minimize(
    lambda w: -roc_auc_score(y, oofs @ (np.abs(w) / np.abs(w).sum())),
    x0=np.ones(len(names)) / len(names), method="Nelder-Mead",
)
w = np.abs(res.x) / np.abs(res.x).sum()
print("Optimised weights:", dict(zip(names, w.round(3))))
print(f"Optimised blend CV AUC: {-res.fun:.5f}")

out = f"submissions/blend_{'_'.join(names)}.csv"
pd.DataFrame({"id": ids, TARGET: tests @ w}).to_csv(out, index=False)
print("Saved", out)
