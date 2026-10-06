"""Fast logistic-regression stacker on logits of saved OOF preds (honest fold-wise CV).

Usage: python stack.py --out file.csv [--C 1.0] name1 name2 ... | --all
  --all  uses every preds/oof_*.npy except EXCLUDE
"""
import glob
import os
import sys

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import PREDS, fold_ids, get_y, write_submission

EXCLUDE = {"lgbm_orig", "feat_baseline"}


def parse():
    a = sys.argv[1:]
    opt = {"out": None, "C": 1.0}
    for k in ("out", "C"):
        if f"--{k}" in a:
            i = a.index(f"--{k}")
            opt[k] = a[i + 1]
            del a[i:i + 2]
    if "--all" in a:
        a = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(PREDS, "oof_*.npy")))
        a = [n for n in a if n not in EXCLUDE]
    return a, opt["out"], float(opt["C"])


def L(m):
    return logit(np.clip(m, 1e-6, 1 - 1e-6))


def main():
    names, out, C = parse()
    y, folds = get_y(), fold_ids()
    oof = L(np.column_stack([np.load(os.path.join(PREDS, f"oof_{n}.npy")) for n in names]))
    test = L(np.column_stack([np.load(os.path.join(PREDS, f"test_{n}.npy")) for n in names]))
    mu, sd = oof.mean(0), oof.std(0)
    oof, test = (oof - mu) / sd, (test - mu) / sd

    pred = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        pred[va] = LogisticRegression(C=C, max_iter=2000).fit(oof[tr], y[tr]).decision_function(oof[va])
    fold_aucs = [roc_auc_score(y[folds == f], pred[folds == f]) for f in range(5)]
    singles = {n: np.mean([roc_auc_score(y[folds == f], oof[folds == f, j]) for f in range(5)]) for j, n in enumerate(names)}
    best = max(singles, key=singles.get)
    print(f"{len(names)} members | best single {best} {singles[best]:.6f} (mean-fold)")
    print(f"Stack honest CV (mean-fold): {np.mean(fold_aucs):.6f}  folds {np.round(fold_aucs, 5)}")

    m = LogisticRegression(C=C, max_iter=2000).fit(oof, y)
    top = sorted(zip(names, m.coef_[0]), key=lambda t: -abs(t[1]))[:12]
    print("Largest weights:", ", ".join(f"{n}={w:.2f}" for n, w in top))
    if out:
        print("Wrote", write_submission(m.predict_proba(test)[:, 1], out))


if __name__ == "__main__":
    main()
