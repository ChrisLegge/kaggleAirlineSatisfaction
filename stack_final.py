"""Final stacker: L2 LogisticRegression (C=0.02) on standardised logits of the members in stack_members.txt.

Honest CV: standardisation + LR fit on 4 folds of common.fold_ids(), scored on the 5th, mean-fold AUC reported.
The submission model is refit on all train rows.

Usage:
  python stack_final.py                       # members from stack_members.txt
  python stack_final.py --refresh-members     # rewrite stack_members.txt = every eligible preds/oof_*.npy
                                              # (excludes naji_* / blend_* label-fit blends, lgbm_orig, feat_baseline)
  python stack_final.py --out stack_v3_x.csv  # also write submissions/<file> and log to experiments/stack_candidates.csv
  python stack_final.py --C 0.05
"""
import csv
import datetime as dt
import glob
import os
import sys

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import PREDS, ROOT, fold_ids, get_y, write_submission

EXCLUDE = {"lgbm_orig", "feat_baseline"}
MEMBERS = os.path.join(ROOT, "stack_members.txt")
CANDS = os.path.join(ROOT, "experiments", "stack_candidates.csv")


def eligible():
    names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(PREDS, "oof_*.npy")))
    return [n for n in names if n not in EXCLUDE and not n.startswith(("naji_", "blend_"))]


def L(m):
    return logit(np.clip(m, 1e-6, 1 - 1e-6)).astype(np.float32)


def main():
    a = sys.argv[1:]
    C = float(a[a.index("--C") + 1]) if "--C" in a else 0.02
    out = a[a.index("--out") + 1] if "--out" in a else None
    if "--refresh-members" in a or not os.path.exists(MEMBERS):
        open(MEMBERS, "w").write("\n".join(eligible()) + "\n")
    names = open(MEMBERS).read().split()
    bad = [n for n in names if n.startswith(("naji_", "blend_"))]
    assert not bad, f"label-fit blends in member list: {bad}"

    y, folds = get_y(), fold_ids()
    X = np.column_stack([L(np.load(os.path.join(PREDS, f"oof_{n}.npy"))) for n in names])
    T = np.column_stack([L(np.load(os.path.join(PREDS, f"test_{n}.npy"))) for n in names])

    pred = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        pred[va] = LogisticRegression(C=C, max_iter=3000).fit((X[tr] - mu) / sd, y[tr]).decision_function((X[va] - mu) / sd)
    fa = [roc_auc_score(y[folds == f], pred[folds == f]) for f in range(5)]
    cv = float(np.mean(fa))
    print(f"{len(names)} members, LR C={C}: honest CV (mean-fold) {cv:.6f}  folds {np.round(fa, 5)}")

    mu, sd = X.mean(0), X.std(0)
    m = LogisticRegression(C=C, max_iter=3000).fit((X - mu) / sd, y)
    top = sorted(zip(names, m.coef_[0]), key=lambda t: -abs(t[1]))[:10]
    print("Largest weights:", ", ".join(f"{n}={w:.2f}" for n, w in top))
    if out:
        p = m.predict_proba((T - mu) / sd)[:, 1]
        print("Wrote", write_submission(p, out))
        new = not os.path.exists(CANDS)
        with open(CANDS, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "file", "honest_cv", "n_members", "members", "method"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), out, f"{cv:.6f}", len(names),
                        " ".join(names), f"LogReg L2 C={C} on standardised logits (stack_final.py)"])


if __name__ == "__main__":
    main()
