"""Final stacker v5: L2 LogisticRegression on fold-standardised logits of the members in stack_members_v5.txt,
optionally with family-wise regularisation (columns of a family multiplied by a scale after standardisation,
i.e. that family's L2 penalty divided by scale^2).

Same honest CV as stack_final.py (v4, left intact): standardisation + LR fit on 4 folds of common.fold_ids(),
scored on the 5th, mean-fold AUC reported. Submission model refit on all train rows.
Members: v4 list + public single-model pairs from import_public_libs3.py. Label-fit blends / stacks
(prefixes naji_, blend_, nstack_) are refused.

Usage:
  python stack_final_v5.py                                   # defaults below (the chosen v5 config)
  python stack_final_v5.py --C 0.02 --pfn 1 --tfm 1          # flat penalty
  python stack_final_v5.py --out stack_v5_x.csv              # also write submissions/<file> and log to experiments/stack_candidates.csv
"""
import csv
import datetime as dt
import os
import sys

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import PREDS, ROOT, fold_ids, get_y, write_submission

MEMBERS = os.path.join(ROOT, "stack_members_v5.txt")
CANDS = os.path.join(ROOT, "experiments", "stack_candidates.csv")
DEFAULT_C, DEFAULT_PFN, DEFAULT_TFM = 0.02, 2.0, 1.25  # chosen v5: honest CV 0.962243 (306 members)


def family(n):
    s = n.lower()
    if "tabpfn" in s or "pfn" in s.split("_")[:2]:
        return "tabpfn"
    if s.startswith("cdtfm_") and not s.endswith(("realmlp", "xgboost")) or "tabicl" in s or "tabm" in s:
        return "tfm"
    if "realmlp" in s or "rmlp" in s or "mlp" in s or "_nn" in s or "deepfm" in s or "fwfm" in s:
        return "mlp"
    return "gbdt"


def L(m):
    return logit(np.clip(m, 1e-6, 1 - 1e-6)).astype(np.float32)


def arg(a, k, d):
    return type(d)(a[a.index(k) + 1]) if k in a else d


def main():
    a = sys.argv[1:]
    C, pfn, tfm = arg(a, "--C", DEFAULT_C), arg(a, "--pfn", DEFAULT_PFN), arg(a, "--tfm", DEFAULT_TFM)
    out = a[a.index("--out") + 1] if "--out" in a else None
    names = open(arg(a, "--members", MEMBERS)).read().split()
    bad = [n for n in names if n.startswith(("naji_", "blend_", "nstack_"))]
    assert not bad, f"label-fit blends in member list: {bad}"
    scale = np.array([{"tabpfn": pfn, "tfm": tfm}.get(family(n), 1.0) for n in names], np.float32)

    y, folds = get_y(), fold_ids()
    X = np.column_stack([L(np.load(os.path.join(PREDS, f"oof_{n}.npy"))) for n in names])
    T = np.column_stack([L(np.load(os.path.join(PREDS, f"test_{n}.npy"))) for n in names])

    pred = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        m = LogisticRegression(C=C, max_iter=3000).fit((X[tr] - mu) / sd * scale, y[tr])
        pred[va] = m.decision_function((X[va] - mu) / sd * scale)
    fa = [roc_auc_score(y[folds == f], pred[folds == f]) for f in range(5)]
    cv = float(np.mean(fa))
    desc = f"LogReg L2 C={C} on standardised logits" + (f", family scale tabpfn x{pfn} tfm x{tfm}" if (pfn, tfm) != (1, 1) else "")
    print(f"{len(names)} members, {desc}: honest CV (mean-fold) {cv:.6f}  folds {np.round(fa, 6)}")

    mu, sd = X.mean(0), X.std(0)
    m = LogisticRegression(C=C, max_iter=3000).fit((X - mu) / sd * scale, y)
    top = sorted(zip(names, m.coef_[0] * scale), key=lambda t: -abs(t[1]))[:10]
    print("Largest weights:", ", ".join(f"{n}={w:.2f}" for n, w in top))
    if out:
        p = m.predict_proba((T - mu) / sd * scale)[:, 1]
        print("Wrote", write_submission(p, out))
        with open(CANDS, "a", newline="") as fh:
            csv.writer(fh).writerow([dt.datetime.now().isoformat(timespec="seconds"), out, f"{cv:.6f}", len(names),
                                     " ".join(names), desc + " (stack_final_v5.py)"])


if __name__ == "__main__":
    main()
