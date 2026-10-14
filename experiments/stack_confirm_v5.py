"""Confirm family-wise scaling on meta-fold seeds other than the member folds (seed 7, 13)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stack_lab_v5 as S
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
POOL = S.V4 + S.NEW + S.OWN10
X = S.X_ALL[:, [S.IDX[n] for n in POOL]]
fam = np.array([S.family(n) for n in POOL])
for seed in (7, 13):
    fo = np.zeros(len(S.y), int)
    for k, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, S.y)):
        fo[b] = k
    res = {}
    for tp, tf in ((1, 1), (2.0, 1.25), (3.0, 1.5)):
        sc = np.array([{"tabpfn": tp, "tfm": tf}.get(f, 1.0) for f in fam], np.float32)
        pred = np.zeros(len(S.y))
        for f in range(5):
            tr, va = fo != f, fo == f
            mu, sd = X[tr].mean(0), X[tr].std(0)
            pred[va] = LogisticRegression(C=0.02, max_iter=3000).fit((X[tr] - mu) / sd * sc, S.y[tr]).decision_function((X[va] - mu) / sd * sc)
        fa = np.array([roc_auc_score(S.y[fo == f], pred[fo == f]) for f in range(5)])
        res[tp] = fa
        print(f"seed {seed} pfn{tp} tfm{tf}: {fa.mean():.6f}  vs flat {(fa - res[1]).mean()*1e6:+.1f}e-6, {(fa > res[1]).sum()}/5 up", flush=True)
