"""Stack gain vs CURRENT production member set (stack_members.txt), matching stack_final.py:
LogReg C=0.02 on logits standardised with fold-train stats, mean-fold AUC on common.fold_ids().
Each arg is a comma-joined set of candidates (single name = single member). Candidates already in the
member list are removed from the base for that comparison only if --loo given (otherwise base is the full list).
Usage: python experiments/gpu4_eval.py [--C 0.02] a b "a,b" ...
"""
import os, sys
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import common
a = sys.argv[1:]
C = 0.02
if "--C" in a:
    i = a.index("--C"); C = float(a[i + 1]); del a[i:i + 2]
y, folds = common.get_y(), common.fold_ids()
names = open(os.path.join(ROOT, "stack_members.txt")).read().split()
def col(n):
    return logit(np.clip(np.load(os.path.join(common.PREDS, f"oof_{n}.npy")), 1e-6, 1 - 1e-6)).astype(np.float32)
B = np.column_stack([col(n) for n in names])
def st(M):
    r = []
    for f in range(5):
        tr, va = folds != f, folds == f
        mu, sd = M[tr].mean(0), M[tr].std(0)
        r.append(roc_auc_score(y[va], LogisticRegression(C=C, max_iter=3000).fit((M[tr] - mu) / sd, y[tr]).decision_function((M[va] - mu) / sd)))
    return np.array(r)
b = st(B); print(f"base {len(names)} members C={C}: {b.mean():.6f}", flush=True)
for s in a:
    cs = [n for n in s.split(",") if n not in names]
    o = col(cs[0])
    single = np.mean([roc_auc_score(y[folds == f], o[folds == f]) for f in range(5)])
    r = st(np.column_stack([B] + [col(n)[:, None] for n in cs]))
    d = r - b
    print(f"+[{s}] single(first) {single:.6f} stack {r.mean():.6f} gain {d.mean():+.6f} folds+ {(d > 0).sum()}/5 min {d.min():+.6f}", flush=True)
