"""Stack gain of adding SUBSETS of candidates (base = full member list excl. gpu3_*). Usage: python gpu3_subset.py "a,b" "a,b,c" ..."""
import glob, os, sys
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import common
y, folds = common.get_y(), common.fold_ids()
names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(common.PREDS, "oof_*.npy")))
names = [n for n in names if not n.startswith(("naji_", "blend_", "gpu3_")) and n not in ("lgbm_orig", "feat_baseline")]
def col(n):
    o = logit(np.clip(np.load(os.path.join(common.PREDS, f"oof_{n}.npy")), 1e-6, 1 - 1e-6))
    return ((o - o.mean()) / o.std()).astype(np.float32)
B = np.column_stack([col(n) for n in names])
def st(M):
    return np.array([roc_auc_score(y[folds == f], LogisticRegression(C=1.0, max_iter=3000).fit(M[folds != f], y[folds != f]).decision_function(M[folds == f])) for f in range(5)])
b = st(B); print(f"base {len(names)}: {b.mean():.6f}", flush=True)
for s in sys.argv[1:]:
    a = st(np.column_stack([B] + [col(n)[:, None] for n in s.split(",")]))
    d = a - b
    print(f"+[{s}] {a.mean():.6f} gain {d.mean():+.6f} folds+ {(d > 0).sum()}/5", flush=True)
