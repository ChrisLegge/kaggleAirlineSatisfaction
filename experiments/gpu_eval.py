"""Quick eval: correlation with reference and rank-blend gains.
Usage: python experiments/gpu_eval.py base1,base2,... new1 new2 ...   (first base = correlation reference)
"""
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402

y = common.get_y()
bases = sys.argv[1].split(",")
news = sys.argv[2:]
names = bases + news
ref = bases[0]
O = {n: np.load(os.path.join(common.PREDS, f"oof_{n}.npy")) for n in names}
R = {n: rankdata(o) / len(o) for n, o in O.items()}
for n in names:
    print(f"{n:24s} auc {roc_auc_score(y, O[n]):.6f} pearson {np.corrcoef(O[n], O[ref])[0, 1]:.4f} "
          f"spearman {spearmanr(O[n], O[ref])[0]:.4f}")
base = sum(R[b] for b in bases) / len(bases)
print("equal rank blend of bases:", round(roc_auc_score(y, base), 6))
for n in news:
    best = max((roc_auc_score(y, (1 - w) * base + w * R[n]), w) for w in np.arange(0, 0.8, 0.05))
    print(f"  + {n}: best w {best[1]:.2f} -> {best[0]:.6f}")
Z = np.column_stack([np.log(R[n] / (1 - R[n] + 1e-9) + 1e-9) for n in names])
lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y)
print("LR weights", dict(zip(names, np.round(lr.coef_[0], 3))), "auc", round(roc_auc_score(y, Z @ lr.coef_[0]), 6))
