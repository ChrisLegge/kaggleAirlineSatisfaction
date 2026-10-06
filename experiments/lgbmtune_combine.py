"""Combine seed runs of the tuned v2 LGBM and check blend value of variants."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import common

P = common.PREDS
ld = lambda k, n: np.load(os.path.join(P, f"{k}_{n}.npy"))
y = common.get_y()

oof = (ld("oof", "lgbmtune_v2_t34") + 3 * ld("oof", "lgbmtune_v2_t34_seeds123")) / 4
tst = (ld("test", "lgbmtune_v2_t34") + 3 * ld("test", "lgbmtune_v2_t34_seeds123")) / 4
common.save_preds("lgbmtune_v2_t34_4seeds", oof, tst, "v2 feats, optuna trial 34, lr 0.02, seeds 42,1,2,3 averaged")

r = lambda a: rankdata(a) / len(a)
for other in ["lgbmtune_v2_t22", "lgbmtune_v2_shallow"]:
    if not os.path.exists(os.path.join(P, f"oof_{other}.npy")):
        continue
    o = ld("oof", other)
    print(other, "auc", round(roc_auc_score(y, o), 6), "corr", round(np.corrcoef(oof, o)[0, 1], 5))
    for w in [0.1, 0.2, 0.3, 0.5]:
        print(f"  w={w}: {roc_auc_score(y, (1 - w) * r(oof) + w * r(o)):.6f}")
