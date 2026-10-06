"""Stack-gain evaluator over the FULL member list.
Members = all preds/oof_*.npy except naji_*, blend_*, lgbm_orig, feat_baseline and the candidates themselves
(and anything starting with gpu3_ unless --keep_gpu3).
For each candidate: single AUC, max corr with a member, and honest 5-fold LogReg stack AUC (mean-fold, same as
stack.py: standardised logits, C=1) with vs without it. Also reports the gain of adding ALL candidates together.
Usage: python experiments/gpu3_eval.py cand1 cand2 ... [--keep_gpu3] [--C 1.0]
"""
import glob
import os
import sys

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402

args = sys.argv[1:]
keep = "--keep_gpu3" in args
args = [a for a in args if a != "--keep_gpu3"]
C = 1.0
if "--C" in args:
    i = args.index("--C"); C = float(args[i + 1]); del args[i:i + 2]
cands = args

y = common.get_y()
folds = common.fold_ids()
names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(common.PREDS, "oof_*.npy")))
names = [n for n in names if not n.startswith(("naji_", "blend_")) and n not in ("lgbm_orig", "feat_baseline")
         and n not in cands and (keep or not n.startswith("gpu3_"))]


def L(o):
    return logit(np.clip(o, 1e-6, 1 - 1e-6))


def z(a):
    return ((a - a.mean(0)) / a.std(0)).astype(np.float32)


B = z(np.column_stack([L(np.load(os.path.join(common.PREDS, f"oof_{n}.npy"))) for n in names]))


def stack_auc(M):
    aucs = []
    for f in range(common.N_FOLDS):
        tr, va = folds != f, folds == f
        lr = LogisticRegression(C=C, max_iter=3000).fit(M[tr], y[tr])
        aucs.append(roc_auc_score(y[va], lr.decision_function(M[va])))
    return np.mean(aucs), np.array(aucs)


base, base_f = stack_auc(B)
print(f"base stack of {len(names)} members: {base:.6f}", flush=True)
cols = []
for n in cands:
    o = L(np.load(os.path.join(common.PREDS, f"oof_{n}.npy")))
    single = np.mean([roc_auc_score(y[folds == f], o[folds == f]) for f in range(5)])
    c = z(o[:, None])
    cols.append(c)
    cors = B.T @ c[:, 0] / len(c)
    j = int(np.argmax(cors))
    a, af = stack_auc(np.column_stack([B, c]))
    d = af - base_f
    print(f"{n:32s} single {single:.6f} maxcorr {cors[j]:.4f} ({names[j]}) stack {a:.6f} gain {a - base:+.6f} "
          f"(folds +{(d > 0).sum()}/5, min {d.min():+.6f})", flush=True)
if len(cands) > 1:
    a, af = stack_auc(np.column_stack([B] + cols))
    print(f"ALL {len(cands)} candidates: stack {a:.6f} gain {a - base:+.6f}", flush=True)
