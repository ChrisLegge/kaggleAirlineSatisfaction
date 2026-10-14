"""Stacker lab for v5: honest fold-wise CV (common.fold_ids), mean-fold AUC, same logic as stack_final.py.

Loads every eligible member's logits once (float32), then runs the named experiments.
  python experiments/stack_lab_v5.py base sweep dedupe family ...
Results appended to experiments/stack_lab_v5.csv.
"""
import csv
import datetime as dt
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PREDS, ROOT, fold_ids, get_y  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "stack_lab_v5.csv")
EXCL_PREFIX = ("naji_", "blend_", "nstack_")
EXCLUDE = {"lgbm_orig", "feat_baseline"}
y, folds = get_y(), fold_ids()


def L(m):
    return logit(np.clip(m, 1e-6, 1 - 1e-6)).astype(np.float32)


def eligible():
    names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(PREDS, "oof_*.npy")))
    return [n for n in names if n not in EXCLUDE and not n.startswith(EXCL_PREFIX)]


V4 = open(os.path.join(ROOT, "stack_members.txt")).read().split()
ALL = eligible()
IMP = {n for n, _ in json.load(open(os.path.join(ROOT, "experiments", "import3_log.json")))["imported"]}
NEW = [n for n in ALL if n in IMP]                       # new public single members (import_public_libs3.py)
OWN10 = [n for n in ALL if n not in V4 and n not in IMP]  # own members left out of v4
IDX = {n: i for i, n in enumerate(ALL)}
CACHE = os.environ.get("STACK_CACHE")  # local (non-OneDrive) cache of the logit matrix
if CACHE and os.path.exists(CACHE + ".npy") and open(CACHE + ".txt").read().split() == ALL:
    X_ALL = np.load(CACHE + ".npy")
else:
    X_ALL = np.empty((len(y), len(ALL)), np.float32)
    for i, n in enumerate(ALL):
        X_ALL[:, i] = L(np.load(os.path.join(PREDS, f"oof_{n}.npy")))
    if CACHE:
        np.save(CACHE + ".npy", X_ALL)
        open(CACHE + ".txt", "w").write("\n".join(ALL))
AUC1 = {n: roc_auc_score(y, X_ALL[:, IDX[n]]) for n in ALL}
print(f"{len(ALL)} eligible: {len(V4)} v4 + {len(NEW)} new public + {len(OWN10)} own not in v4 {OWN10}", flush=True)


def family(n):
    s = n.lower()
    if "tabpfn" in s or s.startswith("cd_pfn_") or "pfn" in s.split("_")[:2]:
        return "tabpfn"
    if s.startswith("cdtfm_") and not s.endswith(("realmlp", "xgboost")) or "tabicl" in s or "tabm" in s:
        return "tfm"
    if "realmlp" in s or "rmlp" in s or "mlp" in s or "_nn" in s or "deepfm" in s or "fwfm" in s:
        return "mlp"
    return "gbdt"


def cv(names, C=0.02, scale=None, extra=None, label="", log=True):
    """Honest fold-wise CV: standardise + LR on 4 folds, score on the 5th. scale: per-column multiplier
    applied after standardisation (family regularisation). extra: fn(Xtr_std, Xva_std, tr_mask)->(Ftr, Fva)."""
    t0 = time.time()
    cols = [IDX[n] for n in names]
    X = X_ALL[:, cols]
    pred = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        A, B = (X[tr] - mu) / sd, (X[va] - mu) / sd
        if scale is not None:
            A, B = A * scale, B * scale
        if extra is not None:
            A, B = extra(A, B, tr)
        pred[va] = LogisticRegression(C=C, max_iter=3000).fit(A, y[tr]).decision_function(B)
    fa = [roc_auc_score(y[folds == f], pred[folds == f]) for f in range(5)]
    m = float(np.mean(fa))
    print(f"{label:40s} n={len(names):3d} C={C:<6} CV {m:.6f} folds {np.round(fa, 6)} ({time.time() - t0:.0f}s)", flush=True)
    if log:
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "label", "n", "C", "cv", "folds", "members"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), label, len(names), C, f"{m:.6f}",
                        " ".join(f"{a:.6f}" for a in fa), " ".join(names)])
    return m, fa, pred


def dedupe(names, thr):
    """Greedy: walk members by descending OOF AUC, drop any whose |corr| with a kept member > thr."""
    order = sorted(names, key=lambda n: -AUC1[n])
    Z = X_ALL[:, [IDX[n] for n in order]]
    Z = (Z - Z.mean(0)) / Z.std(0)
    keep = []
    for j, n in enumerate(order):
        if keep:
            c = Z[:, keep].T @ Z[:, j] / len(y)
            if c.max() > thr:
                continue
        keep.append(j)
    return sorted(order[j] for j in keep)


if __name__ == "__main__":
    exps = sys.argv[1:]
    POOL = V4 + NEW + (OWN10 if "pool306" in exps else [])
    TAG = "306" if "pool306" in exps else "296"
    if "base" in exps:
        cv(V4, 0.02, label="v4 184 (repro)")
        cv(POOL, 0.02, label="v4+new")
    if "sweep" in exps:
        for C in (0.005, 0.01, 0.05, 0.1):
            cv(POOL, C, label="v4+new sweep")
    if "dedupe" in exps:
        for thr in (0.9995, 0.999):
            d = dedupe(POOL, thr)
            cv(d, 0.02, label=f"{TAG} dedupe{thr}")
    if "own10" in exps:
        cv(POOL + OWN10, 0.02, label="v4+new+own10")
    if "nopq" in exps:
        cv([n for n in POOL if n not in ("golem_p_lgbm_auxev", "golem_q_lgbm_auxev_10fold")], 0.02, label=f"{TAG} -golem_p/q")
    if "family" in exps:
        fam = np.array([family(n) for n in POOL])
        print({f: int((fam == f).sum()) for f in set(fam)})
        for tp, tf in ((3.0, 1.5), (2.0, 1.25), (1.5, 1.0)):
            sc = np.array([{"tabpfn": tp, "tfm": tf}.get(f, 1.0) for f in fam], np.float32)
            cv(POOL, 0.02, scale=sc, label=f"{TAG} family scale pfn{tp} tfm{tf}")
    if "fammean" in exps:  # label-free per-family mean of standardised logits appended (no nesting needed)
        fam = np.array([family(n) for n in POOL])
        groups = [np.where(fam == f)[0] for f in sorted(set(fam))]
        def ext(A, B, tr):
            return (np.column_stack([A] + [A[:, g].mean(1) for g in groups]),
                    np.column_stack([B] + [B[:, g].mean(1) for g in groups]))
        cv(POOL, 0.02, extra=ext, label=f"{TAG} + family means")
        def only(A, B, tr):
            return (np.column_stack([A[:, g].mean(1) for g in groups]), np.column_stack([B[:, g].mean(1) for g in groups]))
        cv(POOL, 0.02, extra=only, label=f"{TAG} family means only")
    if "twolevel" in exps:  # level 1 = per-family LR, its inputs for level 2 are inner-OOF (nested in the outer fold)
        fam = np.array([family(n) for n in POOL])
        groups = [np.where(fam == f)[0] for f in sorted(set(fam))]
        def ext2(A, B, tr):
            yt = y[tr]
            inner = folds[tr]  # the other 4 outer folds serve as inner folds
            FA, FB = np.zeros((len(A), len(groups))), np.zeros((len(B), len(groups)))
            for k, g in enumerate(groups):
                for f2 in np.unique(inner):
                    a2, b2 = inner != f2, inner == f2
                    FA[b2, k] = LogisticRegression(C=0.02, max_iter=3000).fit(A[a2][:, g], yt[a2]).decision_function(A[b2][:, g])
                FB[:, k] = LogisticRegression(C=0.02, max_iter=3000).fit(A[:, g], yt).decision_function(B[:, g])
            mu, sd = FA.mean(0), FA.std(0)
            return (FA - mu) / sd, (FB - mu) / sd
        cv(POOL, 1.0, extra=ext2, label=f"{TAG} two-level family LR (nested)")
    if "raw" in exps:  # public-notebook style: raw logits, no standardisation, tight tol
        for C in (0.002, 0.003, 0.005):
            t0 = time.time(); X = X_ALL[:, [IDX[n] for n in POOL]]; pred = np.zeros(len(y))
            for f in range(5):
                tr, va = folds != f, folds == f
                pred[va] = LogisticRegression(C=C, tol=1e-6, max_iter=20000).fit(X[tr], y[tr]).decision_function(X[va])
            fa = [roc_auc_score(y[folds == f], pred[folds == f]) for f in range(5)]
            print(f"{TAG} raw logits C={C}: CV {np.mean(fa):.6f} folds {np.round(fa, 6)} ({time.time()-t0:.0f}s)", flush=True)
