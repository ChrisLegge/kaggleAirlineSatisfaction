"""Import a second batch of public OOF/test libraries into preds/ (prefixes below), checking row alignment.

Credits (Kaggle datasets):
  * kratosyan/s6e10-busyaprime-five-members-oof (bp_) - re-run of busyaprime's notebook
    "S6E10 | Route or distance? Both | LB 0.96169" (busyaprime / Akmal Xodarev; Apache 2.0). Five single models
    (lgbm, lgbm_xt, cat, xgb, realmlp), own folds (seed 33). Not blends.
  * sadamtorres/ps-s6e10-airline-satisfaction-artifacts (sadam_) - single-model experiments e1..e14 + base/tuned
    LGB/XGB/Cat. e14_rmlpseeds_realmlp is a plain (unweighted) average of 5 RealMLP seeds -> not label-fit, kept.
    Their label-fit blend (exp_blend.json weights) is NOT published as an array, so nothing to skip.
    orig_*_pred.npz are predictions of a model trained on the original dataset (feature, not an OOF member) -> skipped.
  * goodpjw2008/s6e10-tabpfn-member - TabPFN/TabICL members; these are the same members already imported as
    pub_tabpfn_* / pub_tabicl_* by import_public.py. Checked by correlation; only imported (as gpt_) if not a duplicate.

Submission-only datasets (no OOF, cannot be CV'd, not stacked): slimypunk03/ps-s6e10-dataset-2,
nina2025/ps-s6e10-02, nina2025/ps-s6e10-04, arunklenin/ps6e10-best-public-submissions,
slimypunk03/ps-s6e10-benchmark-models-starter.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import DATA, PREDS, get_y

SRC = os.path.join("experiments", "public_outputs")
y = get_y()
train_ids = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["id"])["id"].values
test_ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"].values
out = []


def save(name, oof, test):
    oof, test = np.asarray(oof, float), np.asarray(test, float)
    assert oof.shape == (len(y),) and test.shape == (len(test_ids),), name
    assert np.isfinite(oof).all() and np.isfinite(test).all(), name
    auc = roc_auc_score(y, oof)
    if auc < 0.9:  # misaligned rows would give ~0.5
        print(f"SKIP {name}: AUC {auc:.4f} looks misaligned")
        return
    np.save(os.path.join(PREDS, f"oof_{name}.npy"), oof)
    np.save(os.path.join(PREDS, f"test_{name}.npy"), test)
    out.append((name, auc))


# busyaprime five members: bp_<x>.npz with oof/test in train/test order
for p in sorted(glob.glob(os.path.join(SRC, "s6e10-busyaprime-five-members-oof", "bp_*.npz"))):
    z = np.load(p)
    save(os.path.basename(p)[:-4], z["oof"], z["test"])

# sadamtorres: preds_<x>.npz with oof/test in train/test order
for p in sorted(glob.glob(os.path.join(SRC, "ps-s6e10-airline-satisfaction-artifacts", "preds_*.npz"))):
    z = np.load(p)
    save("sadam_" + os.path.basename(p)[6:-4], z["oof"], z["test"])

# goodpjw2008 TabPFN member: id,pred csvs; import only if not a duplicate of the existing pub_ member
g = os.path.join(SRC, "s6e10-tabpfn-member")
for p in sorted(glob.glob(os.path.join(g, "oof_*.csv"))):
    k = os.path.basename(p)[4:-4]
    o = pd.read_csv(p).set_index("id").loc[train_ids, "pred"].values
    t = pd.read_csv(os.path.join(g, f"test_{k}.csv")).set_index("id").loc[test_ids, "pred"].values
    old = os.path.join(PREDS, f"oof_pub_{k}.npy")
    if os.path.exists(old):
        c = np.corrcoef(o, np.load(old))[0, 1]
        if c > 0.9999:
            print(f"duplicate gpt_{k} vs pub_{k}: corr {c:.6f} -> skipped")
            continue
        print(f"gpt_{k} differs from pub_{k}: corr {c:.6f}")
    save(f"gpt_{k}", o, t)

for n, a in sorted(out, key=lambda t: -t[1]):
    print(f"{n:32s} {a:.6f}")
print(len(out), "members imported")
