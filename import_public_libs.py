"""Import further public OOF/test libraries into preds/ (prefixes below), checking row alignment.

Credits (Kaggle datasets): dariushafshar/s6e10-golem-oof-library (golem_), megayak/s6e10-oof-library (mega_),
sachith7/s6e10-stack-oof-predictions (sach_), arhancanli12/s6e10-oof-test-predictions (arhan_),
najiama/s6e10-oof (naji_).
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


# golem: oof_<x>.npy / test_<x>.npy in train/test row order
g = os.path.join(SRC, "s6e10-golem-oof-library")
for p in sorted(glob.glob(os.path.join(g, "oof_*.npy"))):
    k = os.path.basename(p)[4:-4]
    save(f"golem_{k}", np.load(p), np.load(os.path.join(g, f"test_{k}.npy")))

# sachith: <member>.npz with oof/test arrays
for p in sorted(glob.glob(os.path.join(SRC, "s6e10-stack-oof-predictions", "*.npz"))):
    z = np.load(p)
    save("sach_" + os.path.basename(p)[:-4], z["oof"], z["test"])

# parquet libraries with id columns
for d, pre in (("s6e10-oof-library", "mega_"), ("s6e10-oof-test-predictions", "arhan_")):
    o = pd.read_parquet(os.path.join(SRC, d, "oof_predictions.parquet")).set_index("id").loc[train_ids]
    t = pd.read_parquet(os.path.join(SRC, d, "test_predictions.parquet"))
    t = t.set_index("id").loc[test_ids] if "id" in t.columns else t
    for c in o.columns:
        if c in ("fold", "satisfaction") or c not in t.columns:
            continue
        save(pre + c, o[c].values, t[c].values)

# najiama blends: NN_blend_oof.csv / NN_submission.csv
for p in sorted(glob.glob(os.path.join(SRC, "s6e10-oof", "*_blend_oof.csv"))):
    k = os.path.basename(p).split("_")[0]
    o = pd.read_csv(p).assign(id=lambda d: d["id"].astype(int)).set_index("id").loc[train_ids]
    t = pd.read_csv(os.path.join(SRC, "s6e10-oof", f"{k}_submission.csv")).set_index("id").loc[test_ids]
    save(f"naji_{k}", o["satisfaction"].values, t["satisfaction"].values)

for n, a in sorted(out, key=lambda t: -t[1]):
    print(f"{n:28s} {a:.6f}")
print(len(out), "members imported")
