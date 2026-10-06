"""Import public OOF/test member predictions (goodpjw2008 S6E10 series, Part 8 notebook output)
into preds/ as oof_pub_<name>.npy / test_pub_<name>.npy so ensemble.py can stack them with ours.

Credit: goodpjw2008, "S6E10 | TabPFN + Route Categories | LB 0.96160" (Kaggle notebook output).
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import DATA, PREDS, get_y

SRC = os.path.join("experiments", "public_outputs", "goodpjw_p8")

y = get_y()
train_ids = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["id"])["id"].values
test_ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"].values

oof = pd.read_csv(os.path.join(SRC, "oof_members.csv"), dtype="float32").assign(id=lambda d: d["id"].astype(int))
test = pd.read_csv(os.path.join(SRC, "test_members.csv"), dtype="float32").assign(id=lambda d: d["id"].astype(int))
oof = oof.set_index("id").loc[train_ids]
test = test.set_index("id").loc[test_ids]
assert oof.notna().all().all() and test.notna().all().all(), "missing ids / NaNs"

rows = []
for c in oof.columns:
    np.save(os.path.join(PREDS, f"oof_pub_{c}.npy"), oof[c].values.astype(float))
    np.save(os.path.join(PREDS, f"test_pub_{c}.npy"), test[c].values.astype(float))
    rows.append((c, roc_auc_score(y, oof[c].values)))
for c, a in sorted(rows, key=lambda t: -t[1]):
    print(f"pub_{c:24s} {a:.6f}")
