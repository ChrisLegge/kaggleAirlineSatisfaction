"""Shared helpers so every experiment uses identical folds and saves comparable outputs.

Every model must:
  * use fold_ids() for CV (fixed 5-fold stratified split, seed 42)
  * call save_preds(name, oof, test_pred, notes) at the end
That makes every saved prediction blendable with every other one.
"""
import csv
import datetime as dt
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "playground-series-s6e10")
ORIG = os.path.join(ROOT, "originalData", "data.csv")
PREDS = os.path.join(ROOT, "preds")
SUBS = os.path.join(ROOT, "submissions")
RESULTS = os.path.join(ROOT, "results.csv")
TARGET = "satisfaction"
SEED = 42
N_FOLDS = 5


def load_raw():
    """train (with id + target), test (with id), original (no id, with target)."""
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    orig = pd.read_csv(ORIG)
    for df in (train, orig):
        df[TARGET] = df[TARGET].astype(int)
    return train, test, orig


def fold_ids():
    """Array of fold numbers (0..4) for each train row. Cached so it never changes."""
    path = os.path.join(PREDS, "folds.npy")
    if os.path.exists(path):
        return np.load(path)
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].astype(int).values
    folds = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[va] = f
    os.makedirs(PREDS, exist_ok=True)
    np.save(path, folds)
    return folds


def get_y():
    return pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].astype(int).values


def save_preds(name, oof, test_pred, notes=""):
    """Save OOF/test predictions and log the CV score to results.csv."""
    y = get_y()
    auc = roc_auc_score(y, oof)
    folds = fold_ids()
    fold_aucs = [roc_auc_score(y[folds == f], oof[folds == f]) for f in range(N_FOLDS)]
    os.makedirs(PREDS, exist_ok=True)
    np.save(os.path.join(PREDS, f"oof_{name}.npy"), oof)
    np.save(os.path.join(PREDS, f"test_{name}.npy"), test_pred)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "name", "cv_auc", "fold_std", "notes"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), name,
                    f"{auc:.6f}", f"{np.std(fold_aucs):.6f}", notes])
    print(f"[{name}] CV AUC {auc:.6f}  folds {np.round(fold_aucs, 5)}", flush=True)
    return auc


def write_submission(test_pred, filename):
    ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"].values
    os.makedirs(SUBS, exist_ok=True)
    path = os.path.join(SUBS, filename)
    pd.DataFrame({"id": ids, TARGET: test_pred}).to_csv(path, index=False)
    return path
