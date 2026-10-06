"""Train LightGBM / XGBoost / CatBoost with 5-fold CV on engineered features.

Usage: python train_models.py lgbm xgb cat
Saves preds/oof_<name>.npy and preds/test_<name>.npy for blend.py.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import N_FOLDS, SEED, build


def to_category(X, X_test, X_orig, cats):
    X, X_test, X_orig = X.copy(), X_test.copy(), X_orig.copy()
    for c in cats:
        X[c] = X[c].astype("category")
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)
        X_orig[c] = pd.Categorical(X_orig[c], categories=X[c].cat.categories)
    return X, X_test, X_orig


def fit_lgbm(Xtr, ytr, Xva, yva):
    m = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.03, num_leaves=127, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0,
        cat_smooth=20, n_estimators=10000, random_state=SEED, verbose=-1,
    )
    m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="auc",
          callbacks=[lgb.early_stopping(300, verbose=False)])
    return m, m.best_iteration_


def fit_xgb(Xtr, ytr, Xva, yva):
    m = xgb.XGBClassifier(
        n_estimators=10000, learning_rate=0.03, max_depth=8, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0, tree_method="hist",
        device="cuda", enable_categorical=True, max_cat_to_onehot=1,
        eval_metric="auc", early_stopping_rounds=300, random_state=SEED,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, m.best_iteration


def fit_cat(Xtr, ytr, Xva, yva, cats):
    m = CatBoostClassifier(
        iterations=10000, learning_rate=0.06, depth=8, l2_leaf_reg=3,
        eval_metric="AUC", task_type="GPU", random_seed=SEED,
        od_type="Iter", od_wait=300, verbose=False,
    )
    m.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=cats, use_best_model=True)
    return m, m.get_best_iteration()


def run(name, use_orig):
    X, X_test, y, ids, cats, X_orig, y_orig = build(cat_ratings=(name == "cat"))
    if name != "cat":
        X, X_test, X_orig = to_category(X, X_test, X_orig, cats)
    tag = name + ("_orig" if use_orig else "")

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, ytr = X.iloc[tr], y[tr]
        if use_orig:  # original data only ever goes into the training side
            Xtr = pd.concat([Xtr, X_orig], ignore_index=True)
            ytr = np.concatenate([ytr, y_orig])
        args = (Xtr, ytr, X.iloc[va], y[va])
        if name == "lgbm":
            m, it = fit_lgbm(*args)
        elif name == "xgb":
            m, it = fit_xgb(*args)
        else:
            m, it = fit_cat(*args, cats)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"[{tag}] fold {fold}: AUC {roc_auc_score(y[va], oof[va]):.5f} (iters {it})", flush=True)

    print(f"[{tag}] CV AUC: {roc_auc_score(y, oof):.5f}\n", flush=True)
    os.makedirs("preds", exist_ok=True)
    np.save(f"preds/oof_{tag}.npy", oof)
    np.save(f"preds/test_{tag}.npy", test_pred)


if __name__ == "__main__":
    use_orig = "--orig" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name in names or ["lgbm", "xgb", "cat"]:
        run(name, use_orig)
