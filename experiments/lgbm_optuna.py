"""LightGBM tuning (Optuna proxy on fold 0) + full 5-fold CV with seed averaging.

Run from project root:
  python experiments/lgbm_optuna.py tune --trials 50 [--cat_ratings] [--study lgbm_v1]
  python experiments/lgbm_optuna.py top --study lgbm_v1 --k 5
  python experiments/lgbm_optuna.py cv --params '{"num_leaves":..}' --lr 0.02 --seeds 0 1 2 --name lgbmtune_x
  python experiments/lgbm_optuna.py cv --trial 12 --study lgbm_v1 ...   (params from study trial)
CPU only, num_threads=4.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # features.py uses relative paths

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

import common
from features import build

EXP = os.path.join(ROOT, "experiments")
DB = "sqlite:///" + os.path.join(EXP, "lgbm_optuna.db").replace("\\", "/")
THREADS = 8

DATASET_KEYS = {"max_bin", "min_data_per_group", "cat_smooth", "cat_l2", "max_cat_threshold"}


V2 = "--v2" in sys.argv
if V2:
    DB = "sqlite:///" + os.path.join(EXP, "lgbm_optuna_v2.db").replace("\\", "/")


def load_data(cat_ratings):
    if V2:
        sys.path.insert(0, EXP)
        from features_v2 import build_v2
        X, X_test, y, cats = build_v2()
        return X, X_test, y, cats
    X, X_test, y, ids, cats, _, _ = build(cat_ratings=cat_ratings)
    for c in cats:
        X[c] = X[c].astype("category")
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)
    return X, X_test, y, cats


def base_params(seed=42):
    return dict(objective="binary", metric="auc", verbosity=-1, num_threads=THREADS,
                device_type="cpu", seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                data_random_seed=seed, bagging_freq=1)


def fit(params, X, y, tr, va, cats, lr, es=200, max_rounds=20000):
    p = base_params()
    p.update(params)
    p["learning_rate"] = lr
    ds_params = {k: p[k] for k in p if k in DATASET_KEYS or k == "max_bin"}
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cats, params=ds_params, free_raw_data=True)
    dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr, categorical_feature=cats)
    cbs = [lgb.early_stopping(es, verbose=False)]
    m = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva], callbacks=cbs)
    return m


def suggest(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 512, log=True),
        max_depth=trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12, 16]),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 400, log=True),
        min_sum_hessian_in_leaf=trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 20, log=True),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        feature_fraction=trial.suggest_float("feature_fraction", 0.2, 0.9),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 20, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 50, log=True),
        cat_smooth=trial.suggest_float("cat_smooth", 1, 100, log=True),
        cat_l2=trial.suggest_float("cat_l2", 1, 50, log=True),
        max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511, 1023]),
        min_data_per_group=trial.suggest_int("min_data_per_group", 10, 500, log=True),
        path_smooth=trial.suggest_float("path_smooth", 1e-3, 50, log=True),
    )


def cmd_tune(a):
    X, _, y, cats = load_data(a.cat_ratings)
    folds = common.fold_ids()
    tr, va = np.where(folds != 0)[0], np.where(folds == 0)[0]
    study = optuna.create_study(study_name=a.study, storage=DB, direction="maximize",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=int(time.time()) % 10000,
                                                                   n_startup_trials=12, multivariate=True))
    if len(study.trials) == 0:
        # warm start from the current train_models.py config
        study.enqueue_trial(dict(num_leaves=127, max_depth=-1, min_child_samples=40,
                                 min_sum_hessian_in_leaf=1e-3, bagging_fraction=0.8,
                                 feature_fraction=0.5, lambda_l1=1e-3, lambda_l2=2.0,
                                 cat_smooth=20, cat_l2=10, max_bin=255, min_data_per_group=100,
                                 path_smooth=1e-3))
        if V2:  # best old-feature config (trial 45 of lgbm_v1)
            study.enqueue_trial(dict(num_leaves=145, max_depth=-1, min_child_samples=189,
                                     min_sum_hessian_in_leaf=0.020397, bagging_fraction=0.947101,
                                     feature_fraction=0.382358, lambda_l1=0.241589, lambda_l2=0.007548,
                                     cat_smooth=3.044936, cat_l2=41.931418, max_bin=1023,
                                     min_data_per_group=21, path_smooth=0.033256))

    def objective(trial):
        p = suggest(trial)
        t = time.time()
        m = fit(p, X, y, tr, va, cats, lr=a.lr, es=100)
        auc = m.best_score["valid_0"]["auc"]
        trial.set_user_attr("best_iter", m.best_iteration)
        print(f"trial {trial.number}: auc {auc:.6f} iters {m.best_iteration} "
              f"({time.time() - t:.0f}s) {p}", flush=True)
        return auc

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    study.optimize(objective, n_trials=max(0, a.trials - done), gc_after_trial=True)
    print("BEST", study.best_value, study.best_params, flush=True)


def cmd_top(a):
    study = optuna.load_study(study_name=a.study, storage=DB)
    df = study.trials_dataframe().sort_values("value", ascending=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.head(a.k).T)


def cmd_proxy(a):
    """Fold-0 proxy eval of several param overrides on top of a study trial."""
    base = dict(optuna.load_study(study_name=a.study, storage=DB).trials[a.trial].params)
    X, _, y, cats = load_data(a.cat_ratings)
    folds = common.fold_ids()
    tr, va = np.where(folds != 0)[0], np.where(folds == 0)[0]
    for ov in json.loads(a.params):
        p = dict(base)
        p.update(ov)
        t = time.time()
        m = fit(p, X, y, tr, va, cats, lr=a.lr, es=100)
        print(f"proxy {ov} cat_ratings={a.cat_ratings}: {m.best_score['valid_0']['auc']:.6f} "
              f"iters {m.best_iteration} ({time.time() - t:.0f}s)", flush=True)


def cmd_cv(a):
    if a.trial is not None:
        study = optuna.load_study(study_name=a.study, storage=DB)
        params = dict(study.trials[a.trial].params)
    else:
        params = {}
    if a.params:
        params.update(json.loads(a.params))
    X, X_test, y, cats = load_data(a.cat_ratings)
    folds = common.fold_ids()
    oof_all = np.zeros(len(X))
    test_all = np.zeros(len(X_test))
    for s_i, seed in enumerate(a.seeds):
        oof = np.zeros(len(X))
        tp = np.zeros(len(X_test))
        for f in range(common.N_FOLDS):
            t = time.time()
            tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
            p = dict(params)
            p.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
            m = fit(p, X, y, tr, va, cats, lr=a.lr, es=a.es)
            oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
            tp += m.predict(X_test, num_iteration=m.best_iteration) / common.N_FOLDS
            print(f"[{a.name}] seed {seed} fold {f}: {roc_auc_score(y[va], oof[va]):.6f} "
                  f"iters {m.best_iteration} ({time.time() - t:.0f}s)", flush=True)
        print(f"[{a.name}] seed {seed} CV {roc_auc_score(y, oof):.6f}", flush=True)
        oof_all += oof
        test_all += tp
        n = s_i + 1
        print(f"[{a.name}] avg of {n} seeds CV {roc_auc_score(y, oof_all / n):.6f}", flush=True)
        if n == 1 and len(a.seeds) > 1 and a.save_first:
            common.save_preds(a.name + "_s1", oof, tp, f"{a.notes} seed {seed} lr {a.lr} {json.dumps(params)}")
    n = len(a.seeds)
    common.save_preds(a.name, oof_all / n, test_all / n,
                      f"{a.notes} {n} seeds lr {a.lr} cat_ratings={a.cat_ratings} {json.dumps(params)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--study", default="lgbm_v1")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--cat_ratings", action="store_true")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--es", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--name", default="lgbmtune_tmp")
    ap.add_argument("--notes", default="")
    ap.add_argument("--save_first", action="store_true")
    a = ap.parse_args()
    {"tune": cmd_tune, "top": cmd_top, "cv": cmd_cv, "proxy": cmd_proxy}[a.cmd](a)
