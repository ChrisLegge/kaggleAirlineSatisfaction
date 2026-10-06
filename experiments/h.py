import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import common
os.chdir(common.ROOT)
import features as F

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
              n_estimators=5000, random_state=42, verbose=-1, num_threads=4)

def to_cat(X, Xt, cats):
    X, Xt = X.copy(), Xt.copy()
    for c in cats:
        X[c] = X[c].astype("category")
        Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)
    return X, Xt

def fit(Xtr, ytr, Xva, yva, w=None):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m

def run_cv(name, X, Xt, y, folds_to_run=(0,), save=False, notes="", fold_hook=None):
    """fold_hook(tr_idx, va_idx, X, Xt) -> (Xtr, ytr_extra..) not used; keep simple."""
    folds = common.fold_ids()
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt)); aucs = []
    for f in folds_to_run:
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        if fold_hook:
            Xtr, ytr, Xva, Xte = fold_hook(tr, va)
        else:
            Xtr, ytr, Xva, Xte = X.iloc[tr], y[tr], X.iloc[va], Xt
        m = fit(Xtr, ytr, Xva, y[va])
        p = m.predict_proba(Xva)[:, 1]
        oof[va] = p; tp += m.predict_proba(Xte)[:, 1] / len(folds_to_run)
        aucs.append(roc_auc_score(y[va], p))
        print(f"[{name}] fold {f} auc {aucs[-1]:.6f} iters {m.best_iteration_}", flush=True)
    print(f"[{name}] mean {np.mean(aucs):.6f}", flush=True)
    if save and len(folds_to_run) == 5:
        common.save_preds(name, oof, tp, notes)
    return np.mean(aucs)

def base_data():
    train, test, y = F.load()
    X = F.add_features(train.drop(columns=["id"])); Xt = F.add_features(test.drop(columns=["id"]))
    X, Xt = to_cat(X, Xt, F.CAT_COLS)
    return X, Xt, y


