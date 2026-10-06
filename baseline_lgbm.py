"""5-fold stratified CV LightGBM baseline. Saves OOF + test preds for later ensembling."""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = "playground-series-s6e10"
SEED = 42
N_FOLDS = 5
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

y = train["satisfaction"].astype(int).values
X = train.drop(columns=["id", "satisfaction"])
X_test = test.drop(columns=["id"])
for c in CAT_COLS:
    X[c] = X[c].astype("category")
    X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)

params = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=5000,
    random_state=SEED,
    verbose=-1,
)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr, va) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr], y[tr],
        eval_X=X.iloc[va],
        eval_y=y[va],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"Fold {fold}: AUC {roc_auc_score(y[va], oof[va]):.5f}  (iters {model.best_iteration_})")

print(f"\nOverall CV AUC: {roc_auc_score(y, oof):.5f}")

imp = pd.Series(model.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("\nTop features (gain, last fold):\n", (imp / imp.sum()).round(4).head(10))

os.makedirs("preds", exist_ok=True)
os.makedirs("submissions", exist_ok=True)
np.save("preds/oof_lgbm_baseline.npy", oof)
np.save("preds/test_lgbm_baseline.npy", test_pred)
pd.DataFrame({"id": test["id"], "satisfaction": test_pred}).to_csv(
    "submissions/lgbm_baseline.csv", index=False
)
print("\nSaved submissions/lgbm_baseline.csv")
