import time, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from stack_lab import *
import lightgbm as lgb
y, folds = get_y(), fold_ids()
names, X, T = load_all()
# ridge on standardised logits
for a in [1.0, 1000.0]:
    pred = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        pred[va] = Ridge(alpha=a).fit((X[tr]-mu)/sd, y[tr]).predict((X[va]-mu)/sd)
    print("ridge alpha", a, mean_fold_auc(y, pred, folds), flush=True)
# LR on ranks (rank within train, uniform 0..1)
R = np.column_stack([rankdata(X[:, j]) / len(y) for j in range(X.shape[1])]).astype(np.float32)
print("LR ranks C=0.05", cv_lr(R, y, folds, C=0.05), flush=True)
del R
# LGB stacker: top-40 members by single AUC + meta
tr_df = pd.read_csv(os.path.join(ROOT, "playground-series-s6e10", "train.csv"), usecols=["Type of Travel", "Class", "Flight Distance"])
te_df = pd.read_csv(os.path.join(ROOT, "playground-series-s6e10", "test.csv"), usecols=["Type of Travel", "Class", "Flight Distance"])
fd = pd.concat([tr_df["Flight Distance"], te_df["Flight Distance"]]).value_counts()
meta = np.column_stack([(tr_df["Type of Travel"] == "Business travel").astype(float), tr_df["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2}).astype(float), tr_df["Flight Distance"].map(fd).astype(float)])
single = np.array([mean_fold_auc(y, X[:, j], folds) for j in range(X.shape[1])])
top = np.argsort(-single)[:40]
F = np.hstack([X[:, top], meta]).astype(np.float32)
lin = cv_lr(X[:, top], y, folds, C=0.05)
print("LR on same top40", lin, flush=True)
pred = np.zeros(len(y))
for f in range(5):
    tr, va = folds != f, folds == f
    m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, num_threads=3, verbose=-1),
                  lgb.Dataset(F[tr], y[tr]), num_boost_round=600)
    pred[va] = m.predict(F[va], raw_score=True)
    print(" fold", f, roc_auc_score(y[va], pred[va]), flush=True)
print("LGB stacker top40+meta", mean_fold_auc(y, pred, folds), flush=True)
