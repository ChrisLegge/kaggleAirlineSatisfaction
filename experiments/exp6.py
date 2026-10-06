import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from stack_lab import *
y, folds = get_y(), fold_ids()
names = open(os.path.join(ROOT, "stack_members.txt")).read().split()
allnames, X, T = load_all()
idx = [allnames.index(n) for n in names]
X = X[:, idx]
old = [i for i, n in enumerate(names) if not n.startswith(("bp_", "sadam_", "gpu3_"))]
# 1) robustness: different fold seed
f2 = np.zeros(len(y), int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2024).split(X, y)): f2[va] = k
print("seed2024 folds: old-set C=1", cv_lr(X[:, old], y, f2, C=1.0), " new-all C=0.02", cv_lr(X, y, f2, C=0.02), flush=True)
# 2) LGB residual booster on top of honest LR OOF logit
s, oofp = cv_lr(X, y, folds, C=0.02, return_pred=True)
print("LR", s, flush=True)
tr_df = pd.read_csv(os.path.join(ROOT, "playground-series-s6e10", "train.csv"))
te_df = pd.read_csv(os.path.join(ROOT, "playground-series-s6e10", "test.csv"))
fd = pd.concat([tr_df["Flight Distance"], te_df["Flight Distance"]]).value_counts()
meta = pd.DataFrame({"tt": (tr_df["Type of Travel"] == "Business travel").astype(int), "cls": tr_df["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2}),
                     "fdc": tr_df["Flight Distance"].map(fd), "fd": tr_df["Flight Distance"], "age": tr_df["Age"], "ct": (tr_df["Customer Type"] == "Loyal Customer").astype(int), "lr": oofp})
for rounds in [100, 300]:
    pred = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        m = lgb.train(dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=2000, lambda_l2=10, num_threads=4, verbose=-1),
                      lgb.Dataset(meta[tr], y[tr], init_score=oofp[tr]), num_boost_round=rounds)
        pred[va] = oofp[va] + m.predict(meta[va], raw_score=True)
    print("LR + LGB residual", rounds, mean_fold_auc(y, pred, folds), flush=True)
