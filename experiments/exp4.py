import numpy as np, pandas as pd
from stack_lab import *
y, folds = get_y(), fold_ids()
names, X, T = load_all(refresh=True)
print(len(names), flush=True)
for C in [0.02, 0.05]:
    print("all C", C, cv_lr(X, y, folds, C=C), flush=True)
tr_df = pd.read_csv(os.path.join(ROOT, "playground-series-s6e10", "train.csv"), usecols=["Type of Travel", "Class", "Customer Type"])
M = np.column_stack([(tr_df["Type of Travel"] == "Business travel"), tr_df["Class"] == "Business", tr_df["Class"] == "Eco Plus", tr_df["Customer Type"] == "Loyal Customer"]).astype(np.float32)
print("all+meta onehots C=0.02", cv_lr(np.hstack([X, M]), y, folds, C=0.02), flush=True)
# per-segment LR by Type of Travel x Class(business vs not)
seg = (M[:, 0] * 2 + M[:, 1]).astype(int)
pred = np.zeros(len(y))
for f in range(5):
    for s in np.unique(seg):
        tr, va = (folds != f) & (seg == s), (folds == f) & (seg == s)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        pred[va] = LogisticRegression(C=0.02, max_iter=3000).fit((X[tr]-mu)/sd, y[tr]).decision_function((X[va]-mu)/sd)
print("per-segment LR (travel x business)", mean_fold_auc(y, pred, folds), flush=True)
