import time, numpy as np
from stack_lab import *
y, folds = get_y(), fold_ids()
names, X, T = load_all()
print(len(names), flush=True)
for C in [0.002, 0.01]:
    t = time.time(); pred = np.zeros(len(y)); nz = []
    for f in range(5):
        tr, va = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=200, tol=1e-4).fit((X[tr]-mu)/sd, y[tr])
        pred[va] = m.decision_function((X[va]-mu)/sd); nz.append(int((m.coef_ != 0).sum()))
    print("L1 C", C, mean_fold_auc(y, pred, folds), "nonzero", nz, f"{time.time()-t:.0f}s", flush=True)
