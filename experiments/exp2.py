import time, numpy as np, json
from stack_lab import *
from fwdsel import forward, fast_auc
y, folds = get_y(), fold_ids()
names, X, T = load_all()
X = (X - X.mean(0)) / X.std(0)
single = [np.mean([fast_auc(y[folds==f], X[folds==f, j]) for f in range(5)]) for j in range(X.shape[1])]
dn, keep = dedup(names, X, 0.9995, single)
print("dedup 0.9995:", len(names), "->", len(dn), "dropped:", sorted(set(names)-set(dn)), flush=True)
pass
Xk = np.asfortranarray(X[:, keep]); del X
t=time.time()
sel, hist = forward(Xk, y, folds, max_k=60, patience=10, names=dn)
print("time", time.time()-t)
json.dump({"names": dn, "hist": [([dn[j] for j in s], sc) for s, sc in hist]}, open("fwd_nonnested.json", "w"), indent=0)
