import time, numpy as np
from stack_lab import *
y, folds = get_y(), fold_ids()
names, X, T = load_all(refresh=True)
print(len(names), X.shape, X.dtype)
old = [i for i,n in enumerate(names) if not n.startswith(("bp_","sadam_"))]
print("old set", len(old))
t=time.time(); print("old C=1 std", cv_lr(X[:,old], y, folds), time.time()-t, flush=True)
t=time.time(); print("all C=1 std", cv_lr(X, y, folds), time.time()-t, flush=True)
for C in [0.005, 0.02, 0.05, 0.2]:
    print("all C", C, "std", cv_lr(X, y, folds, C=C), flush=True)
for C in [0.05, 1.0]:
    print("all C", C, "raw", cv_lr(X, y, folds, C=C, standardise=False), flush=True)
