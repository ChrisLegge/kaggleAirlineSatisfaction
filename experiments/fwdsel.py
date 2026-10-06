"""Fast forward member selection for the LogReg stacker.

Candidate scoring: for each CV fold, keep the current fitted linear predictor eta as an offset and fit only
(intercept shift, coef of candidate) by 2-D Newton; score the candidate by mean validation-fold AUC.
The best candidate is added and all coefficients are refit by full Newton (warm start). The recorded score
for each step is the true honest CV of the selected set (full refit on train folds, scored on val fold).
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def newton_lr(X, y, w=None, iters=8, l2=1e-3):
    """Logistic regression with intercept (last coef), tiny L2 on non-intercept coefs."""
    n, k = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(k + 1) if w is None else w.copy()
    reg = np.full(k + 1, l2); reg[-1] = 0
    for _ in range(iters):
        p = sigmoid(Xa @ w)
        g = Xa.T @ (p - y) + reg * w
        H = (Xa * (p * (1 - p))[:, None]).T @ Xa + np.diag(reg)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def offset_fit(x, eta, y, iters=5):
    """Fit c*eta + a + b*x (3-param Newton) so existing weights can shrink as the new member enters."""
    Z = np.column_stack([eta, np.ones_like(x), x])
    w = np.array([1.0, 0.0, 0.0])
    for _ in range(iters):
        p = sigmoid(Z @ w)
        g = Z.T @ (p - y)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-6 * np.eye(3)
        w -= np.linalg.solve(H, g)
    return w


def fast_auc(y, s):
    o = np.argsort(s, kind="stable")
    ys = y[o]
    ranks = np.arange(1, len(s) + 1)
    npos = ys.sum()
    return (ranks[ys == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(s) - npos))


def forward(X, y, folds, max_k=40, start=(), patience=6, cand_sub=None, verbose=True, names=None, sub_n=120000, seed=0):
    """X: standardised logits (n, m). folds: fold label per row. Returns (selected idx list, history of CV)."""
    fl = np.unique(folds)
    tr_idx = [np.where(folds != f)[0] for f in fl]
    va_idx = [np.where(folds == f)[0] for f in fl]
    rng = np.random.default_rng(seed)
    sub_idx = [np.sort(rng.choice(len(t), min(sub_n, len(t)), replace=False)) for t in tr_idx]
    sel = list(start)
    ws = [None] * len(fl)
    hist = []
    best, best_k, since = -1, 0, 0

    def refit():
        etas = []
        for i in range(len(fl)):
            if not sel:
                p0 = y[tr_idx[i]].mean()
                ws[i] = None
                etas.append((np.full(len(tr_idx[i]), np.log(p0 / (1 - p0))), np.full(len(va_idx[i]), np.log(p0 / (1 - p0)))))
                continue
            w0 = None
            if ws[i] is not None and len(ws[i]) == len(sel):  # one new coef: insert 0 before intercept
                w0 = np.concatenate([ws[i][:-1], [0.0], ws[i][-1:]])
            ws[i] = newton_lr(X[np.ix_(tr_idx[i], sel)].astype(np.float64), y[tr_idx[i]], w0)
            etas.append((X[np.ix_(tr_idx[i], sel)] @ ws[i][:-1] + ws[i][-1], X[np.ix_(va_idx[i], sel)] @ ws[i][:-1] + ws[i][-1]))
        return etas

    etas = refit()
    if sel:
        sc = np.mean([fast_auc(y[va_idx[i]], etas[i][1]) for i in range(len(fl))])
        hist.append((list(sel), sc)); best, best_k = sc, len(sel)
    cands = list(range(X.shape[1])) if cand_sub is None else list(cand_sub)
    while len(sel) < max_k:
        scores = {}
        for j in cands:
            if j in sel:
                continue
            s = 0.0
            for i in range(len(fl)):
                si = tr_idx[i][sub_idx[i]]
                xt, xv = X[si, j].astype(np.float64), X[va_idx[i], j].astype(np.float64)
                c, a, b = offset_fit(xt, etas[i][0][sub_idx[i]], y[si])
                s += fast_auc(y[va_idx[i]], c * etas[i][1] + a + b * xv)
            scores[j] = s / len(fl)
        j = max(scores, key=scores.get)
        sel.append(j)
        ws = [None if w is None else w for w in ws]
        etas = refit()
        sc = np.mean([fast_auc(y[va_idx[i]], etas[i][1]) for i in range(len(fl))])
        hist.append((list(sel), sc))
        if verbose:
            print(f"  k={len(sel):3d} +{names[j] if names else j:32s} proxy {scores[j]:.6f} cv {sc:.6f}", flush=True)
        if sc > best + 1e-7:
            best, best_k, since = sc, len(sel), 0
        else:
            since += 1
            if since >= patience:
                break
    return sel[:best_k], hist
