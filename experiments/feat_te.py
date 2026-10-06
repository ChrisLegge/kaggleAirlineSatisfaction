from feat_fd import *
ALLK = pd.concat([train[raw_cols], test[raw_cols], o[raw_cols]], ignore_index=True)
def codes(g):
    if len(g) == 1: k = ALLK[g[0]].values
    else:
        k = ALLK[g[0]].astype("int64").values
        for c in g[1:]:
            k = k * 1000003 + pd.factorize(ALLK[c])[0]
    return pd.factorize(k)[0]
def build_te_hook(Xs, Xts, groups, m=5, inner=5, use_orig=True, seed=7):
    C = {tuple(g): codes(g) for g in groups}
    prior_o = yo.mean()
    def hook(tr, va):
        rng = np.random.RandomState(seed)
        inner_f = rng.randint(0, inner, len(tr))
        Xtr, Xva, Xte = Xs.iloc[tr].copy(), Xs.iloc[va].copy(), Xts.copy()
        for g in groups:
            cd = C[tuple(g)]; n = cd.max() + 1
            ctr = cd[:nt][tr]; cva = cd[:nt][va]; cte = cd[nt:nt+nte]; co = cd[nt+nte:]
            ytr = y[tr]
            so = np.bincount(co, weights=yo, minlength=n) if use_orig else np.zeros(n)
            no = np.bincount(co, minlength=n) if use_orig else np.zeros(n)
            S = np.bincount(ctr, weights=ytr, minlength=n); N = np.bincount(ctr, minlength=n)
            pr = ytr.mean()
            nm = "te_" + "_".join(c[:4] for c in g)
            def enc(s, nn):
                return (s + m * pr) / (nn + m)
            # train rows: inner OOF
            te = np.zeros(len(tr)); cn = np.zeros(len(tr))
            for k in range(inner):
                msk = inner_f == k
                Sk = np.bincount(ctr[msk], weights=ytr[msk], minlength=n); Nk = np.bincount(ctr[msk], minlength=n)
                te[msk] = enc((S - Sk + so)[ctr[msk]], (N - Nk + no)[ctr[msk]])
                cn[msk] = (N - Nk + no)[ctr[msk]]
            Xtr[nm] = te; Xtr[nm + "_n"] = cn
            for d, c in ((Xva, cva), (Xte, cte)):
                d[nm] = enc((S + so)[c], (N + no)[c]); d[nm + "_n"] = (N + no)[c]
        return Xtr, y[tr], Xva, Xte
    return hook
GR = {
 "fd": [("Flight Distance",)],
 "fd3": [("Flight Distance",), ("Flight Distance","Class"), ("Flight Distance","Type of Travel"), ("Flight Distance","Customer Type")],
}
if __name__ == "__main__":
    full = "--full" in sys.argv
    for w in [a for a in sys.argv[1:] if a != "--full"]:
        stat, gr = w.split(":")
        fs = {"gs": [fd_groupstats, fd_log], "none": []}[stat]
        X, Xt = with_feats(*fs) if fs else (Xb.reset_index(drop=True), Xtb.reset_index(drop=True))
        hook = build_te_hook(X, Xt, GR[gr])
        run_cv("feat_te_" + w.replace(":", "_"), X, Xt, y, folds_to_run=range(5) if full else (0,), save=full, notes="fold-safe TE " + w, fold_hook=hook)
