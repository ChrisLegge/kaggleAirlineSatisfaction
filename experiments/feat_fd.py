from feat_screen import *
import warnings; warnings.filterwarnings("ignore")
ALL = pd.concat([train[raw_cols], test[raw_cols], o[raw_cols]], ignore_index=True)
nt, nte = len(train), len(test)
def split(df):
    df = df.reset_index(drop=True)
    return df.iloc[:nt].reset_index(drop=True), df.iloc[nt:nt+nte].reset_index(drop=True)
def combo_cnt(groups, suffix="_cc", pool=ALL):
    out = {}
    for g in groups:
        k = pool[list(g)].astype(str).agg("|".join, axis=1) if len(g) > 1 else pool[g[0]]
        allk = ALL[list(g)].astype(str).agg("|".join, axis=1) if len(g) > 1 else ALL[g[0]]
        out["c_" + "_".join(c[:4] for c in g)] = allk.map(allk.value_counts()).values
    return split(pd.DataFrame(out))
def fd_groupstats():
    g = ALL.groupby("Flight Distance")
    out = pd.DataFrame({"fd_n": g["Age"].transform("size")})
    for c in F.RATINGS[:] + ["Age"]:
        out["fd_m_" + c[:8]] = g[c].transform("mean")
    out["fd_age_std"] = g["Age"].transform("std")
    return split(out)
def fd_orig_te():
    s = pd.DataFrame({"k": o["Flight Distance"], "y": yo}).groupby("k").y.agg(["sum","count"])
    pr = yo.mean(); res = []
    for d in (train, test):
        a = s.reindex(d["Flight Distance"].values)
        res.append(pd.DataFrame({"fd_o_te": ((a["sum"].fillna(0) + 5*pr)/(a["count"].fillna(0)+5)).values, "fd_o_n": a["count"].fillna(0).values}))
    return res[0], res[1]
def fd_log():
    a, b = combo_cnt([("Flight Distance",)]); return a, b
V = {
 "fd_groupstats": [fd_groupstats, fd_log],
 "fd_origte": [fd_orig_te, fd_log],
 "fd_combos": [lambda: combo_cnt([("Flight Distance",), ("Flight Distance","Age"), ("Flight Distance","Class"), ("Flight Distance","Type of Travel"), ("Flight Distance","Customer Type"), ("Flight Distance","Gender")])],
 "fd_ratingcombo": [lambda: combo_cnt([("Flight Distance",)]+[("Flight Distance", r) for r in ["Online boarding","Inflight wifi service","Seat comfort"]])],
}
if __name__ == "__main__":
    for w in sys.argv[1:]:
        X, Xt = with_feats(*V[w]); print(w, X.shape, flush=True)
        run_cv("feat_" + w, X, Xt, y, folds_to_run=(0,))
