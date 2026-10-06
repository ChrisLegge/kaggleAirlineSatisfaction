from feat_te import *
A = ALLK.copy()
A["dep"] = A["Departure Delay in Minutes"]; A["arr"] = A["Arrival Delay in Minutes"].fillna(A["dep"])
for c, v in [("Class","Business"),("Class","Eco"),("Type of Travel","Business travel"),("Customer Type","Loyal Customer"),("Gender","Male")]:
    A["f_"+c[:5]+v[:3]] = (A[c]==v).astype(float)
A["ratmean"] = A[F.RATINGS].replace(0, np.nan).mean(axis=1)
A["nzero"] = (A[F.RATINGS]==0).sum(axis=1)
def gs2(keycols=("Flight Distance",), cols=None, tag="g", dev=False):
    cd = codes(list(keycols)); out = {}
    cols = cols or (F.RATINGS + ["Age","dep","arr","ratmean","nzero"] + [c for c in A.columns if c.startswith("f_")])
    G = A[cols].groupby(cd)
    mean = G.transform("mean"); std = G.transform("std")
    cnt = pd.Series(cd).map(pd.Series(cd).value_counts()).values
    out[tag+"_n"] = cnt
    for c in cols:
        out[tag+"_m_"+c[:9]] = mean[c].values
        if c in ("Online boarding","Inflight wifi service","ratmean","Age","dep","Seat comfort"):
            out[tag+"_s_"+c[:9]] = std[c].values
        if dev and c in ("Online boarding","Inflight wifi service","ratmean","Seat comfort","Inflight entertainment","Age"):
            out[tag+"_d_"+c[:9]] = (A[c] - mean[c]).values
    return split(pd.DataFrame(out))
V2 = {
 "gs2": [lambda: gs2()],
 "gs2dev": [lambda: gs2(dev=True)],
 "gs2_cls": [lambda: gs2(), lambda: gs2(("Flight Distance","Class"), cols=["Online boarding","Inflight wifi service","ratmean","Age","dep"], tag="gc")],
 "gs2_tt": [lambda: gs2(), lambda: gs2(("Flight Distance","Type of Travel"), cols=["Online boarding","Inflight wifi service","ratmean","Age","dep"], tag="gt")],
}
if __name__ == "__main__":
    full = "--full" in sys.argv
    for w in [a for a in sys.argv[1:] if a != "--full"]:
        X, Xt = with_feats(*V2[w]); print(w, X.shape, flush=True)
        run_cv("feat_" + w, X, Xt, y, folds_to_run=range(5) if full else (0,), save=full, notes=w)
