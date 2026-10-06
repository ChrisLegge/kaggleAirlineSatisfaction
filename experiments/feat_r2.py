from feat_gs2 import *
from features_v2 import build_v2
X2, Xt2, _, _ = build_v2()
FDv = ALLK["Flight Distance"].values
def lab_lookup(g, tag, m=5):
    cd = codes(list(g)); co = cd[nt+nte:]; n = cd.max()+1
    so = np.bincount(co, weights=yo, minlength=n); no = np.bincount(co, minlength=n)
    te = (so + m*yo.mean())/(no+m)
    d = pd.DataFrame({tag+"_te": te[cd], tag+"_n": no[cd]})
    return split(d)
def r5():
    parts = [lab_lookup(g, "o_" + "_".join(c[:4] for c in g)) for g in
             [("Flight Distance","Online boarding"),("Flight Distance","Inflight wifi service"),("Flight Distance","Class"),("Flight Distance","Type of Travel")]]
    return (pd.concat([p[0] for p in parts],axis=1), pd.concat([p[1] for p in parts],axis=1))
def buckets():
    out = {}
    for b in (5, 10, 25, 100):
        cd = pd.factorize(FDv // b)[0]
        out[f"b{b}_n"] = pd.Series(cd).map(pd.Series(cd).value_counts()).values
        for c in ("Online boarding","Inflight wifi service","ratmean","Age"):
            out[f"b{b}_m_{c[:6]}"] = A[c].groupby(cd).transform("mean").values
    s = np.sort(FDv)
    for k in (3, 10, 30):
        out[f"win{k}"] = np.searchsorted(s, FDv+k, side="right") - np.searchsorted(s, FDv-k, side="left")
    return split(pd.DataFrame(out))
def other_keys():
    res = []
    for keys, tag in [(("Age",),"a"),(("Age","Gender","Customer Type"),"agc"),(("Departure Delay in Minutes","Arrival Delay in Minutes"),"dd"),(("Age","Class","Type of Travel"),"act")]:
        gs = gs2(keys, cols=["Online boarding","Inflight wifi service","ratmean","Seat comfort","Inflight entertainment","dep"], tag=tag)
        ll = lab_lookup(keys, tag+"_o")
        res.append((pd.concat([gs[0], ll[0]],axis=1), pd.concat([gs[1], ll[1]],axis=1)))
    return pd.concat([r[0] for r in res],axis=1), pd.concat([r[1] for r in res],axis=1)
def dev4():
    out = {}
    for keys, tag in [(("Flight Distance","Class"),"fc"),(("Flight Distance","Type of Travel"),"ft")]:
        cd = codes(list(keys))
        for c in ("Online boarding","Inflight wifi service","ratmean","Seat comfort","Inflight entertainment"):
            out[f"{tag}_d_{c[:6]}"] = (A[c] - A[c].groupby(cd).transform("mean")).values
        out[f"{tag}_n"] = pd.Series(cd).map(pd.Series(cd).value_counts()).values
        for c in ("Online boarding","Inflight wifi service","ratmean","Seat comfort","Inflight entertainment"):
            out[f"{tag}_m_{c[:6]}"] = A[c].groupby(cd).transform("mean").values
    return split(pd.DataFrame(out))
def opred_fd():
    s = pd.Series(np.r_[X2["orig_pred"].values, Xt2["orig_pred"].values])
    fd = pd.Series(FDv[:nt+nte])
    g = s.groupby(fd.values).transform("mean").values
    return split(pd.DataFrame({"op_fd_mean": g}).reindex(range(nt+nte)).pipe(lambda d: pd.concat([d, pd.DataFrame(index=range(nt+nte, nt+nte+1))])).iloc[:nt+nte]) if False else (pd.DataFrame({"op_fd_mean": g[:nt]}), pd.DataFrame({"op_fd_mean": g[nt:]}))
def plus(*fs):
    A_, B_ = [X2], [Xt2]
    for f in fs: a, b = f(); A_.append(a); B_.append(b)
    return pd.concat(A_, axis=1), pd.concat(B_, axis=1)
V5 = {"r5": [r5], "buckets": [buckets], "otherkeys": [other_keys], "dev4": [dev4], "opfd": [opred_fd]}
if __name__ == "__main__":
    full = "--full" in sys.argv
    for w in [a for a in sys.argv[1:] if a != "--full"]:
        fs = [f for k in w.split("+") for f in V5[k]]
        X, Xt = plus(*fs); print(w, X.shape, flush=True)
        run_cv("feat_r2_" + w.replace("+","_"), X, Xt, y, folds_to_run=range(5) if full else (0,), save=full, notes=w)
