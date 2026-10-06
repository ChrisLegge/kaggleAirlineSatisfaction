from feat_comb import *
def cnt_all():
    a, b = {}, {}
    for c in F.RATINGS + F.BASE_CATS:
        vc = ALL[c].value_counts()
        a[c[:8] + "_cA"] = train[c].map(vc).values; b[c[:8] + "_cA"] = test[c].map(vc).values
    return pd.DataFrame(a), pd.DataFrame(b)
def cnt_orig_all():
    a, b = {}, {}
    for c in ["Age","Flight Distance","Departure Delay in Minutes","Arrival Delay in Minutes"]:
        vc = o[c].value_counts()
        a[c[:6] + "_cO"] = train[c].map(vc).fillna(0).values; b[c[:6] + "_cO"] = test[c].map(vc).fillna(0).values
    return pd.DataFrame(a), pd.DataFrame(b)
V4 = {"c5": [fd_groupstats, fd_log, fd_orig_te, feat_origmodel, feat_counts, cnt_all],
      "c6": [fd_groupstats, fd_log, fd_orig_te, feat_origmodel, feat_counts, cnt_orig_all]}
if __name__ == "__main__":
    full = "--full" in sys.argv
    for w in [a for a in sys.argv[1:] if a != "--full"]:
        X, Xt = with_feats(*V4[w]); print(w, X.shape, flush=True)
        run_cv("feat_" + w, X, Xt, y, folds_to_run=range(5) if full else (0,), save=full, notes=w)
