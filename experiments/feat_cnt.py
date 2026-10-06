from feat_screen import *
NUM = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
def cnt(cols, srcs, suffix):
    a, b = {}, {}
    pools = {"tr": train, "te": test, "o": o}
    allr = pd.concat([pools[s][raw_cols] for s in srcs])
    for c in cols:
        vc = allr[c].value_counts()
        n = c[:6] + suffix
        a[n] = train[c].map(vc).fillna(0).values; b[n] = test[c].map(vc).fillna(0).values
    return pd.DataFrame(a), pd.DataFrame(b)
V = {
 "cnt_trte": lambda: cnt(NUM, ["tr","te"], "_ctt"),
 "cnt_orig": lambda: cnt(NUM, ["o"], "_co"),
 "cnt_FD": lambda: cnt(["Flight Distance"], ["tr","te","o"], "_c"),
 "cnt_delay": lambda: cnt(NUM[2:], ["tr","te","o"], "_c"),
 "cnt_Age": lambda: cnt(["Age"], ["tr","te","o"], "_c"),
}
for w in sys.argv[1:]:
    X, Xt = with_feats(V[w]); print(w, X.shape, flush=True)
    run_cv("feat_" + w, X, Xt, y, folds_to_run=(0,))
