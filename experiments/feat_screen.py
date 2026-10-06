import itertools
from h import *

train, test, o = common.load_raw()
y = train[F.TARGET].astype(int).values
yo = o[F.TARGET].astype(int).values
raw_cols = [c for c in o.columns if c != F.TARGET]
Xb, Xtb, _ = base_data()
TOP = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
       "Inflight entertainment", "Seat comfort", "Leg room service"]

def lgb_raw(df):
    d = df[raw_cols].copy()
    for c in F.BASE_CATS: d[c] = pd.Categorical(d[c], categories=sorted(train[c].unique()))
    d["Arrival Delay in Minutes"] = d["Arrival Delay in Minutes"].astype(float)
    return d

def feat_origmodel():
    p = dict(PARAMS); p.update(n_estimators=400, learning_rate=0.03, num_leaves=31)
    m = lgb.LGBMClassifier(**p)
    # choose iterations by internal split
    from sklearn.model_selection import train_test_split
    Xo = lgb_raw(o)
    a, b, ya, yb = train_test_split(Xo, yo, test_size=0.2, random_state=1, stratify=yo)
    m.fit(a, ya, eval_set=[(b, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    print("orig-model holdout auc", roc_auc_score(yb, m.predict_proba(b)[:,1]), m.best_iteration_)
    p["n_estimators"] = m.best_iteration_
    m = lgb.LGBMClassifier(**p).fit(Xo, yo)
    # how good is it on competition train?
    pt = m.predict_proba(lgb_raw(train))[:,1]
    print("orig-model auc on comp train", roc_auc_score(y, pt))
    return pd.DataFrame({"orig_pred": pt}), pd.DataFrame({"orig_pred": m.predict_proba(lgb_raw(test))[:,1]})

def lookup(groups, name, m=10):
    prior = yo.mean()
    out_tr, out_te = {}, {}
    for g in groups:
        g = list(g)
        key = lambda d: pd.MultiIndex.from_frame(d[g]) if len(g) > 1 else pd.Index(d[g[0]])
        s = pd.DataFrame({"k": list(map(tuple, o[g].values)) if len(g) > 1 else o[g[0]].values, "y": yo})
        agg = s.groupby("k").y.agg(["sum", "count"])
        for d, out in ((train, out_tr), (test, out_te)):
            k = list(map(tuple, d[g].values)) if len(g) > 1 else d[g[0]].values
            a = agg.reindex(k)
            cnt = a["count"].fillna(0).values; sm = a["sum"].fillna(0).values
            nm = name + "|".join(c[:6] for c in g)
            out[nm + "_te"] = (sm + m * prior) / (cnt + m)
            out[nm + "_n"] = cnt
    return pd.DataFrame(out_tr), pd.DataFrame(out_te)

def feat_lookups():
    groups = [(c,) for c in TOP + ["Age", "Gate location"]] + list(itertools.combinations(TOP, 2))
    groups += [("Online boarding","Inflight wifi service","Type of Travel","Class"),
               ("Online boarding","Inflight wifi service","Type of Travel","Class","Customer Type"),
               ("Online boarding","Inflight wifi service","Type of Travel","Class","Inflight entertainment","Seat comfort")]
    return lookup(groups, "o_")

def feat_lookup_few():
    groups = [("Online boarding","Inflight wifi service","Type of Travel","Class"),
              ("Online boarding","Inflight wifi service","Type of Travel","Class","Customer Type"),
              ("Online boarding","Inflight wifi service","Type of Travel","Class","Inflight entertainment","Seat comfort")]
    return lookup(groups, "o_")

def feat_counts():
    allr = pd.concat([train[raw_cols], test[raw_cols], o[raw_cols]])
    a, b = {}, {}
    for c in ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        vc = allr[c].value_counts()
        a[c[:6] + "_cnt"] = train[c].map(vc).values; b[c[:6] + "_cnt"] = test[c].map(vc).values
    return pd.DataFrame(a), pd.DataFrame(b)

def with_feats(*fs):
    A = [Xb.reset_index(drop=True)]; B = [Xtb.reset_index(drop=True)]
    for f in fs:
        a, b = f(); A.append(a); B.append(b)
    return pd.concat(A, axis=1), pd.concat(B, axis=1)

def cat_ratings():
    a = pd.DataFrame({c + "_c": train[c].astype("category") for c in F.RATINGS + ["Age"]})
    b = pd.DataFrame({c + "_c": pd.Categorical(test[c], categories=a[c + "_c"].cat.categories) for c in F.RATINGS + ["Age"]})
    return a, b

if __name__ == "__main__":
    which = sys.argv[1:]
    full = "--full" in which
    which = [w for w in which if w != "--full"]
    folds = range(5) if full else (0,)
    for w in which:
        if w == "origmodel": X, Xt = with_feats(feat_origmodel)
        elif w == "lookups": X, Xt = with_feats(feat_lookups)
        elif w == "lookfew": X, Xt = with_feats(feat_lookup_few)
        elif w == "counts": X, Xt = with_feats(feat_counts)
        elif w == "catrat": X, Xt = with_feats(cat_ratings)
        elif w == "all": X, Xt = with_feats(feat_origmodel, feat_lookups, feat_counts)
        print(w, X.shape, flush=True)
        run_cv("feat_" + w, X, Xt, y, folds_to_run=folds, save=full, notes=w)
