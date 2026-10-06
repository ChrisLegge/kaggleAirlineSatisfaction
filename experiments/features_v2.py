"""features_v2: base add_features + flight-distance group stats/counts + original-data lookups + original-model pred.
Usage (from any cwd):  sys.path.insert(0, '<root>/experiments'); from features_v2 import build_v2
    X, X_test, y, cats = build_v2()   # cats = list of categorical columns (already category dtype)
Row order matches train.csv / test.csv. Cached in preds/v2_cache.pkl.
All label-using parts use only the ORIGINAL dataset (disjoint from train), so no CV leakage.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_v2():
    import common
    cache = os.path.join(common.PREDS, "v2_cache.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f: return pickle.load(f)
    from feat_comb import with_feats, fd_groupstats, fd_log, fd_orig_te, feat_origmodel, feat_counts, y
    import features as F
    X, Xt = with_feats(fd_groupstats, fd_log, fd_orig_te, feat_origmodel, feat_counts)
    res = (X, Xt, y, list(F.CAT_COLS))
    with open(cache, "wb") as f: pickle.dump(res, f)
    return res

