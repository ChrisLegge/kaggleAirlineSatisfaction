from feat_gs2 import *
V3 = {
 "c1": [fd_groupstats, fd_log, fd_orig_te],
 "c2": [fd_groupstats, fd_log, fd_orig_te, feat_origmodel],
 "c3": [fd_groupstats, fd_log, fd_orig_te, feat_origmodel, feat_counts],
 "c4": [fd_groupstats, fd_log, fd_orig_te, feat_lookup_few],
}
if __name__ == "__main__":
    full = "--full" in sys.argv
    for w in [a for a in sys.argv[1:] if a != "--full"]:
        X, Xt = with_feats(*V3[w]); print(w, X.shape, flush=True)
        run_cv("feat_" + w, X, Xt, y, folds_to_run=range(5) if full else (0,), save=full, notes=w)
