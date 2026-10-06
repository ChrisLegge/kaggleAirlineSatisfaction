"""PyTorch MLP with embeddings for categoricals + ratings (+ binned age/distance/delays).

Usage: python experiments/nn_mlp.py --name gpu_nn_v1 --folds 0 --seeds 0 [--epochs 12 ...]
Only full 5-fold runs (all folds) are saved via common.save_preds.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402

torch.set_num_threads(4)
DEV = "cuda"

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]


def prepare(n_dist_bins=64, pairs=False, v2=False, fd_emb=False):
    tr = pd.read_csv(os.path.join(common.DATA, "train.csv"))
    te = pd.read_csv(os.path.join(common.DATA, "test.csv"))
    y = tr[common.TARGET].astype(int).values
    n_tr = len(tr)
    df = pd.concat([tr.drop(columns=[common.TARGET]), te], ignore_index=True)
    del tr, te
    arr_na = df["Arrival Delay in Minutes"].isna().astype(np.float32)
    dep = df["Departure Delay in Minutes"].astype(float)
    arr = df["Arrival Delay in Minutes"].fillna(dep).astype(float)

    cat_cols = []
    # base categoricals + crossed cats
    cat_df = pd.DataFrame(index=df.index)
    for c in CATS:
        cat_df[c] = df[c]
    cat_df["cls_trav_cust"] = df["Class"] + "_" + df["Type of Travel"] + "_" + df["Customer Type"]
    for c in RATINGS:
        cat_df[c] = df[c]
    cat_df["age"] = df["Age"]
    cat_df["dist_bin"] = pd.qcut(df["Flight Distance"].rank(method="first"), n_dist_bins, labels=False)
    cat_df["dep_bin"] = np.minimum(dep, 60).round().astype(int) // 5 * 5 + (dep > 0)
    cat_df["arr_bin"] = np.minimum(arr, 60).round().astype(int) // 5 * 5 + (arr > 0)
    cat_df["n_zero"] = (df[RATINGS] == 0).sum(axis=1)
    if fd_emb:
        vc = df["Flight Distance"].iloc[:n_tr].value_counts()
        rare = ~df["Flight Distance"].isin(vc[vc >= 20].index)
        cat_df["fd_raw"] = df["Flight Distance"].where(~rare, -1)
    if pairs:
        key = RATINGS + ["Type of Travel", "Class", "Customer Type"]
        kc = {c: pd.factorize(df[c], sort=True)[0] for c in key}
        for i in range(len(key)):
            for j in range(i + 1, len(key)):
                cat_df[f"p_{i}_{j}"] = kc[key[i]] * 10 + kc[key[j]]
    cards = []
    codes = np.zeros((len(df), cat_df.shape[1]), dtype=np.int64)
    for j, c in enumerate(cat_df.columns):
        k, uniq = pd.factorize(cat_df[c], sort=True)
        codes[:, j] = k
        cards.append(len(uniq))
        cat_cols.append(c)

    r = df[RATINGS].astype(np.float32)
    nz = r.replace(0, np.nan)
    num = pd.DataFrame({
        "age": df["Age"].astype(np.float32),
        "dist": np.log1p(df["Flight Distance"]).astype(np.float32),
        "dep": np.log1p(dep), "arr": np.log1p(arr),
        "gain": np.sign(dep - arr) * np.log1p(np.abs(dep - arr)),
        "arr_na": arr_na,
        "r_mean": nz.mean(axis=1).fillna(0), "r_std": r.std(axis=1), "r_min": nz.min(axis=1).fillna(0),
    })
    for c in RATINGS:
        num["n_" + c] = r[c]
    if v2:
        sys.path.insert(0, os.path.join(ROOT, "experiments"))
        from features_v2 import build_v2
        X2, X2t, _, _ = build_v2()
        extra = [c for c in X2.columns if c.startswith(("fd_", "c_", "orig_")) or c.endswith("_cnt")]
        E = pd.concat([X2[extra], X2t[extra]], ignore_index=True).astype(np.float32)
        del X2, X2t
        for c in extra:
            if c.endswith("_cnt") or c in ("fd_n", "c_Flig", "fd_o_n"):
                E[c] = np.log1p(E[c])
        E = E.fillna(E.median())
        if "orig_pred" in E:
            p = E["orig_pred"].clip(1e-4, 1 - 1e-4)
            E["orig_logit"] = np.log(p / (1 - p))
        print("v2 extra cols", list(E.columns), flush=True)
        num = pd.concat([num.reset_index(drop=True), E], axis=1)
    num = num.astype(np.float32).values
    mu, sd = num[:n_tr].mean(0), num[:n_tr].std(0) + 1e-6
    num = (num - mu) / sd
    return codes, num.astype(np.float32), y, n_tr, cards, cat_cols


class PLR(nn.Module):
    """Periodic-Linear-ReLU numeric embeddings (Gorishniy et al.), one per numeric feature."""
    def __init__(self, n_num, n_freq=16, d_out=8, sigma=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_num, n_freq) * sigma)
        self.w = nn.Parameter(torch.randn(n_num, 2 * n_freq, d_out) / (2 * n_freq) ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_num, d_out))
        self.d = n_num * d_out

    def forward(self, x):
        v = 2 * np.pi * self.c[None] * x[..., None]
        v = torch.cat([torch.sin(v), torch.cos(v)], -1)
        return torch.relu(torch.einsum("bnf,nfd->bnd", v, self.w) + self.b).flatten(1)


class Net(nn.Module):
    def __init__(self, cards, n_num, emb_dim=8, hidden=(512, 256, 128), drop=0.2, emb_drop=0.05,
                 n_cross=0, n_base=None, pair_dim=4, plr=0, plr_sigma=1.0):
        super().__init__()
        self.plr = PLR(n_num, 16, plr, plr_sigma) if plr else None
        n_base = len(cards) if n_base is None else n_base
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb_dim, max(2, c)) if j < n_base else pair_dim)
                                   for j, c in enumerate(cards)])
        d_emb = sum(e.embedding_dim for e in self.embs)
        self.emb_drop = nn.Dropout(emb_drop)
        d = d_emb + n_num + (self.plr.d if plr else 0)
        d0 = d
        self.cross = nn.ModuleList([nn.Linear(d0, d0) for _ in range(n_cross)])
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d + (d0 if n_cross else 0), 1)

    def forward(self, xc, xn):
        e = torch.cat([emb(xc[:, j]) for j, emb in enumerate(self.embs)], 1)
        parts = [self.emb_drop(e), xn]
        if self.plr is not None:
            parts.append(self.plr(xn))
        x0 = torch.cat(parts, 1)
        h = self.mlp(x0)
        if len(self.cross):
            x = x0
            for lin in self.cross:  # DCN-v2 cross layers
                x = x0 * lin(x) + x
            h = torch.cat([h, x], 1)
        return self.head(h).squeeze(1)


@torch.no_grad()
def predict(model, xc, xn, bs=65536):
    model.eval()
    out = []
    for i in range(0, len(xc), bs):
        out.append(torch.sigmoid(model(xc[i:i + bs], xn[i:i + bs])).float())
    return torch.cat(out).cpu().numpy()


def train_fold(args, cards, xc_tr, xn_tr, y_tr, xc_va, xn_va, y_va, seed):
    torch.manual_seed(seed)
    model = Net(cards, xn_tr.shape[1], args.emb, tuple(args.hidden), args.drop, args.emb_drop,
                args.cross, args.n_base, args.pair_dim, args.plr, args.plr_sigma).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n = len(y_tr)
    steps = (n + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps * args.epochs,
                                                pct_start=0.25, div_factor=10, final_div_factor=100)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, best_ep, bad = -1, None, -1, 0
    g = torch.Generator(device=DEV).manual_seed(seed)
    yv = y_va.cpu().numpy()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=DEV, generator=g)
        for i in range(steps):
            idx = perm[i * args.bs:(i + 1) * args.bs]
            logit = model(xc_tr[idx], xn_tr[idx])
            yt = y_tr[idx]
            if args.smooth > 0:
                yt = yt * (1 - args.smooth) + 0.5 * args.smooth
            loss = lossf(logit, yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        p = predict(model, xc_va, xn_va)
        auc = roc_auc_score(yv, p)
        if auc > best:
            best, best_ep, bad = auc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if args.verbose:
            print(f"    ep {ep} auc {auc:.5f}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gpu_nn_dev")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--emb", type=int, default=8)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--emb_drop", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cross", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--pair_dim", type=int, default=4)
    ap.add_argument("--v2", type=int, default=0)
    ap.add_argument("--plr", type=int, default=0, help="PLR embedding dim per numeric (0=off)")
    ap.add_argument("--plr_sigma", type=float, default=1.0)
    ap.add_argument("--ref", default="lgbm")
    ap.add_argument("--fd_emb", type=int, default=0, help="embed raw Flight Distance values")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    folds_run = list(range(common.N_FOLDS)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(vars(args), flush=True)

    codes, num, y, n_tr, cards, cat_cols = prepare(pairs=bool(args.pairs), v2=bool(args.v2), fd_emb=bool(args.fd_emb))
    args.n_base = sum(1 for c in cat_cols if not c.startswith("p_"))
    print("n cat cols", len(cat_cols), "base", args.n_base, flush=True)
    folds = common.fold_ids()
    xc_all = torch.from_numpy(codes).to(DEV)
    xn_all = torch.from_numpy(num).to(DEV)
    del codes, num
    y_t = torch.from_numpy(y.astype(np.float32)).to(DEV)
    xc_te, xn_te = xc_all[n_tr:], xn_all[n_tr:]

    oof = np.zeros(n_tr)
    test_pred = np.zeros(len(xc_te))
    for f in folds_run:
        t0 = time.time()
        tr_idx = torch.from_numpy(np.where(folds != f)[0]).to(DEV)
        va_np = np.where(folds == f)[0]
        va_idx = torch.from_numpy(va_np).to(DEV)
        p_va = np.zeros(len(va_np))
        for s in seeds:
            model, auc, ep = train_fold(args, cards, xc_all[tr_idx], xn_all[tr_idx], y_t[tr_idx],
                                        xc_all[va_idx], xn_all[va_idx], y_t[va_idx], seed=1000 * s + f)
            p_va += predict(model, xc_all[va_idx], xn_all[va_idx]) / len(seeds)
            test_pred += predict(model, xc_te, xn_te) / (len(seeds) * len(folds_run))
            print(f"  fold {f} seed {s}: auc {auc:.5f} best_ep {ep}", flush=True)
            del model
            torch.cuda.empty_cache()
        oof[va_np] = p_va
        print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va_np], p_va):.5f}  ({time.time() - t0:.0f}s)", flush=True)

    if len(folds_run) == common.N_FOLDS:
        common.save_preds(args.name, oof, test_pred, args.notes or f"MLP {vars(args)}")
        lg = np.load(os.path.join(common.PREDS, f"oof_{args.ref}.npy"))
        print(f"corr with {args.ref}:", np.corrcoef(oof, lg)[0, 1], flush=True)


if __name__ == "__main__":
    main()
