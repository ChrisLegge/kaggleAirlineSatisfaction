"""Field-aware-ish factorisation machine on GPU (torch). Every column is a categorical field (exact FD, exact Age,
binned delays, ratings, cats, + optional FD x Class/Travel/Cust crossed fields). Model = bias + linear + FM pairwise
(+ optional field-pair weights, i.e. FwFM). No MLP part, so it differs from DeepFM/RealMLP.
Usage: python -u experiments/gpu3_fm.py --name gpu3_fm --folds all
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import common  # noqa: E402
from features import RATINGS, BASE_CATS  # noqa: E402

torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gpu3_fm_dev")
ap.add_argument("--folds", default="0")
ap.add_argument("--dim", type=int, default=16)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=2048)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-6)
ap.add_argument("--fwfm", type=int, default=1)
ap.add_argument("--cross", type=int, default=1)
ap.add_argument("--min_count", type=int, default=3)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--notes", default="")
args = ap.parse_args()
print(vars(args), flush=True)

train, test, _ = common.load_raw()
y = train[common.TARGET].values
n_tr = len(train)
df = pd.concat([train.drop(columns=[common.TARGET]), test], ignore_index=True)
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
F = pd.DataFrame(index=df.index)
for c in RATINGS + BASE_CATS:
    F[c] = df[c].astype(str)
F["fd"] = df["Flight Distance"].astype(str)
F["age"] = df["Age"].astype(str)
F["fd_b"] = pd.cut(np.log1p(df["Flight Distance"]), 32, labels=False).astype(str)
F["dep"] = df["Departure Delay in Minutes"].clip(upper=200).astype(int).astype(str)
F["arr"] = df["Arrival Delay in Minutes"].clip(upper=200).astype(int).astype(str)
F["dd"] = (df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]).clip(-30, 30).astype(int).astype(str)
if args.cross:
    for c in ["Class", "Type of Travel", "Customer Type"]:
        F["fd|" + c] = F["fd"] + "_" + F[c]
    F["ctc"] = F["Class"] + F["Type of Travel"] + F["Customer Type"]
    F["age|ctc"] = F["age"] + "_" + F["ctc"]
codes, off = [], 0
for c in F.columns:
    cd, uq = pd.factorize(F[c])
    vc = np.bincount(cd[:n_tr], minlength=len(uq))
    keep = vc >= args.min_count
    remap = np.where(keep, np.cumsum(keep) - 1, keep.sum())
    codes.append(remap[cd] + off)
    off += keep.sum() + 1
nf = len(codes)
I = torch.tensor(np.column_stack(codes), dtype=torch.long)
print("fields", nf, "vocab", off, flush=True)
dev = "cuda"
iu = torch.triu_indices(nf, nf, 1)


class FM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(off, args.dim)
        self.lin = torch.nn.Embedding(off, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.r = torch.nn.Parameter(torch.ones(len(iu[0])))

    def forward(self, x):
        e = self.emb(x)  # B, F, D
        lin = self.lin(x).sum((1, 2))
        if args.fwfm:
            g = torch.einsum("bfd,bgd->bfg", e, e)[:, iu[0], iu[1]]
            pair = (g * self.r).sum(1)
        else:
            s = e.sum(1)
            pair = 0.5 * ((s * s).sum(1) - (e * e).sum((1, 2)))
        return self.b + lin + pair


def fit_predict(tr, pred_sets, seed):
    torch.manual_seed(seed)
    m = FM().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    Itr = I[tr].to(dev)
    ytr = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        m.train()
        perm = torch.randperm(len(tr), device=dev)
        for s in range(0, len(tr), args.bs):
            b = perm[s:s + args.bs]
            xb = Itr[b]
            out = m(xb)
            loss = lossf(out, ytr[b])
            # L2 only on the embeddings actually used in the batch
            loss = loss + args.wd * (m.emb(xb) ** 2).sum() / len(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    m.eval()
    res = []
    with torch.no_grad():
        for idx in pred_sets:
            o = np.concatenate([m(I[idx[s:s + 16384]].to(dev)).float().cpu().numpy() for s in range(0, len(idx), 16384)])
            res.append(o)
    return res


folds = common.fold_ids()
folds_run = list(range(5)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
oof = np.zeros(n_tr)
tp = np.zeros(len(I) - n_tr)
te_idx = np.arange(n_tr, len(I))
for f in folds_run:
    t0 = time.time()
    tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
    for sd in range(args.seeds):
        o, t = fit_predict(tr, [va, te_idx], sd)
        oof[va] += o / args.seeds
        tp += t / args.seeds / len(folds_run)
        print(f"   seed {sd} AUC {roc_auc_score(y[va], o):.5f}", flush=True)
    print(f"[{args.name}] fold {f}: AUC {roc_auc_score(y[va], oof[va]):.5f} ({time.time() - t0:.0f}s)", flush=True)
if len(folds_run) == 5:
    sig = lambda z: 1 / (1 + np.exp(-z))
    common.save_preds(args.name, sig(oof), sig(tp), args.notes or f"FwFM GPU {vars(args)}")
