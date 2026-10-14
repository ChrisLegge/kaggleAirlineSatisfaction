"""Import a third batch of public OOF/test libraries (published or updated since 2026-10-06) into preds/.

Checks for every pair: 699,635 OOF rows (train.csv order, aligned by id where the source gives ids) and
299,844 test rows, finite, OOF AUC > 0.9 (misalignment guard), and test predictions correlated > 0.9 with an
existing strong member (test-order guard). Exact duplicates of anything already in preds/ (OOF Pearson or
Spearman corr > 0.99999, also among the members imported here) are skipped.

LEAKAGE RULE: blends/stacks whose weights or meta-model were fit on train labels are imported only under the
prefix "blend_" (excluded by stack_final*.py). A published stack with an honest nested OOF is imported as
"nstack_<author>" and is only evaluated on its own, never stacked.

Credits (Kaggle datasets / notebook outputs, downloaded into experiments/public_outputs/):
  * dariushafshar/s6e10-golem-oof-library v6 (golem_)  - new single members o_tabpfn35, p_lgbm_auxev,
    q_lgbm_auxev_10fold, r_realmlp_auxev, s_realmlp_raw_auxev_10fold, t_xgb_d5_auxev, u_catboost_ctr_auxev
    (same seed-42 folds as ours). stack247_* = his nested 247-pair LR stack (notebook "S6E10 | 247 Public OOFs,
    One Stack | LB 0.96187") -> nstack_dariush (evaluation only).
  * cdeotte/s6e10-124-public-oof-and-test-predictions (cd_<orig prefix>_) - Chris Deotte's packaging of the
    public pool (goodpjw2008, kratosyan collection, megayak, sachith7, golem, arhancanli12, busyaprime,
    mitudru (mi_), thisray (th_), wangxintong111 (wx_), hemingweb / samanyu1808 TabPFNs (pfn_), meh_ ...).
    Most are duplicates of members we already have and are skipped by the corr rule. nj_01 is najiama's
    label-fit blend -> blend_cd_nj_01.
  * cdeotte/s6e10-tfm-oof-and-test-predictions (cdtfm_) - tabular foundation models (TabPFN-3.5, TabICL2,
    Kumo, Limix, Mitra, Causilo, Exaone, TabFM) + RealMLP/XGB.
  * goodpjw2008/s6e10-goodpjw2008-extra-members-oof (gpx_) - 17 single members. stack245 = their nested
    245-member LR stack (seed-7 meta folds) -> blend_gpx_stack245 (author warns it must not be stacked).
  * kagankoral/s6e10-ablation-members-oof (kga_) - 17 single GBDT/RealMLP ablation members.
  * kagankoral notebook "s6e10-encodings-catboost-realmlp-lb-0-961" output (kg_) - lgb/xgb/cat/cat_allcat/realmlp.
  * dmitriykolmikov/s6e10-catboost-all-categorical-oof-and-test (dk_) - one CatBoost.
    (dmitriykolmikov/s6e10-flight-distance-oof-lgb-xgb-cb only contains submission.csv -> nothing to import.)
  * sachith7/s6e10-stack-oof-predictions update (sach_) - new lgb_cpu_strong, xgb_A_auxall.
  * koumeimaki/s6e10-route-tfm-nn-oof (km_) - TabPFN-3.5, TabICL, RealMLP, TabM, LGB, XGB, Cat (ids + folds given).
  * amanatar/s6e10-forge-own-pairs (am_) - 6 single members (own seed-33 folds).
  * amanatar notebook "s6e10-the-original-prior-stack-v5" output (av5_) - 5 single models;
    its logit_blend / rank_blend / lr_stack are label-fit -> blend_av5_*.
  * abdullahsafwan333/oof-dataset (ab_) - TabPFN catfd10 / raw / te_cnt, CatBoost v3, XGB v3 (id csvs).
  * hermengardo notebook "single-catboost-eda-ps6e10" output (hm_) - one CatBoost.
  * kozykappa notebook "s6e10-source-transfer-foundation-model-oof" output (kz_) - source_transfer = a single
    raw LightGBM; meta_oof is a stack -> blend_kz_meta.
  * kratosyan notebook "s6e10-xgboost-realmlp-route-id-teacher" output (kr_) - RealMLP / XGB route-id teacher.
  * yekenot notebook "ps-s6-e10-realmlp-pytabkit" output (yk_) - one RealMLP.
  * najiama/s6e10-oof update (naji_ prefix = label-fit blends, already excluded): new 04/05/06 blends are
    imported as blend_naji_*; 07_XGBoost is imported as a single member (nj07_xgb) only if its OOF AUC is
    in single-model range (< 0.9616), otherwise as a blend.

Submission-only datasets (no OOF, cannot be CV'd, not imported): rabari9999/s6e10-top18-artifacts,
nina2025/ps-s6e10-7, nina2025/ps-s6e10-02, alexanderwang2001/s6e10-frontier-reset-candidates,
anhadmahajan06/ps-s6e10-predicting-airline-satisfaction, dmitriykolmikov/s6e10-flight-distance-oof-lgb-xgb-cb.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import DATA, PREDS, get_y

SRC = os.path.join("experiments", "public_outputs")
y = get_y()
train_ids = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["id"])["id"].values
test_ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"].values
REF_TEST = np.load(os.path.join(PREDS, "test_pub_realmlp_v4fa.npy"))


def _std(v):
    v = np.asarray(v, np.float64)
    return ((v - v.mean()) / v.std()).astype(np.float32)


def _rstd(v):
    return _std(rankdata(v))


# existing members (everything in preds/) for the duplicate check
_ex_names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(PREDS, "oof_*.npy")))
_ex_names = [n for n in _ex_names if not n.startswith(("blend_", "nstack_"))]
EX = np.stack([_std(np.load(os.path.join(PREDS, f"oof_{n}.npy"))) for n in _ex_names])
EXR = np.stack([_rstd(np.load(os.path.join(PREDS, f"oof_{n}.npy"))) for n in _ex_names])  # rank duplicates too
NEW = []  # (name, std oof) imported in this run
out, skipped = [], []


def save(name, oof, test, dedupe=True):
    oof, test = np.asarray(oof, float).ravel(), np.asarray(test, float).ravel()
    assert oof.shape == (len(y),) and test.shape == (len(test_ids),), (name, oof.shape, test.shape)
    assert np.isfinite(oof).all() and np.isfinite(test).all(), name
    if oof.min() < 0 or oof.max() > 1 or test.min() < 0 or test.max() > 1:  # given as logits -> probabilities
        print(f"{name}: values outside [0,1] -> treated as logits, sigmoid applied")
        oof, test = 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-test))
    lo = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
    ratio = lo(test).std() / lo(oof).std()
    if not 0.8 < ratio < 1.25:
        print(f"WARN {name}: test/oof logit std ratio {ratio:.3f}")
    if os.path.exists(os.path.join(PREDS, f"oof_{name}.npy")):
        print(f"exists {name} -> skipped")
        return
    auc = roc_auc_score(y, oof)
    if auc < 0.9:
        print(f"SKIP {name}: AUC {auc:.4f} looks misaligned")
        skipped.append((name, "misaligned"))
        return
    tc = np.corrcoef(test, REF_TEST)[0, 1]
    if tc < 0.9:
        print(f"SKIP {name}: test corr with reference {tc:.3f} looks misaligned")
        skipped.append((name, "test misaligned"))
        return
    s, r = _std(oof), _rstd(oof)
    if dedupe:
        c = np.maximum(EX @ s, EXR @ r) / len(s)
        j = int(np.argmax(c))
        if c[j] > 0.99999:
            print(f"dup {name} == {_ex_names[j]} (corr {c[j]:.7f}) -> skipped")
            skipped.append((name, f"dup of {_ex_names[j]}"))
            return
        for n2, s2, r2 in NEW:
            c2 = max(float(s2 @ s), float(r2 @ r)) / len(s)
            if c2 > 0.99999:
                print(f"dup {name} == {n2} (corr {c2:.7f}) -> skipped")
                skipped.append((name, f"dup of {n2}"))
                return
        NEW.append((name, s, r))
    np.save(os.path.join(PREDS, f"oof_{name}.npy"), oof)
    np.save(os.path.join(PREDS, f"test_{name}.npy"), test)
    out.append((name, auc))


def npy_dir(d, pre, files=None, rename=None):
    d = os.path.join(SRC, d)
    for p in sorted(glob.glob(os.path.join(d, "oof_*.npy"))):
        k = os.path.basename(p)[4:-4]
        if files is not None and k not in files:
            continue
        nm = (rename or {}).get(k, pre + k)
        save(nm, np.load(p), np.load(os.path.join(d, f"test_{k}.npy")), dedupe=not nm.startswith("blend_"))


def by_id(path, ids, col):
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(np.int64)
    return df.set_index("id").loc[ids, col].values


# golem v6: new single members (a..n were imported by import_public_libs.py)
npy_dir("v3_s6e10-golem-oof-library", "golem_")

# Chris Deotte's 124 public pairs; nj_01 is najiama's label-fit blend
npy_dir("v3_cdeotte124", "cd_", rename={"nj_01": "blend_cd_nj_01"})

# Chris Deotte's TFM pairs
npy_dir("v3_s6e10-tfm-oof-and-test-predictions", "cdtfm_")

# goodpjw2008 extras; stack245 is a label-fit stack
npy_dir("v3_s6e10-goodpjw2008-extra-members-oof", "gpx_", rename={"stack245": "blend_gpx_stack245"})

# kagankoral ablation members + encodings notebook members
npy_dir("v3_s6e10-ablation-members-oof", "kga_")
d = os.path.join(SRC, "v3k_s6e10-encodings-catboost-realmlp-lb-0-961")
if os.path.exists(os.path.join(d, "test_members.csv")):
    o = pd.read_csv(os.path.join(d, "oof_members.csv")).set_index("id").loc[train_ids]
    t = pd.read_csv(os.path.join(d, "test_members.csv")).set_index("id").loc[test_ids]
    for c in o.columns:
        if c in t.columns:
            save("kg_" + c, o[c].values, t[c].values)

# dmitriykolmikov CatBoost all-categorical
d = os.path.join(SRC, "v3_s6e10-catboost-all-categorical-oof-and-test")
save("dk_cb_cat", np.load(os.path.join(d, "oof_cb_cat.npy")), np.load(os.path.join(d, "test_cb_cat.npy")))

# sachith7 new members
for k in ("lgb_cpu_strong", "xgb_A_auxall"):
    z = np.load(os.path.join(SRC, "v3_s6e10-stack-oof-predictions", f"{k}.npz"))
    save("sach_" + k, z["oof"], z["test"])

# koumeimaki (train/test ids verified to be in train.csv/test.csv order, folds = ours)
npy_dir("v3_s6e10-route-tfm-nn-oof", "km_")

# amanatar forge pairs + original-prior-stack v5 singles (blends -> blend_)
npy_dir("v3_s6e10-forge-own-pairs", "am_")
z = np.load(os.path.join(SRC, "v3k_s6e10-the-original-prior-stack-v5", "oof_artifacts.npz"))
assert (z["y"] == y).all()
for k in ("lgbm_champion", "lgbm_te", "lgbm_te_rows", "xgb_gpu", "catboost_gpu"):
    save("av5_" + k, z["oof_" + k], z["test_" + k])
for k in ("logit_blend", "rank_blend", "lr_stack"):
    save("blend_av5_" + k, z["oof_" + k], z["test_" + k], dedupe=False)

# abdullahsafwan333 id csvs
d = os.path.join(SRC, "v3_oof-dataset")
for p in sorted(glob.glob(os.path.join(d, "*_oof.csv"))):
    k = os.path.basename(p)[:-8]
    save("ab_" + k, by_id(p, train_ids, "pred"), by_id(os.path.join(d, f"{k}_test.csv"), test_ids, "pred"))

# hermengardo single CatBoost
d = os.path.join(SRC, "v3k_single-catboost-eda-ps6e10")
save("hm_cat", np.load(os.path.join(d, "oof_train.npy")), np.load(os.path.join(d, "oof_test.npy")))

# kozykappa source transfer (single raw LGB); meta = stack
d = os.path.join(SRC, "v3k_s6e10-source-transfer-foundation-model-oof")
save("kz_source_transfer", np.load(os.path.join(d, "source_transfer_oof.npy")), np.load(os.path.join(d, "source_transfer_test.npy")))
save("blend_kz_meta", np.load(os.path.join(d, "meta_oof.npy")), np.load(os.path.join(d, "meta_test.npy")), dedupe=False)

# kratosyan route-id teacher members
for p in sorted(glob.glob(os.path.join(SRC, "v3k_s6e10-xgboost-realmlp-route-id-teacher", "**", "*.npz"), recursive=True)):
    z = np.load(p)
    if "oof" in z.files and "test" in z.files:
        save("kr_" + os.path.basename(p)[:-4], z["oof"], z["test"])

# yekenot RealMLP
d = os.path.join(SRC, "v3k_ps-s6-e10-realmlp-pytabkit")
if os.path.exists(os.path.join(d, "oof_preds.csv")):
    save("yk_realmlp", by_id(os.path.join(d, "oof_preds.csv"), train_ids, "satisfaction"),
         by_id(os.path.join(d, "submission.csv"), test_ids, "satisfaction"))

# najiama: 04-06 blends (label-fit) + 07 XGBoost
d = os.path.join(SRC, "v3_s6e10-oof")
for k in ("04", "05", "06"):
    save(f"blend_naji_{k}", by_id(os.path.join(d, f"{k}_blend_oof.csv"), train_ids, "satisfaction"),
         by_id(os.path.join(d, f"{k}_submission.csv"), test_ids, "satisfaction"), dedupe=False)
p = os.path.join(d, "07_XGBoost_oof.csv")
o7, t7 = by_id(p, train_ids, "oof_pred"), by_id(os.path.join(d, "07_submission.csv"), test_ids, "satisfaction")
a7 = roc_auc_score(y, o7)
print(f"najiama 07_XGBoost OOF AUC {a7:.6f}")
save("nj07_xgb" if a7 < 0.9616 else "blend_naji_07", o7, t7, dedupe=a7 < 0.9616)

# Dariush Afshar's nested 247 stack: evaluation only
d = os.path.join(SRC, "v3_s6e10-golem-oof-library")
save("nstack_dariush247", np.load(os.path.join(d, "stack247_oof.npy")), np.load(os.path.join(d, "stack247_test.npy")), dedupe=False)

for n, a in sorted(out, key=lambda t: -t[1]):
    print(f"{n:40s} {a:.6f}")
print(len(out), "pairs imported;", len(skipped), "skipped")
json.dump({"imported": out, "skipped": skipped}, open(os.path.join("experiments", "import3_log.json"), "w"), indent=1)
