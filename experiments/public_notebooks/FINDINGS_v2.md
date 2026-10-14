# Public notebook research v2 (2026-10-14)

24 notebooks read (pulled into `experiments/public_notebooks/v2_*/`, not executed).

## Headline
- No new modelling technique in 8 days. The public top band (LB 0.96187–0.96190) is one shared pool of
  ~245–253 public OOF pairs restacked with logistic regression on logits; test predictions rank-correlate ≥ 0.999.
- The 0.96188+ public files are blends of other people's submission CSVs with LB-chosen weights (not models).
- CV→LB gap is stable across independent stacks: nested CV − public LB ≈ 0.00034 ± 0.00001 (ours: 0.00031).
- Our shortfall is pool contents, not stacker design (their nested CV ≈ 0.96222 ↔ LB ≈ 0.96188).
- The 0.96243 / 0.96234 / 0.96219 leaders are not explained by any public notebook.

## Stacker changes reported (small)
1. LR C≈0.003 on raw clipped logits (≈ +3e-5 vs C=1 on 187 members); converged solver (tol 1e-8): +1–2e-6.
2. Per-family L2 via column scaling (tabpfn ×3, other foundation models ×1.5): CV +3–7e-6, LB neutral.
3. Member hygiene: OOF AUC window 0.90–0.9625, test-vs-OOF agreement gap, mean drift < 0.03, drop near-copies (rank ρ > 0.99995).
4. Admit members with a paired-delta rule (Δ > 2 SE and ≥ 4/5 folds up, on two meta seeds).
5. CV deltas below ~1e-5 did not transfer to the LB.

## Leaky / LB-fitted — avoid
Submission-CSV blends with LB-tuned weights (rabari9999 roadmap, saddamaditya apex, lucifer19 superblend,
nina2025 50x50, secrets-to-satisfaction final file, consensus-tuner); stack/blend OOFs used as members
(najiama 05/06, stack245/247, busyaprime stack53); denpugovkin RankNet in-sample correction; pseudo-labels from
stacks; members early-stopped on the fold they score (golem `p_lgbm_auxev`, `q_lgbm_auxev_10fold`) are mildly optimistic.

## Possibly additive members
Chris Deotte's foundation-model OOFs (TabPFN-3.5, TabICL-2, Kumo, Mitra-2, LimiX-2, TabFM, EXAONE, Causilo),
Koumei Maki's TabPFN/TabICL/TabM set, AbdullahSafwan333's 15-fold×3-seed TabPFN, golem `o_tabpfn35`.
Same-feature GBDT/MLP members add ≈ 0.

## Feature notes (catsrcul ablations, LightGBM)
Original-data marginal target priors +0.00094 (5/5 folds); `Departure/Arrival time convenient` is reverse-signed;
aux expected-rating features −0.00026 for trees. Generator: each row is a source row with 2–4 columns re-drawn;
Flight Distance keeps its original value in only 1.8 % of rows.
