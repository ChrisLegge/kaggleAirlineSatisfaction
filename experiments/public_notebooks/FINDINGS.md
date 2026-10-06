# Public notebook research (S6E10) — summary

Source: 13 public notebooks read on 2026-10-05 (not executed). Gains below are the authors' own CV numbers.

## Leaderboard context
- The cluster at 0.96168–0.96177 is mostly blends of one shared submission file from goodpjw2008's
  8-part series (Part 8 "TabPFN + Route Categories", LB 0.96164). Public LB noise ≈ ±0.0003, CV→LB gap ≈ −0.0004.
- We do **not** use shared submission files; everything here is trained from scratch.

## Techniques we were not using (ranked)
1. **Auxiliary "expected rating" features (label-free)** — for each of 13 ratings + Class/Travel/Customer Type,
   a multiclass XGB predicts that column from the other 20 (OOF over train+test, or fit on original and applied).
   Keep `aux_ev_<c> = P @ values` (ratings) and `aux_p_<c> = P[actual]`, plus `aux_sum_logp`.
   Trees: +0.00010–0.00013 using the 13 `aux_ev` only.
2. **TabPFN-3.5 with full fold context + FD categorical** — +0.00015 in stack, but needs ~17.5 GB VRAM. *Not feasible on RTX 3050 (8 GB).*
3. **LogReg stacker on logits of many diverse members** — +0.00027 over best single. Diversity > strength.
4. **RealMLP (pytabkit)** — best single model in the series (0.9613–0.9614). Every numeric/rating column also
   given a categorical twin; `lr_sched="flat_anneal"`, 3–5 epochs; add aux features and engineered inputs.
5. **Original-data models as features** — XGB(600, lr .05, depth 8) on deduplicated original (+0.0007 on XGB);
   plus an original-trained RealMLP (+0.0002).
6. **Fold-safe target encodings** on [FD], [Age], [FD,Age], [FD,Class], [FD,Travel], [Age,Class,Travel,Customer].
7. **CatBoost with numerics (int copies) + ratings as cat_features**, depth 6–8, l2 10, border_count 254: +0.00015 single.
8. 10-fold members — doubtful (no LB gain reported).

## Reported as not working
Appending original rows; rating sum/std features; ratings as categoricals for trees; TE of rating combos;
FD as native LGB categorical; kNN to original; TabM / MLP-PLR; more seeds; pseudo-labels; LGB stackers.
