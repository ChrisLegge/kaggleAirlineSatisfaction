# Kaggle Playground S6E10 — Predicting Airline Satisfaction

Binary classification of airline passenger satisfaction ([competition page](https://www.kaggle.com/competitions/playground-series-s6e10)),
scored by ROC AUC. 700k synthetic training rows generated from the real
[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) dataset.

**Best result so far:** public LB **0.96177** (rank 4 / 885 on 2026-10-06), 5-fold CV 0.96208.

## Progression

| Step | CV AUC | Public LB |
|---|---|---|
| LightGBM, raw features | 0.95886 | 0.95849 |
| + rank blend with XGBoost / CatBoost | 0.95898 | 0.95841 |
| + **Flight-Distance "route" features** (see below) | 0.96071 (single LGBM) | — |
| LogReg stack of own models on the new features | 0.96084 | 0.96047 |
| + 48 public OOF members (goodpjw2008) | 0.96192 | 0.96160 |
| + further public OOF libraries (125 members) | 0.96199 | 0.96164 |
| + more public libraries (busyaprime, sadamtorres), LogReg C=0.02 (175 members) | 0.96205 | 0.96173 |
| + own 10-fold RealMLP / CatBoost members with auxiliary features (184 members) | 0.96208 | 0.96177 |

The public LB was consistently ~0.0004 below CV, so all model selection was done on CV
(the public LB uses only ~20 % of the test set and is noisy at the 4th decimal).

## What worked

1. **Flight Distance behaves like a route ID.** It has ~3,500 distinct values and passengers sharing a
   value answer the survey similarly. Label-free group statistics per Flight Distance over
   train + test + original data (row count, mean of every rating and of age), plus the original data's
   satisfaction rate per Flight Distance, lifted a single LightGBM from 0.95874 → 0.96071
   (`experiments/features_v2.py`, `experiments/feat_fd.py`). This was ~10× every other idea.
2. **A model trained only on the original dataset**, used as a feature (its predictions are label-safe
   because the original rows are disjoint from the competition data).
3. **Hyper-parameter tuning** with Optuna (`experiments/lgbm_optuna.py`): +0.0002 on the new features
   (best: 67 leaves, strong L1, max_bin 1023).
4. **Diverse model families** on GPU: CatBoost (depthwise, depth 10), XGBoost, a PyTorch MLP with
   embeddings + periodic numeric encodings, and RealMLP (pytabkit).
5. **Honest stacking**: logistic regression on standardised logits of out-of-fold predictions, weights fit
   on 4 folds and scored on the 5th (`stack.py`). Diversity mattered more than single-model strength.

## What didn't work

- Appending the original rows to the training set (−0.0002).
- Simple aggregate features (rating means/stds/counts), ratings as categoricals, extra target encodings.
- "Expected rating" auxiliary features (predict each rating from the others) — helped public RealMLPs but not our LightGBM.
- Seed averaging (+0.00003) and more Optuna trials after ~40.
- TabPFN with full context needs ~17.5 GB VRAM — not possible on an 8 GB RTX 3050.

## Credits / inspiration

The final stacks combine our own models with **publicly shared out-of-fold predictions** from other
competitors. Their work made a large part of the final score possible:

- **goodpjw2008** — the S6E10 notebook series (route features, auxiliary-task features, RealMLP recipe,
  TabPFN-3.5 members); OOF/test members from *"S6E10 | TabPFN + Route Categories | LB 0.96160"*.
- **Vladimir Demidov (yekenot)** — RealMLP · PyTabKit recipe.
- **Busya PRIME** (five members re-run and shared by **kratosyan**) and **sadamtorres** (artifacts library).
- Public OOF libraries: **dariushafshar** (golem OOF library), **sachith7** (stack OOF predictions),
  **megayak** (OOF library), **arhancanli12** (OOF & test predictions).
- Notebooks consulted: najiama, sarunpm, nina2025, matterhorn3838 and others
  (summary in `experiments/public_notebooks/FINDINGS.md`).

Public members are imported with `import_public.py` / `import_public_libs.py` and are prefixed
(`pub_`, `golem_`, `sach_`, `mega_`, `arhan_`) so they are always distinguishable from our own models.

## Repository layout

| Path | Purpose |
|---|---|
| `common.py` | Fixed 5-fold split, data loading, `save_preds` (saves OOF/test preds + logs CV to `results.csv`) |
| `features.py` | v1 feature engineering |
| `experiments/features_v2.py` | Final feature set (route group stats, counts, original-data lookups, original-model feature) |
| `train_models.py`, `baseline_lgbm.py` | First LightGBM / XGBoost / CatBoost pipeline |
| `experiments/lgbm_optuna.py` | LightGBM tuning |
| `experiments/cat_gpu.py`, `xgb_gpu.py`, `nn_mlp.py`, `realmlp_gpu.py` | GPU models |
| `ensemble.py`, `stack.py` | Hill-climbing and logistic-regression stacking with honest CV |
| `results.csv` | CV score of every saved model |
| `submissions_log.csv` | Every Kaggle submission with CV and public score |
| `submissions/` | Submission files |

## Reproducing

1. `pip install pandas scikit-learn lightgbm xgboost catboost optuna torch pytabkit kaggle`
2. Download the competition data into `playground-series-s6e10/`
   (`kaggle competitions download -c playground-series-s6e10`) and the original dataset into
   `originalData/data.csv`. Data files are not committed (competition rules).
3. Build features and models, e.g. `python experiments/lgbm_optuna.py`, `python experiments/cat_gpu.py --v2 1 ...`
   (each script saves to `preds/` via `common.save_preds`).
4. Optional: download the public OOF datasets listed above and run `import_public.py` / `import_public_libs.py`.
5. `python stack.py --all --out my_submission.csv`

Hardware: Intel i5-12400, 16 GB RAM, RTX 3050 8 GB.
