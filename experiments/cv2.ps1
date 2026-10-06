python -u experiments/lgbm_optuna.py cv --v2 --study lgbm_v2 --trial 34 --lr 0.02 --name lgbmtune_v2_t34 --notes "v2 feats, optuna v2 trial 34" *> experiments/cv_v2_t34.log
python -u experiments/lgbm_optuna.py cv --v2 --study lgbm_v2 --trial 22 --lr 0.02 --name lgbmtune_v2_t22 --notes "v2 feats, optuna v2 trial 22" *> experiments/cv_v2_t22.log
