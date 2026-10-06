python -u experiments/lgbm_optuna.py cv --trial 43 --params '{\"max_bin\": 2047}' --lr 0.02 --name lgbmtune_t43 --notes "optuna trial 43" *> experiments/cv_t43.log
python -u experiments/lgbm_optuna.py cv --trial 66 --params '{\"max_bin\": 2047}' --lr 0.02 --name lgbmtune_t66 --notes "optuna trial 66" *> experiments/cv_t66.log
python -u experiments/lgbm_optuna.py cv --trial 45 --params '{\"max_bin\": 2047}' --lr 0.02 --name lgbmtune_t45 --notes "optuna trial 45" *> experiments/cv_t45.log
