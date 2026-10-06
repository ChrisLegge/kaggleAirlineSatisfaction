python -u experiments/lgbm_optuna.py proxy --trial 43 --params '[{}, {\"max_bin\": 2047}, {\"max_bin\": 4095}, {\"max_bin\": 255}, {\"extra_trees\": true}]' *> experiments/proxy1.log
python -u experiments/lgbm_optuna.py proxy --trial 43 --cat_ratings --params '[{}, {\"max_bin\": 4095}]' *> experiments/proxy2.log
