from h import *
X,Xt,y=base_data()
run_cv("feat_baseline",X,Xt,y,folds_to_run=range(5),save=True,notes="LGBM fixed fast cfg on add_features")
