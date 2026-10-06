import sys; sys.path.insert(0,'.')
from h import *
train, test, o = common.load_raw()
y=train.satisfaction.values; yo=o.satisfaction.values
vc=o["Flight Distance"].value_counts()
print("orig FD count dist", vc.value_counts().sort_index().head(8).to_dict())
print("train FD in orig:", train["Flight Distance"].isin(vc.index).mean(), "test", test["Flight Distance"].isin(vc.index).mean())
tv=train["Flight Distance"].value_counts(); print("train FD count dist", tv.value_counts().sort_index().head(8).to_dict(), "n unique", len(tv))
u=vc[vc==1].index; ou=o[o["Flight Distance"].isin(u)].set_index("Flight Distance")
tu=train[train["Flight Distance"].isin(u)]
j=tu.join(ou,on="Flight Distance",rsuffix="_o")
print(len(j))
for c in ["Age","Class","Gender","Type of Travel","Customer Type","Online boarding","Inflight wifi service","Seat comfort","Departure Delay in Minutes"]:
    print(c, (j[c]==j[c+"_o"]).mean())
print("label agree", (j.satisfaction==j.satisfaction_o).mean(), "base", y.mean())
from sklearn.metrics import roc_auc_score
print("auc of orig label alone", roc_auc_score(j.satisfaction,j.satisfaction_o))
