import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, common
tr,te,o=common.load_raw()
print(tr.shape,te.shape,o.shape, tr.satisfaction.mean(), o.satisfaction.mean())
cols=[c for c in o.columns if c!='satisfaction']
print(tr[cols].nunique().to_dict())
print(o[cols].nunique().to_dict())
for d in (tr,te,o): d['Arrival Delay in Minutes']=d['Arrival Delay in Minutes'].fillna(-1)
k=lambda d:d[cols].astype(str).agg('|'.join,axis=1)
ks=set(k(o)); ktr=k(tr)
m=ktr.isin(ks); print('train rows matching orig exactly',m.mean(), 'test',k(te).isin(ks).mean())
print('orig dup rate',k(o).duplicated().mean(),'train dup',ktr.duplicated().mean())
og=o.assign(k=k(o)).groupby('k').satisfaction.mean()
mm=tr[m].assign(k=ktr[m]); print('agreement', ((mm.k.map(og)>0.5)==mm.satisfaction).mean())
