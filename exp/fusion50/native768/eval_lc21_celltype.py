"""Five-fold body-disjoint LC21 cell-type probe on instance CLS embeddings."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--matrix',type=Path,required=True);ap.add_argument('--instance-map',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 x=np.load(a.matrix,mmap_mode='r').astype(np.float64);mp=pd.read_csv(a.instance_map);lab=pd.read_csv(a.labels)
 if x.ndim!=2 or len(x)!=len(mp) or not np.isfinite(x).all():raise ValueError('Invalid embedding matrix')
 if mp.record_id.duplicated().any() or lab.record_id.duplicated().any():raise ValueError('Duplicate record_id')
 d=mp.merge(lab,on='record_id',how='outer',validate='one_to_one',indicator=True)
 if not (d._merge=='both').all() or not (d.body_id_x.astype(str)==d.body_id_y.astype(str)).all():raise ValueError('record_id/body mismatch; no intersection allowed')
 if len(d)!=len(x):raise ValueError('Unmapped embeddings')
 d=d.sort_values('embedding_row').reset_index(drop=True); y=d.cell_type.astype(str).to_numpy();groups=d.body_id_x.astype(str).to_numpy();classes=np.array(sorted(set(y)))
 body_labels=d.groupby('body_id_x').cell_type.nunique()
 if (body_labels!=1).any():raise ValueError('Cell type must be constant within a body')
 cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261113);proba=np.full((len(y),len(classes)),np.nan);fold=np.full(len(y),-1)
 for f,(tr,te) in enumerate(cv.split(x,y,groups),1):
  if set(y[tr])!=set(classes):raise ValueError(f'Fold {f} misses a training class')
  scaler=StandardScaler().fit(x[tr]);model=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=3000,random_state=20261113+f)
  model.fit(scaler.transform(x[tr]),y[tr]);p=model.predict_proba(scaler.transform(x[te]))
  for j,c in enumerate(model.classes_):proba[te,np.where(classes==c)[0][0]]=p[:,j]
  fold[te]=f
 if not np.isfinite(proba).all() or (fold<0).any():raise ValueError('Incomplete OOF')
 pred=classes[proba.argmax(1)];body_records=[]
 for body,idx in d.groupby('body_id_x').groups.items():
  idx=np.asarray(list(idx));truth=set(y[idx])
  if len(truth)!=1 or len(set(fold[idx]))!=1:raise ValueError('Body was split across folds')
  logp=np.log10(np.clip(proba[idx],1e-15,1)).mean(0);body_records.append((str(body),truth.pop(),classes[logp.argmax()],int(fold[idx[0]])))
 body=pd.DataFrame(body_records,columns=['body_id','true','pred','fold']);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 d[['record_id','body_id_x','cell_type']].rename(columns={'body_id_x':'body_id'}).assign(fold=fold,predicted=pred).to_csv(out/'instance_oof.csv',index=False)
 body.to_csv(out/'body_oof.csv',index=False)
 result={'dataset':'LC21 results instances','instances':int(len(x)),'bodies':int(body.body_id.nunique()),'classes':classes.tolist(),'body_disjoint_folds':5,'seed':20261113,
  'instance':{'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro'))},
  'body':{'accuracy':float(accuracy_score(body.true,body.pred)),'balanced_accuracy':float(balanced_accuracy_score(body.true,body.pred)),'macro_f1':float(f1_score(body.true,body.pred,average='macro'))},
  'confusion_labels':classes.tolist(),'body_confusion':confusion_matrix(body.true,body.pred,labels=classes).tolist(),
  'fold_body_counts':body.groupby(['fold','true']).body_id.nunique().unstack(fill_value=0).to_dict()}
 (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
