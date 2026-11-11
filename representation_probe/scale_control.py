"""Auxiliary incremental morphology readability after a train-fitted log-volume control."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from common import *
SELECTED=['fusion50_point_12','fusion50_image_11','fusion50_image_12',
 'fusion50_balanced_pair_11','fusion50_balanced_pair_12','fusion50_fused_12',
 'fusion50_point_mean_l11_raw','fusion50_em_mean_l11_raw']
def main(out,phase):
    out=Path(out);folder=out/phase;df=pd.read_csv(out/'instances.csv',dtype={'body_id':str})
    morph=pd.read_csv(out/'morphology.csv')
    d=df.merge(morph[['stable_key','sphericity','elongation']],on='stable_key',validate='one_to_one',sort=False)
    assert np.array_equal(d.stable_key,df.stable_key)
    folds=json.loads((folder/'folds.json').read_text())['tasks']
    registry={q['name']:q for q in json.loads((out/'feature_registry.json').read_text())}
    grid=json.loads((out/'protocol.json').read_text())['ridge_alpha'];records=[]
    for target,transform in [('sphericity','identity'),('elongation','ln')]:
        f=folds[target]['folds'][0];tr,va,te=[np.array(f[s],int) for s in ['train','val','test']]
        y=d[target].to_numpy(float)
        if transform=='ln':y=np.log(y)
        covariate=np.column_stack([np.ones(len(d)),np.log(d.volume_nm3.to_numpy(float))])
        coef=np.linalg.lstsq(covariate[tr],y[tr],rcond=None)[0];base=covariate@coef;residual=y-base
        baseline=r2_score(y[te],base[te])
        for name in SELECTED:
            if name not in registry:continue
            x=np.load(registry[name]['path']).astype(np.float64)
            loading=np.linalg.lstsq(covariate[tr],x[tr],rcond=None)[0]
            xr=x-covariate@loading
            scaler=StandardScaler().fit(xr[tr]);a,b,c=[scaler.transform(xr[k]) for k in [tr,va,te]]
            best=None;trace=[]
            for alpha in grid:
                model=Ridge(alpha=alpha,solver='cholesky').fit(a,residual[tr])
                score=float(r2_score(y[va],base[va]+model.predict(b)));trace.append(dict(alpha=alpha,validation_r2=score))
                if best is None or score>best[0]+1e-12:best=(score,alpha,model)
            pred=base[te]+best[2].predict(c)
            sse=float(((y[te]-pred)**2).sum());ssebase=float(((y[te]-base[te])**2).sum())
            records.append(dict(feature=name,target=target,scale=transform,alpha=best[1],volume_only_r2=float(baseline),
               augmented_r2=float(r2_score(y[te],pred)),increment_r2=float(r2_score(y[te],pred)-baseline),
               partial_r2_relative_to_volume=1-sse/ssebase,mae=float(mean_absolute_error(y[te],pred)),
               n_test=len(te),test_bodies=d.loc[te,'body_id'].nunique(),validation_trace=trace))
    pd.DataFrame([{k:v for k,v in r.items() if k!='validation_trace'} for r in records]).to_csv(folder/'scale_control.csv',index=False)
    save(folder/'scale_control.json',dict(method='Frisch-Waugh residualization: fit target and every embedding channel on intercept+ln(volume) using train only; train-standardize residual channels; Ridge residual target; validation selects alpha; add volume prediction on test',
        targets='sphericity identity and elongation ln',features_predefined=SELECTED,records=records))
    print('scale controls',len(records))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--phase',default='final');a=p.parse_args();main(a.output,a.phase)
