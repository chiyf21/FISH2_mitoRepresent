"""Paired test-body bootstrap of prespecified contrasts; no encoder retraining."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
def aligned(a,b):
    assert len(set(a['stable_key']))==len(a['stable_key'])
    lookup={k:i for i,k in enumerate(b['stable_key'])}
    order=np.array([lookup[k] for k in a['stable_key']])
    assert len(order)==len(b['stable_key'])
    bb={k:(v[order] if v.ndim and len(v)==len(order) and k!='classes' else v) for k,v in b.items()}
    assert np.array_equal(a['y'],bb['y']) and np.array_equal(a['body_id'],bb['body_id'])
    return bb
def bootstrap(a,b,kind,task,nboot,seed):
    bodies=np.array(sorted(set(a['body_id'])));g=a['body_id']
    rng=np.random.default_rng(seed);weights=rng.multinomial(len(bodies),np.full(len(bodies),1/len(bodies)),size=nboot)
    if kind=='regression':
        y=a['y'].astype(float);pa=a['pred'].astype(float);pb=b['pred'].astype(float)
        n=np.array([(g==q).sum() for q in bodies]);sy=np.array([y[g==q].sum() for q in bodies]);s2=np.array([(y[g==q]**2).sum() for q in bodies])
        ea=np.array([((y[g==q]-pa[g==q])**2).sum() for q in bodies]);eb=np.array([((y[g==q]-pb[g==q])**2).sum() for q in bodies])
        sst=weights@s2-(weights@sy)**2/(weights@n)
        delta=weights@(ea-eb)/sst
        point=float((((y-pa)**2).sum()-((y-pb)**2).sum())/((y-y.mean())**2).sum());metric='r2'
    else:
        classes=a['classes']
        if task=='celltype_six':
            y=np.array([a['y'][g==q][0] for q in bodies])
            pa=np.array([classes[np.argmax(a['probabilities'][g==q].mean(0))] for q in bodies])
            pb=np.array([classes[np.argmax(b['probabilities'][g==q].mean(0))] for q in bodies])
            n=np.stack([(y==c).astype(float) for c in classes],1)
            ta=np.stack([((y==c)&(pa==c)).astype(float) for c in classes],1)
            tb=np.stack([((y==c)&(pb==c)).astype(float) for c in classes],1)
            metric='body_balanced_accuracy'
        else:
            y=a['y'];pa=a['pred'];pb=b['pred']
            n=np.array([[np.sum((g==q)&(y==c)) for c in classes] for q in bodies])
            ta=np.array([[np.sum((g==q)&(y==c)&(pa==c)) for c in classes] for q in bodies])
            tb=np.array([[np.sum((g==q)&(y==c)&(pb==c)) for c in classes] for q in bodies])
            metric='balanced_accuracy'
        den=weights@n;keep=np.all(den>0,axis=1)
        delta=np.mean((weights[keep]@(tb-ta))/den[keep],axis=1)
        point=float(np.mean((tb-ta).sum(0)/n.sum(0)))
    delta=delta[np.isfinite(delta)]
    return dict(metric=metric,delta_second_minus_first=point,ci_low=float(np.quantile(delta,.025)),
        ci_high=float(np.quantile(delta,.975)),bootstrap_valid=len(delta),test_bodies=len(bodies),test_instances=len(a['y']))
def main(out,phase):
    out=Path(out);folder=out/phase;plan=json.loads((Path(__file__).parent/'comparisons.json').read_text())
    scores=pd.read_csv(folder/'scores_long.csv');available=set(scores.feature);contrasts=[];missing=[]
    for fa,fb in plan['pairs']:
        if fa not in available or fb not in available:missing.append([fa,fb]);continue
        tasks=sorted(set(scores.loc[scores.feature==fa,'task'])&set(scores.loc[scores.feature==fb,'task']))
        for task in tasks:
            pa=folder/'predictions'/f'{fa}__{task}.npz';pb=folder/'predictions'/f'{fb}__{task}.npz'
            with np.load(pa) as z:a={k:z[k] for k in z.files}
            with np.load(pb) as z:b={k:z[k] for k in z.files}
            b=aligned(a,b);kind=scores.loc[(scores.feature==fa)&(scores.task==task),'kind'].iloc[0]
            contrast=bootstrap(a,b,kind,task,plan['bootstrap_replicates'],plan['seed'])
            contrasts.append(dict(first=fa,second=fb,task=task,**contrast))
    pd.DataFrame(contrasts).to_csv(folder/'paired_differences.csv',index=False)
    save(folder/'paired_analysis.json',dict(plan=plan,contrasts=len(contrasts),missing_features=missing,
       uncertainty='conditional held-out-body cluster bootstrap; exploratory, not multiplicity corrected; no retraining across bootstrap replicates'))
    if (out/'morphology.csv').exists():
        d=pd.read_csv(out/'morphology.csv')
        cols=['volume_nm3','surface_nm2','sphericity','elongation','surface_to_volume_nm_inv']
        d.loc[d.ok,cols].corr(method='spearman').to_csv(folder/'target_spearman.csv')
    print(json.dumps(dict(contrasts=len(contrasts),missing_pairs=missing)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--phase',default='final');a=p.parse_args();main(a.output,a.phase)
