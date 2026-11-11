"""Frozen representation probes with persisted group splits and train-only scaling."""
import argparse,hashlib,json,time,warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import balanced_accuracy_score,f1_score,classification_report,confusion_matrix,r2_score,mean_absolute_error,recall_score
from sklearn.exceptions import ConvergenceWarning
from common import *
def encoded(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def classification_metrics(y,p,prob,classes,bodies):
    out=dict(balanced_accuracy=float(balanced_accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=classes,average='macro',zero_division=0)),
      class_report=classification_report(y,p,labels=classes,output_dict=True,zero_division=0),
      confusion=confusion_matrix(y,p,labels=classes).tolist())
    counts=Counter(bodies);w=np.array([1/counts[b] for b in bodies])
    out['body_equal_balanced_accuracy']=float(recall_score(y,p,labels=classes,average='macro',sample_weight=w,zero_division=0))
    out['body_mean_instance_accuracy']=float(np.mean([np.mean(y[bodies==b]==p[bodies==b]) for b in sorted(counts)]))
    # Cell type is constant within a body. Mean calibrated class probabilities over instances.
    if all(len(set(y[bodies==b]))==1 for b in counts):
        bt=np.array([y[bodies==b][0] for b in sorted(counts)])
        bp=np.array([classes[np.argmax(prob[bodies==b].mean(0))] for b in sorted(counts)])
        out.update(body_balanced_accuracy=float(balanced_accuracy_score(bt,bp)),
          body_macro_f1=float(f1_score(bt,bp,labels=classes,average='macro',zero_division=0)),
          body_confusion=confusion_matrix(bt,bp,labels=classes).tolist(),
          body_class_report=classification_report(bt,bp,labels=classes,output_dict=True,zero_division=0))
    return out
def reg_metrics(y,p):
    corr=None if np.ptp(p)==0 else float(spearmanr(y,p).statistic)
    if corr is not None and not np.isfinite(corr):corr=None
    return dict(r2=float(r2_score(y,p)),mae=float(mean_absolute_error(y,p)),spearman=corr)
def make_tasks(df,protocol,smoke):
    tasks={}
    for name,allowed in [('compartment_binary',['axon','dendrite']),('compartment_three',['axon','dendrite','neurite_other'])]:
        ids=df.index[df.compartment.isin(allowed)].to_numpy()
        folds=[]
        parts={s:ids[df.loc[ids,'split'].to_numpy()==s] for s in ['train','val','test']}
        if smoke:
            rng=np.random.default_rng(SEED)
            parts={s:np.sort(rng.choice(v,min(len(v),120),replace=False)) for s,v in parts.items()}
        folds.append(dict(fold=0,**{s:v.tolist() for s,v in parts.items()}))
        tasks[name]=dict(kind='classification',target='compartment',classes=allowed,folds=folds)
    bt=df[['body_id','cell_type']].drop_duplicates().sort_values('body_id')
    counts=bt.cell_type.value_counts(); classes=sorted([str(t) for t,k in counts.items() if k>=5 and t!='unlabeled'])
    bt=bt[bt.cell_type.isin(classes)].reset_index(drop=True)
    folds=[]
    for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(bt.body_id,bt.cell_type)):
        bb=bt.iloc[tr]
        train_b,val_b=train_test_split(bb.body_id.to_numpy(),test_size=.2,stratify=bb.cell_type,random_state=SEED+f)
        parts={s:df.index[df.body_id.isin(b)].to_numpy() for s,b in [('train',train_b),('val',val_b),('test',bt.iloc[te].body_id)]}
        if smoke:
            parts={s:np.concatenate([v[df.loc[v,'body_id'].to_numpy()==b][:2] for b in sorted(set(df.loc[v,'body_id']))]) for s,v in parts.items()}
        folds.append(dict(fold=f,**{s:v.tolist() for s,v in parts.items()}))
    tasks['celltype_six']=dict(kind='classification',target='cell_type',classes=classes,folds=folds)
    for target,transform in protocol['regression_targets'].items():
        if target not in df:continue
        ids=df.index[np.isfinite(df[target]) & (df[target]>0)].to_numpy()
        if len(ids)==0:continue
        parts={s:ids[df.loc[ids,'split'].to_numpy()==s] for s in ['train','val','test']}
        if smoke:
            rng=np.random.default_rng(SEED)
            parts={s:np.sort(rng.choice(v,min(len(v),120),False)) for s,v in parts.items()}
        tasks[target]=dict(kind='regression',target=target,transform=transform,folds=[dict(fold=0,**{s:v.tolist() for s,v in parts.items()})])
    for task in tasks.values():
        for fold in task['folds']:
            sets=[set(df.loc[fold[s],'body_id']) for s in ['train','val','test']]
            assert not sets[0]&sets[1] and not sets[1]&sets[2] and not sets[0]&sets[2]
            if task['kind']=='classification':
                assert all(set(df.loc[fold[s],task['target']])==set(task['classes']) for s in ['train','val','test'])
    return tasks
def weights(df,ids,task):
    if task['target']!='cell_type':return None
    a=df.loc[ids]; nc=a.groupby('body_id').size(); nb=a[['body_id','cell_type']].drop_duplicates().cell_type.value_counts()
    w=np.array([1/(nc[b]*nb[t]) for b,t in zip(a.body_id,a.cell_type)],float)
    return w/w.mean()
def main(out,phase,smoke,features):
    out=Path(out);df=pd.read_csv(out/'instances.csv',dtype={'body_id':str})
    if (out/'morphology.csv').exists():
        morph=pd.read_csv(out/'morphology.csv',dtype={'body_id':str})
        assert morph.stable_key.is_unique
        # All target joins use stable keys, with an explicit one-to-one cardinality check.
        cols=['stable_key','surface_nm2','sphericity','elongation','surface_to_volume_nm_inv']
        if not morph.ok.all():morph=morph[morph.ok].copy()
        joined=df[['stable_key']].merge(morph[cols],on='stable_key',how='left',validate='one_to_one',sort=False)
        assert np.array_equal(df.stable_key,joined.stable_key)
        for c in cols[1:]:df[c]=joined[c].to_numpy()
        v=df[['stable_key','volume_nm3']].merge(morph[['stable_key','volume_nm3']],on='stable_key',validate='one_to_one',suffixes=('_a','_b'))
        assert np.array_equal(v.volume_nm3_a,v.volume_nm3_b)
    protocol=json.loads((out/'protocol.json').read_text());tasks=make_tasks(df,protocol,smoke)
    folder=out/('smoke' if smoke else phase);folder.mkdir(exist_ok=True);(folder/'predictions').mkdir(exist_ok=True)
    schema=dict(protocol=protocol,tasks=tasks,stable_keys_sha256=encoded(df.stable_key.tolist()))
    if (folder/'folds.json').exists():assert json.loads((folder/'folds.json').read_text())==schema
    else:save(folder/'folds.json',schema)
    # Shared tasks must retain exactly the same subjects/folds across phases.
    if not smoke and phase!='phase1' and (out/'phase1/folds.json').exists():
        old=json.loads((out/'phase1/folds.json').read_text())
        for k,t in old['tasks'].items():assert tasks[k]==t, k
    registry=json.loads((out/'feature_registry.json').read_text())
    if features:registry=[r for r in registry if r['name'] in features.split(',')]
    if smoke:registry=registry[:2]
    candidates=[dict(name='dummy'),dict(name='simple_morphology')]+registry
    summaries=[];t0=time.time();evaluation_sha=sha(__file__)
    coverage={}
    for name,task in tasks.items():
        support=[]
        for f in task['folds']:
            support.append(dict(fold=f['fold'],**{s:dict(instances=len(f[s]),bodies=df.loc[f[s],'body_id'].nunique(),
              classes=df.loc[f[s],task['target']].value_counts().to_dict() if task['kind']=='classification' else None) for s in ['train','val','test']}))
        coverage[name]=support
    save(folder/'task_coverage.json',coverage)
    for feature in candidates:
        name=feature['name']
        raw=None if name in ['dummy','simple_morphology'] else np.load(feature['path'],mmap_mode='r')
        if raw is not None:assert len(raw)==len(df) and np.isfinite(raw).all()
        for taskname,task in tasks.items():
            identity=encoded(dict(schema=schema,task=taskname,feature=feature,evaluation_code_sha256=evaluation_sha))
            resultfile=folder/f'{name}__{taskname}.json';predfile=folder/'predictions'/f'{name}__{taskname}.npz'
            if resultfile.exists():
                result=json.loads(resultfile.read_text());assert result['identity']==identity
            else:
                ys=[];ps=[];probs=[];testids=[];foldids=[];foldresults=[]
                for fold in task['folds']:
                    tr,va,te=[np.array(fold[s],int) for s in ['train','val','test']]
                    if name=='simple_morphology':
                        cols=['bbox_z_nm','bbox_y_nm','bbox_x_nm']
                        if task['target']!='volume_nm3':cols=['volume_nm3']+cols
                        # No area/sphericity/elongation or deterministic equivalents of regression targets are input.
                        mat=np.log(df[cols].to_numpy(float))
                    elif name=='dummy':mat=np.zeros((len(df),1))
                    else:mat=np.asarray(raw,dtype=np.float64)
                    scaler=StandardScaler().fit(mat[tr])
                    a,b,c=[scaler.transform(mat[i]) for i in [tr,va,te]]
                    assert np.allclose(a.mean(0),0,atol=5e-5)
                    target=df[task['target']].to_numpy()
                    if task['kind']=='classification':
                        classes=np.array(task['classes']);ytr,yva,yte=[target[i].astype(str) for i in [tr,va,te]]
                        best=None;trace=[];w=weights(df,tr,task)
                        grid=[None] if name=='dummy' else protocol['classification_C']
                        for reg in grid:
                            if reg is None:model=DummyClassifier(strategy='most_frequent')
                            else:model=LogisticRegression(C=reg,solver='lbfgs',max_iter=10000,tol=1e-4,
                               class_weight='balanced' if task['target']=='compartment' else None,random_state=SEED)
                            with warnings.catch_warnings(record=True) as ws:
                                warnings.simplefilter('always',ConvergenceWarning)
                                model.fit(a,ytr,sample_weight=None if name=='dummy' else w)
                            if any(issubclass(q.category,ConvergenceWarning) for q in ws):raise RuntimeError('Logistic probe failed convergence')
                            pv=model.predict_proba(b); order=[list(model.classes_).index(k) for k in classes];pv=pv[:,order]
                            pred=classes[pv.argmax(1)]
                            vm=classification_metrics(yva,pred,pv,classes,df.loc[va,'body_id'].to_numpy())
                            score=vm['body_balanced_accuracy'] if task['target']=='cell_type' else vm['balanced_accuracy']
                            trace.append(dict(C=reg,validation_score=score))
                            if best is None or score>best[0]+1e-12:best=(score,reg,model,order)
                        score,reg,model,order=best
                        pp=model.predict_proba(c)[:,order];pred=classes[pp.argmax(1)]
                        metrics=classification_metrics(yte,pred,pp,classes,df.loc[te,'body_id'].to_numpy())
                        probs.append(pp)
                    else:
                        transformed=np.log(target.astype(float)) if task['transform']=='ln' else target.astype(float)
                        ytr,yva,yte=[transformed[i] for i in [tr,va,te]];best=None;trace=[]
                        grid=[None] if name=='dummy' else protocol['ridge_alpha']
                        for reg in grid:
                            model=DummyRegressor(strategy='mean') if reg is None else Ridge(alpha=reg,solver='cholesky')
                            model.fit(a,ytr);score=float(r2_score(yva,model.predict(b)));trace.append(dict(alpha=reg,validation_score=score))
                            if best is None or score>best[0]+1e-12:best=(score,reg,model)
                        score,reg,model=best;pred=model.predict(c);metrics=reg_metrics(yte,pred)
                    ys.append(yte);ps.append(pred);testids.extend(te.tolist());foldids.extend([fold['fold']]*len(te))
                    foldresults.append(dict(fold=fold['fold'],hyperparameter=reg,validation_trace=trace,metrics=metrics,
                      n_train=len(tr),n_val=len(va),n_test=len(te),bodies_train=df.loc[tr,'body_id'].nunique(),
                      bodies_val=df.loc[va,'body_id'].nunique(),bodies_test=df.loc[te,'body_id'].nunique(),
                      scaler_training_rows_sha256=encoded(df.loc[tr,'stable_key'].tolist()),
                      scaler_mean_sha256=hashlib.sha256(scaler.mean_.tobytes()).hexdigest()))
                y=np.concatenate(ys);pred=np.concatenate(ps);testids=np.array(testids,int);bodies=df.loc[testids,'body_id'].to_numpy()
                assert len(np.unique(testids))==len(testids)
                if task['kind']=='classification':
                    pp=np.concatenate(probs);metrics=classification_metrics(y,pred,pp,np.array(task['classes']),bodies)
                    extra=dict(probabilities=pp,classes=np.array(task['classes']))
                else:metrics=reg_metrics(y,pred);extra={}
                np.savez_compressed(predfile,y=y,pred=pred,cohort_index=testids,stable_key=df.loc[testids,'stable_key'].to_numpy(dtype=str),
                    body_id=bodies.astype(str),fold=np.array(foldids),**extra)
                result=dict(identity=identity,feature=name,task=taskname,kind=task['kind'],dimension=mat.shape[1],
                    transform=task.get('transform'),metrics=metrics,folds=foldresults,n_test=len(testids),bodies_test=len(set(bodies)),
                    feature_provenance=feature,predictions=str(predfile))
                if len(foldresults)>1:
                    metric='body_balanced_accuracy' if task['kind']=='classification' else 'r2'
                    values=[f['metrics'][metric] for f in foldresults]
                    result['fold_metric_mean']=float(np.mean(values));result['fold_metric_std']=float(np.std(values,ddof=1))
                save(resultfile,result)
            flat={k:result[k] for k in ['feature','task','kind','dimension','n_test','bodies_test']}
            flat.update({k:v for k,v in result['metrics'].items() if isinstance(v,(float,int)) or v is None})
            if 'fold_metric_mean' in result:flat.update(fold_metric_mean=result['fold_metric_mean'],fold_metric_std=result['fold_metric_std'])
            summaries.append(flat)
            pd.DataFrame(summaries).to_csv(folder/'scores_long.csv',index=False)
            print(json.dumps(dict(feature=name,task=taskname,metrics={k:v for k,v in flat.items() if k in ['balanced_accuracy','body_balanced_accuracy','macro_f1','r2']},elapsed=time.time()-t0)),flush=True)
    scores=pd.DataFrame(summaries);wide=[]
    for feature,g in scores.groupby('feature',sort=False):
        r=dict(feature=feature)
        for _,q in g.iterrows():
            metric='body_balanced_accuracy' if q.task=='celltype_six' else ('balanced_accuracy' if q.kind=='classification' else 'r2')
            r[q.task+'__'+metric]=q[metric]
        wide.append(r)
    pd.DataFrame(wide).to_csv(folder/'performance.csv',index=False)
    save(folder/'complete.json',dict(features=len(candidates),results=len(summaries),elapsed_seconds=time.time()-t0,smoke=smoke,protocol_sha256=sha(out/'protocol.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--phase',default='phase1')
    p.add_argument('--smoke',action='store_true');p.add_argument('--features',default='');a=p.parse_args();main(a.output,a.phase,a.smoke,a.features)
