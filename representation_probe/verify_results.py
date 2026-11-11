"""Data contracts and independently recomputed held-out metrics on actual artifacts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,f1_score,r2_score,mean_absolute_error
from common import *
def main(out,phase):
    out=Path(out);df=pd.read_csv(out/'instances.csv',dtype={'body_id':str});registry=json.loads((out/'feature_registry.json').read_text())
    order=hashlib.sha256('\n'.join(df.stable_key).encode()).hexdigest()
    audit=json.loads((out/'asset_audit.json').read_text())
    assert sha(X/'checkpoints/refine_epoch050.pt')==audit['checkpoint_sha256']
    assert sha(X/'cohort.json')==registry[0]['input_cohort_sha256']
    feature_qc=[]
    cp_hashes={q['checkpoint']:q.get('checkpoint_sha256') for q in registry}
    for initial in json.loads((X/'initial_weights/manifest.json').read_text()):
        path=str(X/initial['path'])
        if path in cp_hashes and cp_hashes[path] is not None:assert cp_hashes[path]==initial['sha256']
    target_table=pd.read_csv(out/'morphology.csv').set_index('stable_key') if (out/'morphology.csv').exists() else None
    protocol=json.loads((out/'protocol.json').read_text())
    for q in registry:
        assert q['stable_key_order_sha256']==order
        a=np.load(q['path'],mmap_mode='r');assert a.shape==(10678,q['dimension']) and np.isfinite(a).all()
        assert sha(q['path'])==q['feature_file_sha256']
        feature_qc.append(dict(name=q['name'],shape=list(a.shape),finite=True))
    cache_agreement={}
    e=out/'extraction/fusion50'
    for exported,cached in [('point_cls_l11_raw','point_11'),('point_cls_l12_raw','point_12'),
       ('em_cls_l11_raw','image_11'),('em_cls_l12_finalnorm','image_12')]:
        if (e/(exported+'.npy')).exists():
            a=np.load(e/(exported+'.npy'));b=np.load(CACHE/(cached+'.npy'))
            assert np.array_equal(a,b),cached
            cache_agreement[cached]='all 10678 rows exactly equal'
    folder=out/phase;scores=pd.read_csv(folder/'scores_long.csv');reference={};tested=0
    for _,q in scores.iterrows():
        p=folder/'predictions'/f'{q.feature}__{q.task}.npz'
        with np.load(p) as z:
            ids=z['cohort_index'];keys=z['stable_key'];y=z['y'];pred=z['pred'];body=z['body_id']
            assert np.array_equal(keys,df.loc[ids,'stable_key'].to_numpy(dtype=str))
            assert np.array_equal(body,df.loc[ids,'body_id'].to_numpy(dtype=str))
            assert len(set(keys))==len(keys)
            if q.task in reference:assert np.array_equal(keys,reference[q.task]),'different subjects across representations'
            else:reference[q.task]=keys
            if q.kind=='classification':
                assert abs(balanced_accuracy_score(y,pred)-q.balanced_accuracy)<1e-10
                assert abs(f1_score(y,pred,labels=z['classes'],average='macro',zero_division=0)-q.macro_f1)<1e-10
                assert np.allclose(z['probabilities'].sum(1),1)
                target='cell_type' if q.task=='celltype_six' else 'compartment'
                assert np.array_equal(y,df.loc[ids,target].to_numpy())
            else:
                assert abs(r2_score(y,pred)-q.r2)<1e-10
                assert abs(mean_absolute_error(y,pred)-q.mae)<1e-10
                source=df.set_index('stable_key')['volume_nm3'] if q.task=='volume_nm3' else target_table[q.task]
                true=np.array([source[k] for k in keys],float)
                if protocol['regression_targets'][q.task]=='ln':true=np.log(true)
                assert np.allclose(y,true,rtol=1e-12,atol=1e-12),'regression target identity mismatch'
            tested+=1
    schema=json.loads((folder/'folds.json').read_text())
    for task in schema['tasks'].values():
        for f in task['folds']:
            groups=[set(df.loc[f[s],'body_id']) for s in ['train','val','test']]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    if (out/'morphology.csv').exists():
        m=pd.read_csv(out/'morphology.csv')
        assert m.stable_key.is_unique and len(m)==10678
        valid=m[m.ok]
        assert valid.metadata_volume_match.all() and valid.mesh_closed.all()
        assert np.isfinite(valid[['volume_nm3','surface_nm2','sphericity','elongation','surface_to_volume_nm_inv']]).all().all()
    replay={}
    if phase=='final':
        assert set(scores.feature)==set(q['name'] for q in registry)|{'dummy','simple_morphology'}
        old=pd.read_csv(out/'phase1/scores_long.csv');merged=old.merge(scores,on=['feature','task'],suffixes=('_old','_new'))
        assert len(merged)==len(old)
        for metric in ['balanced_accuracy','body_balanced_accuracy','macro_f1','r2','mae','spearman']:
            if metric+'_old' not in merged:continue
            delta=(merged[metric+'_old']-merged[metric+'_new']).dropna()
            maximum=float(delta.abs().max()) if len(delta) else 0.
            assert maximum<1e-9,(metric,maximum)
            replay[metric]=maximum
    result=dict(status='PASS',feature_count=len(registry),features=feature_qc,metric_records_recomputed=tested,
      cache_agreement=cache_agreement,task_subjects={k:len(v) for k,v in reference.items()},
      body_leakage=False,stable_identity_verified=True,checkpoint_identity_verified=True,first_formal_phase_replay_max_abs=replay)
    save(folder/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='features'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--phase',default='final');a=p.parse_args();main(a.output,a.phase)
