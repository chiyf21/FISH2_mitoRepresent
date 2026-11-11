"""Validate existing caches and write stable-key labels/registry, never row-join datasets."""
import argparse, csv, json, hashlib, platform, time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
def main(out):
    out=Path(out); rows=load_rows(); n=len(rows)
    assert n==10678
    keys=[key(r) for r in rows]; assert len(set(keys))==n
    pt=pd.read_csv(F/'point_cache/manifest.tsv',sep='\t',dtype={'body_id':str})
    assert len(pt)==n
    labels=json.loads((D/'compartment/cohort_compartments.json').read_text())
    bodytypes=defaultdict(set); records=[]
    for j,r in enumerate(rows):
        q=pt.iloc[int(r['point_row'])]
        assert str(q.body_id)==str(r['body_id']) and int(q.global_instance)==int(r['key'][2])
        assert str(q.mask_path)==r['mask_path'] and str(q.em_path)==r['em_path']
        assert int(q.voxel_count_actual)==int(r['metadata']['voxel_count_actual'])
        assert r['mask_path'].startswith(str(F/'filled_masks')+'/') or (int(r['metadata']['hole_fill_audit']['added_voxels'])==0 and r['mask_path']==r['metadata']['original_mask_path'])
        m=r['metadata']; bodytypes[str(r['body_id'])].add(m['cell_type'])
        shape=np.array(m['shape_zyx'],float)*np.array([30,8,8])
        record=dict(stable_key=key(r),cohort_index=j,body_id=str(r['body_id']),split=r['split'],
                    cell_type=m['cell_type'],compartment=labels.get(f"{r['body_id']}:{r['key'][2]}",'unknown'),
                    volume_nm3=int(m['voxel_count_actual'])*1920.,
                    bbox_z_nm=shape[0],bbox_y_nm=shape[1],bbox_x_nm=shape[2],
                    bbox_aspect=shape.max()/shape.min(),mask_path=r['mask_path'])
        records.append(record)
    assert all(len(ts)==1 for ts in bodytypes.values())
    df=pd.DataFrame(records); df.to_csv(out/'instances.csv',index=False)
    split_bodies={s:set(df.loc[df.split==s,'body_id']) for s in ['train','val','test']}
    assert all(not split_bodies[a]&split_bodies[b] for a,b in [('train','val'),('val','test'),('train','test')])
    ids=np.load(D/'cache/18f11251ca1129e59d16/ids.npy')
    req=json.loads((D/'cache/18f11251ca1129e59d16/request.json').read_text())
    cp=Path(req['checkpoint']); final=X/'checkpoints/refine_epoch050.pt'
    assert cp.samefile(final)
    fk=hashlib.sha256(json.dumps([str(X),50,8050,cp.stat().st_ino,ids.tolist()]).encode()).hexdigest()[:20]
    assert fk=='b9a9cbbcd15a81c16c36' and np.array_equal(ids,np.arange(n))
    import torch,sklearn,scipy,skimage
    ck=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
    assert (ck['stage'],ck['epoch'],ck['step'])==('refine',50,8050)
    init=torch.load(X/'initial_weights/point_mae_epoch100.pt',map_location='cpu',mmap=True,weights_only=False)
    assert init['epoch']==100
    registry=[]; checks={}
    with np.load(X/'refine_epoch50_probe.npz') as probe:
        for mod,dim in [('point',384),('image',768),('balanced_pair',1152),('fused',384)]:
            for l in [9,10,11,12]:
                name=f'{mod}_{l}'; path=CACHE/(name+'.npy')
                a=np.load(path,mmap_mode='r',allow_pickle=False)
                assert a.shape==(n,dim) and np.isfinite(a).all()
                if name in probe:
                    diff=np.asarray(a[probe['ids']],float)-np.asarray(probe[name],float)
                    checks[name]=dict(probe_max_abs=float(abs(diff).max()),probe_relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(probe[name]),1e-12)))
                registry.append(dict(name='fusion50_'+name,path=str(path),stage='Fusion50',
                  network_position=f'{mod} layer {l}',layer=l,dimension=dim,
                  checkpoint=str(final),checkpoint_stat=stat(final),
                  pooling='CLS' if mod in ['point','image'] else ('same-layer balanced CLS concat' if mod=='balanced_pair' else 'shared nonlinear MLP of balanced CLS'),
                  normalization=('EM final LayerNorm before CLS readout' if mod=='image' and l==12 else
                    'checkpoint train-only EMA balance' if mod=='balanced_pair' else
                    'modality LayerNorm, image Linear, shared MLP with final LayerNorm' if mod=='fused' else 'raw block output'),
                  input_protocol='core.prepare epoch0 training=False; point surface sample0, seeded per-instance FPS; EM minmax full crop, mask selects <=2048 tokens; no augmentation',
                  mask_version=str(F/'filled_masks'),ids_file=str(out/'instances.csv'),
                  stable_key_order_sha256=hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
                  input_cohort_sha256=sha(X/'cohort.json'),code=str(D/'worker.py'),
                  feature_file_sha256=sha(path),available=True))
    assert np.array_equal(np.load(CACHE/'fused_12.npy'),np.load(D/'cache/18f11251ca1129e59d16/embeddings.npy'))
    for l in [9,10,11,12]:
        p=np.load(CACHE/f'point_{l}.npy');i=np.load(CACHE/f'image_{l}.npy')
        state=ck['model'];mp=state[f'balance.mu_p_{l}'].numpy();mi=state[f'balance.mu_i_{l}'].numpy()
        vp,vi=state[f'balance.v_{l}'].numpy();t=(vp+vi)/2
        pair=np.concatenate([mp+np.sqrt(t/vp)*(p-mp),mi+np.sqrt(t/vi)*(i-mi)],1)
        assert np.allclose(pair,np.load(CACHE/f'balanced_pair_{l}.npy'),rtol=2e-5,atol=2e-5)
    del ck,init
    source_paths=[X/'cohort.json',X/'train.py',X/'code/core.py',X/'code/base.py',
       X/'config.json',X/'initialization_check.json',X/'initial_weights/manifest.json',
       X/'vendor/image_interface/code/instance_tokens.py',D/'worker.py',D/'compartment/label_compartments.py',
       D/'compartment/merge_compartments.py',D/'compartment/cohort_compartments.json',
       F/'point_cache/manifest.tsv',F/'point_cache/dataset_manifest.json',X/'manifests/split.json']
    counts={}
    for col in ['compartment','cell_type']:
        counts[col]=[dict(label=str(v),instances=len(g),bodies=g.body_id.nunique(),
          split_counts=g.groupby('split').size().to_dict(),split_bodies=g.groupby('split').body_id.nunique().to_dict())
          for v,g in df.groupby(col)]
    canonical=pd.read_csv(R.parents[1]/'neuron_mito_dataset/metadata/instances.csv',dtype={'body_id':str})
    save(out/'feature_registry.json',registry)
    save(out/'asset_audit.json',dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      cache_feature_key=fk,checkpoint=stat(final),checkpoint_sha256=sha(final),
      point_initialization_epoch=100,point_best_epoch_is_not_initialization=98,
      cohort_count=n,canonical_count=len(canonical),canonical_bodies=canonical.body_id.nunique(),
      split={s:dict(instances=int((df.split==s).sum()),bodies=len(bs)) for s,bs in split_bodies.items()},
      counts=counts,cache_monitor_comparison=checks,source_files=[dict(**stat(p),sha256=sha(p)) for p in source_paths],
      software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                    torch=torch.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,skimage=skimage.__version__),
      compartment_status='algorithm-derived weak labels; no independent GT validation; physical mapping assumptions require caution',
      existing_morphology_status='paper_replication surface_nm2 uses voxel exposed-face count; not reused as mesh area'))
    protocol=dict(seed=SEED,classification_C=[0.01,0.1,1.,10.],
      ridge_alpha=[0.1,1.,10.,100.,1000.],primary_split='original train/val/test body partition; do not refit on val',
      celltype='original label classes with >=5 independent bodies; outer5 body-stratified folds, inner body-stratified 20% holdout',
      standardization='fit StandardScaler only on training instances per fit',
      classifier='L2 logistic; compartment class-balanced; celltype inverse body-size and class-body-frequency weighting',
      regression_targets={'volume_nm3':'ln','surface_nm2':'ln','sphericity':'identity','elongation':'ln','surface_to_volume_nm_inv':'ln'},
      selection='compartment balanced accuracy; celltype body-level balanced accuracy; regression validation R2',
      numerical_precision='float64 StandardScaler/probe inputs; L2 logistic lbfgs max_iter10000 tol1e-4; require convergence',
      bootstrap='1000 paired cluster resamples by test body; descriptive uncertainty conditional on trained probes',
      scope='downstream readability of existing SSL models; EM pretraining exposed validation/test bodies')
    save(out/'protocol.json',protocol)
    print(json.dumps(dict(status='prepared',features=len(registry),counts=counts,probe_checks=checks),ensure_ascii=False),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();main(a.output)
