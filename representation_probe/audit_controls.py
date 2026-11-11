"""Reproducible metadata, frozen-front, input-scaling and full-cohort geometry checks."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import *
def main(out):
    out=Path(out);rows=load_rows();reg=json.loads((out/'feature_registry.json').read_text());by={q['name']:q for q in reg}
    split_a=json.loads((F/'training/split.json').read_text());split_b=json.loads((X/'manifests/split.json').read_text())
    assert all(split_a[k]==split_b[k] for k in ['train','val','test','body_ids'])
    source_changes=[]
    for q in json.loads((out/'asset_audit.json').read_text())['source_files']:
        if sha(q['path'])!=q['sha256']:source_changes.append(q['path'])
    assert not source_changes,source_changes
    initial=torch.load(X/'initial_weights/point_mae_epoch100.pt',map_location='cpu',mmap=True,weights_only=False)
    ck=torch.load(X/'checkpoints/pretrain_epoch000.pt',map_location='cpu',mmap=True,weights_only=False)
    compared=0
    for k,v in ck['model'].items():
        if k.startswith('point.'):
            assert torch.equal(v,initial['model']['MAE_encoder.'+k[len('point.'):]]);compared+=1
    assert ck['epoch']==0 and ck['step']==0 and initial['epoch']==100
    best=torch.load(F/'training/mae/best.pt',map_location='cpu',mmap=True,weights_only=False);assert best['epoch']==98
    del ck,initial,best
    spec=importlib.util.spec_from_file_location('audit_runtime_base',X/'code/base.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    points=np.load(F/'point_cache/pointclouds.npy',mmap_mode='r')
    rescaling=[]
    for j in [0,1,50,500,5000,10000]:
        r=rows[j];other=dict(r,normalization_radius_voxels=2*r['normalization_radius_voxels'])
        a=base.point_xyz(r,points,0,torch.device('cpu')).numpy();b=base.point_xyz(other,points,0,torch.device('cpu')).numpy()
        assert np.array_equal(a,b)
        rescaling.append(dict(cohort_index=j,max_abs=float(abs(a-b).max())))
    front=[]
    for mod,names in [('point',['pointmae100_point_group_mean_pre_pe','nna0_point_group_mean_pre_pe','nna20_point_group_mean_pre_pe','fusion50_point_group_mean_pre_pe']),
      ('em',['em_base_em_patch_mean_pre_pe','fusion50_em_patch_mean_pre_pe'])]:
        a=np.load(by[names[0]]['path'])
        for name in names[1:]:
            b=np.load(by[name]['path']);assert np.array_equal(a,b)
            front.append(dict(modality=mod,first=names[0],second=name,all_10678_equal=True))
    labels=json.loads((D/'compartment/cohort_compartments.json').read_text())
    body_qc=[]
    for p in sorted((D/'compartment/compartment_labels').glob('*.json')):
        q=json.loads(p.read_text())
        if q.get('error') or q.get('body_id') not in {r['body_id'] for r in rows}:continue
        body_qc.append(dict(body_id=q['body_id'],**q.get('stats',{})))
    pd.DataFrame(body_qc).to_csv(out/'compartment_body_qc.csv',index=False)
    geometry=[]
    selected=['fusion50_point_12','fusion50_image_11','fusion50_image_12','fusion50_em_cls_l12_raw',
      'fusion50_em_mean_l11_raw','fusion50_em_mean_l12_raw','fusion50_em_mean_l12_finalnorm',
      'fusion50_fused_12','fusion50_balanced_pair_11']
    for name in selected:
        a=np.load(by[name]['path']).astype(np.float64);a-=a.mean(0)
        ev=np.maximum(np.linalg.eigvalsh(a.T@a/(len(a)-1)),0);s=ev.sum();q=ev[ev>0]/s
        geometry.append(dict(feature=name,instances=len(a),dimension=a.shape[1],raw_effective_rank=float(np.exp(-(q*np.log(q)).sum())),
          total_variance=float(s),top_pc_share=float(ev[-1]/s)))
    pd.DataFrame(geometry).to_csv(out/'representation_geometry.csv',index=False)
    save(out/'controls_audit.json',dict(status='PASS',point_split_equals_nna_split=True,nna0_point_tensors_identical_to_mae100=compared,
      input_rescaling_test=rescaling,frozen_front_comparisons=front,source_changes=source_changes,
      compartment_bodies_qc=len(body_qc),arbor_conflict_bodies=sum(q.get('arbor_conflict',False) for q in body_qc),
      geometry_scope='descriptive raw covariance on all10678; no transforms fitted here are used by supervised probes'))
    import inspect,timm
    from timm.models.vision_transformer import Block,Attention
    bp=Path(inspect.getsourcefile(Block));ap=Path(inspect.getsourcefile(Attention))
    save(out/'attention_implementation.json',dict(timm_version=timm.__version__,block_file=str(bp),block_file_sha256=sha(bp),attention_file=str(ap),attention_file_sha256=sha(ap),block_forward=inspect.getsource(Block.forward),attention_forward=inspect.getsource(Attention.forward),mask_path='valid[:,None,None,:] -> Block -> Attention -> SDPA; True denotes allowed keys; pooling excludes padded queries'))
    print(json.dumps(dict(status='PASS',nna0_matching_tensors=compared,front_checks=len(front),label_bodies=len(body_qc))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
