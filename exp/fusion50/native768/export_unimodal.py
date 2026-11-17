"""Export normalized L12 CLS features for Fish2 reviewed81 or LC21 instances."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np,torch,torch.distributed as dist
from train import load_models,load_points,records
import core

FISH2=Path('/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/neuron_mito_dataset/embeddings/fish2_reviewed_81/instances')
LC21=Path('/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/flyem_lc21_source_instances_20261110/results')

def dataset_rows(name):
    rows=[]
    if name=='fish2':
        with (FISH2/'instance_paths.csv').open() as f:
            for j,r in enumerate(csv.DictReader(f)):
                rows.append(dict(cohort_index=j,record_id=r['record_id'],body_id=str(r['body_id']),
                    em_path=str(FISH2/r['em_path']),mask_path=str(FISH2/r['mask_path'])))
    else:
        for nf in sorted(LC21.glob('*/neuron_metadata.json')):
            neuron=json.loads(nf.read_text()); body=str(neuron['bodyId'])
            for mf in sorted((nf.parent/'instances').glob('instance_*/metadata.json')):
                md=json.loads(mf.read_text()); folder=mf.parent
                rows.append(dict(cohort_index=len(rows),record_id=f"flyem_lc21_source_instances_20261110__{body}__{int(md['global_instance']):06d}",
                    body_id=body,cell_type=md['cell_type'],em_path=str(folder/'em.npy'),mask_path=str(folder/'mask.npy')))
    if not rows:raise ValueError(f'No rows for dataset {name}')
    for r in rows:
        if not Path(r['em_path']).is_file() or not Path(r['mask_path']).is_file():raise FileNotFoundError(r)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dataset',choices=('fish2','lc21'),required=True)
    ap.add_argument('--modality',choices=('point','image'),required=True);ap.add_argument('--batch-size',type=int,default=2)
    a=ap.parse_args()
    if a.modality=='point' and a.dataset!='fish2':raise ValueError('LC21 export currently uses image CLS only')
    cfg=json.loads(a.config.read_text());ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    phase='unimodal_'+a.modality
    if ck.get('stage')!=phase:raise ValueError(f'Expected {phase} checkpoint')
    rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'));world=int(os.getenv('WORLD_SIZE','1'))
    torch.cuda.set_device(local);dev=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl')
    rows=dataset_rows(a.dataset);point,group,image=load_models(cfg,dev)
    model=core.System(point,image).to(dev)
    result=model.load_state_dict(ck['model'],strict=False)
    if result.unexpected_keys:raise ValueError(f'Unexpected checkpoint keys: {result.unexpected_keys[:5]}')
    prefix='point.' if a.modality=='point' else 'image.'
    clskey='point_cls' if a.modality=='point' else 'image.cls_token'
    if clskey not in ck['model'] or not any(k.startswith(prefix+'blocks.8.') for k in ck['model']):
        raise ValueError('Checkpoint lacks trained CLS or block-9 weights')
    model.phase=phase;model.eval()
    pointdata=None
    cohort=None
    if a.modality=='point':
        cohort=records(cfg);lookup={r['record_id']:r for r in cohort}; ordered=[]
        with (FISH2/'instance_paths.csv').open() as f:
            for r in csv.DictReader(f):
                if r['record_id'] not in lookup:raise ValueError(f"Unmatched record_id {r['record_id']}")
                q=lookup[r['record_id']]
                if q['body_id']!=r['body_id']:raise ValueError(f"Body mismatch {r['record_id']}")
                ordered.append(q)
        rows=ordered;pointdata=load_points(cfg,cohort)
    out=a.output.resolve()
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    if world>1:dist.barrier()
    indexed=list(enumerate(rows));local_rows=indexed[rank::world];features=[]
    with torch.inference_mode():
        for start in range(0,len(local_rows),a.batch_size):
            batch=local_rows[start:start+a.batch_size];ix=[x[0] for x in batch];rr=[x[1] for x in batch]
            inputs=core.prepare(rr,pointdata,point,group,image,dev,0,False,
                point_only=(a.modality=='point'),image_only=(a.modality=='image'))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p,i,_,_=model(inputs,views=1,loss=False)
            values=(p if a.modality=='point' else i)[12][:,0].float().cpu().numpy()
            if not np.isfinite(values).all():raise FloatingPointError(f'Nonfinite rows {ix[0]}..{ix[-1]}')
            features.append((ix,values))
    if features:
        index=np.concatenate([v[0] for v in features]); matrix=np.concatenate([v[1] for v in features])
    else:index=np.empty(0,dtype=int);matrix=np.empty((0,768),dtype=np.float32)
    np.savez(out/f'rank{rank}.npz',index=index,matrix=matrix)
    if world>1:dist.barrier()
    if rank==0:
        full=np.empty((len(rows),768),dtype=np.float32)
        for rnk in range(world):
            part=np.load(out/f'rank{rnk}.npz');full[part['index']]=part['matrix']
        np.save(out/'image_or_point_cls_l12.npy',full)
        with (out/'instance_map.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['embedding_row','record_id','body_id'])
            for j,r in enumerate(rows):w.writerow([j,r['record_id'],r['body_id']])
        if a.dataset=='lc21':
            with (out/'labels.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['record_id','body_id','cell_type'])
                for r in rows:w.writerow([r['record_id'],r['body_id'],r['cell_type']])
        print(json.dumps(dict(dataset=a.dataset,modality=a.modality,instances=len(rows),shape=list(full.shape),output=str(out)),flush=True))
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
