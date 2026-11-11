"""Merge completed prepared records without copying or modifying source arrays."""
import argparse,csv,json,time,concurrent.futures
from pathlib import Path
import numpy as np
from .prepare import atomic_json
from .data import point_support

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--original',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();source=Path(a.original)
    rows={r['record_id']:r for r in csv.DictReader((source/'metadata/instances.csv').open())}
    selected={};errors=[];start=time.time()
    def inspect(meta,input_root):
        try:
            r=json.loads(meta.read_text());base=meta.parent
            if r['record_id'] not in rows:raise ValueError('Not in original dataset')
            row=rows[r['record_id']]
            em=np.load(base/'em.npy',mmap_mode='r');mask=np.load(base/'mask.npy',mmap_mode='r');valid=np.load(base/'valid.npy',mmap_mode='r')
            if em.shape!=mask.shape or em.shape!=valid.shape:raise ValueError('shape mismatch')
            lo=np.array(r['tight_lo_zyx']);size=np.array(r['tight_shape_zyx'])
            sl=tuple(slice(int(x),int(x+n)) for x,n in zip(lo,size))
            original_mask=np.load(source/row['mask_path'],mmap_mode='r');original_em=np.load(source/row['em_path'],mmap_mode='r')
            fg=original_mask!=0
            if not np.array_equal(mask[sl],original_mask):raise ValueError('Original mask differs')
            if not np.array_equal(em[sl][fg],original_em[fg]):raise ValueError('Original foreground EM differs')
            if not valid[sl][fg].all():raise ValueError('missing foreground raw')
            point_support(mask.astype(bool),lo,size,np.array(r['spacing_zyx_nm']))
            r['relative_dir']=str(base.resolve());r['prepared_source']=str(input_root)
            return r,None
        except Exception as e:return None,{'metadata':str(meta),'error':str(e)}
    for input_root in map(Path,a.inputs):
        # Also handle scp placing the source instances directory directly at its destination.
        if (input_root/'manifest.json').exists():
            source_doc=json.loads((input_root/'manifest.json').read_text())
            paths=[input_root/r['relative_dir']/'prepared.json' for r in source_doc['records']]
        else:paths=list(input_root.glob('*/prepared.json'))+list(input_root.glob('instances/*/prepared.json'))
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            futures=[pool.submit(inspect,meta,input_root) for meta in paths]
            for i,f in enumerate(concurrent.futures.as_completed(futures)):
                r,error=f.result()
                if error:errors.append(error)
                else:selected[r['record_id']]=r
                if (i+1)%1000==0:print(json.dumps({'audited':i+1,'unique':len(selected),'errors':len(errors),'seconds':round(time.time()-start)}),flush=True)
        print(json.dumps({'input':str(input_root),'unique_prepared':len(selected),'errors':len(errors)}),flush=True)
    bodies=sorted({row['body_id'] for row in rows.values()});np.random.default_rng(42).shuffle(bodies)
    n=max(1,round(.1*len(bodies)));splits={'test':bodies[:n],'val':bodies[n:n*2],'train':bodies[n*2:]}
    records=sorted(selected.values(),key=lambda r:r['record_id'])
    for r in records:r['split']=next(k for k,v in splits.items() if r['body_id'] in v)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Choose a new merged version to preserve training cohort')
    atomic_json(out/'manifest.json',{'schema':1,'records':records,'body_splits':splits,'original_dataset':str(source),'inputs':a.inputs})
    summary={'source_instances':len(rows),'unique_instances':len(records),'not_in_this_cohort':len(rows)-len(records),
        'split_instances':{k:sum(r['split']==k for r in records) for k in splits},
        'split_bodies_available':{k:len({r['body_id'] for r in records if r['split']==k}) for k in splits},
        'train_views_per_epoch':4*sum(r['split']=='train' for r in records),'validation_limit_for_training':0,
        'source_foreground_consistency_checked':True,'rejected_cache_records':len(errors),'seconds':time.time()-start}
    atomic_json(out/'summary.json',summary);atomic_json(out/'merge_errors.json',errors)
    atomic_json(out/'not_in_this_cohort.json',[row['record_id'] for row in rows.values() if row['record_id'] not in selected])
    print(json.dumps(summary),flush=True)
    if not all(summary['split_instances'][s]>0 for s in ('train','val','test')):raise RuntimeError('Missing split; prepare more bodies before training')
if __name__=='__main__':main()
