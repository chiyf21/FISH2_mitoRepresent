"""Read-only source ingestion; recover real raw context and explicit validity."""
from __future__ import annotations
import argparse, concurrent.futures, csv, hashlib, json, math, os, re, time
from pathlib import Path
import numpy as np
from scipy import ndimage

SPACING=np.array([30.,8.,8.])
SCHEMA=1

def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)

def source_index(root,body):
    directory=root/body/'blocks'
    paths={}
    for p in directory.iterdir():
        m=re.fullmatch(r'x(-?\d+)_y(-?\d+)_z(-?\d+)',p.name)
        # Directory names already encode the grid; avoid thousands of slow per-file
        # stat calls. Missing intersecting blocks are caught on actual np.load.
        if m: paths[tuple(map(int,m.groups()))]=p/'em.npy'
    if not paths: raise FileNotFoundError(f'No raw blocks: {directory}')
    origins=np.array(list(paths),dtype=np.int64)
    first=np.load(next(iter(paths.values())),mmap_mode='r')
    return origins,list(paths.values()),np.array(first.shape)

def source_stats(index,seed=42):
    origins,paths,shape=index
    # Match reference percentile clipping, using a reproducible source-block sample.
    # Full source collection can contain thousands of blocks: cap the statistics IO.
    rng=np.random.default_rng(seed); chosen=np.sort(rng.choice(len(paths),min(4,len(paths)),replace=False))
    hist=np.zeros(256,dtype=np.int64)
    for i in chosen:
        a=np.load(paths[i],mmap_mode='r')
        sample=np.asarray(a[::4,::4,::4]).reshape(-1)
        if len(sample)>32768:sample=sample[::math.ceil(len(sample)/32768)]
        hist+=np.bincount(sample.astype(np.uint8),minlength=256)
    cum=hist.cumsum(); lo=float(np.searchsorted(cum,cum[-1]*.005)); hi=float(np.searchsorted(cum,cum[-1]*.998))
    return {'lo':lo,'hi':hi,'method':'source-body raw block sample, p0.5/p99.8; base stride4; max4 blocks x32768 values','n_samples':int(hist.sum()),'n_source_blocks':len(paths),'seed':seed}

def assemble(index,lo_zyx,hi_zyx):
    origins,paths,size=index
    lo=lo_zyx[::-1]; hi=hi_zyx[::-1]
    em=np.zeros(tuple(hi_zyx-lo_zyx),dtype=np.uint8); valid=np.zeros_like(em,dtype=bool)
    use=np.flatnonzero(((origins<hi)&(origins+size>lo)).all(1))
    for i in use:
        origin=origins[i]; a=np.load(paths[i],mmap_mode='r')
        end=origin+np.array(a.shape)
        start=np.maximum(lo,origin); stop=np.minimum(hi,end)
        src=tuple(slice(int(x),int(y)) for x,y in zip(start-origin,stop-origin))
        dst=tuple(slice(int(x),int(y)) for x,y in zip((start-lo)[::-1],(stop-lo)[::-1]))
        em[dst]=np.asarray(a[src]).transpose(2,1,0);valid[dst]=True
    return em,valid

def prepare_body(body,rows,args):
    root=Path(args.dataset); cache=Path(args.out); results=[]; errors=[]
    try:
        index=source_index(Path(args.raw_root),body)
        stats_path=cache/'statistics'/f'{body}.json'
        stats=json.loads(stats_path.read_text()) if stats_path.exists() else source_stats(index)
        if not stats_path.exists():atomic_json(stats_path,stats)
    except Exception as e:
        return [],[{'record_id':r['record_id'],'error':f'raw_source: {e}'} for r in rows]
    for row in rows:
        try:
            target=cache/'instances'/row['record_id'];meta=target/'prepared.json'
            if meta.exists():
                record=json.loads(meta.read_text())
                if record['schema']!=SCHEMA or record['max_context_voxels']!=args.max_voxels:raise ValueError('Existing cache configuration differs')
                results.append(record);continue
            mask=np.load(root/row['mask_path'],mmap_mode='r')
            if mask.ndim!=3 or not mask.any(): raise ValueError('empty/non3D mask')
            bbox=np.array(json.loads(row['bbox_xyz']),dtype=np.int64)
            native_lo=bbox[:3][::-1];native_shape=(bbox[3:]-bbox[:3])[::-1]
            if tuple(native_shape)!=mask.shape:raise ValueError('bbox/shape mismatch')
            margin=np.ceil(native_shape*(2/17)).astype(int)+2
            shape=native_shape+2*margin
            if math.prod(map(int,shape))>args.max_voxels:
                raise ValueError(f'context exceeds initial native-resolution voxel budget: {math.prod(map(int,shape))}')
            em,valid=assemble(index,native_lo-margin,native_lo+native_shape+margin)
            extended=np.zeros_like(em,dtype=np.uint8)
            sl=tuple(slice(int(m),int(m+n)) for m,n in zip(margin,native_shape))
            extended[sl]=mask
            if np.any((extended!=0)&~valid):raise ValueError('raw missing under foreground')
            if int(extended.sum())!=int(row['voxel_count']):raise ValueError('foreground count mismatch')
            target.mkdir(parents=True,exist_ok=True)
            # NPY memmaps let training sample native target voxels without decompression.
            np.save(target/'em.npy',em);np.save(target/'mask.npy',extended);np.save(target/'valid.npy',valid)
            record={'schema':SCHEMA,'record_id':row['record_id'],'body_id':body,'cell_type':row['cell_type'],
                    'relative_dir':str(target.relative_to(cache)),'shape_zyx':list(map(int,shape)),
                    'tight_lo_zyx':list(map(int,margin)),'tight_shape_zyx':list(map(int,native_shape)),
                    'spacing_zyx_nm':SPACING.tolist(),'lo':stats['lo'],'hi':stats['hi'],
                    'max_context_voxels':args.max_voxels,'foreground_voxels':int(extended.sum()),
                    'missing_context_voxels':int((~valid).sum()),'source_status':row['source_neuron_status'],
                    'source_em':row['em_path'],'source_mask':row['mask_path'],'bbox_xyz':bbox.tolist()}
            atomic_json(meta,record);results.append(record)
        except Exception as e:errors.append({'record_id':row['record_id'],'body_id':body,'error':str(e)})
    return results,errors

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--raw-root',required=True)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--max-voxels',type=int,default=1500000);p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'statistics').mkdir(exist_ok=True);(out/'instances').mkdir(exist_ok=True)
    rows=list(csv.DictReader((Path(args.dataset)/'metadata/instances.csv').open()))
    all_bodies=sorted({r['body_id'] for r in rows})
    if args.limit:
        rng=np.random.default_rng(42)
        rows=[rows[i] for i in sorted(rng.choice(len(rows),min(args.limit,len(rows)),replace=False))]
    grouped={}
    for row in rows:grouped.setdefault(row['body_id'],[]).append(row)
    records=[];errors=[];start=time.time()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures={pool.submit(prepare_body,body,rs,args):body for body,rs in grouped.items()}
        for f in concurrent.futures.as_completed(futures):
            good,bad=f.result();records.extend(good);errors.extend(bad)
            print(json.dumps({'body':futures[f],'prepared':len(records),'rejected':len(errors),'seconds':round(time.time()-start)}),flush=True)
    records.sort(key=lambda r:r['record_id']);bodies=all_bodies
    rng=np.random.default_rng(42);rng.shuffle(bodies);nval=max(1,round(.1*len(bodies)));ntest=nval
    splits={'test':bodies[:ntest],'val':bodies[ntest:ntest+nval],'train':bodies[ntest+nval:]}
    for record in records:record['split']=next(k for k,v in splits.items() if record['body_id'] in v)
    atomic_json(out/'manifest.json',{'schema':SCHEMA,'records':records,'body_splits':splits,'preparation':vars(args)})
    atomic_json(out/'rejections.json',errors)
    summary={'source_records':len(rows),'prepared_records':len(records),'rejected_records':len(errors),
             'split_instances':{k:sum(r['split']==k for r in records) for k in splits},
             'split_bodies':{k:len(v) for k,v in splits.items()},'max_context_voxels':args.max_voxels,
             'missing_context_instances':sum(r['missing_context_voxels']>0 for r in records),
             'elapsed_seconds':round(time.time()-start),'online_views_per_train_instance':4}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    if not records:raise RuntimeError('No valid training records prepared')

if __name__=='__main__':main()
