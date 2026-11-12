"""Prepare a private surface cache; never mutate the source dataset."""
import csv, json, hashlib
from pathlib import Path
import numpy as np

def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)

def load_rows(path):
    path=Path(path)
    if path.suffix=='.json':return json.loads(path.read_text(encoding='utf-8'))
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t' if path.suffix=='.tsv' else ','))

def source_rows(cfg):
    mode=cfg['input']['kind'];p=Path(cfg['input']['path'])
    if mode=='cache':
        rows=load_rows(p/'manifest.tsv');arr=np.load(p/'pointclouds.npy',mmap_mode='r',allow_pickle=False)
        if arr.ndim!=4 or arr.shape[1:]!=(2,2048,3) or len(rows)!=len(arr):raise ValueError('Cache must be [N,2,2048,3] with matching manifest.tsv')
        indices=list(range(len(rows)))
        if cfg['input'].get('cohort'):
            cohort=load_rows(cfg['input']['cohort']);indices=[int(r.get('point_row',r.get('point_index',-1))) for r in cohort]
            if len(indices)!=len(set(indices)) or not indices or min(indices)<0 or max(indices)>=len(rows):raise ValueError('Invalid/duplicate source point rows in cohort')
        return [(i,dict(rows[i])) for i in indices],arr
    raise ValueError('Only existing point-cloud caches are accepted: input.kind must be cache')

def prepare(cfg, out, limit=None):
    out=Path(out)
    if (out/'dataset_manifest.json').exists():raise FileExistsError(f'Prepared data exists: {out}; use resume, not overwrite')
    rows,source=source_rows(cfg)
    if limit and len(rows)>limit:
        # Include multiple bodies even when the source cache is sorted by body.
        first={}
        for j,(_,r) in enumerate(rows):first.setdefault(str(r['body_id']),j)
        chosen=list(first.values())[:min(limit,3)]
        remaining=np.setdiff1d(np.arange(len(rows)),chosen)
        chosen+=np.random.default_rng(cfg['seed']).choice(remaining,limit-len(chosen),replace=False).tolist()
        rows=[rows[j] for j in chosen]
    if not rows:raise ValueError('Empty input')
    out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(cfg['seed']);manifest=[]
    points=np.lib.format.open_memmap(out/'pointclouds.npy',mode='w+',dtype='float32',shape=(len(rows),2,2048,3))
    for j,(source_row,row) in enumerate(rows):
        xyz=np.asarray(source[source_row],dtype=np.float32)
        if not np.isfinite(xyz).all() or float(row.get('normalization_radius_voxels',0))<=0:raise ValueError(f'Invalid source row {source_row}')
        points[j]=xyz
        row.update(instance_index=j,source_point_row=source_row)
        row.setdefault('surface_voxel_count',2048);row.setdefault('cell_type','unknown')
        row.setdefault('instance_id',row.get('global_instance',source_row))
        manifest.append(row)
    points.flush();del points
    fields=sorted(set().union(*(r.keys() for r in manifest)))
    with (out/'manifest.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(manifest)
    write_json(out/'instances.json',manifest)
    write_json(out/'dataset_manifest.json',dict(instances=len(rows),array_file='pointclouds.npy',array_shape=[len(rows),2,2048,3],dtype='float32',point_sampling='uniform 6-neighbor voxel surface, two independent samples; inherited for cache input',spacing_xyz_nm=cfg['spacing_xyz_nm'],input=cfg['input'],seed=cfg['seed']))
    return manifest

def split_rows(rows, cfg):
    if any(row.get('split') for row in rows):
        assigned={name:[] for name in ('train','val','test')}
        bodies={}
        for index,row in enumerate(rows):
            split=row['split']
            body=str(row['body_id'])
            if split not in assigned or bodies.setdefault(body,split)!=split:
                raise ValueError(f'Invalid or conflicting split for body {body}')
            assigned[split].append(index)
        if not all(assigned.values()):
            raise ValueError('The provided split needs train, val and test rows')
        assigned['body_ids']={name:sorted(b for b,s in bodies.items() if s==name) for name in ('train','val','test')}
        return assigned
    groups={}
    for j,row in enumerate(rows):groups.setdefault(str(row['body_id']),[]).append(j)
    bodies=sorted(groups)
    if len(bodies)<3:raise ValueError('Need >=3 distinct body_id groups for train/val/test')
    rng=np.random.default_rng(cfg['seed']);rng.shuffle(bodies)
    v=max(1,round(len(bodies)*cfg['split']['val_fraction']));t=max(1,round(len(bodies)*cfg['split']['test_fraction']))
    if v+t>=len(bodies):raise ValueError('Requested split leaves no training body')
    assignments=dict(test=bodies[:t],val=bodies[t:t+v],train=bodies[t+v:])
    return {**{s:sorted(j for b in bs for j in groups[b]) for s,bs in assignments.items()},'body_ids':assignments}

def signature(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()

def coordinates(points, row, spacing, normalization):
    x=np.array(points,dtype=np.float32,copy=True)*float(row['normalization_radius_voxels'])*np.asarray(spacing,np.float32)
    x-=x.mean(0,keepdims=True)
    if normalization=='unit_sphere':x/=max(float(np.linalg.norm(x,axis=1).max()),1e-8)
    else:raise ValueError('Only unit_sphere is implemented; do not silently reinterpret physical coordinates')
    return x

def views(points, row, cfg, rng, jitter=None):
    result=[]
    for v in range(2):
        x=coordinates(points[v],row,cfg['spacing_xyz_nm'],cfg['normalization'])
        angle=rng.uniform(0,2*np.pi);c,s=np.cos(angle),np.sin(angle)
        x=x@np.array([[c,s,0],[-s,c,0],[0,0,1]],np.float32)
        if rng.random()<.5:x[:,0]*=-1
        if rng.random()<.5:x[:,1]*=-1
        noise=0.0 if jitter is None else jitter
        if noise:x+=np.clip(rng.normal(0,noise,x.shape),-3*noise,3*noise).astype(np.float32)
        result.append(x)
    return np.stack(result)
