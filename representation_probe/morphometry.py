"""Exact voxel volume/moments and physical Lewiner mesh targets, incrementally."""
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import marching_cubes, mesh_surface_area
from common import *
SPACING=np.array([30.,8.,8.]) # ZYX nm
def measure(r):
    p=Path(r['mask_path']);before=stat(p);raw=np.load(p,mmap_mode='r',allow_pickle=False)
    if raw.ndim!=3 or not np.isfinite(raw).all() or np.any((raw!=0)&(raw!=1)):raise ValueError('nonbinary/nonfinite/3D')
    mask=np.asarray(raw,dtype=bool); n=int(mask.sum())
    if n<4:raise ValueError('degenerate mask')
    occupied=[np.flatnonzero(np.any(mask,axis=tuple(a for a in range(3) if a!=k))) for k in range(3)]
    lo=np.array([v[0] for v in occupied]);hi=np.array([v[-1]+1 for v in occupied])
    tight=mask[tuple(slice(a,b) for a,b in zip(lo,hi))]
    # Exact coordinate moments in slices: bounded memory even for large instances.
    summ=np.zeros(3);cross=np.zeros((3,3))
    for z in range(tight.shape[0]):
        y,x=np.nonzero(tight[z])
        if not len(x):continue
        xyz=np.column_stack([np.full(len(x),z),y,x])*SPACING
        summ+=xyz.sum(0);cross+=xyz.T@xyz
    cov=cross/n-np.outer(summ/n,summ/n)
    eig=np.linalg.eigvalsh(cov)
    if eig[0]<=0 or not np.isfinite(eig).all():raise ValueError('degenerate physical covariance')
    # One layer of exterior zeros closes all crop-edge surfaces at the half-voxel isosurface.
    field=np.pad(tight,1).astype(np.float32)
    vertices,faces,_,_=marching_cubes(field,level=.5,spacing=tuple(SPACING),step_size=1,method='lewiner',allow_degenerate=False)
    area=float(mesh_surface_area(vertices,faces))
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,cnt=np.unique(edges,axis=0,return_counts=True);closed=bool(np.all(cnt==2))
    tri=vertices[faces].astype(np.float64)
    meshvol=float(abs(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6))
    volume=n*float(np.prod(SPACING))
    psi=float(np.pi**(1/3)*(6*volume)**(2/3)/area)
    assert stat(p)==before,'source mask changed during read'
    return dict(stable_key=r['stable_key'],cohort_index=int(r['cohort_index']),body_id=str(r['body_id']),
      volume_nm3=volume,surface_nm2=area,sphericity=psi,elongation=float(np.sqrt(eig[-1]/eig[0])),
      surface_to_volume_nm_inv=area/volume,bbox_z_nm=float((hi-lo)[0]*30),bbox_y_nm=float((hi-lo)[1]*8),
      bbox_x_nm=float((hi-lo)[2]*8),mesh_closed=closed,mesh_vertices=len(vertices),mesh_faces=len(faces),
      mesh_volume_nm3=meshvol,mesh_to_voxel_volume=meshvol/volume,voxel_count=n,
      metadata_volume_match=bool(volume==float(r['volume_nm3'])),touches_crop=bool((lo==0).any() or (hi==mask.shape).any()),
      source_mask_stat=before,ok=bool(closed and area>0 and np.isfinite([area,psi,meshvol]).all()))
def main(out,limit):
    out=Path(out);df=pd.read_csv(out/'instances.csv',dtype={'body_id':str});rows=df.to_dict('records')
    if limit:rows=rows[:limit]
    dest=out/('morphology_smoke.jsonl' if limit else 'morphology.jsonl')
    done={}
    if dest.exists():
        for line in dest.read_text().splitlines():
            try:
                q=json.loads(line);done[q['stable_key']]=q
            except json.JSONDecodeError:raise RuntimeError('truncated cache; repair only this owned output before resuming')
    t=time.time()
    with dest.open('a',buffering=1) as h:
        for r in rows:
            k=r['stable_key']
            if k in done:
                if done[k].get('ok'):assert done[k]['source_mask_stat']==stat(r['mask_path'])
                continue
            try:q=measure(r)
            except Exception as e:q=dict(stable_key=k,cohort_index=int(r['cohort_index']),body_id=str(r['body_id']),ok=False,error=repr(e))
            h.write(json.dumps(q,allow_nan=False)+'\n');done[k]=q
            if len(done)%100==0:print(json.dumps(dict(done=len(done),total=len(rows),elapsed_sec=time.time()-t)),flush=True)
    result=pd.DataFrame([done[r['stable_key']] for r in rows])
    result.to_csv(out/('morphology_smoke.csv' if limit else 'morphology.csv'),index=False)
    numeric=['volume_nm3','surface_nm2','sphericity','elongation','surface_to_volume_nm_inv','mesh_to_voxel_volume']
    qc=dict(count=len(result),ok=int(result.ok.sum()),errors=result.loc[~result.ok,['stable_key','error']].to_dict('records') if 'error' in result else [],
      summary={c:dict(finite=int(np.isfinite(result[c]).sum()),min=float(result[c].min()),max=float(result[c].max()),median=float(result[c].median())) for c in numeric},
      metadata_volume_mismatches=int((~result.metadata_volume_match.fillna(False)).sum()),
      sphericity_above_one=int((result.sphericity>1).sum()),nonclosed=int((~result.mesh_closed.fillna(False)).sum()),
      protocol=dict(spacing_zyx_nm=SPACING.tolist(),volume='occupied voxel count * 1920 nm^3',
        area='skimage Lewiner marching cubes level0.5 step1, no degenerate triangles, physical spacing, one exterior-zero voxel pad',
        elongation='sqrt(largest/smallest eigenvalue of occupied voxel-center physical covariance)',
        target_mask='training filled masks; no canonical replacement',crop_boundary='zero pad closes observed crop; cannot recover physical truncation beyond original acquisition'))
    save(out/('morphology_smoke_qc.json' if limit else 'morphology_qc.json'),qc)
    print(json.dumps(qc),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--limit',type=int,default=0);q=a.parse_args();main(q.output,q.limit)
