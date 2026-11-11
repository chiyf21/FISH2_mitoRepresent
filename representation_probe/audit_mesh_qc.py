"""Diagnose strict manifold QC flags without changing any primary subject set."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from skimage.measure import marching_cubes
from common import *
def main(out):
    out=Path(out);rows=load_rows();diagnostics=[];volume_checks=[]
    for line in (out/'morphology.jsonl').read_text().splitlines():
        q=json.loads(line)
        if q['ok']:continue
        r=rows[q['cohort_index']];mask=np.asarray(np.load(r['mask_path'],mmap_mode='r'),dtype=bool)
        count=int(mask.sum());assert count==int(r['metadata']['voxel_count_actual'])
        if 'volume_nm3' not in q:
            volume_checks.append(dict(stable_key=q['stable_key'],voxel_count=count,metadata_match=True,shape=list(mask.shape)))
            continue
        occ=[np.flatnonzero(np.any(mask,axis=tuple(j for j in range(3) if j!=k))) for k in range(3)]
        tight=mask[tuple(slice(v[0],v[-1]+1) for v in occ)]
        vertices,faces,_,_=marching_cubes(np.pad(tight,1).astype(np.float32),level=.5,spacing=(30.,8.,8.),step_size=1,method='lewiner',allow_degenerate=False)
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        _,cnt=np.unique(edges,axis=0,return_counts=True)
        diagnostics.append(dict(stable_key=q['stable_key'],boundary_edges=int((cnt==1).sum()),
            nonmanifold_edges=int((cnt>2).sum()),maximum_edge_incidence=int(cnt.max())))
    save(out/'degenerate_volume_checks.json',volume_checks)
    save(out/'mesh_qc_diagnostics.json',dict(instances=diagnostics,summary=dict(flagged_meshes=len(diagnostics),
      with_boundary_edges=sum(q['boundary_edges']>0 for q in diagnostics),with_nonmanifold_edges=sum(q['nonmanifold_edges']>0 for q in diagnostics),
      degenerate_covariance=len(volume_checks)),interpretation='primary geometry probes conservatively exclude every strict manifold-QC flag; topology flags are not independent proof of segmentation error'))
    print('mesh diagnostics',len(diagnostics),'degenerate volume checks',len(volume_checks))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
