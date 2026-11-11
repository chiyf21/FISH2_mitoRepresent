"""Verify prepared objects against the user's original instance packages."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from .data import point_support

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--original',required=True)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args();root=Path(a.cache);source=Path(a.original)
    doc=json.loads((root/'manifest.json').read_text());rows={r['record_id']:r for r in csv.DictReader((source/'metadata/instances.csv').open())}
    records=doc['records'][:a.limit] if a.limit else doc['records'];ids=set();errors=[];counts={};t0=time.time()
    for i,r in enumerate(records):
        try:
            assert r['record_id'] not in ids;ids.add(r['record_id']);row=rows[r['record_id']]
            assert r['body_id']==row['body_id'];base=root/r['relative_dir']
            em=np.load(base/'em.npy',mmap_mode='r');mask=np.load(base/'mask.npy',mmap_mode='r');valid=np.load(base/'valid.npy',mmap_mode='r')
            assert em.shape==mask.shape==valid.shape==tuple(r['shape_zyx'])
            lo=np.array(r['tight_lo_zyx']);size=np.array(r['tight_shape_zyx']);sl=tuple(slice(int(x),int(x+n)) for x,n in zip(lo,size))
            original_mask=np.load(source/row['mask_path'],mmap_mode='r');original_em=np.load(source/row['em_path'],mmap_mode='r')
            assert np.array_equal(mask[sl],original_mask)
            fg=original_mask!=0
            assert np.array_equal(em[sl][fg],original_em[fg]),'Original foreground EM differs'
            assert valid[sl][fg].all()
            t,rr,vox=point_support(mask.astype(bool),lo,size,np.array(r['spacing_zyx_nm']))
            counts[r['split']]=counts.get(r['split'],0)+1
        except Exception as e:errors.append({'record_id':r['record_id'],'error':str(e)})
        if i%1000==0:print(json.dumps({'checked':i+1,'errors':len(errors)}),flush=True)
    sets={k:set(v) for k,v in doc['body_splits'].items()}
    assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['test']&sets['val'])
    result={'checked':len(records),'errors':errors,'split_counts':counts,'seconds':time.time()-t0,
            'original_dataset':str(source),'foreground_raw_and_mask_match':not errors}
    (root/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if errors:raise RuntimeError(f'{len(errors)} audit failures; inspect audit.json')
if __name__=='__main__':main()
