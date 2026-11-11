"""Copy 5000 unaugmented native instances with volume-stratified selection."""
import concurrent.futures
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp

ROOT=Path('/path/to/project_data/EM/point-MAE/runs/v1_ablation_5000_20261018')
SOURCE=Path('/path/to/project_data/EM/mito_voxel_mae_prepared_v2_20261015')
RAW=Path('/path/to/project_data/EM/neuron_mito_dataset')


def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def allocate(n,probability,capacity):
    ideal=n*probability
    quota=np.minimum(np.floor(ideal).astype(int),capacity)
    while quota.sum()<n:
        candidates=np.flatnonzero(quota<capacity)
        if not len(candidates):raise ValueError('Insufficient source capacity')
        j=max(candidates,key=lambda i:(ideal[i]-quota[i],-i))
        quota[j]+=1
    return quota


def main():
    target=ROOT/'data';target.mkdir(parents=True,exist_ok=True)
    assert not (target/'manifest.json').exists(),'Already exported; inspect rather than resample'
    reference=json.loads((SOURCE/'manifest.json').read_text())
    rows=list(csv.DictReader((RAW/'metadata/instances.csv').open()))
    raw_by_id={r['record_id']:r for r in rows}
    records=reference['records']
    assert len(records)==19947
    volume=np.array([int(r['voxel_count']) for r in rows])
    logv=np.log10(volume)
    edges=np.unique(np.quantile(logv,np.linspace(0,1,21)))
    edges[0]=-np.inf;edges[-1]=np.inf
    bin_of=lambda n:int(np.searchsorted(edges,np.log10(n),side='right')-1)
    counts=np.bincount([bin_of(int(r['voxel_count'])) for r in rows],minlength=len(edges)-1)
    probability=counts/counts.sum()
    train_count=sum(r['split']=='train' for r in records)
    val_count=sum(r['split']=='val' for r in records)
    nval=round(5000*val_count/(train_count+val_count));ntrain=5000-nval
    assert (train_count,val_count,ntrain,nval)==(15508,1953,4441,559)
    rng=np.random.default_rng(20261018)
    chosen=[];quotas={}
    for split,n in (('train',ntrain),('val',nval)):
        groups=[[] for _ in probability]
        for r in records:
            if r['split']==split:groups[bin_of(int(raw_by_id[r['record_id']]['voxel_count']))].append(r)
        quota=allocate(n,probability,np.array([len(g) for g in groups]))
        quotas[split]=quota.tolist()
        for group,k in zip(groups,quota):
            chosen.extend(group[int(i)] for i in rng.choice(len(group),int(k),replace=False))
    chosen.sort(key=lambda r:r['record_id'])
    assert len(chosen)==5000 and len({r['record_id'] for r in chosen})==5000
    split_bodies={s:sorted({r['body_id'] for r in chosen if r['split']==s}) for s in ('train','val')}
    assert not set(split_bodies['train'])&set(split_bodies['val'])
    atomic(target/'selection_plan.json',{'seed':20261018,'selected_ids':[r['record_id'] for r in chosen],
           'original_train_val':[train_count,val_count],'selected_train_val':[ntrain,nval],
           'sampling':'20 quantile strata of log10 target foreground voxel count; match all19947 distribution within each existing train/val split',
           'quota':quotas,'bin_edges_log10':[None if not np.isfinite(e) else float(e) for e in edges]})
    started=time.time()
    def copy_one(record):
        raw=raw_by_id[record['record_id']]
        source=SOURCE/record['relative_dir'];dest=target/'instances'/record['record_id']
        dest.mkdir(parents=True,exist_ok=True)
        hashes={}
        for name in ('em.npy','mask.npy','valid.npy'):
            a=source/name;b=dest/name
            shutil.copy2(a,b)
            assert a.stat().st_size==b.stat().st_size
            # Streamed hashes verify every exported byte, without retaining huge
            # volumes twice in memory. Original files are only read.
            hashes[name]=hashlib.file_digest(b.open('rb'),'sha256').hexdigest()
            assert hashes[name]==hashlib.file_digest(a.open('rb'),'sha256').hexdigest()
        mask=np.load(dest/'mask.npy',mmap_mode='r')
        assert int(mask.sum(dtype=np.int64))==int(raw['voxel_count'])
        lo=np.array(record['tight_lo_zyx']);shape=np.array(record['tight_shape_zyx'])
        sl=tuple(slice(int(a),int(a+b)) for a,b in zip(lo,shape))
        original=np.load(RAW/raw['mask_path'],mmap_mode='r')
        assert np.array_equal(mask[sl],original)
        original_em=np.load(RAW/raw['em_path'],mmap_mode='r')
        exported_em=np.load(dest/'em.npy',mmap_mode='r')
        fg=np.asarray(original,dtype=bool)
        assert np.array_equal(exported_em[sl][fg],original_em[fg])
        result={**record,'relative_dir':str(dest.relative_to(target)),
                'foreground_voxels':int(raw['voxel_count']),'volume_um3':int(raw['voxel_count'])*1920/1e9,
                'sha256':hashes,'source_cache':str(source),'augmentation':'none'}
        atomic(dest/'prepared.json',result)
        return result
    exported=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(copy_one,chosen),1):
            exported.append(r)
            if i%100==0:print(json.dumps({'exported':i,'total':5000,'seconds':time.time()-started}),flush=True)
    manifest={'schema':1,'records':exported,'body_splits':{**split_bodies,'test':[]},
              'preparation':{'dataset':str(RAW),'source_cache':str(SOURCE),'no_augmentation':True,'seed':20261018}}
    atomic(target/'manifest.json',manifest)
    selected=np.array([r['foreground_voxels'] for r in exported])
    def stats(a):return {'n':len(a),'quantiles':dict(zip(['min','p01','p10','p50','p90','p99','max'],np.quantile(a,[0,.01,.1,.5,.9,.99,1]).tolist()))}
    summary={'original':stats(volume),'selected':stats(selected),'split_instances':{'train':ntrain,'val':nval},
             'split_bodies':{s:len(v) for s,v in split_bodies.items()},
             'ks_distance_all_original_vs_selected':float(ks_2samp(volume,selected).statistic),
             'split_ks':{s:float(ks_2samp(volume,[r['foreground_voxels'] for r in exported if r['split']==s]).statistic) for s in ('train','val')},
             'export_seconds':time.time()-started,'byte_hash_and_mask_checks_passed':5000,
             'note':'Original test bodies were not sampled. No size cutoff; no resampling; uint8 native EM/mask and original validity retained.'}
    atomic(target/'summary.json',summary)
    with (target/'instances.csv').open('w',newline='') as f:
        fields=('record_id','body_id','split','foreground_voxels','volume_um3','relative_dir')
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in exported)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    bins=np.linspace(logv.min(),logv.max(),36)
    for a,label in ((volume,'Original19947'),(selected,'Selected5000')):
        axs[0].hist(np.log10(a),bins=bins,density=True,histtype='step',linewidth=1.5,label=label)
        ordered=np.sort(np.log10(a));axs[1].plot(ordered,np.arange(1,len(a)+1)/len(a),label=label)
    axs[0].set_ylabel('Density');axs[1].set_ylabel('ECDF')
    for ax in axs:ax.set_xlabel('log10 foreground voxel count');ax.legend();ax.grid(alpha=.2)
    fig.savefig(target/'volume_distribution.png',dpi=180);plt.close(fig)
    atomic(target/'completed.json',{'instances':5000,'verified':True})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
