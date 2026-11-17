"""Extend a verified native Fusion50 export to a larger instance manifest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def mount_shards(rows, points, out, cards):
    from fish2_downstream.export_primary_native import dump
    ranges = np.array_split(np.arange(len(rows)),len(cards))
    directories = []
    for start, indices in enumerate(ranges):
        d = out / f'gpu_shard_{start:02d}'
        d.mkdir()
        os.link(points, d/'pointclouds.npy')
        dump(d/'inputs.json',[rows[j] for j in indices])
        directories.append((d, indices))
    return directories


def run_exports(directories, cards, checkpoint):
    busy = subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],
        text=True).splitlines()
    rows = subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],
        text=True)
    gpus = {i:(uuid,int(mem)) for i,uuid,mem in
            (map(str.strip,row.split(',')) for row in rows.splitlines())}
    for gpu in cards:
        uuid, mem = gpus[gpu]
        if uuid in busy or mem > 256:
            raise RuntimeError(f'Physical GPU {gpu} is occupied; choose idle GPUs')
    processes = []
    for gpu,(d,_) in zip(cards,directories):
        env = os.environ | dict(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        argv = [sys.executable,'-Bu','fish2_downstream/export_primary_native.py',
                '--panel',str(d),'--reference',str(checkpoint.parent.parent),
                '--output',str(d),'--batch-size','8',str(checkpoint)]
        with (d/'export.log').open('w') as log:
            processes.append(subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,
                                              stdout=log,stderr=subprocess.STDOUT))
    failures = [(d.name,rc) for (d,_),proc in zip(directories,processes)
                if (rc := proc.wait()) != 0]
    if failures:
        raise RuntimeError(f'Shard export failed: {failures}')


def embed_stage(args):
    from fish2_downstream.export_primary_native import READOUTS, prepare, sha
    a = args
    manifest = read(a.dataset/'dataset_manifest.json')
    for fn, expected in manifest['sha256'].items():
        if sha(a.dataset/fn) != expected:
            raise ValueError(f'Source changed: {fn}')
    dtypes = {'body_id':str}
    full = pd.read_csv(a.dataset/'instances.csv',dtype=dtypes)
    prior = pd.read_csv(a.reuse/'instance_map.csv',dtype=dtypes)
    assert full.record_id.is_unique and prior.record_id.is_unique
    assert len(full)==manifest['instance_count'] and full.body_id.nunique()==manifest['neuron_count']
    assert set(prior.record_id) <= set(full.record_id)
    check=['body_id','cell_type','source_run_id','global_instance','em_path','mask_path',
           'voxel_count','bbox_xyz','array_axis_order','voxel_size_xyz_nm']
    pd.testing.assert_frame_equal(
        full.set_index('record_id').loc[prior.record_id,check],prior.set_index('record_id')[check])
    reuse_run = a.reuse / a.checkpoint.parent.parent.name
    receipt = read(reuse_run/'complete.json')
    assert receipt['sha256']==sha(a.checkpoint)
    assert receipt['dimensions']==READOUTS and receipt['instances']==len(prior)
    assert not receipt['balance'] and receipt['epoch']==50

    full['embedding_row'] = np.arange(len(full))
    old_row = full.record_id.map(prior.set_index('record_id').embedding_row)
    reuse_mask = old_row.notna().to_numpy()
    old_indices = old_row[reuse_mask].to_numpy(dtype=int)
    extra_indices = np.flatnonzero(~reuse_mask)
    full['inference_source'] = np.where(reuse_mask,'primary_panel_reused','new_inference')
    cohort = read(a.checkpoint.parent.parent/'cohort.json')
    body_split = {r['body_id']:r['split'] for r in cohort}
    full['refine_split'] = full.body_id.map(body_split).fillna('external')
    full['record_in_refine_cohort'] = full.record_id.isin({r['key'][2] for r in cohort})

    novel = a.output/'novel'
    novel.mkdir()
    data = novel/'dataset'
    data.mkdir()
    extra = full.iloc[extra_indices].reset_index(drop=True)
    extra.to_csv(data/'instances.csv',index=False)
    write(data/'dataset_manifest.json',dict(instance_count=len(extra),neuron_count=extra.body_id.nunique(),
                                           sha256={'instances.csv':sha(data/'instances.csv')}))
    full.to_csv(a.output/'instance_map.csv',index=False)
    cache = novel/'cache'
    cache.mkdir()
    print('Preparing additional instances',len(extra),flush=True)
    prepare(data,a.checkpoint.parent.parent,a.checkpoint,cache,a.workers)
    dirs = mount_shards(read(cache/'inputs.json'),cache/'pointclouds.npy',novel,a.gpus)
    run_exports(dirs,a.gpus,a.checkpoint)

    dest = a.output/'embeddings'
    dest.mkdir()
    for readout,dim in READOUTS.items():
        x = np.lib.format.open_memmap(dest/f'{readout}.partial.npy',mode='w+',dtype='float32',
                                      shape=(len(full),dim))
        reused = np.load(reuse_run/f'{readout}.npy',mmap_mode='r')
        assert reused.shape==(len(prior),dim)
        x[reuse_mask] = reused[old_indices]
        for directory,indices in dirs:
            exported = directory/a.checkpoint.parent.parent.name
            assert read(exported/'complete.json')['sha256']==receipt['sha256']
            values = np.load(exported/f'{readout}.npy',mmap_mode='r')
            assert values.shape==(len(indices),dim)
            x[extra_indices[indices]] = values
        assert np.isfinite(x).all()
        x.flush()
        del x
        (dest/f'{readout}.partial.npy').replace(dest/f'{readout}.npy')
    write(a.output/'embeddings_complete.json',dict(
        dataset=manifest['dataset_id'],source_sha256=manifest['sha256'],instances=len(full),
        bodies=full.body_id.nunique(),reused=int(reuse_mask.sum()),new=len(extra),
        checkpoint=receipt['checkpoint'],checkpoint_sha256=receipt['sha256'],dimensions=READOUTS,
        row_alignment='zero-based embedding_row in instance_map.csv'))
    print('Embeddings verified',len(full),flush=True)


def fit_umap(output,key):
    from sklearn.decomposition import PCA
    from umap import UMAP
    import numba
    numba.set_num_threads(1)
    x=np.load(output/'embeddings'/f'{key}.npy',mmap_mode='r')
    pca=PCA(30,svd_solver='randomized',random_state=20261116)
    values=pca.fit_transform(x)
    params=dict(n_neighbors=30,min_dist=.1,metric='cosine',random_state=20261116,n_jobs=1)
    uv=UMAP(**params).fit_transform(values)
    assert uv.shape==(len(x),2) and np.isfinite(uv).all()
    np.save(output/f'UMAP_{key}.npy',uv,allow_pickle=False)
    write(output/f'UMAP_{key}.json',dict(pca30_variance=float(pca.explained_variance_ratio_.sum()),
                                       pca_dimensions=30,umap=params))
    print('UMAP ready',key,flush=True)


def plot_umaps(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    panel=pd.read_csv(a.output/'instance_map.csv',dtype={'body_id':str})
    readouts={'image':'Image CLS · 768D','point':'Point CLS · 768D',
              'concat':'Concat · 1536D','projection':'Projection · 384D'}
    env=os.environ | dict(CUDA_VISIBLE_DEVICES='',NUMBA_NUM_THREADS='1',
                          OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    jobs=[]
    for key in readouts:
        argv=[sys.executable,'-Bu','-m','fish2_downstream.full_native_umap',
              '--output',str(a.output),'--layout',key]
        with (a.output/f'UMAP_{key}.log').open('w') as log:
            jobs.append(subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT))
    if any([proc.wait()!=0 for proc in jobs]):
        raise RuntimeError('At least one UMAP layout failed')
    coords={key:np.load(a.output/f'UMAP_{key}.npy') for key in readouts}
    columns=['embedding_row','record_id','body_id','cell_type','refine_split']
    for key, uv in coords.items():
        panel[columns].assign(umap_x=uv[:,0],umap_y=uv[:,1]).to_csv(a.output/f'UMAP_{key}.csv',index=False)

    labels=panel.cell_type.fillna('unlabeled')
    shown=labels[labels.ne('unlabeled')].value_counts().head(12).index.tolist()
    grouped=labels.where(labels.isin(shown+['unlabeled']),'Other labeled types')
    volume=np.log10(panel.voxel_count.to_numpy()*1920/1e9)
    norm=Normalize(*np.quantile(volume,[.01,.99]),clip=True)
    for overlay,title in [('cell_type','Cell type'),('volume','Volume')]:
        fig,axes=plt.subplots(2,2,figsize=(12,9.5),layout='constrained')
        fig.suptitle(f'Fish2 full panel · 29,752 mitochondria · E50 no Balance · {title}',fontsize=15)
        handles=[]
        for j,(ax,(key,label)) in enumerate(zip(axes.flat,readouts.items())):
            uv=coords[key]
            if overlay=='cell_type':
                entries=[('unlabeled','#c4c8cf','Unlabeled'),
                         ('Other labeled types','#737e88','Other labeled types')]
                entries += [(cl,plt.get_cmap('tab20')(i),cl) for i,cl in enumerate(shown)]
                for cl,color,name in entries:
                    selected=grouped.eq(cl).to_numpy()
                    dot=ax.scatter(*uv[selected].T,s=2,c=[color],alpha=.53,linewidths=0,label=name)
                    if j==0: handles.append(dot)
            else:
                dots=ax.scatter(*uv.T,s=2,c=volume,cmap='viridis',norm=norm,alpha=.65,linewidths=0)
            ax.set(title=label,xticks=[],yticks=[],aspect='equal')
        if overlay=='cell_type':
            fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False,markerscale=3)
        else:
            fig.colorbar(dots,ax=axes,label='log10 volume (µm³)',shrink=.75)
        fig.savefig(a.output/f'UMAP_full326_{overlay}.png',dpi=160,facecolor='white')
        plt.close(fig)
    write(a.output/'complete.json',dict(instances=len(panel),layouts=list(readouts),
       plots=['UMAP_full326_cell_type.png','UMAP_full326_volume.png'],top_labeled_types=shown))
    print('Full-panel plots verified',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--reuse',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--gpus',nargs='+')
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--layout',choices=['point','image','concat','projection'])
    a=p.parse_args()
    if a.layout:
        fit_umap(a.output,a.layout)
        return
    embed_stage(a)
    plot_umaps(a)


if __name__=='__main__':
    main()
