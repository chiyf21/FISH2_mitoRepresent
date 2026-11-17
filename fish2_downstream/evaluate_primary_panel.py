"""Compare frozen Fish2 panel readouts with shared body-disjoint probes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from fish2_downstream.assets import folds
from fish2_downstream.evaluate import evaluate


def read(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def aligned(path, panel, fields):
    table = pd.read_csv(path,dtype={'body_id':str})
    table = table.set_index('record_id',verify_integrity=True).loc[panel.record_id]
    assert np.array_equal(table.body_id, panel.body_id)
    assert np.array_equal(table.cell_type, panel.cell_type)
    return table[fields].to_numpy(dtype=np.float64)


def evaluate_readout(vectors, specs, out):
    directory = out
    directory.mkdir()
    rows = []
    for name, frame, config in specs:
        log_name = directory/name
        print(directory.name,name,flush=True)
        features = np.asarray(vectors[frame.embedding_row], dtype=np.float64)
        scores, instances, groups, training = evaluate(features,frame,config,'lr',2)
        instances.to_csv(log_name.with_suffix('.instance_oof.csv'),index=False)
        groups.to_csv(log_name.with_suffix('.group_oof.csv'),index=False)
        write_json(log_name.with_suffix('.metrics.json'), scores)
        write_json(log_name.with_suffix('.training.json'), training)
        for unit,result in scores.items():
            rows.append(dict(task=name,unit=unit,instances=len(frame),
              bodies=frame.body_id.nunique(),groups=config['groups'],
              balanced_accuracy=result['balanced_accuracy'],macro_f1=result['macro_f1']))
    return pd.DataFrame(rows)


def task_specs(panel, labels, out):
    result = []
    labelled = panel.merge(labels[['record_id','body_id','cell_type','compartment','compartment_qc']],
                          on=['record_id','body_id','cell_type'],how='inner',validate='one_to_one')
    assert len(labelled) == len(set(panel.record_id) & set(labels.record_id))
    labelled = labelled[labelled.compartment_qc.eq('ok') & labelled.compartment.isin(['axon','dendrite'])]
    for name, data, label, scope in [
      ('cell_type',panel,'cell_type','pooled'),
      ('cell_type_refine_unseen',panel[panel.refine_split.ne('train')],'cell_type','pooled'),
      ('compartment_reviewed_subset',labelled,'compartment','by_type'),
    ]:
        frame = data.copy().reset_index(drop=True)
        assignment = folds(frame)
        assignment.to_csv(out/f'{name}_body_folds.csv',index=False)
        frame['fold'] = frame.body_id.map(assignment.set_index('body_id').fold)
        classes = sorted(frame[label].unique())
        frame['label'] = frame[label].map({label:i for i,label in enumerate(classes)})
        group = ['body_id'] if label=='cell_type' else ['body_id','compartment']
        result.append((name,frame,dict(label=label,scope=scope,classes=classes,group=group,
                                       groups=frame.groupby(group).ngroups)))
    return result,labelled


def plot_umap(panel,labelled,embeddings,out):
    from umap import UMAP
    stages = [('image','Image CLS (768D)'),('point','Point CLS (768D)'),
              ('concat','Concat (1536D)'),('projection','Projection (384D)')]
    args = dict(n_neighbors=30,min_dist=.1,metric='cosine',random_state=20261116,n_jobs=1)
    coords = {}
    meta = {}
    for key,title in stages:
        print('UMAP',key,flush=True)
        x = np.load(embeddings/f'{key}.npy',mmap_mode='r')
        pca = PCA(n_components=30,svd_solver='randomized',random_state=args['random_state'])
        uv = UMAP(**args).fit_transform(pca.fit_transform(x))
        assert uv.shape==(len(panel),2) and np.isfinite(uv).all()
        coords[key] = uv
        meta[key] = dict(pca30_variance=float(pca.explained_variance_ratio_.sum()))
        panel[['record_id','body_id','cell_type','refine_split']].assign(
            x=uv[:,0],y=uv[:,1]).to_csv(out/f'UMAP_E50_{key}.csv',index=False)
    write_json(out/'umap_config.json',dict(pca_dimensions=30,umap=args,readouts=meta,
       fit='independent PCA and UMAP per readout; identical inputs and parameters'))
    plot_overlays(panel, labelled, coords, stages, out)


def plot_overlays(panel, labelled, coords, stages, out, compartments_only=False,
                  compartment_filename='UMAP_E50_compartment.png'):
    labels_by_id = labelled.set_index('record_id').compartment
    for label in (('compartment',) if compartments_only else ('cell_type','compartment')):
        values = panel.cell_type if label=='cell_type' else panel.record_id.map(labels_by_id)
        classes = sorted(values.dropna().unique())
        fig,axes = plt.subplots(2,2,figsize=(11,8.7),layout='constrained')
        title = 'Cell type' if label=='cell_type' else 'Reviewed compartments'
        fig.suptitle(f'Fusion50 native 768D | E50 without Balance | {title}',fontsize=16)
        handles = []
        for ax,(key,stage_title) in zip(axes.flat,stages):
            uv = coords[key]
            if values.isna().any():
                keep = values.isna().to_numpy()
                artist = ax.scatter(*uv[keep].T,c='#c3c9d2',s=2,alpha=.30,linewidths=0,label='Outside displayed classes')
                if key == 'image':
                    handles.append(artist)
            for i,cls in enumerate(classes):
                keep = values.eq(cls).to_numpy()
                artist = ax.scatter(*uv[keep].T,color=plt.get_cmap('tab10')(i),s=3,alpha=.60,
                                   linewidths=0,label=cls,rasterized=True)
                if key == 'image':
                    handles.append(artist)
            ax.set(title=stage_title,xticks=[],yticks=[])
            ax.set_aspect('equal',adjustable='datalim')
        fig.legend(handles=handles,loc='outside lower center',ncol=min(5,len(handles)),markerscale=3,
                   frameon=False)
        name = compartment_filename if label=='compartment' else 'UMAP_E50_cell_type.png'
        fig.savefig(out/name,dpi=175,facecolor='white')
        plt.close(fig)


def wait_for(path, launcher):
    while not path.is_file():
        try:
            status = Path(f'/proc/{launcher}/stat').read_text().split(') ',1)[1][0]
        except FileNotFoundError:
            status = 'Z'
        if status == 'Z':
            raise RuntimeError(f'Export ended before producing {path}')
        time.sleep(10)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',type=Path,required=True)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--export',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    a.output.mkdir()
    launch=read(a.export/'launcher.json')
    wait_for(a.export/'instance_map.csv',launch['pid'])
    panel = pd.read_csv(a.export/'instance_map.csv',dtype={'body_id':str})
    labels = pd.read_csv(a.labels,dtype={'body_id':str})
    specs, labelled = task_specs(panel,labels,a.output)
    manifest = read(a.panel/'dataset_manifest.json')
    hand = aligned(a.panel/'handcrafted_features22.csv',panel,manifest['handcrafted_columns'])
    assert np.isfinite(hand).all()
    write_json(a.output/'protocol.json',dict(panel=manifest, label_source=str(a.labels),
      label_sha256=sha(a.labels), folds_seed=20261113,
      cohorts={k:dict(instances=len(f),bodies=f.body_id.nunique(),groups=c['groups'],
       classes=c['classes']) for k,f,c in specs},
      splits={str(k):dict(instances=len(f),bodies=f.body_id.nunique())
              for k,f in panel.groupby('refine_split')},
      exact_refine_records=int(panel.record_in_refine_cohort.sum()),
      exact_train_records=int((panel.record_in_refine_cohort & panel.refine_split.eq('train')).sum()),
      compartment_scope='Reviewed81 exact record_id and body/type agreement; QC=ok, axon/dendrite. Per-body-and-true-compartment grouping.',
      classifier='L2 logistic regression, C=1, class_weight=balanced, train-fold StandardScaler'))
    with threadpool_limits(limits=2):
        scores = evaluate_readout(hand,specs,a.output/'handcrafted').assign(model='Handcrafted',stage='22D')
        matrices = [
          ('Off E50','mito_e16_native_fusion768_nobalance_long12h_20261117_01'),
          ('Off E5','mito_e16_native_fusion768_from_nna20_5ep_20261116_01'),
          ('On E5','mito_e16_native_fusion768_balance_from_nna20_5ep_20261116_01'),
        ]
        scores.to_csv(a.output/'downstream_metrics.csv',index=False)
        for model,name in matrices:
            folder=a.export/name
            wait_for(folder/'complete.json',launch['pid'])
            receipt=read(folder/'complete.json')
            assert receipt['instances']==len(panel)
            assert receipt['epoch']==(50 if model=='Off E50' else 5)
            assert receipt['balance']==(model=='On E5')
            target=a.output/name
            target.mkdir()
            for stage in ('concat','projection'):
                vectors=np.load(folder/f'{stage}.npy',mmap_mode='r')
                assert len(vectors)==len(panel)
                batch=evaluate_readout(vectors,specs,target/stage).assign(model=model,stage=stage)
                scores=pd.concat([scores,batch],ignore_index=True)
                scores.to_csv(a.output/'downstream_metrics.csv',index=False)
            if model=='Off E50':
                plot_umap(panel,labelled,folder,a.output)
        write_json(a.output/'complete.json',dict(rows=len(scores),models=[x[0] for x in matrices]))
    print('Panel evaluation complete',flush=True)


if __name__=='__main__':
    main()
