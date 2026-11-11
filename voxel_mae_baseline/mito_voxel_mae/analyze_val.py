"""Validation latent visualization, colored by physical mask morphometrics."""
from __future__ import annotations
import argparse, csv, hashlib, json, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from skimage import measure

FEATURES = [('volume', 'Volume', 'um^3'), ('surface_area', 'Surface area', 'um^2')]
for i in range(1, 4):
    FEATURES += [(f'pc{i}_{k}', f'PC{i} {label}', unit) for k, label, unit in [
        ('length', 'length', 'um'), ('inertia', 'inertia moment', 'um^5'),
        ('symmetry', 'symmetry', 'score'), ('area', 'cross-sectional area', 'um^2'),
        ('perimeter', 'circumference', 'um')]]
FEATURES += [('diameter', 'Diameter', 'um'), ('sphericity', 'Sphericity', 'score')]
EXTRA = ('convex_hull_area', 'Convex hull surface area (author label: Convexity)', 'um^2')

def section(mask, spacing, center, basis, axis):
    other = [j for j in range(3) if j != axis]
    u, v = basis[:, other[0]], basis[:, other[1]]
    # A physical isotropic 8nm plane, with a pixel exactly at the centroid.
    corners = np.array(np.meshgrid(*[[-.5, n-.5] for n in mask.shape], indexing='ij')).reshape(3, -1).T * spacing
    centered = corners - center
    h = .008
    a = int(np.ceil(np.max(np.abs(centered @ u)) / h)) + 2
    b = int(np.ceil(np.max(np.abs(centered @ v)) / h)) + 2
    x, y = np.meshgrid(np.arange(-a, a+1)*h, np.arange(-b, b+1)*h, indexing='ij')
    xyz = center[:, None, None] + u[:, None, None]*x + v[:, None, None]*y
    plane = ndimage.map_coordinates(mask.astype(np.uint8), xyz / spacing[:, None, None], order=0, mode='constant', cval=0) > 0
    labels, n = ndimage.label(plane, np.ones((3,3)))
    if not n:
        return 0., 0.
    # Match author's hole-filled component nearest centroid rule.
    points = np.argwhere(plane)
    nearest = points[np.argmin(((points-np.array([a,b]))**2).sum(1))]
    plane = ndimage.binary_fill_holes(labels == labels[tuple(nearest)])
    contours = measure.find_contours(np.pad(plane.astype(float), 1), .5)
    perimeter = sum(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in contours)*h
    return float(plane.sum()*h*h), float(perimeter)

def morphometry(record):
    mask = np.load(Path(record['relative_dir'])/'mask.npy').astype(bool)
    spacing = np.array(record['spacing_zyx_nm'], dtype=float)/1000
    coords = np.argwhere(mask).astype(float) * spacing
    center = coords.mean(0)
    centered = coords-center
    covariance = centered.T @ centered / len(coords)
    eig, basis = np.linalg.eigh(covariance)
    basis = basis[:, np.argsort(eig)[::-1]]
    pc = centered @ basis
    # Physical principal-frame quantization, analogous to author's integer PC occupancy.
    q = np.unique(np.trunc(pc/.008).astype(np.int32), axis=0)
    qset = np.ascontiguousarray(q).view(np.dtype((np.void, q.dtype.itemsize*3))).ravel()
    padded = np.pad(mask.astype(np.uint8), 1)
    vertices, faces, _, _ = measure.marching_cubes(padded, .5, spacing=tuple(spacing))
    vertices -= spacing
    sa = float(measure.mesh_surface_area(vertices, faces))
    hull = ConvexHull(vertices)
    hv = vertices[hull.vertices]
    diameter = max(float(cdist(hv[i:i+256], hv).max()) for i in range(0,len(hv),256))
    vv = float(np.prod(spacing))
    volume = len(coords)*vv
    result = {'record_id': record['record_id'], 'body_id': record['body_id'], 'volume': volume,
              'surface_area': sa, 'diameter': diameter, 'sphericity': np.pi**(1/3)*(6*volume)**(2/3)/sa,
              'convex_hull_area': float(hull.area), 'mask_voxels': len(coords)}
    for i in range(3):
        flipped = q.copy(); flipped[:,i] *= -1
        fset = np.ascontiguousarray(flipped).view(qset.dtype).ravel()
        area, perimeter = section(mask, spacing, center, basis, i)
        result.update({f'pc{i+1}_length': float(np.ptp(pc[:,i])+np.sum(np.abs(basis[:,i])*spacing)),
                       f'pc{i+1}_inertia': float((pc[:,[j for j in range(3) if j!=i]]**2).sum()*vv),
                       f'pc{i+1}_symmetry': len(np.intersect1d(qset,fset,assume_unique=True))/len(q),
                       f'pc{i+1}_area': area, f'pc{i+1}_perimeter': perimeter})
    assert np.isfinite([result[k] for k,_,_ in FEATURES+[EXTRA]]).all()
    return result

def checks():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mask = np.ones((5,9,13), np.uint8)
        np.save(Path(tmp)/'mask.npy',mask)
        rec = {'relative_dir':tmp,'spacing_zyx_nm':[30,8,8],'body_id':0,'record_id':'box'}
        a = morphometry(rec)
        rec['spacing_zyx_nm'] = [60,16,16]
        b = morphometry(rec)
        for k in ['volume','surface_area','diameter','pc1_length','pc1_inertia']:
            power = {'volume':3,'surface_area':2,'diameter':1,'pc1_length':1,'pc1_inertia':5}[k]
            assert np.isclose(b[k]/a[k],2**power,rtol=1e-5), (k,a[k],b[k])
        assert np.isclose(a['volume'],mask.sum()*.03*.008*.008)
        assert all(0 <= a[f'pc{i}_symmetry'] <= 1 for i in range(1,4))
    print('Physical volume/scale powers and symmetry bounds checks passed.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--checks-only',action='store_true')
    a=p.parse_args();checks()
    if a.checks_only:return
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ts=time.time()
    import torch
    from .data import MitoDataset,collate_one,to_device
    from .model import VoxelMAE
    torch.set_num_threads(4);torch.manual_seed(42)
    ds=MitoDataset(a.cache,'val',queries=1,views=1,augmentations=False,masking=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embfile=out/'val_embeddings.npz'
    if not embfile.exists():
        state=torch.load(a.checkpoint,map_location=device,weights_only=False)
        model=VoxelMAE(**state['model_config']).to(device).eval();model.load_state_dict(state['model'])
        embeddings=[]
        with torch.inference_mode():
            for i in range(len(ds)):
                b=to_device(collate_one([ds[i]]),device)
                embeddings.append(model.encode(b).float().cpu().numpy()[0])
                if (i+1)%50==0:print(json.dumps({'encoded':i+1,'total':len(ds),'seconds':time.time()-ts}),flush=True)
        np.savez_compressed(embfile,embedding=np.stack(embeddings),record_id=np.array([r['record_id'] for r in ds.records]))
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    data=np.load(embfile);z=data['embedding']
    assert list(data['record_id']) == [r['record_id'] for r in ds.records]
    ffile=out/'morphometrics.json'
    if not ffile.exists():
        results=[]
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,row in enumerate(pool.map(morphometry,ds.records,chunksize=1)):
                results.append(row)
                if (i+1)%50==0:print(json.dumps({'morphometrics':i+1,'total':len(ds),'seconds':time.time()-ts}),flush=True)
        ffile.write_text(json.dumps(results,indent=2))
    results=json.loads(ffile.read_text());assert len(results)==len(ds)
    assert [r['record_id'] for r in results]==list(data['record_id'])
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness
    import umap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    # Center latent; do not standardize its already learned dimensions independently.
    pca=PCA(n_components=min(50,z.shape[1],len(z)-1),svd_solver='full')
    pcs=pca.fit_transform(z)
    # Same 50-PC input for deterministic Euclidean UMAP.
    reduced=umap.UMAP(n_neighbors=30,min_dist=.1,metric='euclidean',random_state=42,n_jobs=1).fit_transform(pcs)
    xy=[pcs[:,:2],reduced]
    np.savez_compressed(out/'reduction.npz',pca=pcs,umap=reduced,explained_variance_ratio=pca.explained_variance_ratio_,pca_components=pca.components_,pca_mean=pca.mean_)
    fields=['record_id','body_id','mask_voxels']+[k for k,_,_ in FEATURES+[EXTRA]]
    with (out/'val_morphometrics_coordinates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+['pca1','pca2','umap1','umap2']);writer.writeheader()
        for i,r in enumerate(results):writer.writerow({**r,'pca1':pcs[i,0],'pca2':pcs[i,1],'umap1':reduced[i,0],'umap2':reduced[i,1]})
    plots=out/'plots';plots.mkdir(exist_ok=True)
    rng=np.random.default_rng(42);order=rng.permutation(len(z))
    correlations=[]
    with PdfPages(out/'19_features_PCA_UMAP.pdf') as pdf:
        for index,(key,title,unit) in enumerate(FEATURES+[EXTRA]):
            values=np.array([r[key] for r in results])
            # No data removal: color saturation only; CSV keeps exact physical values.
            vmin,vmax=np.percentile(values,[1,99]);vmax=max(vmax,vmin+1e-12)
            fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
            for ax,coords,name in zip(axes,xy,['PCA','UMAP']):
                sc=ax.scatter(coords[order,0],coords[order,1],c=values[order],s=13,cmap='viridis',vmin=vmin,vmax=vmax,alpha=.85,linewidths=0,rasterized=True)
                ax.set(title=name,xlabel=f'{name} 1',ylabel=f'{name} 2');ax.grid(alpha=.12)
            axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
            axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
            fig.suptitle(f'{title} | n={len(z)} | best checkpoint, unmasked latent')
            fig.colorbar(sc,ax=axes,label=f'{unit}; color limits: P1-P99')
            filename=f'{index+1:02d}_{key}.png' if index<19 else 'supplement_convexity.png'
            fig.savefig(plots/filename,dpi=180)
            if index<19:pdf.savefig(fig)
            plt.close(fig)
            correlations.append({'feature':key,'rho_pc1':float(spearmanr(values,pcs[:,0]).statistic),'rho_pc2':float(spearmanr(values,pcs[:,1]).statistic)})
    fig,axes=plt.subplots(5,4,figsize=(16,18),constrained_layout=True)
    for ax,(key,title,unit) in zip(axes.flat,FEATURES):
        values=np.array([r[key] for r in results]);lo,hi=np.percentile(values,[1,99])
        sc=ax.scatter(reduced[order,0],reduced[order,1],c=values[order],s=4,cmap='viridis',vmin=lo,vmax=max(hi,lo+1e-12),linewidths=0)
        ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(sc,ax=ax,shrink=.7,label=unit)
    axes.flat[-1].axis('off');fig.suptitle('19 mask morphometrics on the same model-latent UMAP',fontsize=17)
    fig.savefig(out/'UMAP_19_overview.png',dpi=150);plt.close(fig)
    with (out/'feature_PC_correlations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(correlations[0]));w.writeheader();w.writerows(correlations)
    summary={'instances':len(z),'latent_dim':z.shape[1],'checkpoint':a.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
             'pca_variance_first2':pca.explained_variance_ratio_[:2].tolist(),'pca_variance_first50':float(pca.explained_variance_ratio_.sum()),
             'umap':{'input':'50 unwhitened PCA components of centered latent','n_neighbors':30,'min_dist':.1,'metric':'euclidean','seed':42},
             'umap_trustworthiness_k15':float(trustworthiness(z,reduced,n_neighbors=15)),
             'features':FEATURES,'extra_feature':EXTRA,'seconds':time.time()-ts}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    top=sorted(correlations[:19],key=lambda r:abs(r['rho_pc1']),reverse=True)[:5]
    feature_table='\n'.join(f'| {i+1} | {title} | {unit} |' for i,(_,title,unit) in enumerate(FEATURES))
    (out/'README.md').write_text(f'''# 验证集编码 PCA / UMAP 与 19 个形态特征

全部 {len(z)} 个 val 实例，使用 best.pt 无遮挡、无增强的 256 维编码，FP32 推理。形态特征来自输入真值 mask；它们没有参与降维拟合。PCA 直接中心化 latent，不逐维标准化。前两维解释方差 {sum(summary['pca_variance_first2'])*100:.2f}%；前50维 {summary['pca_variance_first50']*100:.2f}%。UMAP 参数：50维 PCA 输入、Euclidean、neighbors=30、min_dist=0.1、seed=42。相对于原256维 latent 的 k=15 trustworthiness={summary['umap_trustworthiness_k15']:.4f}。

## 图与原始数据

`plots/01_*.png` 至 `plots/19_*.png` 为19张主图，每张左 PCA、右 UMAP，颜色表示同一个形态特征。`19_features_PCA_UMAP.pdf` 为19页合集。`UMAP_19_overview.png` 为总览。颜色只在1–99百分位饱和，保留全部实例；CSV存实际值。`supplement_convexity.png` 是第20张附图。`val_embeddings.npz`、`reduction.npz`、`val_morphometrics_coordinates.csv` 可复用。

| 编号 | 特征 | 单位 |
|---|---|---|
{feature_table}

## 论文对应关系与测量约定

来源：[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12906887/)；[作者特征生成代码](https://github.com/ClarkLabCode/MitochondrialMorphologyPosition/blob/main/HPC/save_mito_synapse/mito_synapse_main.py)；[作者分析 notebook](https://github.com/ClarkLabCode/MitochondrialMorphologyPosition/blob/main/morphology%20embeddings%20-%20LC%20neurons.ipynb)。作者 notebook 实列22项，论文称21维；按用户确认，排除依赖神经元几何的 mito CA 与 length along skeleton，Convexity单列附图，剩19项为主图。作者 Convexity 字段实际为 scipy ConvexHull.area，本实现也绘制凸包表面积，不冒充无量纲紧致度。

本数据为8/8/30nm各向异性 mask，不能直接照搬作者等距体素代码。体积按原始体素数乘体素体积；表面积用带padding的物理 marching cubes网格。主轴从所有前景体素中心的物理协方差得到，长度包含体素投影宽度；惯性矩按单位密度的体积权重求和（um^5）。对称性在质心居中的PC框架按8nm量化后作轴反射，取离散集合重叠比例。截面通过真实质心、以各主轴为法向，在8nm平面最近邻采样，取最近中心的连通分量并填洞，周长从marching-squares轮廓得到。直径在物理表面凸包顶点上分块求精确最大距离；球形度=pi^(1/3)*(6V)^(2/3)/SA（主图为原始球形度，作者notebook使用log10）。这些是适配本数据的物理测量，不是作者原始等距体素数值的逐位复刻。小/薄实例的截面、对称性对网格分辨率敏感。

## 阅读建议

PC1相关最强的5项：{json.dumps(top,ensure_ascii=False)}。相关表为探索性Spearman统计，没有独立测试集或多重检验校正。UMAP的方向、全局间距、簇间空隙不能直接解释为生物学距离。绝对尺寸本来就作为模型输入和监督，体积/长度出现梯度不能单独证明编码学到了尺寸之外的形态。验证集只有10个神经元，实例也不是完全独立的生物学重复。
''',encoding='utf-8')
    print(json.dumps({'complete':True,**summary}),flush=True)

if __name__=='__main__':main()
