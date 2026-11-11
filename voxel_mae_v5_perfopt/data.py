import json,time,os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from mito_voxel_mae_v2.data import point_support
from mito_voxel_mae_v4.native_data import tight_box
from .augmentation import augment
from .targets import OFFSETS,safe_query_centres
from .trace import trace

TILE_THRESHOLD=10**18

class MitoDataset(Dataset):
    def __init__(self,cache,split='train',queries=1000,views=4,augmentations=True,masking=True,seed=42,limit=0):
        self.root=Path(cache);doc=json.loads((self.root/'manifest.json').read_text())
        self.records=[r for r in doc['records'] if split=='all' or r['split']==split]
        if limit:self.records=self.records[:limit]
        self.queries=queries;self.views=views if split=='train' else 1
        self.augmentations=augmentations and split=='train';self.masking=masking
        self.seed=seed;self.epoch=0;self.split=split
        if not self.records:raise ValueError('Empty split')
    def __len__(self):return len(self.records)*self.views
    def __getitem__(self,index):
        actual=index[0] if isinstance(index,tuple) else index
        r=self.records[actual%len(self.records)]
        if isinstance(index,tuple) and index[1]==0:
            return dict(skip=True,record_id=r['record_id'],reason='ddp_padding',sample_weight=0.)
        if int(np.prod(r['shape_zyx']))<=64000000:
            return self._get_item(index)
        # Large elastic volumes allocate several dense coordinate arrays. Bound
        # their concurrent preparation across both experiments, without dropping
        # any instance or reducing its native resolution.
        import fcntl
        directory=Path(os.getenv('MITOMAE_LARGE_GATE','/home/cyf/.cache/mitomae_large_preprocess'))
        directory.mkdir(parents=True,exist_ok=True)
        while True:
            for slot in range(2):
                handle=(directory/f'slot{slot}.lock').open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                try:return self._get_item(index)
                finally:fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
            time.sleep(.1)

    def _get_item(self,index):
        weight=1.
        if isinstance(index,tuple):index,weight=index
        r=self.records[index%len(self.records)];path=self.root/r['relative_dir'];start=time.time()
        trace('sample_start',record=r['record_id'],index=index)
        epoch=self.epoch if self.split=='train' else 0
        rng=np.random.default_rng(self.seed+index*100003+epoch*10000019)
        raw=np.load(path/'em.npy',mmap_mode='r');mask=np.load(path/'mask.npy',mmap_mode='r').astype(bool)
        valid=np.load(path/'valid.npy',mmap_mode='r');spacing=np.asarray(r['spacing_zyx_nm'],dtype=np.float32)
        image=np.asarray(raw,dtype=np.float32)
        image-=r['lo'];image/=max(r['hi']-r['lo'],1e-6);np.clip(image,0,1,out=image)
        kind=(index//len(self.records))%4 if self.augmentations else 0
        image,mask,valid=augment(image,mask,valid,spacing,rng,kind)
        trace('sample_augmented',record=r['record_id'],kind=kind,seconds=time.time()-start)
        lo,size=tight_box(mask);centre=lo+size/2
        pt,pr,vox=point_support(mask,lo,size,spacing,grid=17)
        visible=np.ones((4,4,4),bool)
        if self.masking:visible.reshape(-1)[rng.choice(64,38,replace=False)]=False
        point_bins=np.clip(np.floor((pt+.5)*4).astype(int),0,3)
        pix=np.clip(np.rint(vox).astype(int),0,np.array(mask.shape)-1)
        pv=visible[tuple(point_bins.T)]&valid[tuple(pix.T)]
        if not pv.any():
            choices=np.flatnonzero(valid[tuple(pix.T)])
            if not len(choices):return dict(skip=True,record_id=r['record_id'],reason='no_visible_raw_support',sample_weight=weight)
            j=int(rng.choice(choices));visible[tuple(point_bins[j])]=True
            pv=visible[tuple(point_bins.T)]&valid[tuple(pix.T)]
        origin=np.asarray(r.get('bbox_xyz',[0,0,0])[:3])[::-1]-np.asarray(r.get('tight_lo_zyx',[0,0,0]))
        candidates=safe_query_centres(mask,valid,visible,lo,size,origin)
        trace('sample_candidates',record=r['record_id'],seconds=time.time()-start,count=len(candidates))
        if not len(candidates):
            return dict(skip=True,record_id=r['record_id'],reason='no_safe_333_query',sample_weight=weight)
        total=len(candidates);q=candidates[rng.choice(total,min(self.queries,total),replace=False)]
        patch=q[:,None,:]+OFFSETS[None];idx=tuple(patch.transpose(2,0,1))
        occ=mask[idx].astype(np.float32)
        assert (occ.sum(1)>=2).all() and valid[idx].all()
        bins=[np.clip(np.floor(((np.arange(n)+.5-l)/s)*4).astype(int),0,3) for n,l,s in zip(mask.shape,lo,size)]
        observed=visible[np.ix_(*bins)]&valid
        assert not observed[idx].any(),'Query patch intersects visible input'
        # Compute normalization from visible pixels only: hidden targets must not
        # affect the input's z-score statistics.
        sl=tuple(slice(int(l),int(l+n)) for l,n in zip(lo,size))
        values=image[sl][observed[sl]]
        if not len(values):values=image[observed]
        mu=float(values.mean());std=max(float(values.std()),1e-6)
        target=((image[idx]-mu)/std).astype(np.float32)
        input_image=image
        np.subtract(input_image,mu,out=input_image);np.divide(input_image,std,out=input_image)
        if self.augmentations:input_image+=rng.normal(0,rng.uniform(0,.035),image.shape).astype(np.float32)
        input_image[~observed]=0
        size_log=np.log(size*spacing/1000).astype(np.float32)
        result={'image':input_image[None],'target_mask':(mask&observed)[None].astype(np.float32),
                'observed':observed[None].astype(np.float32),'image_valid':np.asarray(valid,dtype=np.float32)[None],
                'points_t':pt,'points_r':pr,'points_vox':vox,'points_valid':pv,
                'points_um':((vox+.5-centre)*spacing/1000).astype(np.float32),
                'query_um':((q+.5-centre)*spacing/1000).astype(np.float32),
                'query_vox':q.astype(np.float32),'query_valid':np.ones(len(q),bool),
                'occupancy':occ,'intensity':target,'size_target':size_log,'sample_weight':np.array(weight,dtype=np.float32)}
        trace('sample_done',record=r['record_id'],seconds=time.time()-start)
        return {**{k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in result.items()},
                'record_id':r['record_id'],'context_shape':np.array(mask.shape),'tight_lo':lo,'tight_shape':size,
                'query_candidates':total,'query_unique_count':len(q),'augmentation_kind':kind,
                'spacing_zyx_nm':spacing.tolist(),'patch_extent_zyx_nm':(3*spacing).tolist(),
                'preprocess_seconds':time.time()-start,'skip':False}

def dummy_batch():
    # Graph-connected zero-weight filler only for an empty local DDP rank.
    one=lambda *shape:torch.ones(shape)
    zero=lambda *shape:torch.zeros(shape)
    b={'volumes':[(zero(1,3,8,8),zero(1,3,8,8),zero(1,3,8,8),one(1,3,8,8))],
       'points_um':zero(1,1,3),'points_vox':zero(1,1,3),'points_valid':one(1,1).bool(),
       'query_um':zero(1,1,3),'query_valid':zero(1,1).bool(),'occupancy':zero(1,1,27),
       'intensity':zero(1,1,27),'size_target':zero(1,3),'sample_weight':zero(1),
       'record_id':['DDP_ZERO_WEIGHT'],'padded_shape':[3,8,8],'tiled':True,'empty':False,
       'query_unique_count':[0],'query_candidates':[0],'skipped_records':[]}
    return b

def collate_batch(items):
    skipped=[{'record_id':s['record_id'],'reason':s['reason']} for s in items if s['skip'] and float(s['sample_weight'])>0]
    items=[s for s in items if not s['skip']]
    if not items:return {'empty':True,'skipped_records':skipped}
    out={'volumes':[tuple(s[k] for k in ('image','target_mask','observed','image_valid')) for s in items]}
    nmax=max(len(s['points_vox']) for s in items);qmax=max(len(s['query_um']) for s in items)
    for k in items[0]:
        if k in ('image','target_mask','observed','image_valid'):continue
        vals=[s[k] for s in items]
        if k in ('points_t','points_r','points_vox','points_um','points_valid'):
            out[k]=torch.stack([F.pad(v,(0,0,0,nmax-len(v)) if v.ndim==2 else (0,nmax-len(v))) for v in vals])
        elif k in ('query_um','query_vox','query_valid','occupancy','intensity'):
            out[k]=torch.stack([F.pad(v,(0,0,0,qmax-len(v)) if v.ndim==2 else (0,qmax-len(v))) for v in vals])
        elif torch.is_tensor(vals[0]):out[k]=torch.stack(vals)
        else:out[k]=vals
    out.update(empty=False,tiled=True,skipped_records=skipped,
               padded_shape=[int(max(s['context_shape'][j] for s in items)) for j in range(3)])
    return out

def to_device(b,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
