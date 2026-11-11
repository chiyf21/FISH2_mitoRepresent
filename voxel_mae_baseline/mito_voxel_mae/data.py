from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import Dataset


def tight_box(mask):
    slices=ndimage.find_objects(mask.astype(np.uint8),max_label=1)[0]
    if slices is None:raise ValueError('empty transformed instance')
    lo=np.array([s.start for s in slices]); shape=np.array([s.stop-s.start for s in slices])
    return lo,shape


def augment(em,mask,valid,spacing,rng,kind):
    """Transform clean EM/mask/valid together; noise is applied later to input only."""
    if kind==0:return em,mask,valid
    shape=np.array(em.shape); center=(shape-1)/2
    # Small physical affine: preserves spacing anisotropy rather than rotating indices.
    angle=rng.uniform(-10,10)*np.pi/180 if kind==2 else rng.uniform(-3,3)*np.pi/180
    cs,sn=np.cos(angle),np.sin(angle)
    rot=np.array([[1,0,0],[0,cs,-sn],[0,sn,cs]])
    scale=np.diag(rng.uniform(.97,1.03,3)) if kind==1 else np.eye(3)
    shear=np.eye(3)
    if kind==1:shear[1,2]=rng.uniform(-.025,.025)
    inv=np.linalg.inv(rot@shear@scale)
    matrix=np.diag(1/spacing)@inv@np.diag(spacing)
    shift=rng.uniform(-.015,.015,3)*shape
    offset=center-matrix@(center+shift)
    if kind==3:
        coords=np.indices(tuple(shape),dtype=np.float32)
        source=np.einsum('ij,j...->i...',matrix,coords)+offset[:,None,None,None]
        for axis in range(3):
            coarse=rng.normal(size=(4,4,4)).astype(np.float32)
            smooth=ndimage.zoom(coarse,shape/4,order=3)
            source[axis]+=smooth[:shape[0],:shape[1],:shape[2]]*.008*shape[axis]
        fn=lambda a,order:ndimage.map_coordinates(a,source,order=order,mode='constant',cval=0,prefilter=False)
    else:
        fn=lambda a,order:ndimage.affine_transform(a,matrix,offset,output_shape=tuple(shape),order=order,mode='constant',cval=0,prefilter=False)
    m=fn(mask,0)>0; v=fn(valid.astype(np.float32),1)>.999; e=fn(em,1)
    # Reject augmentation that truncates the object or moves foreground into missing EM.
    if not m.any() or any(m.take(i,axis=a).any() for a in range(3) for i in (0,-1)) or (m&~v).any():
        return em,mask,valid
    return e,m,v


def point_support(mask,lo,size,spacing,grid=17):
    # Include an optional one-cell halo, so outer neighbors across bbox faces survive.
    axes=[(np.arange(-1,grid+1)+.5)/grid for _ in range(3)]
    t=np.stack(np.meshgrid(*axes,indexing='ij'),-1)-.5
    vox=lo+(t+.5)*size-.5
    idx=np.rint(vox).astype(int); inside_bounds=((idx>=0)&(idx<np.array(mask.shape))).all(-1)
    idx=np.clip(idx,0,np.array(mask.shape)-1)
    inside=inside_bounds & mask[tuple(np.moveaxis(idx,-1,0))]
    # Only tight-bbox seeds; halo points can only be selected as neighbors.
    inside[[0,-1],:,:]=False;inside[:,[0,-1],:]=False;inside[:,:,[0,-1]]=False
    keep=ndimage.binary_dilation(inside,structure=np.ones((3,3,3),bool))&inside_bounds
    if not keep.any():raise ValueError('No coarse-grid support; increase sampling resolution')
    t=t[keep].astype(np.float32);vox=vox[keep].astype(np.float32)
    r=t*(size*spacing)/(size*spacing).max()
    return t,r.astype(np.float32),vox


class MitoDataset(Dataset):
    def __init__(self,cache,split='train',queries=8192,views=4,augmentations=True,masking=True,seed=42,limit=0):
        self.root=Path(cache);doc=json.loads((self.root/'manifest.json').read_text())
        self.records=[r for r in doc['records'] if split=='all' or r['split']==split]
        if limit:self.records=self.records[:limit]
        self.queries=queries;self.views=views if split=='train' else 1
        self.augmentations=augmentations and split=='train';self.masking=masking
        self.seed=seed;self.epoch=0;self.split=split
        if not self.records:raise ValueError(f'Empty split {split}')
    def __len__(self):return len(self.records)*self.views
    def __getitem__(self,index):
        record=self.records[index%len(self.records)];path=self.root/record['relative_dir']
        # Workers receive fresh deterministic streams each epoch; validation is fixed.
        epoch=self.epoch if self.split=='train' else 0
        rng=np.random.default_rng(self.seed+index*100003+epoch*10000019)
        raw=np.load(path/'em.npy',mmap_mode='r');mask=np.load(path/'mask.npy',mmap_mode='r').astype(bool)
        valid=np.load(path/'valid.npy',mmap_mode='r')
        spacing=np.asarray(record['spacing_zyx_nm'],dtype=np.float32)
        image=np.clip((raw.astype(np.float32)-record['lo'])/max(record['hi']-record['lo'],1e-6),0,1)
        kind=(index//len(self.records))%4 if self.augmentations else 0
        image,mask,valid=augment(image,mask,valid,spacing,rng,kind)
        lo,size=tight_box(mask);sl=tuple(slice(int(a),int(a+n)) for a,n in zip(lo,size))
        vals=image[sl][valid[sl]];mu=float(vals.mean());std=max(float(vals.std()),1e-6)
        image=(image-mu)/std;image=np.where(valid,image,0).astype(np.float32)
        points_t,points_r,vox=point_support(mask,lo,size,spacing)
        block_visible=np.ones((4,4,4),dtype=bool)
        if self.masking:block_visible.reshape(-1)[rng.choice(64,38,replace=False)]=False
        pblock=np.clip(np.floor((points_t+.5)*4).astype(int),0,3)
        pv=block_visible[tuple(pblock.T)]
        pix=np.clip(np.rint(vox).astype(int),0,np.array(mask.shape)-1)
        pv&=valid[tuple(pix.T)]
        if not pv.any():
            # Reveal one independently selected support block, never expose target labels.
            choices=np.flatnonzero(valid[tuple(pix.T)])
            if not len(choices):raise ValueError('No valid raw at any support point')
            j=int(rng.choice(choices));block_visible[tuple(pblock[j])]=True
            pv=block_visible[tuple(pblock.T)]&valid[tuple(pix.T)]
        axes=[np.clip(np.floor(((np.arange(n)+.5-lo[a])/size[a])*4).astype(int),0,3) for a,n in enumerate(mask.shape)]
        observed=block_visible[np.ix_(*axes)]&valid
        input_image=image.copy()
        if self.augmentations:input_image+=rng.normal(0,rng.uniform(0,.035),image.shape).astype(np.float32)
        input_image[~observed]=0
        q=np.stack([rng.integers(0,int(n),self.queries) for n in size],-1)
        qt=(q+.5)/size-.5;qr=qt*(size*spacing)/(size*spacing).max();qi=q+lo
        target_index=tuple(qi.T);qh=~block_visible[tuple(np.clip(np.floor((qt+.5)*4).astype(int),0,3).T)]
        # grid_sample coordinates are XYZ; source arrays and geometric coordinates are ZYX.
        grid_xyz=(2*(vox+.5)/np.array(image.shape)-1)[:,::-1].copy()
        log_size=np.log(size*spacing/1000).astype(np.float32) # log micrometers
        result={'image':input_image[None],'observed':observed[None].astype(np.float32),
                'image_valid':np.asarray(valid,dtype=np.float32)[None],'points_t':points_t,'points_r':points_r,
                'points_grid':grid_xyz.astype(np.float32),'points_valid':pv,
                'query_t':qt.astype(np.float32),'query_r':qr.astype(np.float32),'query_hidden':qh,
                'occupancy':mask[target_index].astype(np.float32),'intensity':image[target_index],
                'query_valid':valid[target_index],'log_size':log_size,'size_target':log_size.copy()}
        return {**{k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in result.items()},
                'record_id':record['record_id'],'tight_lo':lo,'tight_shape':size,'context_shape':np.array(image.shape),
                'intensity_mean':mu,'intensity_std':std,'clip_lo':record['lo'],'clip_hi':record['hi'],'augmentation_kind':kind}


def collate_one(items):
    if len(items)!=1:raise ValueError('Native varying-sized CNN crops use batch1 + gradient accumulation')
    return {k:(v.unsqueeze(0) if torch.is_tensor(v) else v) for k,v in items[0].items()}

def to_device(b,device):return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
