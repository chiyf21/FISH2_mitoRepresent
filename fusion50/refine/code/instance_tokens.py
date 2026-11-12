"""Trainable, variable-length instance tokens from a pretrained Focus3D MAE.

ZYX arrays; local crop-origin patch grid; mask selects tokens, never masks EM.
No decoder, instance pooling, resizing, random MAE masking, or optimizer step.
"""
from pathlib import Path
from functools import partial
import sys
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def patch_counts(mask, patch_size=(2, 8, 8)):
    mask=np.asarray(mask)
    if mask.ndim!=3 or not all(mask.shape): raise ValueError('Expected nonempty ZYX mask')
    if not np.all((mask==0)|(mask==1)): raise ValueError('Provide a binary single-instance mask')
    pads=[(0,(-n)%p) for n,p in zip(mask.shape,patch_size)]
    m=np.pad(mask.astype(bool),pads)
    z,y,x=(n//p for n,p in zip(m.shape,patch_size)); pz,py,px=patch_size
    counts=m.reshape(z,pz,y,py,x,px).sum(axis=(1,3,5))
    coords=np.argwhere(counts>0)
    return coords,counts[counts>0],(z,y,x)


def spatial_subsample(coords, cap, spacing_zyx, patch_size, seed=0):
    """Equal-width physical-space occupied bins, one random token/bin + random fill.

    No O(N*K) FPS. Sparse branches get a representative when they occupy a
    distinct cell; this is not a guarantee that every terminal branch survives.
    """
    if cap is None or len(coords)<=cap: return np.arange(len(coords))
    if cap<1: raise ValueError('max_tokens must be positive or None')
    scale=np.asarray(spacing_zyx,dtype=float)*np.asarray(patch_size)
    if scale.shape!=(3,) or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('spacing_zyx must contain three finite positive values')
    xyz=coords*scale; xyz-=xyz.min(axis=0)
    lo=0.; hi=float(np.max(np.ptp(xyz,axis=0)))+1.
    for _ in range(32):
        size=(lo+hi)/2
        cells=np.floor(xyz/max(size,1e-12)).astype(np.int64)
        if len(np.unique(cells,axis=0))>cap: lo=size
        else: hi=size
    cells=np.floor(xyz/hi).astype(np.int64)
    rng=np.random.default_rng(seed); shuffled=rng.permutation(len(coords))
    _,first=np.unique(cells[shuffled],axis=0,return_index=True)
    selected=shuffled[first]
    assert len(selected)<=cap
    remaining=np.setdiff1d(np.arange(len(coords)),selected)
    if len(selected)<cap: selected=np.concatenate([selected,rng.choice(remaining,cap-len(selected),replace=False)])
    return np.sort(selected)


class InstanceTokenEncoder(nn.Module):
    def __init__(self, mae, max_tokens=2048):
        super().__init__()
        if max_tokens is not None and max_tokens<1: raise ValueError('Invalid max_tokens')
        self.patch_embed=mae.patch_embed
        self.cls_token=mae.cls_token
        self.blocks=mae.blocks
        self.norm=mae.norm
        self.max_tokens=max_tokens
        self.training_grid=tuple(n//p for n,p in zip(mae.patch_embed.img_size,mae.patch_embed.patch_size))
        self._encode_axis=None

    @classmethod
    def from_checkpoint(cls, checkpoint, project, max_tokens=2048):
        # torch.load(weights_only=False) is for this trusted local training checkpoint only.
        sys.path.insert(0,str(Path(project).resolve()))
        from mae3d.models_mae import MaskedAutoencoderViT3D
        from mae3d.util.pos_embed import get_1d_sincos_pos_embed_from_grid
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        args=vars(ck['args']) if hasattr(ck['args'],'__dict__') else ck['args']
        if args['in_chans']!=1: raise ValueError('This interface expects grayscale EM')
        mae=MaskedAutoencoderViT3D(img_size=args['input_size'],patch_size=args['patch_size'],
            in_chans=1,embed_dim=args['embed_dim'],depth=args['depth'],num_heads=args['num_heads'],
            decoder_embed_dim=args['decoder_embed_dim'],decoder_depth=args['decoder_depth'],
            decoder_num_heads=args['decoder_num_heads'],mlp_ratio=args['mlp_ratio'],
            norm_pix_loss=args['norm_pix_loss'],norm_layer=partial(nn.LayerNorm,eps=1e-6))
        mae.load_state_dict(ck['model'],strict=True)
        result=cls(mae,max_tokens)
        result._encode_axis=get_1d_sincos_pos_embed_from_grid
        result.checkpoint_epoch=int(ck['epoch'])+1
        return result

    def forward(self, em, mask, *, spacing_zyx=(1.,1.,1.), seed=0, warn=True):
        """One instance. Returns N×D contextual tokens; gradients remain enabled.

        spacing defaults to voxel units, NOT a claim about physical voxel size.
        Pass actual ZYX spacing for physically uniform spatial subsampling.
        """
        if em.ndim!=3 or tuple(em.shape)!=tuple(mask.shape): raise ValueError('EM/mask must have identical 3D ZYX shape')
        p=self.patch_embed.patch_size
        mask_np=mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
        coords,counts,grid=patch_counts(mask_np,p)
        raw_count=len(coords)
        if not raw_count: raise ValueError('Empty instance mask')
        ids=spatial_subsample(coords,self.max_tokens,spacing_zyx,p,seed)
        coords=coords[ids]; counts=counts[ids]
        outside=bool(np.any(coords>=np.asarray(self.training_grid)))
        if warn and len(ids)<raw_count:
            warnings.warn(f'Instance tokens capped: {raw_count} -> {len(ids)} ({len(ids)/raw_count:.1%} retained)',stacklevel=2)
        if warn and outside: warnings.warn('Some token coordinates exceed pretrained positional range',stacklevel=2)
        weight=self.patch_embed.proj.weight
        x=torch.as_tensor(em,device=weight.device,dtype=weight.dtype)
        if not bool(torch.isfinite(x).all()): raise ValueError('Nonfinite EM values')
        # Match pretraining per-volume min-max; do not zero EM outside instance.
        x=(x-x.amin())/(x.amax()-x.amin()+1e-8)
        dz,dy,dx=[(-n)%q for n,q in zip(x.shape,p)]
        x=F.pad(x[None,None],(0,dx,0,dy,0,dz),mode='replicate')[0,0]
        c=torch.as_tensor(coords,device=x.device,dtype=torch.long)
        patches=x.unfold(0,p[0],p[0]).unfold(1,p[1],p[1]).unfold(2,p[2],p[2])
        patches=patches[c[:,0],c[:,1],c[:,2]].reshape(len(c),-1)
        # Exactly equivalent to stride=kernel Conv3d for selected grayscale patches.
        projected=F.linear(patches,weight.reshape(weight.shape[0],-1),self.patch_embed.proj.bias)
        dim=projected.shape[-1]
        pe=np.concatenate([self._encode_axis(dim//3,coords[:,axis].astype(np.float32)) for axis in range(3)],axis=1)
        tokens=projected+torch.as_tensor(pe,device=x.device,dtype=projected.dtype)
        # Keep the pretrained CLS internally; do not choose CLS vs pooling as a readout.
        h=torch.cat([self.cls_token.expand(1,-1,-1),tokens[None]],dim=1)
        for block in self.blocks: h=block(h)
        h=self.norm(h)
        return dict(tokens=h[0,1:],cls_token=h[0,0],patch_tokens=projected,
            coordinates_zyx=c,foreground_voxels=torch.as_tensor(counts,device=x.device),
            raw_token_count=raw_count,retained_token_count=len(c),retained_fraction=len(c)/raw_count,
            grid_zyx=grid,position_out_of_training_range=outside)

    def forward_paths(self, em_path, mask_path, *, axes='zyx', **kwargs):
        em=np.load(em_path,allow_pickle=False); mask=np.load(mask_path,allow_pickle=False)
        if axes=='xyz': em=em.transpose(2,1,0); mask=mask.transpose(2,1,0)
        elif axes!='zyx': raise ValueError('axes must be zyx or xyz')
        return self(em,mask,**kwargs)
