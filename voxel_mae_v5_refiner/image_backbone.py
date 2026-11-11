"""Physical-scale 512px tiles, packed 2D slices, unchanged-Z residual fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from mito_voxel_mae_v3.model import SliceUNet,ZResidual
from mito_voxel_mae_v2.model import VoxelMAE as BaseMAE,loss_terms

class VoxelMAE(BaseMAE):
    def __init__(self,*args,slice_microbatch=4,tile_batch=2,activation_checkpointing=True,**kwargs):
        super().__init__(*args,**kwargs);self.channels=self.unet.a.c1.out_channels
        del self.unet
        self.slice_unet=SliceUNet(self.channels)
        self.z_blocks=nn.ModuleList([ZResidual(self.channels) for _ in range(3)])
        self.slice_microbatch=slice_microbatch;self.tile_batch=tile_batch
        self.activation_checkpointing=activation_checkpointing
        self.core=(16,256,256);self.xy_halo=128;self.z_halo=3

    def extract_tiles(self,*args):
        """CPU tile tensors precede GPU interpolation grids; no loop-state closures."""
        m=len(args)//2;tiles=args[:m];grids=args[m:]
        lengths=[len(t) for t in tiles];device=next(self.parameters()).device
        slices=torch.cat(tiles);features=[]
        for x in slices.split(self.slice_microbatch):
            features.append(self.slice_unet(x.to(device,non_blocking=True)))
        packed=torch.cat(features);nz=max(lengths);maps=[];masks=[]
        for f,n in zip(packed.split(lengths),lengths):
            maps.append(F.pad(f.permute(1,0,2,3),(0,0,0,0,0,nz-n)))
            masks.append(F.pad(torch.ones((1,n,1,1),device=device),(0,0,0,0,0,nz-n)))
        fmap=torch.stack(maps);domain=torch.stack(masks).to(fmap.dtype)
        for block in self.z_blocks:fmap=block(fmap,domain)
        output=[]
        for i,(grid,n) in enumerate(zip(grids,lengths)):
            grid=grid.clone();grid[...,2]=(grid[...,2]+1)*(n/nz)-1
            value=F.grid_sample(fmap[i:i+1],grid[None,:,None,None],mode='bilinear',align_corners=False)
            output.append(value[0,:,:,0,0].T.float())
        return tuple(output)

    def point_features(self,b):
        device=next(self.parameters()).device
        result=torch.zeros((*b['points_vox'].shape[:2],self.channels),device=device)
        jobs=[]
        def flush(result):
            if not jobs:return result
            args=tuple(j[2] for j in jobs)+tuple(j[3] for j in jobs)
            # eval() controls Dropout, not whether an autograd graph needs memory.
            use_checkpoint=self.activation_checkpointing and torch.is_grad_enabled()
            values=checkpoint(self.extract_tiles,*args,use_reentrant=False) if use_checkpoint else self.extract_tiles(*args)
            # Scatter into a flattened buffer, preserving autograd for every tile.
            flat=result.flatten(0,1);N=result.shape[1]
            for (instance,ids,_,_),value in zip(jobs,values):
                flat=flat.index_copy(0,ids+instance*N,value)
            jobs.clear()
            return flat.reshape_as(result)
        for i,channels in enumerate(b['volumes']):
            vox=b['points_vox'][i];valid=b['points_valid'][i]
            cells=torch.floor(vox/torch.tensor(self.core,device=device)).long()
            shape=channels[0].shape[-3:]
            for cell in torch.unique(cells[valid],dim=0).tolist():
                ids=((cells==torch.tensor(cell,device=device)).all(-1)&valid).nonzero().flatten()
                z0=max(0,cell[0]*16-self.z_halo);z1=min(shape[0],(cell[0]+1)*16+self.z_halo)
                y0=cell[1]*256-self.xy_halo;x0=cell[2]*256-self.xy_halo
                ys=max(0,y0);ye=min(shape[1],y0+512);xs=max(0,x0);xe=min(shape[2],x0+512)
                values=[]
                for channel in channels:
                    crop=channel[0,z0:z1,ys:ye,xs:xe]
                    values.append(F.pad(crop,(xs-x0,x0+512-xe,ys-y0,y0+512-ye)))
                tile=torch.stack(values,1) # [Z,4,512,512], no interpolation/resizing
                low=torch.tensor([z0,y0,x0],device=device)
                extent=torch.tensor([z1-z0,512,512],device=device)
                grid=(2*(vox[ids]+.5-low)/extent-1).flip(-1)
                jobs.append((i,ids,tile,grid))
                if len(jobs)>=self.tile_batch:result=flush(result)
        return flush(result)
