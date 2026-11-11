from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.GELU(),
                               nn.Conv3d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.GELU())
    def forward(self,x): return self.net(x)


class AnisotropicUNet(nn.Module):
    """Two-level U-Net; XY downsampling supports crops with Z=1."""
    def __init__(self, channels=24):
        super().__init__()
        self.a=ConvBlock(3,channels); self.b=ConvBlock(channels,channels*2)
        self.c=ConvBlock(channels*2,channels*4)
        self.upb=ConvBlock(channels*6,channels*2); self.upa=ConvBlock(channels*3,channels)
    def forward(self,x):
        a=self.a(x); b=self.b(F.avg_pool3d(a,(1,2,2),ceil_mode=True))
        c=self.c(F.avg_pool3d(b,(1,2,2),ceil_mode=True))
        b=self.upb(torch.cat((b,F.interpolate(c,size=b.shape[-3:],mode='trilinear',align_corners=False)),1))
        return self.upa(torch.cat((a,F.interpolate(b,size=a.shape[-3:],mode='trilinear',align_corners=False)),1))


def fps_groups(xyz,valid,n):
    """Padding never participates. Short clouds repeat centers only, not observations."""
    b,total,_=xyz.shape
    if not valid.any(1).all(): raise ValueError('Empty support cloud')
    selected=torch.zeros(b,n,device=xyz.device,dtype=torch.long)
    dist=torch.full((b,total),float('inf'),device=xyz.device)
    center=valid.long().argmax(1); batch=torch.arange(b,device=xyz.device)
    for i in range(n):
        selected[:,i]=center
        d=(xyz-xyz[batch,center,None]).square().sum(-1)
        dist=torch.minimum(dist,d).masked_fill(~valid,-1)
        center=dist.argmax(1)
    return selected


class VoxelMAE(nn.Module):
    def __init__(self, dim=256,depth=6,heads=8,latent=256,groups=64,k=32,channels=24):
        super().__init__(); self.groups=groups; self.k=k; self.dim=dim
        self.unet=AnisotropicUNet(channels)
        self.point=nn.Sequential(nn.Linear(channels+6,dim),nn.GELU(),nn.Linear(dim,dim))
        self.pos=nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.scale=nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        layer=nn.TransformerEncoderLayer(dim,heads,dim*4,dropout=.1,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False)
        self.bottleneck=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,latent),nn.GELU(),nn.Linear(latent,latent))
        self.voxel=nn.Sequential(nn.Linear(latent+6+36,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,2))
        self.size=nn.Sequential(nn.Linear(latent,128),nn.GELU(),nn.Linear(128,3))

    def encode(self,b):
        fmap=self.unet(torch.cat((b['image'],b['observed'],b['image_valid']),1))
        xyz=b['points_r']; t=b['points_t']; valid=b['points_valid']
        grid=b['points_grid'].unsqueeze(2).unsqueeze(2)
        feat=F.grid_sample(fmap,grid,mode='bilinear',align_corners=False,padding_mode='zeros')[:,:, :,0,0].transpose(1,2)
        center_idx=fps_groups(xyz,valid,self.groups)
        batch=torch.arange(xyz.shape[0],device=xyz.device)[:,None]
        centers=xyz[batch,center_idx]
        d=torch.cdist(centers.float(),xyz.float()).masked_fill(~valid[:,None],float('inf'))
        idx=d.topk(min(self.k,xyz.shape[1]),largest=False).indices
        bi=torch.arange(xyz.shape[0],device=xyz.device)[:,None,None]
        local=xyz[bi,idx]-centers[:,:,None]
        local_t=t[bi,idx]-t[batch,center_idx][:,:,None]
        token=self.point(torch.cat((local,local_t,feat[bi,idx]),-1))
        token=token.masked_fill(~valid[bi,idx,None],float('-inf')).amax(2)
        token=token+self.pos(centers)+self.scale(b['log_size'])[:,None]
        return self.bottleneck(self.encoder(token).mean(1))

    def decode(self,z,q_t,q_r):
        frequencies=2.**torch.arange(6,device=q_t.device,dtype=q_t.dtype)
        angles=q_t[...,None]*frequencies*torch.pi*2
        fourier=torch.cat((angles.sin(),angles.cos()),-1).flatten(-2)
        return self.voxel(torch.cat((z[:,None].expand(-1,q_t.shape[1],-1),q_t,q_r,fourier),-1))

    def forward(self,b):
        z=self.encode(b)
        return {'z':z,'voxels':self.decode(z,b['query_t'],b['query_r']),'log_size':self.size(z)}


def loss_terms(out,b):
    pred=out['voxels'].float(); occ=b['occupancy'].float()
    hidden=b['query_hidden'].float(); weight=(.25+.75*hidden)*b['query_valid'].float()
    bce=F.binary_cross_entropy_with_logits(pred[...,0],occ,reduction='none')
    # Per-instance foreground/background balance rather than all-background shortcut.
    fg=(weight*occ); bg=(weight*(1-occ))
    fg_loss=(bce*fg).sum()/fg.sum().clamp_min(1)
    bg_loss=(bce*bg).sum()/bg.sum().clamp_min(1)
    prob=pred[...,0].sigmoid()
    dice=1-(2*(prob*occ*weight).sum()+1)/((prob*weight).sum()+(occ*weight).sum()+1)
    valid=b['query_valid'].float()*weight
    residual=F.smooth_l1_loss(pred[...,1],b['intensity'].float(),reduction='none')
    # Foreground and external regions each retain a meaningful contribution.
    iv_fg=valid*occ; iv_bg=valid*(1-occ)
    img=.5*((residual*iv_fg).sum()/iv_fg.sum().clamp_min(1)+(residual*iv_bg).sum()/iv_bg.sum().clamp_min(1))
    size=F.smooth_l1_loss(out['log_size'].float(),b['size_target'].float())
    total=fg_loss*.5+bg_loss*.5+dice*.5+img+size*.2
    with torch.no_grad():
        hit=prob>.5; truth=occ>.5
        iou=(hit&truth).sum().float()/(hit|truth).sum().clamp_min(1)
        err=(out['log_size'].float()-b['size_target']).exp().sub(1).abs().mean()
    return {'loss':total,'occupancy':.5*(fg_loss+bg_loss),'dice_loss':dice,'intensity':img,'size':size,'iou':iou,'size_relative_error':err}
