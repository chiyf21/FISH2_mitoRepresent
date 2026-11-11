import math
import torch
from torch import nn
from torch.nn import functional as F
from mito_voxel_mae_v4.model import VoxelMAE as ImagePointMAE
from mito_voxel_mae_v2.model import fps_groups
from .targets import low_frequency_basis

class VoxelMAE(ImagePointMAE):
    def __init__(self,*args,objective='voxel',lf_with_occ=False,**kwargs):
        super().__init__(*args,**kwargs);self.objective=objective;self.lf_with_occ=lf_with_occ
        del self.scale;del self.voxel
        self.point=nn.Sequential(nn.Linear(self.channels+3,self.dim),nn.GELU(),nn.Linear(self.dim,self.dim))
        latent=self.size[0].in_features
        output=54 if objective=='voxel' else 10+(27 if lf_with_occ else 0)
        self.block=nn.Sequential(nn.Linear(latent+3+36,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,output))
        basis,modes,frequencies=low_frequency_basis()
        self.register_buffer('lf_basis',torch.from_numpy(basis))
        self.register_buffer('wavelength_um',torch.tensor([.024,.048,.096,.192,.384,.768]))
        self.lf_modes_zyx=modes;self.lf_frequencies_cycles_nm=frequencies

    def encode(self,b):
        xyz=b['points_um'];valid=b['points_valid'];feat=self.point_features(b)
        center_idx=fps_groups(xyz,valid,self.groups)
        batch=torch.arange(xyz.shape[0],device=xyz.device)[:,None];centers=xyz[batch,center_idx]
        distance=(centers.float()[:,:,None]-xyz.float()[:,None]).square().sum(-1).masked_fill(~valid[:,None],float('inf'))
        idx=torch.argsort(distance,dim=-1,stable=True)[...,:min(self.k,xyz.shape[1])]
        bi=torch.arange(xyz.shape[0],device=xyz.device)[:,None,None]
        token=self.point(torch.cat((xyz[bi,idx]-centers[:,:,None],feat[bi,idx]),-1))
        token=token.masked_fill(~valid[bi,idx,None],float('-inf')).amax(2)
        token=token+self.pos(centers)
        return self.bottleneck(self.encoder(token).mean(1))

    def forward(self,b):
        z=self.encode(b);q=b['query_um']
        angles=q[...,None]/self.wavelength_um*2*math.pi
        fourier=torch.cat((angles.sin(),angles.cos()),-1).flatten(-2)
        pred=self.block(torch.cat((z[:,None].expand(-1,q.shape[1],-1),q,fourier),-1)).float()
        out={'z':z,'log_size':self.size(z),'basis':self.lf_basis,'objective':self.objective}
        if self.objective=='voxel':
            pred=pred.reshape(*pred.shape[:2],27,2)
            out['occ']=pred[...,0];out['image']=pred[...,1]
            with torch.autocast(device_type=z.device.type,enabled=False):
                out['lf']=out['image']@self.lf_basis.T/math.sqrt(27)
        else:
            out['lf']=pred[...,:10]
            with torch.autocast(device_type=z.device.type,enabled=False):
                out['image']=out['lf']@self.lf_basis*math.sqrt(27)
            if self.lf_with_occ:out['occ']=pred[...,10:]
        return out

def metric_keys(objective,lf_with_occ=False):
    keys=['loss','lowfreq','size','size_relative_error','intensity_mae','dc_mae','ac_mae']
    if objective=='voxel':keys+=['intensity']
    if objective=='voxel' or lf_with_occ:keys+=['occupancy','dice_loss','iou']
    return keys

def loss_terms(out,b):
    qvalid=b['query_valid'].float();occ=b['occupancy'].float();truth=b['intensity'].float()
    # Every item first averages its valid queries. Padding and skipped rank fillers
    # never contribute, regardless of the number of safe crops in an instance.
    qmean=lambda x:(x*qvalid).sum(1)/qvalid.sum(1).clamp_min(1)
    with torch.autocast(device_type=truth.device.type,enabled=False):
        target_lf=truth@out['basis'].T/math.sqrt(27)
        low_image=out['lf']@out['basis']*math.sqrt(27)
        target_low_image=target_lf@out['basis']*math.sqrt(27)
    # Compare in image units, not a mean over 10 tiny coefficients. This makes
    # the auxiliary image-task weight comparable with the voxel intensity loss.
    low=qmean(F.smooth_l1_loss(low_image,target_low_image,reduction='none').mean(-1))
    size=F.smooth_l1_loss(out['log_size'].float(),b['size_target'].float(),reduction='none').mean(1)
    total=(low if out['objective']=='lowfreq' else torch.zeros_like(low))+size*.2
    values={'lowfreq':low,'size':size}
    if out['objective']=='voxel':
        residual=F.smooth_l1_loss(out['image'],truth,reduction='none')
        weight=qvalid[:,:,None];fg=occ*weight;bg=(1-occ)*weight
        img_fg=(residual*fg).sum((1,2))/fg.sum((1,2)).clamp_min(1)
        img_bg=(residual*bg).sum((1,2))/bg.sum((1,2)).clamp_min(1)
        image=.7*img_fg+.3*img_bg
        values['intensity']=image;total=total+image
    if 'occ' in out:
        weight=qvalid[:,:,None];fg=occ*weight;bg=(1-occ)*weight
        bce=F.binary_cross_entropy_with_logits(out['occ'],occ,reduction='none')
        bce=.5*(bce*fg).sum((1,2))/fg.sum((1,2)).clamp_min(1)+.5*(bce*bg).sum((1,2))/bg.sum((1,2)).clamp_min(1)
        probability=out['occ'].sigmoid()
        dice=1-(2*(probability*occ*weight).sum((1,2))+1)/((probability*weight).sum((1,2))+(occ*weight).sum((1,2))+1)
        values.update(occupancy=bce,dice_loss=dice);total=total+bce+.5*dice
        with torch.no_grad():
            hit=probability>.5;gt=occ>.5
            values['iou']=((hit & gt)*weight).sum((1,2))/((hit|gt)*weight).sum((1,2)).clamp_min(1)
    with torch.no_grad():
        values['intensity_mae']=qmean((out['image']-truth).abs().mean(-1))
        values['dc_mae']=qmean((out['lf'][...,0]-target_lf[...,0]).abs())
        values['ac_mae']=qmean((out['lf'][...,1:]-target_lf[...,1:]).abs().mean(-1))
        values['size_relative_error']=(out['log_size'].float()-b['size_target']).exp().sub(1).abs().mean(1)
    values['loss']=total
    sw=b['sample_weight'].reshape(-1)
    return {k:(v*sw).sum()/sw.sum().clamp_min(1) for k,v in values.items()}
