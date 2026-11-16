import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pad_sequence
import base
LAYERS=(9,10,11,12)
SEED=20261029
def prepare(rows,pointdata,point,grouping,image,device,epoch,training,point_only=False):
    from instance_tokens import patch_counts,spatial_subsample
    pts=[];images=[]
    for r in rows:
        for view in range(2 if training else 1):
            if training and point_only:
                seed=base.SEED+r['cohort_index']*1009+view*9176+(epoch+1)*100003
                rng=np.random.default_rng(seed);k=int(rng.integers(0,4));fx=bool(rng.integers(0,2));fy=bool(rng.integers(0,2))
                quarter=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],np.float32);rotation=np.linalg.matrix_power(quarter,k)
                if fx:rotation=rotation@np.diag([-1.,1.,1.]).astype(np.float32)
                if fy:rotation=rotation@np.diag([1.,-1.,1.]).astype(np.float32)
                xyz=base.point_xyz(r,pointdata,view,device,rotation,.01,seed+31)
            elif training:
                em,mask,xyz,seed=base.view_transform(r,pointdata,view,epoch,True,device)
            else:
                seed=SEED+r['cohort_index']*1009
                if not point_only:
                    em=torch.as_tensor(np.array(np.load(r['em_path'],mmap_mode='r'),dtype=np.float32),device=device)
                    mask=torch.as_tensor(np.array(np.load(r['mask_path'],mmap_mode='r'),dtype=np.uint8),device=device)
                xyz=base.point_xyz(r,pointdata,0,device)
            with torch.no_grad():
                with torch.random.fork_rng(devices=[device.index]):
                    torch.cuda.manual_seed(seed+17);g,c=grouping(xyz)
                tokens=point.encoder(g);pos=point.pos_embed(c);pts.append((tokens[0],pos[0]))
                if point_only:continue
                p=image.patch_embed.patch_size
                co,_,_=patch_counts(mask.cpu().numpy(),p)
                if len(co)==0:raise ValueError('empty mask')
                co=co[spatial_subsample(co,image.max_tokens,(1.,1.,1.),p,seed)]
                x=(em-em.amin())/(em.amax()-em.amin()+1e-8)
                dz,dy,dx=[(-n)%q for n,q in zip(x.shape,p)]
                x=F.pad(x[None,None],(0,dx,0,dy,0,dz),mode='replicate')[0,0]
                ci=torch.as_tensor(co,device=device,dtype=torch.long)
                patches=x.unfold(0,p[0],p[0]).unfold(1,p[1],p[1]).unfold(2,p[2],p[2])
                patches=patches[ci[:,0],ci[:,1],ci[:,2]].reshape(len(co),-1)
                w=image.patch_embed.proj.weight
                projected=F.linear(patches,w.reshape(w.shape[0],-1),image.patch_embed.proj.bias)
                pe=np.concatenate([image._encode_axis(256,co[:,a].astype(np.float32)) for a in range(3)],1)
                images.append(projected+torch.as_tensor(pe,device=device,dtype=projected.dtype))
    pt=torch.stack([v[0] for v in pts]);pp=torch.stack([v[1] for v in pts])
    if point_only:return pt,pp,None,None
    im=pad_sequence(images,batch_first=True)
    lengths=torch.tensor([len(t) for t in images],device=device)
    valid=torch.arange(im.shape[1],device=device)[None]<lengths[:,None]
    valid=torch.cat([torch.ones(len(valid),1,device=device,dtype=torch.bool),valid],1)
    return pt,pp,im,valid

class System(nn.Module):
    def __init__(self, point, image):
        super().__init__()
        self.point = point
        self.image = image
        width = point.trans_dim
        if width != 768 or image.cls_token.shape[-1] != width:
            raise ValueError('Fusion requires native 768D Point and Image CLS tokens')
        self.point_cls = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.point.requires_grad_(False)
        self.image.requires_grad_(False)
        for block in point.blocks.blocks[-4:]:
            block.requires_grad_(True)
        self.fusion = nn.Sequential(nn.Linear(2 * width, 384), nn.GELU(),
                                    nn.Linear(384, 384), nn.LayerNorm(384))
        self.fusion.requires_grad_(False)
        self.point_nna = None
        self.phase = 'pretrain'

    def train(self,mode=True):
        super().train(mode)
        # Disabling stochastic depth permits exact gradient recomputation/checkpointing.
        self.point.eval();self.image.eval()
        return self
    def encode(self,inputs,views):
        pt,pp,im,valid=inputs;b=len(pt)
        h=torch.cat([self.point_cls.expand(b,-1,-1),pt],1)
        pos=torch.cat([pp.new_zeros(b, 1, pt.shape[-1]),pp],1)
        pv={}
        for j,block in enumerate(self.point.blocks.blocks,1):
            if torch.is_grad_enabled():h=checkpoint(block,h+pos,use_reentrant=False)
            else:h=block(h+pos)
            if j in LAYERS:
                cls = h[:, 0].float()
                cls = self.point.norm(cls) if j == 12 else F.layer_norm(cls, (cls.shape[-1],))
                pv[j] = cls.reshape(-1, views, cls.shape[-1])
        if im is None:return pv,{}
        h=torch.cat([self.image.cls_token.expand(b,-1,-1),im],1)
        mask=valid[:,None,None,:]
        iv={}
        for j,block in enumerate(self.image.blocks,1):
            if torch.is_grad_enabled():h=checkpoint(lambda x,m,bl=block:bl(x,attn_mask=m),h,mask,use_reentrant=False)
            else:h=block(h,attn_mask=mask)
            if j in LAYERS:
                cls = h[:, 0].float()
                cls = self.image.norm(cls) if j == 12 else F.layer_norm(cls, (cls.shape[-1],))
                iv[j] = cls.reshape(-1, views, cls.shape[-1])
        return pv,iv
    def forward(self,inputs,ids=None,views=2,loss=True):
        p,i=self.encode(inputs,views)
        if self.phase=='pretrain':
            if not loss:return p,i,{},{}
            total, parts = self.point_nna(p, ids)
            return total, parts, {}
        concat = {layer: torch.cat((p[layer], i[layer]), dim=-1) for layer in LAYERS}
        fused = {layer: self.fusion(z) for layer, z in concat.items()}
        if not loss:
            return p, i, fused, concat
        parts={};losses=[]
        for l in LAYERS:
            value, terms=vicreg(fused[l])
            contrastive, extra=filtered_infonce(fused[l],ids,topk=20,temperature=0.2)
            point_reg, point_terms = cls_regularizer(p[l])
            image_reg, image_terms = cls_regularizer(i[l])
            losses.append(value+contrastive+point_reg+image_reg)
            terms.update({'point_'+k:v for k,v in point_terms.items()})
            terms.update({'image_'+k:v for k,v in image_terms.items()})
            terms=dict(terms,infonce=contrastive.detach(),**extra)
            for k,v in terms.items():parts[f'{l}/{k}']=v
        return torch.stack(losses).mean(), parts, {}

def vicreg(z):
    # Differentiable global-batch gather; pool both views as in the old VICReg implementation.
    if dist.is_initialized() and dist.get_world_size()>1:
        from torch.distributed.nn.functional import all_gather
        z=torch.cat(all_gather(z.float()),dim=0)
    a,b=z[:,0].float(),z[:,1].float()
    inv=F.mse_loss(a,b)
    both=torch.cat((a,b),dim=0)
    std=torch.sqrt(both.var(dim=0,unbiased=False)+1e-4)
    var=F.relu(1-std).mean()
    centered=both-both.mean(0,keepdim=True)
    cov=centered.T@centered/(len(centered)-1)
    off=~torch.eye(a.shape[-1],device=a.device,dtype=torch.bool)
    covloss=cov[off].square().sum()/a.shape[-1]
    return 25*inv+5*var+0.2*covloss,dict(invariance=inv.detach(),variance=var.detach(),covariance=covloss.detach())

def cls_regularizer(z):
    # Apply the regularizer to the same normalized CLS seen by fusion.
    if dist.is_initialized() and dist.get_world_size()>1:
        from torch.distributed.nn.functional import all_gather
        z=torch.cat(all_gather(z.float()),dim=0)
    both=torch.cat((z[:,0].float(),z[:,1].float()),dim=0)
    var=F.relu(1-torch.sqrt(both.var(dim=0,unbiased=False)+1e-4)).mean()
    centered=both-both.mean(0,keepdim=True)
    cov=centered.T@centered/(len(centered)-1)
    off=~torch.eye(both.shape[-1],device=both.device,dtype=torch.bool)
    covloss=cov[off].square().sum()/both.shape[-1]
    weighted=1.25*var+0.05*covloss
    return weighted,dict(variance=var.detach(),covariance=covloss.detach(),regularizer=weighted.detach())

def filtered_infonce(z,ids,topk=20,temperature=0.2,debug=False):
    if dist.is_initialized() and dist.get_world_size()>1:
        from torch.distributed.nn.functional import all_gather
        z=torch.cat(all_gather(z.float()),dim=0)
        gathered=[torch.empty_like(ids) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered,ids);ids=torch.cat(gathered)
    b,v,d=z.shape
    assert v==2 and b>=2
    assert len(torch.unique(ids))==b, 'Repeated instance IDs must not be treated as negatives'
    u=F.normalize(z.float(),dim=-1)
    # Rank OTHER INSTANCES using the stop-gradient, unit-normalized two-view mean.
    centers=F.normalize(u.detach().mean(1),dim=-1)
    similarities=centers@centers.T
    similarities.fill_diagonal_(-torch.inf)
    k=min(topk,max(0,b-2))  # retain at least one negative instance for small batches
    excluded=torch.zeros(b,b,device=z.device,dtype=torch.bool)
    if k:excluded.scatter_(1,similarities.topk(k,dim=1).indices,True)
    ignore=excluded.repeat_interleave(2,0).repeat_interleave(2,1)
    ignore=ignore|torch.eye(2*b,device=z.device,dtype=torch.bool)
    flat=u.reshape(2*b,d);logits=flat@flat.T/temperature
    labels=torch.arange(2*b,device=z.device)^1
    assert not ignore[torch.arange(2*b,device=z.device),labels].any()
    logits=logits.masked_fill(ignore,-torch.inf)
    loss=F.cross_entropy(logits,labels)
    with torch.no_grad():
        negative_mask=~ignore.clone()
        negative_mask[torch.arange(2*b,device=z.device),labels]=False
        cosines=flat@flat.T
        extra=dict(negatives_per_anchor=negative_mask.sum(1).float().mean(),excluded_instances=loss.new_tensor(float(k)),positive_cosine=cosines[torch.arange(2*b,device=z.device),labels].mean(),negative_cosine=cosines[negative_mask].mean())
    return (loss,extra,ignore,labels) if debug else (loss,extra)

def cov_metrics(z):
    x=np.array(z,dtype=np.float64);mu=x.mean(0);x-=mu
    gram = x @ x.T if len(x) < x.shape[1] else x.T @ x
    ev=np.maximum(np.linalg.eigvalsh(gram/(len(x)-1)),0)[::-1];q=ev/max(ev.sum(),1e-30);pos=q>0
    return dict(effective_rank=float(np.exp(-(q[pos]*np.log(q[pos])).sum())),total_variance=float(ev.sum()),pc1=float(q[0]),pc95=int(np.searchsorted(np.cumsum(q),.95)+1),mean_norm=float(np.linalg.norm(mu)),instances=len(z))

def monitor(model,recs,indices,points,grouping,device,rank,world,epoch,phase,export=False):
    model.eval();local=indices[rank::world];result={};seen=[]
    with torch.inference_mode():
        for start in range(0,len(local),8):
            pos=local[start:start+8];rr=[recs[int(j)] for j in pos]
            inp=prepare(rr,points,model.point,grouping,model.image,device,0,False,point_only=model.phase=='pretrain')
            with torch.autocast('cuda',dtype=torch.bfloat16):p,i,f,c=model(inp,views=1,loss=False)
            for name,values in [('point',p),('image',i),('fused',f),('concat',c)]:
                for l,x in values.items():result.setdefault(f'{name}_{l}',[]).append(x[:,0].float().cpu().numpy())
            seen.extend(pos)
    packed=dict(ids=seen,features={k:np.concatenate(v) for k,v in result.items()})
    gathered=[None]*world;dist.all_gather_object(gathered,packed)
    metrics=None
    if rank==0:
        ix=np.concatenate([v['ids'] for v in gathered]);order=np.argsort(ix)
        feats={k:np.concatenate([v['features'][k] for v in gathered])[order] for k in result}
        metrics={k:cov_metrics(v) for k,v in feats.items()}
        if phase == 'refine':
            for layer in LAYERS:
                vp = metrics[f'point_{layer}']['total_variance']
                vi = metrics[f'image_{layer}']['total_variance']
                metrics[f'concat_{layer}']['point_variance_share'] = vp / (vp + vi) if vp + vi else None
        record=dict(phase=phase,epoch=epoch,time=time.time(),metrics=metrics,probe='fixed canonical input; no augmentation; point sample0 fixed FPS; same IDs every epoch')
        append(ROOT/'epoch_metrics.jsonl',record)
        np.savez(ROOT/f'{phase}_epoch{epoch:02d}_probe.npz',ids=ix[order],**feats)
        if export:
            for k,v in feats.items():np.save(ROOT/'results'/f'{k}_embeddings.npy',v)
        print(json.dumps(record),flush=True)
    return metrics
