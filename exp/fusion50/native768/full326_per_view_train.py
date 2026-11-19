"""Train matched Fusion runs from Point CLS initialization with the official per-view VICReg statistics."""
import argparse
import json
import math
import random
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import core, load_models, rng_state


def per_view(z):
    with torch.autocast('cuda',enabled=False):
        a,b=z[:,0].float(),z[:,1].float()
        inv=F.mse_loss(a,b)
        a=torch.cat(all_gather(a),dim=0)
        b=torch.cat(all_gather(b),dim=0)
        if len(a)<=a.shape[-1]:
            raise ValueError('Per-view global batch must exceed the projection dimension')
        terms=[]
        for x in (a,b):
            x=x-x.mean(0)
            v=F.relu(1-torch.sqrt(x.var(0,unbiased=True)+1e-4)).mean()
            c=(x.T@x)/(len(x)-1)
            off=c-torch.diag_embed(c.diagonal())
            terms.append((v,off.square().sum()/x.shape[-1]))
        return inv,(terms[0][0]+terms[1][0])/2,terms[0][1]+terms[1][1]


def negative_only_bce(z, margin, temperature, topk):
    z = torch.cat(all_gather(z.float()), dim=0)
    u = F.normalize(z, dim=-1)
    centers = F.normalize(u.detach().mean(1), dim=1)
    similarity = centers @ centers.T
    similarity.fill_diagonal_(-torch.inf)
    exclude = torch.eye(len(z), device=z.device, dtype=torch.bool)
    exclude.scatter_(1, similarity.topk(min(topk, len(z)-2), dim=1).indices, True)
    u = u.flatten(0,1)
    cosines = u @ u.T
    eligible = ~exclude.repeat_interleave(2, 0).repeat_interleave(2, 1)
    return F.softplus((cosines[eligible] - margin) / temperature).mean()


def step(model, batch, indices, points, grouping, device, epoch, scales, rank, opt, micro, *, negative=None):
    records=[batch[int(i)] for i in indices]
    ids=torch.tensor(indices,device=device)
    chunks=[]
    local=[[[] for _ in core.LAYERS] for _ in (0,1)]
    for start in range(0,len(indices),micro):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            t=core.prepare(records[start:start+micro],points,model.point,grouping,model.image,device,epoch,True)
            encoded=model.encode(t,2)
        if opt is not None:
            chunks.append(tuple(x.cpu() for x in t))
        for dest,enc in zip(local,encoded):
            for dst,l in zip(dest,core.LAYERS):
                dst.append(enc[l])
    pooled=[[torch.cat(v).requires_grad_(opt is not None) for v in m] for m in local]
    total=[]; stats=[]
    coeffs=[*scales, negative["weight"] if negative else 1, 1, 1]
    with torch.set_grad_enabled(opt is not None),torch.autocast('cuda',dtype=torch.bfloat16):
        for a,b in zip(*pooled):
            z=model.fusion(torch.cat((a,b),dim=-1))
            inv,var,cov=per_view(z)
            if negative is None:
                info,_=core.filtered_infonce(z,ids,topk=20,temperature=.2)
            else:
                info=negative_only_bce(z,negative["margin"],negative["temperature"],negative["topk"])
            pr,_=core.cls_regularizer(a)
            ir,_=core.cls_regularizer(b)
            vals=[inv,var,cov,info,pr,ir]
            total.append(sum(v*k for v,k in zip(vals,coeffs)))
            stats.append(torch.stack([v.detach().float() for v in vals]))
    loss=torch.stack(total).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    if opt is not None:
        loss.backward()
        start=0
        for chunk in chunks:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a,b=model.encode(tuple(x.to(device) for x in chunk),2)
            count=a[core.LAYERS[0]].shape[0]
            outputs,grads=[],[]
            for enc,leaves in zip((a,b),pooled):
                for l,leaf in zip(core.LAYERS,leaves):
                    outputs.append(enc[l])
                    grads.append(leaf.grad[start:start+count])
            torch.autograd.backward(outputs,grads)
            start+=count
        params=[p for p in model.parameters() if p.requires_grad]
        # Each rank computes its own paired invariance and the shared global terms.
        # Average parameter gradients as DDP would, after differentiable gathers.
        groups={}
        for p in params:
            if p.grad is None:
                raise RuntimeError('Missing gradient on a trainable parameter')
            groups.setdefault(p.grad.dtype,[]).append(p.grad)
        for grads in groups.values():
            flat=torch.cat([v.reshape(-1) for v in grads])
            dist.all_reduce(flat)
            flat.div_(4.)
            offset=0
            for g in grads:
                g.copy_(flat[offset:offset+g.numel()].view_as(g))
                offset+=g.numel()
        norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
        opt.step()
    raw=torch.stack(stats).mean(0)
    dist.all_reduce(raw)
    raw/=4
    name='negative_bce' if negative else 'infonce'
    obj=dict(zip(('invariance','variance','covariance',name,'point_cls','image_cls'),raw.tolist()))
    obj['loss']=sum(v*k for v,k in zip(raw.tolist(),coeffs))
    if negative:
        obj['negative_weighted']=obj[name]*negative['weight']
        if opt is not None:
            obj['grad_norm']=float(norm)
        return obj
    obj['score_25_5_02']=sum(v*k for v,k in zip(raw.tolist(),[25,5,.2,1,1,1]))
    obj['score_25_5_1']=sum(v*k for v,k in zip(raw.tolist(),[25,5,1,1,1,1]))
    obj['score_25_25_1']=sum(v*k for v,k in zip(raw.tolist(),[25,25,1,1,1,1]))
    if opt is not None:
        obj['grad_norm']=float(norm)
    return obj


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','split','point-cls-init','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--weights',nargs=3,type=float,required=True)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--micro',type=int,default=4)
    ap.add_argument('--batch-per-rank',type=int,default=128)
    ap.add_argument('--schedule-epochs',type=int,default=30)
    a=ap.parse_args()
    if not 1 <= a.epochs <= a.schedule_epochs:
        ap.error('epochs must be between 1 and schedule-epochs')
    rank=int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    assert dist.get_world_size()==4
    batch_size=a.batch_per_rank*dist.get_world_size()
    device=torch.device('cuda',rank)
    torch.set_num_threads(2)
    seed=20261118
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=json.loads((a.inputs/'rows.json').read_text())
    paths=json.loads((a.inputs/'caches.json').read_text())
    cache=[np.load(p,mmap_mode='r') for p in paths]

    class Points:
        def __getitem__(self,key):
            i,v=key
            r=rows[i]
            return cache[r['cache_index']][r['point_row'],v]

    records=[dict(row,cohort_index=i,point_row=i) for i,row in enumerate(rows)]
    points=Points()
    split=json.loads(a.split.read_text())
    train=np.array(split['train'])
    val=np.array(split['validation_batches']).ravel()
    if len(val)<batch_size or len(np.unique(val))!=len(val):
        raise ValueError('Validation batches must contain enough distinct instance IDs')
    val=val[:len(val)//batch_size*batch_size].reshape(-1,batch_size)
    steps_per_epoch=len(train)//batch_size
    ck=torch.load(a.point_cls_init,map_location='cpu',mmap=True,weights_only=False)
    config=dict(ck['config'], point_cls_init_checkpoint=str(a.point_cls_init),
                pair_balance=False, fusion_epochs=a.epochs)
    if (ck['stage'], ck['epoch'], ck['world_size']) != ('pretrain', config['point_cls_epochs'], 4):
        raise ValueError('Expected the final Point-only NNA checkpoint')
    point,grouping,image=load_models(config,device)
    model=core.System(point,image).to(device)
    state=ck['model']
    model.point.load_state_dict({k.removeprefix('point.'): v for k,v in state.items() if k.startswith('point.')},strict=True)
    with torch.no_grad():
        model.point_cls.copy_(state['point_cls'])
    del ck, state
    model.phase='refine'
    model.fusion.requires_grad_(True)
    model.image.cls_token.requires_grad_(True)
    for b in model.image.blocks[-4:]:
        b.requires_grad_(True)
    enc=[p for b in list(point.blocks.blocks[-4:])+list(image.blocks[-4:]) for p in b.parameters()]
    heads=[model.point_cls,model.image.cls_token]+list(model.fusion.parameters())
    peak_lr=(1e-5,2e-4)
    opt=torch.optim.AdamW([dict(params=enc,lr=peak_lr[0]),dict(params=heads,lr=peak_lr[1])],
                          weight_decay=1e-4,betas=(.9,.95))
    core.ROOT=a.output
    settings=dict(model=config,weights=a.weights,epochs=a.epochs,batch_per_rank=a.batch_per_rank,
                  global_batch=batch_size,views_per_instance=2,microbatch=a.micro,steps_per_epoch=steps_per_epoch,
                  split=str(a.split),lr_peak=peak_lr,warmup_steps=steps_per_epoch,schedule_epochs=a.schedule_epochs,
                  seed=seed,projection_init='fresh',optimizer_init='fresh')
    if rank==0:
        (a.output/'config.json').write_text(json.dumps(settings,indent=2))
    dist.barrier()
    updates=0
    for epoch in range(a.epochs+1):
        since=time.time()
        if epoch:
            model.train()
            order=np.random.default_rng(seed+epoch).permutation(train)
            stats=[]
            for i,start in enumerate(range(0,len(order)-batch_size+1,batch_size),1):
                global_ids=order[start:start+batch_size]
                ix=global_ids[rank::4]
                opt.zero_grad(set_to_none=True)
                progress=updates/steps_per_epoch
                factor=min(1.,(updates+1)/steps_per_epoch)*(.5+.5*math.cos(math.pi*progress/a.schedule_epochs))
                for g,lr in zip(opt.param_groups,peak_lr):
                    g['lr']=lr*factor
                item=step(model,records,ix,points,grouping,device,1000+epoch,a.weights,rank,opt,a.micro)
                updates+=1
                stats.append(item)
                if rank==0:
                    row=dict(epoch=epoch,batch=i,updates=updates,lr=[x['lr'] for x in opt.param_groups],**item)
                    core.append(a.output/'steps.jsonl',row)
                    if updates<=3 or updates%10==0: print(json.dumps(row),flush=True)
        model.eval()
        val_stats=[step(model,records,z[rank::4],points,grouping,device,6000,a.weights,rank,None,a.micro) for z in val]
        core.monitor(model,records,split['monitor'],points,grouping,device,rank,4,epoch,'refine')
        states=[None]*4
        dist.all_gather_object(states,rng_state(device))
        if rank==0:
            avg=lambda seq:{k:float(np.mean([r[k] for r in seq])) for k in seq[0]}
            row=dict(epoch=epoch,updates=updates,validation=avg(val_stats),seconds=time.time()-since)
            if epoch:
                row['train']=avg(stats)
            path=a.output/f'refine_epoch{epoch:03d}.pt'
            tmp=path.with_suffix('.tmp')
            torch.save(dict(stage='refine',epoch=epoch,step=updates,world_size=4,
                      config=config,model=model.state_dict(),optimizer=opt.state_dict(),
                      rng_per_rank=states,experiment_config=settings),tmp)
            tmp.replace(path)
            core.append(a.output/'epochs.jsonl',row)
            print(json.dumps(row),flush=True)
        dist.barrier()
    if rank==0:
        (a.output/'complete.json').write_text(json.dumps(dict(epochs=a.epochs,updates=updates)))
    dist.destroy_process_group()


if __name__=='__main__':
    main()
