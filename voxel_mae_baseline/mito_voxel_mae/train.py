from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import MitoDataset,collate_one,to_device
from .model import VoxelMAE,loss_terms


def configure(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True

def save(path,state):
    tmp=path.with_suffix('.tmp.pt');torch.save(state,tmp);tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--queries',type=int,default=8192)
    p.add_argument('--views',type=int,default=4);p.add_argument('--accumulate',type=int,default=8)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume');p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=256)
    p.add_argument('--no-augment',action='store_true');p.add_argument('--overfit-record',type=int,default=-1)
    p.add_argument('--dim',type=int,default=256);p.add_argument('--depth',type=int,default=6)
    p.add_argument('--latent',type=int,default=256);p.add_argument('--channels',type=int,default=24)
    p.add_argument('--device',default='cuda');a=p.parse_args();configure(a.seed)
    device=torch.device(a.device);amp=device.type=='cuda'
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume:raise FileExistsError('Use --resume to continue an existing run')
    train=MitoDataset(a.cache,'train',a.queries,a.views,not a.no_augment,True,a.seed,a.limit)
    val=MitoDataset(a.cache,'val',a.queries,1,False,True,a.seed+1,a.val_limit)
    if a.overfit_record>=0:
        train.records=[train.records[a.overfit_record]];train.views=max(a.views,32)
        val.records=train.records;val.seed=train.seed;val.split='train'
    kwargs=dict(batch_size=1,collate_fn=collate_one,num_workers=a.workers,pin_memory=amp)
    tr=DataLoader(train,shuffle=True,**kwargs);vl=DataLoader(val,shuffle=False,**kwargs)
    model_cfg={'dim':a.dim,'depth':a.depth,'latent':a.latent,'channels':a.channels}
    model=VoxelMAE(**model_cfg).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.05)
    start=0;global_step=0;best=float('inf')
    if a.resume:
        state=torch.load(a.resume,map_location=device,weights_only=False);model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer']);start=state['epoch']+1;global_step=state['global_step'];best=state['best']
    (out/'config.json').write_text(json.dumps({**vars(a),'model':model_cfg,'parameters':sum(x.numel() for x in model.parameters())},indent=2))
    manifest=json.loads((Path(a.cache)/'manifest.json').read_text())
    sets={k:set(v) for k,v in manifest['body_splits'].items()}
    assert not any(sets[x]&sets[y] for x,y in [('train','val'),('train','test'),('val','test')])
    print(json.dumps({'start':True,'train_instances':len(train.records),'train_views_per_epoch':len(train),
                      'val_instances':len(val.records),'device':str(device),'model':model_cfg,'parameters':sum(x.numel() for x in model.parameters())}),flush=True)
    stop=False
    for epoch in range(start,a.epochs):
        train.epoch=epoch;model.train();optimizer.zero_grad(set_to_none=True);seen=0;totals={};ts=time.time()
        for step,b in enumerate(tr):
            b=to_device(b,device)
            # LR warmup and cosine scheduling, based on the full configured run.
            progress=global_step/max(1,len(tr)*a.epochs)
            warmup=min(500,max(32,len(tr)//2))
            lr=a.lr*min(1,(global_step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for g in optimizer.param_groups:g['lr']=lr
            remaining=len(tr)-(step//a.accumulate)*a.accumulate
            divisor=min(a.accumulate,remaining)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
                result=model(b);terms=loss_terms(result,b);loss=terms['loss']/divisor
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss: {b["record_id"]}')
            loss.backward();global_step+=1;seen+=1
            if (step+1)%a.accumulate==0 or step+1==len(tr) or (a.max_steps and global_step>=a.max_steps):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            for k,v in terms.items():totals[k]=totals.get(k,0)+float(v.detach())
            if step%20==0:
                row={'epoch':epoch,'step':step,'global_step':global_step,'lr':lr,'seconds':round(time.time()-ts,2),
                     'record_id':b['record_id'],'augmentation_kind':b['augmentation_kind'],
                     'visible_points':int(b['points_valid'].sum()),'shape':list(b['image'].shape),
                     'gpu_peak_gb':torch.cuda.max_memory_allocated()/1e9 if amp else 0,
                     **{k:float(v.detach()) for k,v in terms.items()}}
                print(json.dumps(row),flush=True)
                with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if global_step%500==0:
                save(out/'recovery.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch-1,
                     'global_step':global_step,'best':best,'config':vars(a),'model_config':model_cfg,
                     'note':'Epoch replay on recovery; last.pt is exact epoch boundary'})
            if a.max_steps and global_step>=a.max_steps:stop=True;break
        model.eval();vt={};n=0
        with torch.no_grad():
            for b in vl:
                b=to_device(b,device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):terms=loss_terms(model(b),b)
                for k,v in terms.items():vt[k]=vt.get(k,0)+float(v)
                n+=1
        metrics={'epoch':epoch,'global_step':global_step,'seconds':round(time.time()-ts,2),
                 'train':{k:v/seen for k,v in totals.items()},'val':{k:v/n for k,v in vt.items()}}
        print(json.dumps(metrics),flush=True)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        improved=metrics['val']['loss']<best
        best=min(best,metrics['val']['loss'])
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'global_step':global_step,
               'best':best,'config':vars(a),'model_config':model_cfg}
        save(out/'last.pt',state)
        if improved:save(out/'best.pt',state)
        if stop:break
    (out/'completed.json').write_text(json.dumps({'global_step':global_step,'epoch':epoch,'limited_run':bool(a.max_steps),'best_val':best}))

if __name__=='__main__':main()
