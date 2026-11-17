"""Train either unimodal branch for the fixed 50-epoch ablation."""
import argparse,json,math,os,random,time
from datetime import timedelta
from pathlib import Path
import numpy as np, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
import train as pipe, core

def atomic(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--modality',choices=('point','image'),required=True);a=ap.parse_args()
 cfg=json.loads(a.config.read_text())
 if cfg.get('single_epochs',50)!=50 or cfg.get('pair_balance',False):raise ValueError('Requires 50 epochs and PairBalance off')
 modality=a.modality; phase='unimodal_'+modality
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
 if 64%world or 64//world<2:raise ValueError('Global batch must be 64')
 torch.cuda.set_device(local);dev=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=30));torch.set_num_threads(4)
 seed=cfg.get('seed',20261029);random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True);core.ROOT=out
 if (out/'complete.json').exists():raise FileExistsError(out)
 rows=pipe.records(cfg);points=pipe.load_points(cfg,rows);tr=[j for j,r in enumerate(rows) if r['split']=='train'];va=[j for j,r in enumerate(rows) if r['split']=='val']
 if len(rows)<10000:raise ValueError(f'Expected 10k+ instances, got {len(rows)}')
 point,group,image=pipe.load_models(cfg,dev);model=core.System(point,image).to(dev)
 model.point.requires_grad_(False);model.image.requires_grad_(False);model.point_cls.requires_grad_(False);model.fusion.requires_grad_(False)
 if modality=='point':
  for b in model.point.blocks.blocks[-4:]:b.requires_grad_(True)
  model.point_cls.requires_grad_(True)
  if rank==0:
   with torch.no_grad():
    mean,std,_=pipe.base.estimate_point_cls_stats(point,group,[rows[j] for j in tr],points,dev,64)
    model.point_cls.copy_(mean[None,None]+.01*std[None,None]*torch.randn_like(model.point_cls))
  dist.broadcast(model.point_cls.data,src=0)
  enc=[p for b in model.point.blocks.blocks[-4:] for p in b.parameters()];cls=[model.point_cls]
 else:
  model.image.cls_token.requires_grad_(True)
  for b in model.image.blocks[-4:]:b.requires_grad_(True)
  enc=[p for b in model.image.blocks[-4:] for p in b.parameters()];cls=[model.image.cls_token]
 model.phase=phase
 sampler=DistributedSampler(tr,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=True);loader=DataLoader(tr,batch_size=64//world,sampler=sampler,drop_last=True,num_workers=0)
 opt=torch.optim.AdamW([dict(params=enc,lr=1e-5,base_lr=1e-5),dict(params=cls,lr=2e-4,base_lr=2e-4)],weight_decay=1e-4,betas=(.9,.95))
 ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
 rng=np.random.default_rng(seed);n=min(256,len(tr),len(va));probe=sorted(rng.choice(tr,n,False).tolist()+rng.choice(va,n,False).tolist())
 if rank==0:pipe.write(out/'cohort.json',rows);pipe.write(out/'monitor_ids.json',probe);pipe.write(out/'config.resolved.json',cfg)
 dist.barrier();best=float('inf');steps=0
 for epoch in range(1,51):
  sampler.set_epoch(epoch);ddp.train();beg=time.monotonic();total=0.;count=0
  for batch,pos in enumerate(loader,1):
   rr=[rows[int(j)] for j in pos];ids=torch.tensor([r['cohort_index'] for r in rr],device=dev)
   inp=core.prepare(rr,points,point,group,image,dev,epoch,True,point_only=(modality=='point'))
   frac=epoch-1+(batch-1)/len(loader);factor=min(1.,frac+1/len(loader))*(.5+.5*math.cos(math.pi*frac/50))
   for g in opt.param_groups:g['lr']=g['base_lr']*factor
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts,_=ddp(inp,ids)
   if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss epoch={epoch}, batch={batch}')
   loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
   opt.step();steps+=1;total+=float(loss.detach())*len(rr);count+=len(rr)
   if rank==0:pipe.append(out/'step_metrics.jsonl',dict(epoch=epoch,batch=batch,step=steps,loss=float(loss.detach()),parts={k:float(v) for k,v in parts.items()},grad_norm=float(grad),time=time.time()))
  agg=torch.tensor([total,count],dtype=torch.float64,device=dev);dist.all_reduce(agg);mean=float(agg[0]/agg[1])
  core.monitor(model,rows,probe,points,group,dev,rank,world,epoch,phase)
  if rank==0:
   prefix='point.blocks.blocks.' if modality=='point' else 'image.blocks.';clskey='point_cls' if modality=='point' else 'image.cls_token'
   learned={k:v.detach().cpu() for k,v in model.state_dict().items() if k==clskey or any(k.startswith(prefix+f'{i}.') for i in (8,9,10,11))}
   ck=dict(stage=phase,epoch=epoch,step=steps,config=cfg,model=learned,mean_loss=mean)
   for name in ('last','best'):
    if name=='best' and mean>=best:continue
    if name=='best':best=mean;ck['best_loss']=best
    tmp=out/'checkpoints'/f'{name}.partial.pt';torch.save(ck,tmp);tmp.replace(out/'checkpoints'/f'{name}.pt')
   rec=dict(status='epoch_complete',modality=modality,epoch=epoch,epochs=50,step=steps,mean_loss=mean,best_loss=best,elapsed_epoch_sec=time.monotonic()-beg,time=time.time())
   pipe.append(out/'epoch_metrics.jsonl',rec);atomic(out/'status.json',rec)
  dist.barrier()
 if rank==0:atomic(out/'complete.json',dict(status='training_complete',modality=modality,epochs=50,steps=steps,best_loss=best,time=time.time()))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
