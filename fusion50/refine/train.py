"""New Point-MAE -> point NNA20 -> reference VICReg reg1/5 CLS1/20 fusion50."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,hashlib,json,math,random,sys,time,fcntl
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT.parent/'shared')]
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
import core,base,point_mae_adapter as adapter
from instance_tokens import InstanceTokenEncoder
from nna import Supervision,hashes

def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)
def append(path,obj):
 with Path(path).open('a') as f:f.write(json.dumps(obj)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
core.time=time;core.json=json;core.append=append;core.write=write
def rng_state(device):return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
def restore(s,device):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state(s['cuda'].cpu(),device)
class LegacyNNA(nn.Module):
 def __init__(self):
  super().__init__();self.sup=Supervision(ROOT.parent/'vendor/mim_refiner',dict(layers=[9,10,11,12],queue_size=65536,topk=1,temperature=.2))
 def forward(self,features,ids):
  mean,parts=self.sup(dict(zip(core.LAYERS,features)),ids);return mean*4,[parts[str(l)] for l in core.LAYERS]

def records(config):
 point_run=Path(config['point_run'])
 source=json.loads((point_run/'data/instances.json').read_text())
 split=json.loads((point_run/'split.json').read_text())
 point_config=json.loads((point_run/'config.resolved.json').read_text())
 spacing=point_config['spacing_xyz_nm']
 assign={j:name for name in ('train','val','test') for j in split[name]}
 if sorted(assign)!=list(range(len(source))) or sum(len(split[name]) for name in ('train','val','test'))!=len(source):
  raise ValueError('Point-run split must assign each instance exactly once')
 rows=[]
 keys=set()
 for j,row in enumerate(source):
  body=str(row['body_id'])
  key=[str(row.get('source_run_id','dataset')),body,str(row.get('record_id') or row['instance_id'])]
  if tuple(key) in keys:raise ValueError(f'Duplicate instance key: {key}')
  keys.add(tuple(key))
  rows.append(dict(cohort_index=j,key=key,body_id=body,point_row=j,em_path=row['em_path'],mask_path=row['mask_path'],normalization_radius_voxels=float(row['normalization_radius_voxels']),spacing_xyz_nm=spacing,split=assign[j]))
 if not rows:raise ValueError('No instances')
 return rows

def load_points(config,rows):
 path=Path(config['point_run'])/'data/pointclouds.npy'
 points=np.load(path,mmap_mode='r',allow_pickle=False)
 if points.shape!=(len(rows),2,2048,3):
  raise ValueError(f'Invalid point cache shape: {points.shape}')
 return points

def load_models(config,device):
 args=SimpleNamespace(group_size=32,num_group=64,mask_ratio=.7,trans_dim=384,encoder_dims=384,depth=12,drop_path=.1,heads=6,decoder_depth=4,decoder_heads=6)
 full=adapter.build_model(ROOT.parent/'vendor/point_mae',args,torch.device('cpu'))
 ck=torch.load(config['point_checkpoint'],map_location='cpu',weights_only=False);full.load_state_dict(ck['model'],strict=True)
 point=full.MAE_encoder.to(device).eval().requires_grad_(False);grouping=full.group_divider.to(device).eval();del full,ck
 image=InstanceTokenEncoder.from_checkpoint(config['image_checkpoint'],ROOT.parent/'image',max_tokens=2048).to(device).eval().requires_grad_(False)
 return point,grouping,image

def save_epoch(model,opt,phase,epoch,step,out,rank,world,device,cfg,smoke=False):
 states=[None]*world;dist.all_gather_object(states,rng_state(device))
 if rank==0:
  name=f'{phase}_epoch{epoch:03d}.pt';path=out/'checkpoints'/name;path.parent.mkdir(exist_ok=True)
  obj=dict(stage=phase,epoch=epoch,step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng_per_rank=states,world_size=world,config=cfg,point_last4=[b.state_dict() for b in model.point.blocks.blocks[-4:]],image_last4=[b.state_dict() for b in model.image.blocks[-4:]])
  tmp=path.with_suffix('.partial');torch.save(obj,tmp);tmp.replace(path)
  latest=out/'last.new';os.link(path,latest);latest.replace(out/'last.pt')
 dist.barrier()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--budget-seconds',type=float,default=float('inf'));args=ap.parse_args()
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 cfg=json.loads(Path(args.config).read_text());out=Path(args.output).resolve();out.mkdir(exist_ok=True);(out/'results').mkdir(exist_ok=True);core.ROOT=out
 lock=None
 if rank==0:lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 seed=20261029;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 recs=records(cfg);tr=[i for i,r in enumerate(recs) if r['split']=='train'];va=[i for i,r in enumerate(recs) if r['split']=='val'];rng=np.random.default_rng(seed);nprobe=16 if args.smoke else 256;probe=sorted(rng.choice(tr,min(len(tr),nprobe),False).tolist()+rng.choice(va,min(len(va),nprobe),False).tolist())
 if rank==0:
  write(out/'cohort.json',recs);write(out/'monitor_ids.json',probe);write(out/'status.json',dict(status='initializing',phase='pretrain',epoch=0))
 points=load_points(cfg,recs)
 point,grouping,image=load_models(cfg,device)
 model=core.System(point,image,normalize_image_cls_regularizer=cfg.get('normalize_image_cls_regularizer',False)).to(device)
 if rank==0:write(out/'initialization_check.json',dict(status='PASS',point='surface Point-MAE, strict full-model load',point_path=cfg['point_checkpoint'],image=cfg['image_checkpoint'],old_point_nna_weights_loaded=False))
 resume=torch.load(out/'last.pt',map_location='cpu',weights_only=False) if args.resume and (out/'last.pt').exists() else None
 if (out/'last.pt').exists() and not args.resume:raise FileExistsError('Existing run requires --resume')
 if resume is None:
  mean,std,_=base.estimate_point_cls_stats(point,grouping,[recs[j] for j in tr],points,device,64)
  with torch.no_grad():model.point_cls.copy_(mean[None,None]+.01*std[None,None]*torch.randn_like(model.point_cls))
 if len(tr)<world*2 or 64%world or 64//world<2:
  raise ValueError('Fusion50 requires a global batch of 64 and at least two instances per rank')
 sampler=DistributedSampler(tr,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=True)
 loader=DataLoader(tr,batch_size=64//world,sampler=sampler,drop_last=False,num_workers=0)
 step=0 if resume is None else resume['step'];start_time=time.monotonic();deadline=start_time+args.budget_seconds
 for phase,epochs in [('pretrain',20),('refine',50)]:
  if resume is not None and resume['stage']=='refine' and phase=='pretrain':continue
  continuing=resume is not None and resume['stage']==phase
  if phase=='pretrain':model.point_nna=LegacyNNA().to(device);model.phase='pretrain'
  else:
   model.point_nna=None;model.phase='refine';model.fusion.requires_grad_(True);model.image.cls_token.requires_grad_(True)
   for b in model.image.blocks[-4:]:b.requires_grad_(True)
  if continuing:
   model.load_state_dict(resume['model'],strict=True)
   if resume.get('rng_per_rank'):restore(resume['rng_per_rank'][rank],device)
  elif phase=='refine':
   model.phase='calibration';model.eval();core.monitor(model,recs,probe,points,grouping,device,rank,world,0,'calibration');dist.barrier()
   data=np.load(out/'calibration_epoch00_probe.npz');keep=np.isin(data['ids'],tr)
   with torch.no_grad():
    for l in core.LAYERS:
     values=[]
     for mod in ('point','image'):
      x=np.asarray(data[f'{mod}_{l}'][keep],np.float64);mu=x.mean(0);v=np.mean(np.sum((x-mu)**2,axis=1));getattr(model.balance,f'mu_{"p" if mod=="point" else "i"}_{l}').copy_(torch.tensor(mu,device=device,dtype=torch.float32));values.append(v)
     getattr(model.balance,f'v_{l}').copy_(torch.tensor(values,device=device,dtype=torch.float32))
    model.balance.updates.fill_(1)
   model.phase='refine'
  enc=[p for b in list(point.blocks.blocks[-4:])+(list(image.blocks[-4:]) if phase=='refine' else []) for p in b.parameters()]
  cls=[model.point_cls]+([image.cls_token] if phase=='refine' else []);heads=list(model.point_nna.parameters()) if phase=='pretrain' else list(model.fusion.parameters())
  opt=torch.optim.AdamW([dict(params=enc,lr=1e-5,base_lr=1e-5),dict(params=cls+heads,lr=2e-4,base_lr=2e-4)],weight_decay=1e-4,betas=(.9,.95))
  if continuing:opt.load_state_dict(resume['optimizer'])
  ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
  if not continuing:
   model.eval();core.monitor(model,recs,probe,points,grouping,device,rank,world,0,phase);save_epoch(model,opt,phase,0,step,out,rank,world,device,cfg,args.smoke)
  first=resume['epoch']+1 if continuing else 1;last_epoch_seconds=0.
  for epoch in range(first,epochs+1):
   stop=(out/'STOP_AFTER_EPOCH').exists() or time.monotonic()+max(last_epoch_seconds,60)+120>deadline
   flag=torch.tensor(int(stop),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
   if flag.item():
    if rank==0:write(out/'paused.json',dict(stage=phase,last_completed_epoch=epoch-1,reason='time budget or stop request',time=time.time()))
    dist.barrier();dist.destroy_process_group();return
   sampler.set_epoch(epoch+(0 if phase=='pretrain' else 1000));ddp.train();beg=time.monotonic();summ=0.;count=0
   for batchno,positions in enumerate(loader,1):
    if len(positions)<2:continue
    rr=[recs[int(j)] for j in positions];ids=torch.tensor([r['cohort_index'] for r in rr],device=device)
    inp=core.prepare(rr,points,point,grouping,image,device,epoch+(0 if phase=='pretrain' else 1000),True,point_only=phase=='pretrain')
    frac=(epoch-1)+(batchno-1)/len(loader);factor=min(1.,frac+1/len(loader))*(.5+.5*math.cos(math.pi*frac/epochs))
    for g in opt.param_groups:g['lr']=g['base_lr']*factor
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts,stats=ddp(inp,ids)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
    opt.step();step+=1;summ+=float(loss)*len(rr);count+=len(rr)
    if rank==0:
     record=dict(phase=phase,epoch=epoch,batch=batchno,batches_per_epoch=len(loader),optimizer_step=step,loss=float(loss),parts={k:float(v) for k,v in parts.items()},grad_norm=float(grad),elapsed_epoch_sec=time.monotonic()-beg,time=time.time())
     if stats:record['balance']={k:dict(point_variance=float(s[0]),image_variance=float(s[1]),point_gain=float(s[2]),image_gain=float(s[3])) for k,s in stats.items()}
     append(out/'step_metrics.jsonl',record);write(out/'status.json',dict(status='training',**record))
    if args.smoke:
     check=dict(phase=phase,loss=float(loss),point_cls_grad=float(model.point_cls.grad.abs().sum()),point_last4_grad=float(sum(p.grad.abs().sum() for b in point.blocks.blocks[-4:] for p in b.parameters() if p.grad is not None)),image_cls_grad=0. if image.cls_token.grad is None else float(image.cls_token.grad.abs().sum()),image_last4_grad=float(sum(p.grad.abs().sum() for b in image.blocks[-4:] for p in b.parameters() if p.grad is not None)),prefix_grads_absent=all(p.grad is None for b in list(point.blocks.blocks[:8])+list(image.blocks[:8]) for p in b.parameters()))
     assert check['prefix_grads_absent'] and check['point_cls_grad']>0 and check['point_last4_grad']>0
     if phase=='refine':assert check['image_cls_grad']>0 and check['image_last4_grad']>0
     if rank==0:write(out/f'{phase}_smoke.json',check)
     if batchno==2:break
   core.monitor(model,recs,probe,points,grouping,device,rank,world,epoch,phase);save_epoch(model,opt,phase,epoch,step,out,rank,world,device,cfg,args.smoke)
   last_epoch_seconds=time.monotonic()-beg
   if rank==0:write(out/'status.json',dict(status='epoch_complete',phase=phase,epoch=epoch,epochs=epochs,optimizer_step=step,mean_loss=summ/count,elapsed_epoch_sec=last_epoch_seconds,time=time.time()))
   dist.barrier()
   if args.smoke:break
  del ddp,opt;dist.barrier();resume=None
 if rank==0:write(out/'complete.json',dict(status='smoke_passed' if args.smoke else 'training_complete',steps=step,time=time.time()))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
