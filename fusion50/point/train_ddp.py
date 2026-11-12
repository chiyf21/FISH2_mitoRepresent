"""Point-MAE only; exact global-batch partitioning, SyncBN, continuation and timed smoke."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,datetime,hashlib,json,math,random,sys,time,fcntl
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from train import Trainer,save,append
from data import write_json

def rng_state(device):return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[torch.cuda.get_rng_state(device)])
def restore(s,device):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state(s['cuda'][0].cpu(),device)
def digest(model):
 h=hashlib.sha256()
 for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def assert_equal_states(model,world):
 hashes=[None]*world;dist.all_gather_object(hashes,digest(model));assert len(set(hashes))==1,hashes;return hashes[0]
def validation(t,model,rank):
 dist.barrier();torch.cuda.synchronize();started=time.perf_counter();score=None;model.eval()
 if rank==0:
  s=rng_state(t.device);values=[]
  with torch.no_grad():
   for j in range(0,len(t.split['val']),t.cfg['mae']['batch_size']):
    ids=t.split['val'][j:j+t.cfg['mae']['batch_size']];torch.manual_seed(t.cfg['seed']+j)
    with t.autocast():loss=model(t.xyz(ids))
    values.append((float(loss),len(ids)))
  score=sum(x*n for x,n in values)/sum(n for _,n in values);restore(s,t.device)
 torch.cuda.synchronize();dist.barrier();elapsed=time.perf_counter()-started
 score_t=torch.tensor(0. if score is None else score,device=t.device,dtype=torch.float64);dist.broadcast(score_t,0)
 return float(score_t),elapsed

def main():
 p=argparse.ArgumentParser(description='Distributed surface Point-MAE training')
 p.add_argument('--config',required=True,help='Resolved config from pipeline.py --prepare-only')
 p.add_argument('--checkpoint',help='Optional checkpoint to resume')
 a=p.parse_args()
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(local)
 dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180));cfg=json.loads(Path(a.config).read_text());cfg['device']=f'cuda:{local}';t=Trainer(cfg);c=cfg['mae'];dest=t.out/'mae'
 if not 1 <= world <= c['batch_size']:
  raise ValueError('Global batch must contain at least one instance per rank')
 if rank==0:
  dest.mkdir(exist_ok=True)
  lock=(dest/'ddp.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (dest/'last.pt').exists() and not a.checkpoint:
   raise FileExistsError('Existing run requires --checkpoint')
 dist.barrier()
 model=nn.SyncBatchNorm.convert_sync_batchnorm(t.mae).to(t.device)
 params=[x for x in model.parameters() if x.requires_grad]
 optimizer=torch.optim.AdamW(params,lr=c['lr'],weight_decay=c['weight_decay'],betas=(.9,.95))
 start=step=0
 best=float('inf')
 restored=False
 if a.checkpoint:
  state=torch.load(a.checkpoint,map_location=t.device,weights_only=False)
  model.load_state_dict(state['model'],strict=True)
  optimizer.load_state_dict(state['optimizer'])
  start=int(state['epoch']);step=int(state['step']);best=float(state['best'])
  if state.get('world_size')==world and state.get('rng_per_rank'):
   restore(state['rng_per_rank'][rank],t.device);restored=True
  del state
 seed=cfg['seed']+start*100003+rank
 if not restored:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 ddp=DDP(model,device_ids=[local],output_device=local,broadcast_buffers=True,find_unused_parameters=False,gradient_as_bucket_view=True,static_graph=True)
 def update(ids,epoch,check_ids=False):
  local_ids=np.array_split(np.asarray(ids),world)[rank]
  if not len(local_ids):raise ValueError('Empty rank batch')
  if check_ids:
   chunks=[None]*world;dist.all_gather_object(chunks,local_ids.tolist());assert [j for chunk in chunks for j in chunk]==list(ids)
  optimizer.zero_grad(set_to_none=True)
  xyz=t.xyz(local_ids,epoch,True,jitter=c.get('jitter_std',0))
  with t.autocast():loss=ddp(xyz[:,epoch%2]);weighted=loss*(world*len(local_ids)/len(ids))
  weighted.backward();norm=torch.nn.utils.clip_grad_norm_(params,c['grad_clip'])
  if not torch.isfinite(norm) or not torch.isfinite(loss):raise FloatingPointError('Nonfinite distributed loss or gradients')
  optimizer.step()
  pair=torch.stack([loss.detach().double()*len(local_ids),loss.new_tensor(float(len(local_ids)),dtype=torch.float64)]);dist.all_reduce(pair)
  return float(pair[0]/pair[1]),float(norm),len(local_ids)
 for epoch in range(start,c['epochs']):
  torch.cuda.synchronize();tick=time.perf_counter();factor=min(1.,(epoch+1)/max(c['warmup_epochs'],1)) if epoch<c['warmup_epochs'] else .01+.99*.5*(1+math.cos(math.pi*(epoch-c['warmup_epochs'])/max(1,c['epochs']-c['warmup_epochs'])))
  for group in optimizer.param_groups:group['lr']=c['lr']*factor
  ddp.train();batches=[ids for ids in t.batch_ids(epoch,c['batch_size']) if len(ids)>=world];losses=[]
  if not batches:raise ValueError('No complete distributed batch')
  for j,ids in enumerate(batches):
   loss,grad,n=update(ids,epoch,j==0);step+=1;losses.append(loss)
   if rank==0 and (j==0 or step%25==0):append(dest/'step_metrics.jsonl',dict(stage='mae',epoch=epoch+1,step=step,loss=loss,batch_instances=len(ids),global_batch=c['batch_size'],world_size=world,batches_per_epoch=len(batches),time=time.time(),gradient_norm=grad))
  val,val_seconds=validation(t,model,rank);elapsed=time.perf_counter()-tick
  states=[None]*world;dist.all_gather_object(states,rng_state(t.device))
  if rank==0:
   improved=val<best;best=min(best,val)
   record=dict(stage='mae',epoch=epoch+1,step=step,train_loss=float(np.mean(losses)),layers={},lr=[g['lr'] for g in optimizer.param_groups],val_loss=val,seconds=elapsed,world_size=world,global_batch=c['batch_size'],validation_seconds=val_seconds,physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'))
   append(dest/'metrics.jsonl',record)
   saved=dict(stage='mae',epoch=epoch+1,step=step,model=model.state_dict(),heads=None,optimizer=optimizer.state_dict(),rng=states[0],rng_per_rank=states,world_size=world,best=best,config=cfg)
   save(dest/'last.pt',saved)
   if improved:save(dest/'best.pt',saved)
   if (epoch+1)%c['save_every']==0 or epoch+1==c['epochs']:save(dest/f'epoch_{epoch+1:03d}.pt',saved)
  dist.barrier()
 if rank==0:write_json(dest/'complete.json',dict(stage='mae',epochs=c['epochs'],step=step,world_size=world,global_batch=c['batch_size'],mae_only=True,checkpoint=str(dest/'last.pt')))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
