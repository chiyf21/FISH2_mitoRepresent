import argparse,json,math,os,random,time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .data import MitoDataset,collate_batch,to_device,dummy_batch
from .model import VoxelMAE,loss_terms,metric_keys
from .batching import BucketBatches
from .trace import trace,worker_init

def save(path,state):
    tmp=path.with_suffix('.tmp.pt');torch.save(state,tmp);tmp.replace(path)

def entropy_effective_rank(sum_z,sum_zz,n):
    if n<2:return 0.0
    cov=(sum_zz-torch.outer(sum_z,sum_z)/n)/(n-1)
    eig=torch.linalg.eigvalsh((cov+cov.T)*.5).clamp_min_(0)
    total=eig.sum()
    if float(total)<=1e-30:return 0.0
    p=eig[eig>0]/total
    return float(torch.exp(-(p*p.log()).sum()))

def main():
    import resource
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(max(soft,65536),hard),hard))
    torch.multiprocessing.set_sharing_strategy('file_system')
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--objective',choices=['voxel','lowfreq'],required=True);p.add_argument('--lf-with-occ',action='store_true')
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--queries',type=int,default=1000);p.add_argument('--views',type=int,default=4)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--prefetch-factor',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--budget',type=int,default=64);p.add_argument('--max-batch',type=int,default=8)
    p.add_argument('--slice-microbatch',type=int,default=4);p.add_argument('--tile-batch',type=int,default=2)
    p.add_argument('--effective-batch',type=int,default=512);p.add_argument('--save-every',type=int,default=5)
    p.add_argument('--resume');p.add_argument('--max-steps',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--val-limit',type=int,default=0);p.add_argument('--no-augment',action='store_true')
    p.add_argument('--memory-fraction',type=float,default=.4)
    a=p.parse_args()
    # Shared run policy permits checkpoint interval changes during pressure tests
    # or training without interrupting either distributed experiment.
    policy=Path(a.out).parent/'checkpoint_policy.json'
    def checkpoint_interval():
        interval=int(json.loads(policy.read_text())['save_every']) if policy.exists() else a.save_every
        if interval<1:raise ValueError('save_every must be positive')
        return interval
    a.save_every=checkpoint_interval()
    rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if world>1:dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=60))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume:raise FileExistsError('Explicit resume required')
    train=MitoDataset(a.cache,'train',a.queries,a.views,not a.no_augment,True,a.seed,a.limit)
    val=MitoDataset(a.cache,'val',a.queries,1,False,True,a.seed+1,a.val_limit)
    bs=BucketBatches(train,rank,world,a.budget,a.max_batch,a.seed)
    vs=BucketBatches(val,rank,world,a.budget,a.max_batch,a.seed,False)
    kwargs=dict(collate_fn=collate_batch,num_workers=a.workers,pin_memory=False,worker_init_fn=worker_init)
    if a.workers:kwargs.update(prefetch_factor=max(1,a.prefetch_factor),multiprocessing_context='spawn')
    tr=DataLoader(train,batch_sampler=bs,**kwargs);vl=DataLoader(val,batch_sampler=vs,**kwargs)
    cfg=dict(objective=a.objective,lf_with_occ=a.lf_with_occ,slice_microbatch=a.slice_microbatch,tile_batch=a.tile_batch)
    model=VoxelMAE(**cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.05)
    start=0;step_count=0;best=float('inf');optimizer_updates=0;resume_lr=None
    if a.resume:
        state=torch.load(a.resume,map_location=device,weights_only=False)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        start=state['epoch']+1;step_count=state['global_step'];best=state['best']
        resume_lr=float(opt.param_groups[0]['lr'])
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    torch.manual_seed(a.seed+rank);torch.cuda.manual_seed_all(a.seed+rank)
    doc=json.loads((Path(a.cache)/'manifest.json').read_text());sets=[set(v) for v in doc['body_splits'].values()]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    config={**vars(a),'model':cfg,'world_size':world,'parameters':sum(p.numel() for p in model.parameters()),
        'input_points':'relative17^3 grid, target interior +26grid neighbours, visible and valid only',
        'query_policy':'32x32x30nm centre lattice; entire3^3 hidden/valid, >=2 target voxels; <=1000 unique queries; no700/300quota',
        'no_query_policy':'skip that instance/view, logged; DDP empty ranks use zero-weight graph filler',
        'patch_extent_xyz_nm':[24,24,90],'position':'centred physical um, no bbox-size input or per-instance axis normalization in encoder/decoder',
        'normalization':'z-score from visible voxels only; clean hidden target receives same transform',
        'lowfreq':'10 orthonormal3D-DCT modes sorted by physical frequency; coefficients divided bysqrt27; SmoothL1 on inverse low-pass images in intensity units',
        'lf_modes_zyx':model.lf_modes_zyx,'lf_frequencies_cycles_nm':model.lf_frequencies_cycles_nm,
        'validation_note':'overlapping patch voxel metrics, selected >=2foreground/hidden crops; not unbiased dense-volume metrics',
        'validation_embedding_rank':'entropy effective rank of centered covariance of bottleneck z over eligible validation instances; fixed masked validation view'}
    if resume_lr is not None:
        config['resume_start_lr']=resume_lr
        config['resume_epochs_additional']=max(0,a.epochs-start)
    if rank==0:
        (out/'config.json').write_text(json.dumps(config,indent=2));(out/'train.pid').write_text(str(os.getpid()))
        print(json.dumps({'start':True,'train_instances':len(train.records),'val_instances':len(val.records),'steps_per_epoch':len(tr),'config':config}),flush=True)
    keys=metric_keys(a.objective,a.lf_with_occ);last_recovery=step_count;stop=False
    def record_skips(b,epoch,split):
        if b['skipped_records']:
            with (out/f'skipped_rank{rank}.jsonl').open('a') as f:
                for row in b['skipped_records']:f.write(json.dumps(dict(epoch=epoch,split=split,**row))+'\n')
        return len(b['skipped_records'])
    def reduce(sums,n,skipped):
        values=torch.tensor([n,skipped]+[sums[k] for k in keys],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(values)
        return {k:float(values[j+2]/values[0].clamp_min(1)) for j,k in enumerate(keys)},int(values[0]),int(values[1])
    for epoch in range(start,a.epochs):
        train.epoch=epoch;bs.epoch=epoch;wrapped.train();ts=time.time()
        sums={k:0. for k in keys};seen=0;skipped=0;window=0;opt.zero_grad(set_to_none=True)
        def update():
            nonlocal optimizer_updates
            for parameter in model.parameters():
                if parameter.grad is not None:parameter.grad.div_(max(window,1))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True)
            optimizer_updates+=1
        trace('epoch_start',epoch=epoch)
        for step,b in enumerate(tr):
            trace('data_ready',epoch=epoch,step=step,empty=b['empty'])
            skipped+=record_skips(b,epoch,'train')
            local_count=0. if b['empty'] else float(b['sample_weight'].sum())
            count=torch.tensor(local_count,device=device)
            if world>1:dist.all_reduce(count)
            trace('count_reduced',step=step,count=float(count))
            if not float(count):continue
            if b['empty']:b=dummy_batch()
            b=to_device(b,device)
            if resume_lr is None:
                progress=(epoch*len(tr)+step)/max(1,len(tr)*a.epochs);warmup=min(200,max(20,len(tr)//4))
                lr=a.lr*min(1,(step_count+1)/warmup)*(.1+.45*(1+math.cos(math.pi*progress)))
            else:
                progress=((epoch-start)*len(tr)+step)/max(1,len(tr)*max(1,a.epochs-start))
                lr=resume_lr*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in opt.param_groups:group['lr']=lr
            trace('forward_start',step=step,local_batch=local_count)
            with torch.autocast('cuda',dtype=torch.bfloat16):terms=loss_terms(wrapped(b),b)
            trace('forward_return',step=step)
            loss=terms['loss']*local_count*world
            if not torch.isfinite(loss):raise FloatingPointError(str(b['record_id']))
            trace('backward_start',step=step)
            loss.backward();window+=float(count);step_count+=1;seen+=local_count
            trace('backward_return',step=step)
            if window>=a.effective_batch or step+1==len(tr) or (a.max_steps and step_count>=a.max_steps):
                update();window=0
            for k in keys:sums[k]+=float(terms[k].detach())*local_count
            if rank==0 and step%5==0:
                row={'epoch':epoch,'step':step,'global_step':step_count,'seconds':round(time.time()-ts,2),
                     'global_batch':float(count),'local_batch':local_count,'lr':lr,'optimizer_updates_this_process':optimizer_updates,'accumulated_views':window,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,
                     'queries':b['query_unique_count'],'query_candidates':b['query_candidates'],'skipped_views':skipped,
                     **{k:float(v.detach()) for k,v in terms.items()}}
                print(json.dumps(row),flush=True)
                with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if rank==0 and (step_count-last_recovery>=250 or last_recovery==0 and optimizer_updates>0) and window==0:
                save(out/'recovery.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch-1,
                     global_step=step_count,best=best,config=config,model_config=cfg,note='Replay current epoch on recovery'))
                last_recovery=step_count
            if a.max_steps and step_count>=a.max_steps:stop=True;break
        if window:update();window=0
        wrapped.eval();vsum={k:0. for k in keys};vseen=0;vskip=0
        rank_dim=model.size[0].in_features
        zsum=torch.zeros(rank_dim,device=device,dtype=torch.float64)
        zsecond=torch.zeros(rank_dim,rank_dim,device=device,dtype=torch.float64)
        zn=0.
        with torch.no_grad():
            for b in vl:
                vskip+=record_skips(b,epoch,'val')
                if b['empty']:continue
                n=float(b['sample_weight'].sum());b=to_device(b,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b);terms=loss_terms(pred,b)
                for k in keys:vsum[k]+=float(terms[k])*n
                vseen+=n
                z=pred['z'].double();weight=b['sample_weight'].double().reshape(-1,1)
                zsum+=(z*weight).sum(0);zsecond+=z.T@(z*weight);zn+=float(weight.sum())
        moments=torch.cat((torch.tensor([zn],device=device,dtype=torch.float64),zsum,zsecond.flatten()))
        if world>1:dist.all_reduce(moments)
        embedding_n=int(round(float(moments[0])))
        embedding_rank=entropy_effective_rank(moments[1:1+rank_dim],moments[1+rank_dim:].reshape(rank_dim,rank_dim),embedding_n) if rank==0 else None
        training,ntrain,skipped=reduce(sums,seen,skipped);validation,nval,vskip=reduce(vsum,vseen,vskip)
        if not ntrain or not nval:raise RuntimeError(f'No eligible train/val samples: {ntrain}/{nval}')
        improved=validation['loss']<best;best=min(best,validation['loss'])
        if rank==0:
            row=dict(epoch=epoch,global_step=step_count,seconds=time.time()-ts,train_views_seen=ntrain,
                     skipped_train_views=skipped,val_instances_seen=nval,skipped_val_instances=vskip,
                     val_embedding_n=embedding_n,val_embedding_entropy_rank=embedding_rank,train=training,val=validation)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,global_step=step_count,best=best,config=config,model_config=cfg)
            save(out/'last.pt',state)
            if (epoch+1)%checkpoint_interval()==0 or epoch+1==a.epochs:save(out/f'epoch_{epoch+1:02d}.pt',state)
            if improved:save(out/'best.pt',state)
        if world>1:dist.barrier()
        if stop:break
    if rank==0:(out/'completed.json').write_text(json.dumps(dict(epochs_completed=epoch+1,global_step=step_count,limited_run=stop,best_val=best)))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
