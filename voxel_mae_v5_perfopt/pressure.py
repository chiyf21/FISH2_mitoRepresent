"""Isolated distributed pressure probes; a failed probe cannot kill training."""
import argparse,json,os,time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import VoxelMAE,loss_terms
from .data import MitoDataset,collate_batch,to_device

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--objective',required=True);p.add_argument('--kind',choices=['tiles','instances'],required=True)
    p.add_argument('--batch',type=int,required=True);p.add_argument('--tile-batch',type=int,default=4)
    p.add_argument('--slice-microbatch',type=int,default=64);p.add_argument('--memory-fraction',type=float,default=.72)
    a=p.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,local)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=timedelta(minutes=12))
    model=VoxelMAE(objective=a.objective,lf_with_occ=True,tile_batch=a.tile_batch,slice_microbatch=a.slice_microbatch).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    start=time.time();torch.cuda.reset_peak_memory_stats()
    if a.kind=='tiles':
        # Worst tile depth:16 core slices +3 context slices at either end.
        tiles=[torch.randn(22,4,512,512) for _ in range(a.batch)]
        grids=[torch.rand(128,3,device='cuda')*1.5-.75 for _ in tiles]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=model.extract_tiles(*(tuple(tiles)+tuple(grids)))
            loss=sum(x.square().mean() for x in features)
        loss.backward();opt.step()
        sample=None
    else:
        dataset=MitoDataset(a.cache,'train',1000,1,False,True,42,0)
        # Use a real median-size instance, not an artificially tiny volume.
        ordered=sorted(range(len(dataset.records)),key=lambda i:np.prod(dataset.records[i]['shape_zyx']))
        sample=None
        for index in ordered[len(ordered)//2:len(ordered)//2+100]:
            candidate=dataset[index]
            if not candidate['skip'] and candidate['query_unique_count']>=300:
                sample=candidate;break
        if sample is None:raise RuntimeError('No representative pressure-test instance')
        wrapped=DDP(model,device_ids=[local],broadcast_buffers=False)
        batch=to_device(collate_batch([sample]*a.batch),'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):terms=loss_terms(wrapped(batch),batch)
        loss=terms['loss'];loss.backward();opt.step()
    torch.cuda.synchronize()
    value=torch.tensor([torch.cuda.max_memory_allocated()/1e9,time.time()-start],device='cuda')
    dist.all_reduce(value,op=dist.ReduceOp.MAX)
    if rank==0:
        row=dict(kind=a.kind,batch_per_gpu=a.batch,world_size=dist.get_world_size(),objective=a.objective,
                 peak_gpu_gb=float(value[0]),seconds=float(value[1]),passed=True,
                 tile_batch=a.tile_batch,slice_microbatch=a.slice_microbatch,
                 representative=None if sample is None else sample['record_id'])
        Path(a.out).write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
