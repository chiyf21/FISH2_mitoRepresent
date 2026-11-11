import argparse,json,time
from pathlib import Path
import torch
from .model import VoxelMAE

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.38)
tiles=[torch.randn(22,4,512,512) for _ in range(4)]
grids=[torch.rand(128,3,device='cuda')*1.5-.75 for _ in tiles]
rows=[]
for micro in (4,8,16,32,64):
    model=VoxelMAE(slice_microbatch=micro,tile_batch=4).cuda().train()
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            value=model.extract_tiles(*(tuple(tiles)+tuple(grids)))
            loss=sum(v.square().mean() for v in value)
        loss.backward();torch.cuda.synchronize()
        row=dict(slice_microbatch=micro,tiles=4,slices=88,seconds=time.time()-start,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,passed=True)
    except torch.OutOfMemoryError:
        row=dict(slice_microbatch=micro,passed=False)
    rows.append(row);print(json.dumps(row),flush=True)
    del model
    if 'value' in locals():del value,loss
    torch.cuda.empty_cache()
Path(a.out).write_text(json.dumps(rows,indent=2))
