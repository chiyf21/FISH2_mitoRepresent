"""Encode once, query every original voxel center in chunks."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import MitoDataset,collate_one,to_device
from .model import VoxelMAE

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--record-id')
    p.add_argument('--chunk',type=int,default=8192);p.add_argument('--embeddings-only',action='store_true');a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model=VoxelMAE(**state['model_config']).to(device).eval()
    model.load_state_dict(state['model']);ds=MitoDataset(a.cache,'all',queries=1,views=1,augmentations=False,masking=False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.embeddings_only:
        embeddings=[]
        with torch.no_grad():
            for i in range(len(ds)):embeddings.append(model.encode(to_device(collate_one([ds[i]]),device)).cpu().numpy()[0])
        np.savez_compressed(out/'embeddings.npz',embedding=np.stack(embeddings),record_id=np.array([r['record_id'] for r in ds.records]),body_id=np.array([r['body_id'] for r in ds.records]));return
    if a.record_id:
        matches=[i for i,r in enumerate(ds.records) if r['record_id']==a.record_id]
        if len(matches)!=1:raise ValueError('record-id must match exactly one prepared instance')
        a.index=matches[0]
    sample=ds[a.index];b=to_device(collate_one([sample]),device)
    shape=tuple(map(int,sample['tight_shape']));count=int(np.prod(shape))
    probability=np.lib.format.open_memmap(out/'occupancy_probability.npy',mode='w+',dtype='float32',shape=shape)
    intensity=np.lib.format.open_memmap(out/'intensity_normalized.npy',mode='w+',dtype='float32',shape=shape)
    with torch.no_grad():
        z=model.encode(b);np.save(out/'embedding.npy',z.cpu().numpy()[0])
        spacing=np.array(ds.records[a.index]['spacing_zyx_nm']);length=np.array(shape)*spacing
        for start in range(0,count,a.chunk):
            ids=np.arange(start,min(start+a.chunk,count));idx=np.stack(np.unravel_index(ids,shape),-1)
            t=(idx+.5)/np.array(shape)-.5;r=t*length/length.max()
            pred=model.decode(z,torch.tensor(t[None],device=device,dtype=torch.float32),torch.tensor(r[None],device=device,dtype=torch.float32))[0]
            probability.reshape(-1)[ids]=pred[:,0].sigmoid().cpu().numpy();intensity.reshape(-1)[ids]=pred[:,1].cpu().numpy()
        size=model.size(z).exp().cpu().numpy()[0]*1000
    probability.flush();intensity.flush()
    (out/'metadata.json').write_text(json.dumps({'record_id':sample['record_id'],'shape_zyx':shape,'predicted_size_zyx_nm':size.tolist(),
        'intensity_mean':sample['intensity_mean'],'intensity_std':sample['intensity_std'],
        'clip_lo':sample['clip_lo'],'clip_hi':sample['clip_hi'],'note':'Intensity is crop z-score after source clipping; clipped extremes are not recoverable'},indent=2))

if __name__=='__main__':main()
