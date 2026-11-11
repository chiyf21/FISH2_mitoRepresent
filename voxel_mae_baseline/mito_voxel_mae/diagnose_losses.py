"""Read-only diagnostics on 64 evenly spaced validation instances; no updates."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import MitoDataset,collate_one,to_device
from .model import VoxelMAE,loss_terms

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda');ds=MitoDataset(a.cache,'val',8192,1,False,True,43)
    state=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model=VoxelMAE(**state['model_config']).to(device).eval();model.load_state_dict(state['model'])
    rows=[]
    for i in np.linspace(0,len(ds)-1,64,dtype=int):
        b=to_device(collate_one([ds[int(i)]]),device)
        with torch.no_grad():z=model.encode(b)
        z=z.detach().requires_grad_(True)
        pred=model.decode(z,b['query_t'],b['query_r'])
        out={'voxels':pred,'log_size':model.size(z),'z':z};terms=loss_terms(out,b)
        losses=[terms['occupancy']+.5*terms['dice_loss'],terms['intensity'],.2*terms['size']]
        grads=[torch.autograd.grad(l,z,retain_graph=True)[0].flatten() for l in losses]
        zero={'voxels':torch.zeros_like(pred),'log_size':out['log_size'],'z':z}
        baseline=loss_terms(zero,b)
        row={'record_id':b['record_id'],'intensity':float(terms['intensity'].detach()),'intensity_zero_baseline':float(baseline['intensity'].detach()),
             'occupancy_bce':float(terms['occupancy'].detach()),'occupancy_zero_baseline':float(baseline['occupancy'].detach())}
        w=(.25+.75*b['query_hidden'].float())*b['query_valid'].float()
        target=b['intensity'];occ=b['occupancy']>.5;error=pred[...,1]-target
        for group,g in [('fg',occ),('bg',~occ),('hidden',b['query_hidden']),('visible',~b['query_hidden'])]:
            weights=w*g
            den=weights.sum().clamp_min(1)
            row[group+'_mae']=float((weights*error.abs()).sum().detach()/den)
            row[group+'_smoothl1']=float((weights*F.smooth_l1_loss(pred[...,1],target,reduction='none')).sum().detach()/den)
            row[group+'_baseline_smoothl1']=float((weights*F.smooth_l1_loss(torch.zeros_like(target),target,reduction='none')).sum().detach()/den)
        for name,g in zip(['occupancy','intensity','size'],grads):row['grad_z_'+name]=float(g.norm())
        row['grad_z_cos_occ_int']=float(F.cosine_similarity(grads[0],grads[1],dim=0))
        row['grad_z_ratio_int_occ']=float(grads[1].norm()/grads[0].norm().clamp_min(1e-12))
        rows.append(row)
    keys=[k for k in rows[0] if k!='record_id']
    summary={'instances':len(rows),'mode':'fixed validation masks, no augmentation, FP32 eval; evenly spaced subset',
             'means':{k:float(np.mean([r[k] for r in rows])) for k in keys},
             'medians':{k:float(np.median([r[k] for r in rows])) for k in keys},
             'fraction_negative_occ_int_cosine':float(np.mean([r['grad_z_cos_occ_int']<0 for r in rows])),
             'note':'Gradients at the shared latent z, not all encoder parameters. Snapshot diagnostic, not the historical training gradient trajectory.'}
    path=Path(a.out);path.mkdir(parents=True,exist_ok=True)
    (path/'loss_diagnostics.json').write_text(json.dumps({'summary':summary,'records':rows},indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
