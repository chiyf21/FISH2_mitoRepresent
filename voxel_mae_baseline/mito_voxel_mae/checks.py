"""Meaningful pre-launch tests and a real-instance fixed-view overfit check."""
import argparse,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from .data import MitoDataset,collate_one,to_device,point_support,augment
from .model import VoxelMAE,loss_terms

def tests(device):
    torch.manual_seed(42)
    z,y,x=np.indices((9,25,27))
    mask=((z-4)**2/9+(y-12)**2/49+(x-13)**2/64)<1
    spacing=np.array([30.,8.,8.]);lo=np.array([1,5,5]);size=np.array([7,15,17])
    t,r,vox=point_support(mask,lo,size,spacing)
    assert len(np.unique(vox,axis=0))==len(vox)
    assert np.allclose(r,t*(size*spacing)/(size*spacing).max())
    # grid_sample must agree with analytic linear ramps, including ZYX->XYZ.
    ramp=(z*2+y*3+x*5).astype(np.float32)
    grid=(2*(vox+.5)/np.array(mask.shape)-1)[:,::-1].copy()
    sampled=torch.nn.functional.grid_sample(torch.tensor(ramp[None,None],device=device),
        torch.tensor(grid[None,:,None,None],device=device,dtype=torch.float32),align_corners=False,mode='bilinear')
    bounds=((vox>=0)&(vox<=np.array(mask.shape)-1)).all(1)
    expected=vox@np.array([2,3,5])
    assert np.allclose(sampled.cpu().numpy().reshape(-1)[bounds],expected[bounds],atol=1e-4)
    # All geometric augmentation modes preserve aligned shapes and binary occupancy.
    for kind in range(4):
        e,m,v=augment(ramp,mask,np.ones(mask.shape,bool),spacing,np.random.default_rng(kind+42),kind)
        assert e.shape==m.shape==v.shape and m.any() and not (m&~v).any()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);folder=root/'instance';folder.mkdir()
        np.save(folder/'em.npy',np.clip(ramp,0,255).astype(np.uint8));np.save(folder/'mask.npy',mask.astype(np.uint8));np.save(folder/'valid.npy',np.ones(mask.shape,bool))
        record={'record_id':'synthetic','body_id':'A','relative_dir':'instance','spacing_zyx_nm':spacing.tolist(),'lo':0,'hi':255,'split':'train'}
        (root/'manifest.json').write_text(json.dumps({'records':[record]}))
        ds=MitoDataset(root,queries=256,views=4,masking=True)
        for i in range(4):
            b=collate_one([ds[i]])
            assert (b['image'][b['observed']==0]==0).all()
        b=to_device(collate_one([ds[0]]),device)
        cfg={'dim':64,'depth':2,'heads':4,'latent':32,'groups':8,'k':16,'channels':8}
        model=VoxelMAE(**cfg).to(device)
        result=model(b);terms=loss_terms(result,b);terms['loss'].backward()
        missing=[name for name,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not missing,missing
        assert model.bottleneck[-1].weight.grad.abs().sum()>0
        assert model.size[-1].weight.grad.abs().sum()>0
        model.eval()
        with torch.no_grad():
            z1=model.encode(b);changed=dict(b)
            for key in ('points_r','points_t','points_grid'):
                changed[key]=torch.cat((b[key],torch.full((1,10,3),1000.,device=device)),1)
            changed['points_valid']=torch.cat((b['points_valid'],torch.zeros((1,10),device=device,dtype=torch.bool)),1)
            assert torch.allclose(z1,model.encode(changed),atol=1e-5),'padding affected encoding'
            full=model.decode(z1,b['query_t'],b['query_r'])
            parts=torch.cat([model.decode(z1,b['query_t'][:,i:i+31],b['query_r'][:,i:i+31]) for i in range(0,256,31)],1)
            assert torch.allclose(full,parts,atol=1e-5),'query chunk changed predictions'
            changed=dict(b);changed['log_size']=b['log_size']+np.log(2)
            assert not torch.allclose(z1,model.encode(changed)),'scale conditioning ignored'
        # Native thin-Z cases must survive U-Net without a Z pooling failure.
        x=torch.randn(1,3,1,8,8,device=device);assert model.unet(x).shape[-3:]==(1,8,8)
    return {'coordinate_ramp':True,'neighbor_deduplication':True,'augment_alignment':True,'input_masking':True,
            'all_parameter_gradients':True,'latent_and_scale_gradients':True,'padding_invariance':True,
            'query_chunk_equivalence':True,'scale_sensitivity':True,'z1_unet':True}

def overfit(cache,out,device,steps):
    ds=MitoDataset(cache,'train',queries=2048,views=1,augmentations=False,masking=True)
    # Native smallest context to keep this launch check bounded.
    index=min(range(len(ds.records)),key=lambda i:np.prod(ds.records[i]['shape_zyx']))
    b=to_device(collate_one([ds[index]]),device);cfg={'dim':64,'depth':2,'heads':4,'latent':64,'groups':16,'k':16,'channels':8}
    model=VoxelMAE(**cfg).to(device);model.eval() # Fixed-view fit with deterministic dropout.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0)
    losses=[];initial=None;ts=time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True);terms=loss_terms(model(b),b);terms['loss'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();value=float(terms['loss'].detach());losses.append(value)
        if initial is None:initial={k:float(v.detach()) for k,v in terms.items()}
        if step%20==0:print(json.dumps({'overfit_step':step,'loss':value}),flush=True)
    with torch.no_grad():final={k:float(v) for k,v in loss_terms(model(b),b).items()}
    assert final['loss']<initial['loss']*.9,{'initial':initial,'final':final}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'model_config':cfg},out/'overfit.pt')
    return {'record_id':b['record_id'],'steps':steps,'initial':initial,'final':final,'seconds':time.time()-ts,'losses':losses}

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache');p.add_argument('--out',required=True)
    p.add_argument('--overfit-steps',type=int,default=100);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(2);device=torch.device(a.device);result={'tests':tests(device)}
    print(json.dumps(result),flush=True)
    if a.cache:result['real_overfit']=overfit(a.cache,a.out,device,a.overfit_steps)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
