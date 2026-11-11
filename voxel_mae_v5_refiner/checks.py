import argparse,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from .data import MitoDataset,collate_batch,to_device,dummy_batch
from .targets import OFFSETS,safe_query_centres,low_frequency_basis
from .model import VoxelMAE,loss_terms
from .batching import BucketBatches
from mito_voxel_mae_v4.native_data import tight_box

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache');p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);results={}
    # Compare the fast lattice predicate against direct per-crop enumeration.
    rng=np.random.default_rng(7);shape=(9,13,15);mask=rng.random(shape)<.2;valid=rng.random(shape)>.01
    visible=np.ones((4,4,4),bool);visible.reshape(-1)[rng.choice(64,38,replace=False)]=False
    lo=np.array([2,2,2]);size=np.array([5,9,11]);origin=np.array([17,33,42])
    fast=safe_query_centres(mask,valid,visible,lo,size,origin)
    brute=[]
    for z in range(1,shape[0]-1):
        for y in range(1,shape[1]-1):
            for x in range(1,shape[2]-1):
                q=np.array([z,y,x])
                if ((q+origin)%np.array([1,4,4])).any():continue
                patch=q+OFFSETS;idx=tuple(patch.T)
                bins=np.clip(np.floor(((patch+.5-lo)/size)*4).astype(int),0,3)
                if valid[idx].all() and mask[idx].sum()>=2 and not visible[tuple(bins.T)].any():brute.append(q)
    assert len(fast)>0 and np.array_equal(fast,np.array(brute).reshape(-1,3))
    basis,modes,frequencies=low_frequency_basis()
    assert np.allclose(basis@basis.T,np.eye(10),atol=2e-6)
    constant=np.ones(27)*3;coeff=constant@basis.T/np.sqrt(27)
    assert abs(coeff[0]-3)<1e-6 and np.max(np.abs(coeff[1:]))<1e-6
    assert modes[1]==[1,0,0] and modes[2]==[2,0,0]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);records=[]
        for i,shape in enumerate([(9,29,35),(13,35,43),(17,37,45),(7,15,15)]):
            path=root/str(i);path.mkdir();z,y,x=np.indices(shape)
            mask=((z-(shape[0]-1)/2)/max(shape[0]/3,1))**2+((y-shape[1]/2)/(shape[1]/3))**2+((x-shape[2]/2)/(shape[2]/3))**2<1
            if i==3:mask[:]=False;mask[3,7,7]=True
            np.save(path/'em.npy',np.clip(x*3+y*2+z,0,255).astype('uint8'));np.save(path/'mask.npy',mask.astype('uint8'));np.save(path/'valid.npy',np.ones(shape,bool))
            records.append(dict(record_id=str(i),body_id=str(i),relative_dir=str(i),shape_zyx=shape,spacing_zyx_nm=[30,8,8],lo=0,hi=255,split='train'))
        (root/'manifest.json').write_text(json.dumps({'records':records}))
        ds=MitoDataset(root,queries=1000,views=4,augmentations=False)
        samples=[ds[i] for i in range(3)];assert all(not s['skip'] for s in samples)
        assert ds[3]['skip'] and ds[3]['reason']=='no_safe_333_query'
        assert collate_batch([ds[3]])['empty']
        for s in samples:
            patch=s['query_vox'].long().numpy()[:,None]+OFFSETS
            assert not s['observed'][0].numpy()[tuple(patch.transpose(2,0,1))].any()
            assert (s['occupancy'].sum(1)>=2).all()
            assert len(torch.unique(s['query_vox'],dim=0))==len(s['query_vox'])
            grid=((s['points_vox'].numpy()+.5-s['tight_lo'])/s['tight_shape'])*17-.5
            assert np.allclose(grid,np.rint(grid),atol=2e-5),'Input does not use17^3 relative grid'
            assert np.allclose(s['points_um'].numpy(),(s['points_vox'].numpy()+.5-s['tight_lo']-s['tight_shape']/2)*[.03,.008,.008])
        # Hidden image changes must leave the complete encoder input unchanged.
        original=np.load(root/'0/em.npy');changed=original.copy();hidden=~samples[0]['observed'][0].numpy().astype(bool)
        changed[hidden]=(changed[hidden].astype(int)+37)%255;np.save(root/'0/em.npy',changed)
        altered=ds[0]
        assert torch.equal(samples[0]['image'],altered['image']),'Hidden intensities leaked via preprocessing'
        assert not torch.equal(samples[0]['intensity'],altered['intensity'])
        np.save(root/'0/em.npy',original)
        batch=to_device(collate_batch(samples),'cuda');cases={}
        for objective in ('voxel','lowfreq'):
            model=VoxelMAE(objective=objective,lf_with_occ=True,dim=64,depth=2,heads=4,latent=32,groups=8,k=16,channels=8).cuda().eval()
            assert not hasattr(model,'scale')
            with torch.no_grad():
                z=model.encode(batch)
                separate=torch.cat([model.encode(to_device(collate_batch([s]),'cuda')) for s in samples])
                delta=float((z-separate).abs().max());assert torch.allclose(z,separate,atol=5e-5,rtol=5e-5),delta
                altered=dict(batch);altered['occupancy']=1-batch['occupancy'];altered['size_target']=batch['size_target']+5
                assert torch.equal(z,model.encode(altered))
            model.train();terms=loss_terms(model(batch),batch);terms['loss'].backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            model.zero_grad(set_to_none=True);dummy=to_device(dummy_batch(),'cuda')
            zero=loss_terms(model(dummy),dummy)['loss'];assert float(zero.detach())==0;zero.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            cases[objective]={'batch_latent_max_difference':delta,'loss':float(terms['loss'].detach())}
        results['synthetic']={'passed':['fast333predicate equals brute force','all27 hidden and valid','>=2GT foreground voxels','no duplicate queries','17-grid input','no-safe-query skip','hidden-image input isolation incl normalization','physical nm/um positions, no size input','physical DCT orthonormal/DC/frequency order','variable query padding','both objectives finite gradients','DDP zero-weight filler gradients'],'objectives':cases}
    if a.cache:
        ds=MitoDataset(a.cache,queries=1000,views=4,augmentations=True)
        samples=[ds[i+v*len(ds.records)] for i,v in zip(range(min(3,len(ds.records))),(0,1,3))]
        active=[s for s in samples if not s['skip']];assert active
        batch=to_device(collate_batch(samples),'cuda');real={}
        for objective in ('voxel','lowfreq'):
            model=VoxelMAE(objective=objective,lf_with_occ=True).cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
            with torch.autocast('cuda',dtype=torch.bfloat16):terms=loss_terms(model(batch),batch)
            terms['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
            real[objective]={'step_seconds':time.time()-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,
                             'terms':{k:float(v.detach()) for k,v in terms.items()}}
            del model,opt,terms;torch.cuda.empty_cache()
        results['real']={'candidates':[s['query_candidates'] for s in active],'queries':[s['query_unique_count'] for s in active],
                         'skipped':[s['record_id'] for s in samples if s['skip']],'objectives':real}
    (out/'checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
