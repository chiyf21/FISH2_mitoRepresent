"""Read-only hooks on the exact runtime model; no replacement Transformer operations."""
import argparse,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.overrides import TorchFunctionMode
from common import *
LAYERS=[6,8,11,12]
def masked_mean(tokens,valid):
    assert tokens.ndim==3 and valid.shape==tokens.shape[:2] and bool(valid.any(1).all())
    return (tokens.float()*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
class CaptureProjection(TorchFunctionMode):
    """prepare uses functional Linear instead of calling the patch Conv module."""
    def __init__(self,image):self.image=image;self.means=[]
    def __torch_function__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        if getattr(func,'__name__','')=='linear' and len(args)>1:
            weight=args[1]
            if (weight.ndim==2 and weight.shape[0]==768
                    and weight.data_ptr()==self.image.patch_embed.proj.weight.data_ptr()):
                self.means.append(result.float().mean(0).cpu().numpy())
        return result
class Hooks:
    def __init__(self,model,no_cls=False):
        self.model=model;self.no_cls=no_cls;self.valid=None;self.values={};self.handles=[]
        for modality,blocks in [('point',model.point.blocks.blocks),('em',model.image.blocks)]:
            for l in LAYERS:
                self.handles.append(blocks[l-1].register_forward_hook(self.hook(modality,l)))
        self.handles.append(model.image.norm.register_forward_hook(self.norm_hook('em')))
        self.handles.append(model.point.norm.register_forward_hook(self.norm_hook('point')))
    def put(self,name,tensor):
        a=tensor.detach().float().cpu().numpy()
        assert a.ndim==2 and np.isfinite(a).all(),name
        self.values[name]=a
    def hook(self,modality,layer):
        def receive(module,args,h):
            if modality=='point':
                ordinary=h if self.no_cls else h[:,1:]
                self.put(f'point_mean_l{layer}_raw',ordinary.float().mean(1))
                if not self.no_cls:self.put(f'point_cls_l{layer}_raw',h[:,0])
            else:
                self.put(f'em_mean_l{layer}_raw',masked_mean(h[:,1:],self.valid[:,1:]))
                self.put(f'em_cls_l{layer}_raw',h[:,0])
        return receive
    def norm_hook(self,modality):
        def receive(module,args,h):
            if modality=='em':
                self.put('em_mean_l12_finalnorm',masked_mean(h[:,1:],self.valid[:,1:]))
                self.put('em_cls_l12_finalnorm',h[:,0])
            else:self.put('point_mean_l12_finalnorm',(h if self.no_cls else h[:,1:]).float().mean(1))
        return receive
def load(stage,device):
    spec=importlib.util.spec_from_file_location('probe_runtime_train',X/'train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    point,grouping,image=m.load_models(device);model=m.System(point,image).to(device)
    cp=None
    if stage in ['fusion50','nna20','nna0']:
        cp=X/'checkpoints'/('refine_epoch050.pt' if stage=='fusion50' else ('pretrain_epoch020.pt' if stage=='nna20' else 'pretrain_epoch000.pt'))
        ck=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
        assert ck['epoch']==(50 if stage=='fusion50' else (20 if stage=='nna20' else 0))
        if stage in ['nna20','nna0']:model.point_nna=m.LegacyNNA().to(device)
        model.load_state_dict(ck['model'],strict=True)
        model.point_nna=None;del ck
    model.phase='refine' if stage=='fusion50' else 'pretrain'
    model.eval().requires_grad_(False);grouping.eval().requires_grad_(False)
    return m,model,grouping,cp
def batch(m,model,grouping,rows,points,device,stage,book):
    book.values={}
    point_only=stage in ['nna20','nna0','pointmae100']
    capture=CaptureProjection(model.image)
    with capture:inp=m.prepare(rows,points,model.point,grouping,model.image,device,0,False,point_only=point_only)
    book.put('point_group_mean_pre_pe',inp[0].float().mean(1))
    book.put('point_input_mean_post_pe',(inp[0]+inp[1]).float().mean(1))
    if not point_only:
        book.valid=inp[3]
        assert len(capture.means)==len(rows)
        book.values['em_patch_mean_pre_pe']=np.stack(capture.means)
        book.put('em_input_mean_post_pe',masked_mean(inp[2],inp[3][:,1:]))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if stage=='pointmae100':
            h=model.point.blocks(inp[0],inp[1])
            model.point.norm(h)
            result=None
        else:result=model(inp,views=1,loss=False)
    values=dict(book.values)
    counts=None if point_only else inp[3][:,1:].sum(1).cpu().numpy()
    if stage=='em_base':values={k:v for k,v in values.items() if k.startswith('em_')}
    elif point_only:values={k:v for k,v in values.items() if k.startswith('point_')}
    return values,result,counts
def register(out,stage,arrays,checkpoint,manifest):
    registry=json.loads((out/'feature_registry.json').read_text());known={q['name'] for q in registry}
    for name,path in arrays.items():
        # Existing Fusion50 L11/L12 CLS remain the formal cache reads; extracted copies are QA anchors.
        if stage=='fusion50' and name in ['point_cls_l11_raw','point_cls_l12_raw','em_cls_l11_raw','em_cls_l12_finalnorm']:continue
        if stage!='fusion50' and ('cls_l6_' in name or 'cls_l8_' in name):continue
        full=stage+'_'+name
        if full in known:continue
        l=next((j for j in LAYERS if f'l{j}_' in name),0)
        pooling='CLS' if '_cls_' in name else 'mean of valid ordinary tokens only (no CLS, no padding)'
        norm='final LayerNorm applied per token before pooling' if 'finalnorm' in name else 'no final LayerNorm'
        if 'pre_pe' in name:norm='before positional encoding, no final LayerNorm'
        if 'post_pe' in name:norm='after positional encoding at Transformer input, before block1'
        cp=checkpoint or (X/'initial_weights'/('point_mae_epoch100.pt' if stage=='pointmae100' else 'image_mae_checkpoint119.pth'))
        a=np.load(path,mmap_mode='r')
        registry.append(dict(name=full,path=str(path),stage=stage,network_position=name,layer=l,dimension=a.shape[1],
          checkpoint=str(cp),checkpoint_stat=stat(cp),pooling=pooling,normalization=norm,
          input_protocol='same runtime core.prepare(training=False), sample0 per-instance seeded grouping; bf16 blocks, float32 pooling',
          mask_version='cohort mask mapping: 9870 filled files +808 unchanged original masks',
          ids_file=str(out/'instances.csv'),stable_key_order_sha256=manifest['stable_key_order_sha256'],
          input_cohort_sha256=manifest['cohort_sha256'],code=str(Path(__file__).resolve()),configuration=str(out/'extraction'/stage/'manifest.json'),
          feature_file_sha256=sha(path),available=True,
          interpretation='Point-MAE100 has no CLS; native TransformerEncoder wrapper then optional native norm' if stage=='pointmae100' else 'unchanged runtime forward with read-only hooks'))
    save(out/'feature_registry.json',registry)
def main(out,stage,smoke,gpu):
    out=Path(out);dest=out/'extraction'/stage;dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    rows=load_rows();points=np.load(F/'point_cache/pointclouds.npy',mmap_mode='r')
    m,model,grouping,cp=load(stage,device);book=Hooks(model,stage=='pointmae100')
    sources=[X/'train.py',X/'code/core.py',X/'code/base.py',X/'vendor/point_mae/models/Point_MAE.py',X/'vendor/image_interface/code/instance_tokens.py']
    manifest=dict(stage=stage,checkpoint=stat(cp) if cp else None,code_sha256=sha(__file__),sources={str(p):sha(p) for p in sources},
      cohort_sha256=sha(X/'cohort.json'),stable_key_order_sha256=__import__('hashlib').sha256('\n'.join(key(r) for r in rows).encode()).hexdigest(),
      batch_size=8,chunk_size=64,precision='float32 prepare, autocast bf16 native blocks, float32 pooling',
      point_position='native wrapper adds group position at every block; CLS zero position',
      em_position='3-axis sincos once at input in local voxel patch coordinates; actual runtime subsampling uses spacing=(1,1,1)',
      token_mask='input valid[:,1:] excludes CLS and all padding; fixed 64 real Point groups',
      gpu=gpu,scope='readability; no encoder or probe gradients during feature export')
    if (dest/'manifest.json').exists():assert json.loads((dest/'manifest.json').read_text())==manifest
    else:save(dest/'manifest.json',manifest)
    with torch.inference_mode():
        first,outputs,counts=batch(m,model,grouping,rows[:8],points,device,stage,book)
        repeated,_,_=batch(m,model,grouping,rows[:8],points,device,stage,book)
        delta={k:float(np.max(abs(v-repeated[k]))) for k,v in first.items()}
        assert max(delta.values())==0,'nondeterministic inference'
        qa=dict(shapes={k:list(v.shape) for k,v in first.items()},determinism_max_abs=delta,
          valid_em_tokens=None if counts is None else counts.tolist(),point_groups=64,finite=True)
        if stage=='fusion50':
            compare={}
            for mod,vals in [('point',outputs[0]),('image',outputs[1]),('fused',outputs[2])]:
                for l,z in vals.items():
                    original=np.load(CACHE/f'{mod}_{l}.npy',mmap_mode='r')[:8];new=z[:,0].float().cpu().numpy()
                    rel=float(np.linalg.norm(new-original)/max(np.linalg.norm(original),1e-12))
                    compare[f'{mod}_{l}']=dict(max_abs=float(abs(new-original).max()),relative_l2=rel)
                    assert rel<.025,'cache identity mismatch'
            qa['cache_comparison']=compare
            # Pooling correctness with arbitrary padding sentinels, independent of the attention computation.
            v=torch.tensor([[True,True,False],[True,False,False]],device=device)
            h=torch.randn(2,3,5,device=device);old=masked_mean(h,v)
            h[~v]=1e8;assert torch.equal(old,masked_mean(h,v))
            qa['padding_sentinel_pooling_test']=True
        save(dest/('smoke_qa.json' if smoke else 'qa.json'),qa);print(json.dumps(qa),flush=True)
        if smoke:return
        chunks=dest/'chunks';chunks.mkdir(exist_ok=True);started=time.time()
        for start in range(0,len(rows),64):
            file=chunks/f'{start:06d}.npz';end=min(start+64,len(rows));indices=np.arange(start,end)
            if file.exists():
                with np.load(file) as saved:assert np.array_equal(saved['ids'],indices)
                continue
            values={};token_counts=[]
            for pos in range(start,end,8):
                got,_,cnt=batch(m,model,grouping,rows[pos:min(pos+8,end)],points,device,stage,book)
                for k,val in got.items():values.setdefault(k,[]).append(val)
                if cnt is not None:token_counts.extend(cnt.tolist())
            packed={k:np.concatenate(v) for k,v in values.items()}
            assert all(np.isfinite(a).all() and len(a)==len(indices) for a in packed.values())
            tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,ids=indices,em_token_counts=np.array(token_counts),**packed);tmp.replace(file)
            save(dest/'progress.json',dict(done=end,total=len(rows),elapsed_seconds=time.time()-started))
            print(json.dumps(dict(stage=stage,done=end,total=len(rows),elapsed_seconds=time.time()-started)),flush=True)
    paths={}
    sample=next(chunks.glob('*.npz'))
    with np.load(sample) as z:names=[k for k in z.files if k not in ['ids','em_token_counts']]
    for name in names:
        parts=[]
        for start in range(0,len(rows),64):
            with np.load(chunks/f'{start:06d}.npz') as z:
                assert np.array_equal(z['ids'],np.arange(start,min(start+64,len(rows))))
                parts.append(z[name])
        a=np.concatenate(parts);assert a.shape[0]==10678 and np.isfinite(a).all()
        path=dest/(name+'.npy');tmp=path.with_suffix('.tmp.npy');np.save(tmp,a);tmp.replace(path);paths[name]=path
    assert all(sha(p)==h for p,h in [(Path(p),h) for p,h in manifest['sources'].items()])
    register(out,stage,paths,cp,manifest)
    save(dest/'complete.json',dict(instances=len(rows),arrays=len(paths),elapsed_seconds=time.time()-started,paths={k:str(p) for k,p in paths.items()}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stage',choices=['fusion50','nna20','nna0','pointmae100','em_base'],required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--gpu',default='GPU-3ae93c94-1c44-abc6-e01c-3e227609c329');a=p.parse_args();main(a.output,a.stage,a.smoke,a.gpu)
