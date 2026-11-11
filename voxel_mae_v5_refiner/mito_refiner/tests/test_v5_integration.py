"""CPU correctness checks on disposable synthetic fixtures, not a training pilot."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from ... import image_backbone
from ...data import MitoDataset, collate_batch
from ...model import VoxelMAE
from .. import train
from ..data_utils import load_paired_items, load_sample, make_dataset
from ..evaluate import embed_dataset, target_features
from ..models import MultiLayerID
from ..queue import FeatureQueue
from ..residual_adapter import ResidualEmbeddingAdapter
from ..training_step import backward_refinement, configure_encoder
from ..v5_adapter import V5VoxelMAEAdapter


def fixture_cache(root):
    records=[]
    rng=np.random.default_rng(42)
    for i in range(6):
        shape=(7,24+i*2,26+i*2)
        grid=np.indices(shape)
        mask=sum(((grid[a]-(shape[a]-1)/2)/(shape[a]/4))**2 for a in range(3))<1
        raw=rng.integers(0,256,shape,dtype=np.uint8)
        valid=np.ones(shape,bool);valid[:,0,:]=False
        directory=root/str(i);directory.mkdir()
        for name,value in (("em",raw),("mask",mask),("valid",valid)):
            np.save(directory/f"{name}.npy",value)
        records.append(dict(record_id=f"r{i}",body_id=f"body{i}",relative_dir=str(i),
                       split="train" if i<4 else "val" if i==4 else "test",
                       shape_zyx=list(shape),spacing_zyx_nm=[30,8,8],lo=0,hi=255))
    (root/"manifest.json").write_text(json.dumps({"records":records}),encoding="utf-8")


def full_visibility_and_pairing(root):
    ds=make_dataset(root,"train",train_views=False)
    item=ds[0]
    assert not item["skip"] and "query_um" not in item
    assert torch.equal(item["observed"],item["image_valid"])
    assert torch.equal(item["image"],ds[0]["image"])
    batch=load_sample(ds,0,"cpu")
    assert "volumes" in batch and "query_um" not in batch
    paired=make_dataset(root,"train",train_views=True)
    items,kept,skips=load_paired_items(paired,[0,1])
    assert kept==[0,1] and not skips
    assert [i["record_id"] for i in items]==["r0","r0","r1","r1"]
    assert [i["augmentation_kind"] for i in items]==[0,2,0,2]
    packed=collate_batch(items)
    assert packed["points_um"].shape[0]==4
    assert [v[0].shape[-1] for v in packed["volumes"]]==[26,26,28,28]

    class FailedView:
        records=paired.records;views=4;augmentations=True
        def __getitem__(self,index):
            if index==len(self.records)*2:
                return dict(skip=True,record_id="r0",reason="synthetic_failure")
            return paired[index]
    items,kept,skips=load_paired_items(FailedView(),[0,1])
    assert kept==[1] and len(items)==2 and skips[0]["record_id"]=="r0"
    # Compare the original masked MAE path against its source implementation.
    from mito_voxel_mae_v5.data import MitoDataset as OriginalDataset
    old=OriginalDataset(root,"train",queries=8,views=1,augmentations=False,masking=True)
    new=MitoDataset(root,"train",queries=8,views=1,augmentations=False,masking=True)
    for i in range(4):
        a,b=old[i],new[i]
        assert a["skip"]==b["skip"]
        for key,value in a.items():
            if torch.is_tensor(value): assert torch.equal(value,b[key]),key
    return {"full_visibility":True,"pair_order":True,"pairwise_skip":True,
            "masked_mae_regression":"exact tensor equality vs original v5"}


def synthetic_items(batch=4):
    result=[]
    vox=torch.tensor([[0.,5.,5.],[0.,12.,12.],[0.,20.,20.],[0.,30.,30.],
                      [0.,5.,30.],[0.,30.,5.],[0.,14.,25.],[0.,25.,14.]])
    for i in range(batch*2):
        n=8-i%3;v=vox[:n].clone();raw=torch.randn(1,1,40,40)
        result.append(dict(image=raw,target_mask=(raw>0).float(),observed=torch.ones_like(raw),
            image_valid=torch.ones_like(raw),points_vox=v,
            points_um=(v-torch.tensor([0.,20.,20.]))*torch.tensor([.030,.008,.008]),
            points_valid=torch.ones(n,dtype=torch.bool),sample_weight=torch.tensor(1.),
            record_id=f"i{i//2}",context_shape=np.array([1,40,40]),skip=False))
    return result


def forward_interfaces(root):
    """No loss.backward(), optimizer, or real pretrained weights in this check."""
    torch.manual_seed(42)
    cfg=dict(dim=32,depth=6,heads=4,latent=32,groups=8,k=4,channels=4,
             slice_microbatch=2,tile_batch=2)
    model=VoxelMAE(**cfg).eval();adapter=V5VoxelMAEAdapter()
    items=synthetic_items(2);batch=collate_batch(items)
    with torch.no_grad():
        joined=adapter.extract_features(model,batch)
        individual=[adapter.extract_features(model,collate_batch([item])) for item in items]
    error=max(float((joined[n]-torch.cat([r[n] for r in individual])).abs().max()) for n in joined)
    assert error<1e-4
    from mito_voxel_mae_v5.model import VoxelMAE as OriginalModel
    original=OriginalModel(**cfg).eval()
    original.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        source_error=float((original.encode(batch)-joined["final"]).abs().max())
    assert source_error<1e-4
    with mock.patch.object(image_backbone,"checkpoint",wraps=image_backbone.checkpoint) as spy:
        result=model.encode(collate_batch(items[:1]))
        assert spy.call_count>0 and not model.training
        del result
    ds=make_dataset(root,"test")
    views=make_dataset(root,"test",train_views=True)
    payload=embed_dataset(adapter,model,ds,"cpu",two_global_views=True,view_ds=views)
    assert payload["features"]["final"].shape==(1,32)
    assert payload["paired_views"]["final"].shape==(1,2,32)
    return {"max_batch_vs_single_abs_error":error,"max_original_v5_forward_abs_error":source_error,
            "eval_checkpoint_forward":True,"primary_and_stability_interfaces":True,
            "optimizer_updates":0,"fixture_only":True}


def batch_replay_and_checkpoint():
    torch.manual_seed(42)
    items=synthetic_items(4);names=("block4","block5","block6_mean")
    cfg=dict(dim=32,depth=6,heads=4,latent=32,groups=8,k=4,channels=4,
             slice_microbatch=2,tile_batch=2)
    a=VoxelMAE(**cfg).eval();b=copy.deepcopy(a)
    adapter=V5VoxelMAEAdapter()
    head_a=MultiLayerID(input_dim=32,hidden_dim=32,output_dim=16,layers=names)
    head_b=copy.deepcopy(head_a)
    qa=nn.ModuleDict({n:FeatureQueue(32,16) for n in names})
    for q in qa.values():
        q.enqueue(torch.randn(24,16),[f"bank{i}" for i in range(24)],["body"]*24,0)
    qb=copy.deepcopy(qa)
    configure_encoder(a,names,True);configure_encoder(b,names,True)
    rng=torch.get_rng_state()
    out_a=backward_refinement(adapter,a,head_a,qa,items,[f"i{i}" for i in range(4)],
                             "cpu",mode="true",precision="fp32")
    torch.set_rng_state(rng)
    out_b=backward_refinement(adapter,b,head_b,qb,items,[f"i{i}" for i in range(4)],
                             "cpu",mode="replay",precision="fp32")
    loss_error=abs(out_a["loss"]-out_b["loss"])
    abs_errors=[];left=[];right=[]
    for pa,pb in zip(list(a.parameters())+list(head_a.parameters()),list(b.parameters())+list(head_b.parameters())):
        assert (pa.grad is None)==(pb.grad is None)
        if pa.grad is not None:
            abs_errors.append(float((pa.grad-pb.grad).abs().max()))
            left.append(pa.grad.flatten());right.append(pb.grad.flatten())
    ga,gb=torch.cat(left),torch.cat(right)
    relative=float((ga-gb).norm()/ga.norm().clamp_min(1e-12))
    assert loss_error<1e-4 and relative<3e-3,(loss_error,relative)
    assert all(p.grad is None for p in a.bottleneck.parameters())
    for model,heads in ((a,head_a),(b,head_b)):
        parameters=[p for p in list(model.parameters())+list(heads.parameters()) if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        torch.optim.SGD(parameters,lr=.01).step()
    update_error=max(float((pa-pb).abs().max()) for pa,pb in zip(
        list(a.parameters())+list(head_a.parameters()),list(b.parameters())+list(head_b.parameters())))
    assert update_error<1e-5
    a.zero_grad(set_to_none=True)
    with mock.patch.object(image_backbone,"checkpoint",wraps=image_backbone.checkpoint) as spy:
        z=a.encode(collate_batch(items[:1]));z.sum().backward()
        assert spy.call_count>0 and not a.training
    with mock.patch.object(image_backbone,"checkpoint",wraps=image_backbone.checkpoint) as spy:
        with torch.no_grad(): a.encode(collate_batch(items[:1]))
        assert spy.call_count==0
    configure_encoder(a,("block4","block5","final"),True)
    a.zero_grad(set_to_none=True)
    adapter.extract_features(a,collate_batch(items[:1]))["final"].square().mean().backward()
    assert any(p.grad is not None and float(p.grad.norm())>0 for p in a.bottleneck.parameters())
    return dict(loss_max_abs_error=loss_error,gradient_max_abs_error=max(abs_errors),
                gradient_relative_l2_error=relative,parameter_update_max_abs_error=update_error,
                eval_checkpointing=True,bottleneck_frozen_by_default=True,optional_final_gradient=True,
                synthetic_contrastive_batch=[4,2])


def residual_adapter_forward():
    torch.manual_seed(42)
    x=torch.randn(6,32,requires_grad=True)
    module=ResidualEmbeddingAdapter(dim=32,hidden_dim=16,alpha_init=.1)
    initial=module(x)
    assert initial.shape==x.shape
    assert torch.allclose(initial,torch.nn.functional.normalize(x,dim=-1),atol=1e-6)
    torch.nn.init.normal_(module.fc2.weight,std=.01)
    module(x).square().sum().backward()
    assert module.fc1.weight.grad is not None and float(module.fc1.weight.grad.norm())>0
    assert module.fc2.weight.grad is not None and float(module.fc2.weight.grad.norm())>0
    assert module.alpha.grad is not None and float(module.alpha.grad.abs())>0
    return {"output_dim":32,"identity_at_initialization":True,"adapter_gradient":True}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,16))
        self.bottleneck=nn.Linear(16,16)
        self.block=nn.Linear(16,1);self.size=nn.Linear(16,3)


class ToyAdapter:
    identifier="synthetic-v5-test"
    def extract_features(self,model,batch):
        rows=[]
        for volume in batch["volumes"]:
            image,mask,observed,valid=volume
            rows.append(torch.stack([image.mean(),image.std(),mask.mean(),valid.mean()]))
        x=model.encoder(torch.stack(rows))
        return {"block4":x,"block5":x.tanh(),"block6_mean":x*1.1,"final":model.bottleneck(x)}


def trainer_resume(root):
    source=root/"synthetic_source.pt";torch.save({"synthetic":True},source)
    def args(out,phase,residual=False,frozen=False):
        return argparse.Namespace(cache=str(root),checkpoint=str(source),out=str(out),phase=phase,
            backbone_adapter="synthetic:test",head_init="",resume="",limit=4,seed=42,epochs=3,
            batch_size=2,max_steps=3,save_every=1,batch_mode="true",precision="fp32",
            activation_checkpointing=True,slice_microbatch=2,tile_batch=2,queue_size=16,
            temperature=.2,head_dim=8,head_hidden=16,head_lr=1e-3,encoder_lr=1e-3,
            ema_decay=.9,replay_tolerance=2e-3,final_head_input="block6_mean",
            head_layers="block4,block5,final" if residual else "",
            residual_adapter=residual,adapter_hidden=8,adapter_alpha=.1,freeze_encoder=frozen)
    initial=ToyModel().state_dict()
    adapter=ToyAdapter()
    def load(a,device):
        m=ToyModel();m.load_state_dict(initial);return adapter,m,{}
    errors={}
    with mock.patch.object(train,"load_backbone",side_effect=load),mock.patch.object(train,"load_adapter",return_value=adapter):
        for phase in ("init","refine"):
            full=root/f"{phase}_full";partial=root/f"{phase}_resumed"
            full.mkdir();partial.mkdir()
            a=args(full,phase);b=args(partial,phase)
            if phase=="refine":
                a.head_init=b.head_init=str(root/"init_full"/"head_init.pt")
            run=train.run_init if phase=="init" else train.run_refine
            torch.manual_seed(71);np.random.seed(71)
            run(a,torch.device("cpu"))
            torch.manual_seed(71);np.random.seed(71)
            b.max_steps=1;run(b,torch.device("cpu"))
            b.max_steps=3;b.resume=str(partial/f"last_{phase}.pt")
            run(b,torch.device("cpu"))
            name="head_init.pt" if phase=="init" else "refined.pt"
            expected=torch.load(full/name,weights_only=False)
            actual=torch.load(partial/name,weights_only=False)
            max_error=0.
            for key in (("heads",) if phase=="init" else ("heads","model","ema_model")):
                for n,value in expected[key].items():
                    diff=(value-actual[key][n]).abs()
                    max_error=max(max_error,float(diff.max()))
            for n in expected["queues"]:
                assert expected["queues"][n]["_extra_state"]==actual["queues"][n]["_extra_state"]
                assert torch.equal(expected["queues"][n]["features"],actual["queues"][n]["features"])
            assert max_error<1e-7,(phase,max_error)
            assert expected["step"]==actual["step"]==3
            assert expected["cursor_epoch"]==actual["cursor_epoch"]
            assert expected["cursor_batch"]==actual["cursor_batch"]
            errors[phase]=max_error
        adapter_errors={}
        for phase in ("init","refine"):
            full=root/f"adapter_{phase}_full";partial=root/f"adapter_{phase}_resumed"
            full.mkdir();partial.mkdir()
            a=args(full,phase,residual=True,frozen=True)
            b=args(partial,phase,residual=True,frozen=True)
            if phase=="refine":
                a.head_init=b.head_init=str(root/"adapter_init_full"/"head_init.pt")
            run=train.run_init if phase=="init" else train.run_refine
            torch.manual_seed(91);np.random.seed(91)
            run(a,torch.device("cpu"))
            torch.manual_seed(91);np.random.seed(91)
            b.max_steps=1;run(b,torch.device("cpu"))
            b.max_steps=3;b.resume=str(partial/f"last_{phase}.pt")
            run(b,torch.device("cpu"))
            filename="head_init.pt" if phase=="init" else "refined.pt"
            expected=torch.load(full/filename,weights_only=False)
            actual=torch.load(partial/filename,weights_only=False)
            assert expected["embedding_adapter"] is not None
            max_error=max(float((value-actual["embedding_adapter"][key]).abs().max())
                          for key,value in expected["embedding_adapter"].items())
            head_error=max(float((value-actual["heads"][key]).abs().max())
                           for key,value in expected["heads"].items())
            model_error=(max(float((value-actual["model"][key]).abs().max())
                             for key,value in expected["model"].items())
                         if phase=="refine" else 0.)
            max_error=max(max_error,head_error,model_error)
            assert max_error<1e-7 and model_error==0 and expected["step"]==actual["step"]==3
            adapter_errors[phase]=max_error
        errors["residual_adapter_frozen_encoder"] = adapter_errors
    return {"trainer_resume_max_parameter_error":errors,"fixture_only":True}


def main():
    p=argparse.ArgumentParser();p.add_argument("--out")
    p.add_argument("--interfaces-only",action="store_true",
                   help="No backward or optimizer updates; use while training experiments are deferred")
    a=p.parse_args()
    torch.set_num_threads(2)
    results=[]
    with tempfile.TemporaryDirectory(prefix="v5_refiner_unit_") as directory:
        root=Path(directory);fixture_cache(root)
        tests=[("encoder_only_pairing_and_mae_regression",lambda:full_visibility_and_pairing(root)),
               ("v5_forward_and_evaluation_interfaces",lambda:forward_interfaces(root))]
        tests.append(("residual_adapter_forward_and_gradient",residual_adapter_forward))
        if not a.interfaces_only:
            tests += [("v5_true_batch_replay_checkpoint_gradient",batch_replay_and_checkpoint),
                      ("stage_a_b_resume_on_synthetic_fixture",lambda:trainer_resume(root))]
        for name,test in tests:
            try:
                detail=test();results.append({"test":name,"status":"PASS",**detail})
            except Exception as e:
                results.append({"test":name,"status":"FAIL","error":repr(e)})
                if a.out: Path(a.out).write_text(json.dumps(results,indent=2),encoding="utf-8")
                print(json.dumps(results[-1]),flush=True);raise
            print(json.dumps(results[-1]),flush=True)
        ds=make_dataset(root,"test")
        values=target_features(ds,0)
        foreground=np.load(root/"5"/"mask.npy").astype(bool)
        raw=np.load(root/"5"/"em.npy")
        assert np.isclose(values[4],raw[foreground].mean())
        assert 0<values[3]<=1.1
        results.append({"test":"raw_texture_and_signed_surface_area","status":"PASS"})
    if a.out: Path(a.out).write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__": main()
