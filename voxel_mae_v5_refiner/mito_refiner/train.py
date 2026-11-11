"""Separate post-MAE refinement: frozen heads, then encoder + heads.

True batch is the default. Replay changes the encoder memory policy only;
both paths run each head once on all paired features and use the same NNA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .backbone import load_adapter
from .checkpoint import attach_resume_state, restore_resume_state
from .data_utils import groups_of, load_global_views, load_paired_items, make_dataset
from .features import extract_features
from .models import MultiLayerID
from .queue import FeatureQueue
from .residual_adapter import ResidualEmbeddingAdapter
from .training_step import (backward_refinement, configure_encoder,
                            cross_view_heads, precision_context)

DEFAULT_ADAPTER = f"{__package__}.v5_adapter:V5VoxelMAEAdapter"


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def emit(out, row):
    line = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    with (Path(out) / "training_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_layers(args, available):
    tail = "final" if args.final_head_input == "bottleneck" else "block6_mean"
    names = tuple(s.strip() for s in args.head_layers.split(",") if s.strip()) if args.head_layers else (
        "block4", "block5", tail)
    if not names or len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError(f"Invalid head layers {names}; available: {tuple(available)}")
    return names


def new_queues(names, capacity, dim, device):
    return torch.nn.ModuleDict({name: FeatureQueue(capacity, dim).to(device) for name in names})


def lookup_metrics(lookup, body_ids):
    rows = [(entry, body_ids[i]) for i, views in enumerate(lookup)
            for entry in views if entry["valid"]]
    if not rows:
        return {"nn_similarity": None, "nn_age": None, "nn_rank": None,
                "same_body_nn_ratio": None}
    return {
        "nn_similarity": float(np.mean([entry["similarity"] for entry, _ in rows])),
        "nn_rank": float(np.mean([entry["rank"] for entry, _ in rows])),
        "nn_age": float(np.mean([entry["age"] for entry, _ in rows])),
        "same_body_nn_ratio": float(np.mean([entry["body_ids"][0] == body for entry, body in rows])),
    }


def gradient_norm(parameters):
    squares = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.


@torch.no_grad()
def update_ema(ema, online, decay):
    for target, source in zip(ema.parameters(), online.parameters()):
        target.lerp_(source, 1. - decay)
    for target, source in zip(ema.buffers(), online.buffers()):
        target.copy_(source)


def load_backbone(args, device):
    adapter = load_adapter(args.backbone_adapter)
    model, _, config = adapter.load_model(args.checkpoint, device)
    model.slice_microbatch = args.slice_microbatch
    model.tile_batch = args.tile_batch
    model.activation_checkpointing = args.activation_checkpointing
    config = {**config, "slice_microbatch": args.slice_microbatch,
              "tile_batch": args.tile_batch, "activation_checkpointing": args.activation_checkpointing}
    return adapter, model, config


def check_resume(args, state, ids, source_sha):
    if state.get("phase") != args.phase or state["source_checkpoint_sha256"] != source_sha:
        raise ValueError("Resume phase or source checkpoint does not match")
    if list(state["record_ids"]) != list(ids):
        raise ValueError("Resume cohort/order does not match")
    for key in ("seed", "batch_size", "precision", "temperature", "batch_mode", "ema_decay",
                "head_dim", "head_hidden", "queue_size", "head_lr", "encoder_lr",
                "residual_adapter", "adapter_hidden", "adapter_alpha", "freeze_encoder"):
        if state["config"].get(key) != getattr(args, key):
            raise ValueError(f"Resume config mismatch: {key}")


def checkpoint_state(args, model, config, heads, queues, optimizer, ids, bodies,
                     source_sha, cursor, step, updates, ema=None, feature_cache=None,
                     embedding_adapter=None, ema_embedding_adapter=None):
    state = {
        "phase": args.phase, "step": step, "optimizer_updates": updates,
        "cursor_epoch": cursor[0], "cursor_batch": cursor[1],
        "heads": heads.state_dict(), "queues": {n:q.state_dict() for n,q in queues.items()},
        "head_layers": list(heads.heads),
        "feature_dims": {n:head.projector[0].in_features for n,head in heads.heads.items()},
        "record_ids": ids, "body_ids": bodies, "train_record_ids": ids,
        "model_config": config, "backbone_adapter": load_adapter(args.backbone_adapter).identifier,
        "source_checkpoint": str(args.checkpoint), "source_checkpoint_sha256": source_sha,
        "config": vars(args), "bottleneck_refined": "final" in heads.heads,
        "input_protocol": "full visibility; full-instance-mask conditioned; global views 0 and 2",
        "embedding_adapter": embedding_adapter.state_dict() if embedding_adapter is not None else None,
        "embedding_adapter_config": embedding_adapter.config() if embedding_adapter is not None else None,
        "ema_embedding_adapter": (ema_embedding_adapter.state_dict()
                                  if ema_embedding_adapter is not None else None),
    }
    if args.phase == "refine":
        state.update(model=model.state_dict(), ema_model=ema.state_dict() if ema is not None else None)
    if feature_cache is not None:
        state["feature_cache"] = str(feature_cache)
    return attach_resume_state(state, optimizer)


def run_init(args, device):
    adapter, model, config = load_backbone(args, device)
    model.eval().requires_grad_(False)
    ds = make_dataset(args.cache, "train", args.limit, args.seed, train_views=True)
    if len(ds.records) < 2:
        raise ValueError("Head initialization requires >=2 instances")
    arrays, ids, bodies = {}, [], []
    for index, record in enumerate(ds.records):
        rows = []
        for batch in load_global_views(ds, index, device):
            with torch.no_grad(), precision_context(device, args.precision):
                rows.append(extract_features(adapter, model, batch))
        if not arrays:
            arrays = {name: [] for name in rows[0]}
        for name in arrays:
            arrays[name].append(np.stack([row[name][0].float().cpu().numpy() for row in rows]))
        ids.append(str(record["record_id"]))
        bodies.append(str(record["body_id"]))
        if (index + 1) % 16 == 0:
            emit(args.out, {"phase":"feature_cache", "done":index+1, "total":len(ds.records)})
    arrays = {name:np.stack(values).astype(np.float32) for name,values in arrays.items()}
    names = configured_layers(args, arrays)
    embedding_adapter = (ResidualEmbeddingAdapter(arrays["final"].shape[-1],
        args.adapter_hidden, args.adapter_alpha).to(device) if args.residual_adapter else None)
    cache = Path(args.out) / "feature_cache_train.npz"
    np.savez_compressed(cache, **arrays, record_id=np.asarray(ids), body_id=np.asarray(bodies),
                        view_kinds=np.asarray([0,2]))
    features = {name:torch.as_tensor(value, device=device) for name,value in arrays.items()}
    heads = MultiLayerID(input_dims={n:v.shape[-1] for n,v in arrays.items()}, layers=names,
                         output_dim=args.head_dim, hidden_dim=args.head_hidden).to(device)
    queues = new_queues(names, args.queue_size, args.head_dim, device)
    stage_a_parameters = list(heads.parameters())
    if embedding_adapter is not None:
        stage_a_parameters += list(embedding_adapter.parameters())
    optimizer = torch.optim.AdamW(stage_a_parameters, lr=args.head_lr, weight_decay=1e-5)
    source_sha = file_sha256(args.checkpoint)
    start_epoch, start_batch, step, updates = 0, 0, 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        check_resume(args, state, ids, source_sha)
        if state["head_layers"] != list(names):
            raise ValueError("Resume head layer selection differs")
        heads.load_state_dict(state["heads"])
        if embedding_adapter is not None:
            if state.get("embedding_adapter") is None:
                raise ValueError("Resume checkpoint lacks the requested residual adapter")
            embedding_adapter.load_state_dict(state["embedding_adapter"])
        for name in names: queues[name].load_state_dict(state["queues"][name])
        restore_resume_state(state, optimizer)
        start_epoch, start_batch = state["cursor_epoch"], state["cursor_batch"]
        step, updates = state["step"], state["optimizer_updates"]
    cursor = (start_epoch, start_batch)

    def snapshot():
        return checkpoint_state(args, model, config, heads, queues, optimizer, ids, bodies,
                                source_sha, cursor, step, updates, feature_cache=cache,
                                embedding_adapter=embedding_adapter)

    for epoch in range(start_epoch, args.epochs):
        groups = groups_of(np.random.default_rng(args.seed+epoch).permutation(len(ids)).tolist(), args.batch_size)
        for position, group in enumerate(groups):
            if epoch == start_epoch and position < start_batch: continue
            if args.max_steps and step >= args.max_steps: break
            ts = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            heads.train()
            total, losses, keys, counts, lookups, info = cross_view_heads(
                heads, {n:features[n][group].reshape(-1,features[n].shape[-1]) for n in names},
                queues, [ids[i] for i in group], args.temperature, step,
                selection="hard_top1", topk=1, embedding_adapter=embedding_adapter)
            grad = 0.
            if sum(counts.values()):
                total.backward()
                grad = gradient_norm(heads.parameters())
                torch.nn.utils.clip_grad_norm_(heads.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            for n in names:
                queues[n].enqueue(keys[n][:,0], [ids[i] for i in group], [bodies[i] for i in group], step)
            step += 1
            cursor = (epoch,position+1) if position+1 < len(groups) else (epoch+1,0)
            emit(args.out, {"phase":"head_init", "step":step, "optimizer_updates":updates,
                 "loss":float(total.detach()), "head_losses":{n:float(v.detach()) for n,v in losses.items()},
                 "valid_anchors":counts, "nna_diagnostics":info, "head_grad_norm_preclip":grad,
                 "neighbors":{n:lookup_metrics(v,[bodies[i] for i in group]) for n,v in lookups.items()},
                 "queue_fill_ratio":{n:float(q.valid.float().mean()) for n,q in queues.items()},
                 "runtime_s":time.perf_counter()-ts})
            if step % args.save_every == 0: save(Path(args.out)/"last_init.pt",snapshot())
        if args.max_steps and step >= args.max_steps: break
    state = snapshot()
    save(Path(args.out)/"last_init.pt",state)
    save(Path(args.out)/"head_init.pt",state)


def run_refine(args, device):
    adapter, model, config = load_backbone(args, device)
    init = torch.load(args.head_init or Path(args.out)/"head_init.pt", map_location="cpu", weights_only=False)
    source_sha = file_sha256(args.checkpoint)
    if init["source_checkpoint_sha256"] != source_sha or init["backbone_adapter"] != adapter.identifier:
        raise ValueError("Head initialization and source MAE differ")
    names = tuple(init["head_layers"])
    if args.head_layers and tuple(n.strip() for n in args.head_layers.split(",")) != names:
        raise ValueError("Stage B must use the Stage A head layers")
    for key in ("head_dim", "head_hidden", "queue_size", "precision"):
        if init["config"][key] != getattr(args,key):
            raise ValueError(f"Stage A/B mismatch: {key}")
    heads = MultiLayerID(input_dims=init["feature_dims"], layers=names,
                         output_dim=args.head_dim, hidden_dim=args.head_hidden).to(device)
    heads.load_state_dict(init["heads"])
    adapter_config = init.get("embedding_adapter_config")
    if bool(adapter_config) != bool(args.residual_adapter):
        raise ValueError("Stage A/B residual-adapter setting differs")
    embedding_adapter = (ResidualEmbeddingAdapter(**adapter_config).to(device)
                         if adapter_config else None)
    if embedding_adapter is not None:
        embedding_adapter.load_state_dict(init["embedding_adapter"])
    queues = new_queues(names,args.queue_size,args.head_dim,device)
    for name in names: queues[name].load_state_dict(init["queues"][name])
    if args.freeze_encoder and args.batch_mode == "replay":
        raise ValueError("Frozen-encoder refinement uses true batches; replay is unnecessary")
    encoder_parameters = configure_encoder(model,names,args.activation_checkpointing,
                                           freeze_encoder=args.freeze_encoder)
    adapter_parameters = list(embedding_adapter.parameters()) if embedding_adapter is not None else []
    trainable = encoder_parameters + adapter_parameters + list(heads.parameters())
    groups = ([{"params":encoder_parameters,"lr":args.encoder_lr}] if encoder_parameters else [])
    if adapter_parameters:
        groups.append({"params":adapter_parameters,"lr":args.head_lr})
    groups.append({"params":list(heads.parameters()),"lr":args.head_lr})
    optimizer = torch.optim.AdamW(groups,weight_decay=.05)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema_decay > 0 else None
    ema_embedding_adapter = (copy.deepcopy(embedding_adapter).eval().requires_grad_(False)
                             if embedding_adapter is not None and ema is not None else None)
    ds = make_dataset(args.cache,"train",args.limit,args.seed,train_views=True)
    ids = [str(r["record_id"]) for r in ds.records]
    bodies = [str(r["body_id"]) for r in ds.records]
    if ids != list(init["record_ids"]):
        raise ValueError("Stage A/B cohort or record order differs")
    start_epoch,start_batch,step,updates = 0,0,0,0
    if args.resume:
        state = torch.load(args.resume,map_location="cpu",weights_only=False)
        check_resume(args,state,ids,source_sha)
        if state["head_layers"] != list(names): raise ValueError("Resume heads differ")
        model.load_state_dict(state["model"])
        heads.load_state_dict(state["heads"])
        if embedding_adapter is not None:
            embedding_adapter.load_state_dict(state["embedding_adapter"])
        for name in names: queues[name].load_state_dict(state["queues"][name])
        if ema is not None: ema.load_state_dict(state["ema_model"])
        if ema_embedding_adapter is not None:
            ema_embedding_adapter.load_state_dict(state["ema_embedding_adapter"])
        restore_resume_state(state,optimizer)
        start_epoch,start_batch = state["cursor_epoch"],state["cursor_batch"]
        step,updates = state["step"],state["optimizer_updates"]
    cursor = (start_epoch,start_batch)

    def snapshot():
        return checkpoint_state(args,model,config,heads,queues,optimizer,ids,bodies,
                                source_sha,cursor,step,updates,ema,
                                embedding_adapter=embedding_adapter,
                                ema_embedding_adapter=ema_embedding_adapter)

    for epoch in range(start_epoch,args.epochs):
        ds.epoch = epoch
        groups = groups_of(np.random.default_rng(args.seed+epoch).permutation(len(ids)).tolist(),args.batch_size)
        for position, group in enumerate(groups):
            if epoch == start_epoch and position < start_batch: continue
            if args.max_steps and step >= args.max_steps: break
            ts = time.perf_counter()
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            items,kept,skipped = load_paired_items(ds,group)
            cursor = (epoch,position+1) if position+1 < len(groups) else (epoch+1,0)
            if len(kept) < 2:
                emit(args.out,{"phase":"skipped_pair_batch","epoch":epoch,"skipped":skipped,
                               "remaining_instances":len(kept)})
                continue
            batch_ids,batch_bodies = [ids[i] for i in kept],[bodies[i] for i in kept]
            optimizer.zero_grad(set_to_none=True)
            row = backward_refinement(adapter,model,heads,queues,items,batch_ids,device,
                   args.batch_mode,args.precision,args.temperature,step,args.replay_tolerance,
                   embedding_adapter=embedding_adapter)
            encoder_grad,head_grad = gradient_norm(encoder_parameters),gradient_norm(heads.parameters())
            adapter_grad = gradient_norm(adapter_parameters)
            if sum(row["valid_anchors"].values()):
                active_grad = adapter_grad if args.freeze_encoder else encoder_grad
                if (not np.isfinite(active_grad+head_grad) or
                        min(active_grad,head_grad) <= 0):
                    raise FloatingPointError("Expected finite, nonzero active-model and head gradients")
                torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                optimizer.step()
                if ema is not None:
                    update_ema(ema,model,args.ema_decay)
                    if ema_embedding_adapter is not None:
                        update_ema(ema_embedding_adapter,embedding_adapter,args.ema_decay)
                updates += 1
            for name in names: queues[name].enqueue(row["keys"][name][:,0],batch_ids,batch_bodies,step)
            row.pop("keys")
            row["neighbors"] = {n:lookup_metrics(v,batch_bodies) for n,v in row.pop("lookups").items()}
            step += 1
            emit(args.out,{**row,"phase":"encoder_refine","epoch":epoch,"step":step,
                 "optimizer_updates":updates,"batch_mode":args.batch_mode,"batch_instances":len(kept),
                 "batch_views":len(items),"skipped_pairs":skipped,"bottleneck_refined":"final" in names,
                 "encoder_grad_norm_preclip":encoder_grad,"head_grad_norm_preclip":head_grad,
                 "embedding_adapter_grad_norm_preclip":adapter_grad,
                 "encoder_frozen":args.freeze_encoder,"residual_adapter":embedding_adapter is not None,
                 "queue_fill_ratio":{n:float(q.valid.float().mean()) for n,q in queues.items()},
                 "runtime_s":time.perf_counter()-ts,
                 "gpu_peak_allocated_mb":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0.})
            if step % args.save_every == 0: save(Path(args.out)/"last_refine.pt",snapshot())
        if args.max_steps and step >= args.max_steps: break
    state = snapshot()
    save(Path(args.out)/"last_refine.pt",state)
    save(Path(args.out)/"refined.pt",state)


def main():
    p = argparse.ArgumentParser(description="MAE-v5 + separate MIM-Refiner R0")
    p.add_argument("--phase",choices=("init","refine"),required=True)
    for name in ("cache","checkpoint","out"): p.add_argument(f"--{name}",required=True)
    p.add_argument("--backbone-adapter",default=DEFAULT_ADAPTER)
    p.add_argument("--head-init",default="")
    p.add_argument("--resume",default="")
    p.add_argument("--limit",type=int,default=128)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--max-steps",type=int,default=0)
    p.add_argument("--save-every",type=int,default=50)
    p.add_argument("--batch-mode",choices=("true","replay"),default="true")
    p.add_argument("--precision",choices=("fp32","bf16"),default="bf16")
    p.add_argument("--activation-checkpointing",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--slice-microbatch",type=int,default=8)
    p.add_argument("--tile-batch",type=int,default=2)
    p.add_argument("--queue-size",type=int,default=2048)
    p.add_argument("--temperature",type=float,default=.2)
    p.add_argument("--head-dim",type=int,default=256)
    p.add_argument("--head-hidden",type=int,default=512)
    p.add_argument("--head-lr",type=float,default=2e-4)
    p.add_argument("--encoder-lr",type=float,default=1e-5)
    p.add_argument("--ema-decay",type=float,default=.999)
    p.add_argument("--replay-tolerance",type=float,default=2e-3)
    p.add_argument("--final-head-input",choices=("block6_mean","bottleneck"),default="block6_mean")
    p.add_argument("--head-layers",default="")
    p.add_argument("--residual-adapter",action="store_true")
    p.add_argument("--adapter-hidden",type=int,default=256)
    p.add_argument("--adapter-alpha",type=float,default=.1)
    p.add_argument("--freeze-encoder",action="store_true")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if int(os.environ.get("WORLD_SIZE","1")) != 1:
        p.error("This trainer is single-device; global NNA/BN across DDP ranks is not implemented")
    if min(a.epochs,a.queue_size,a.head_dim,a.head_hidden,a.save_every,a.slice_microbatch,a.tile_batch,a.adapter_hidden)<1 or a.batch_size<2:
        p.error("Positive sizes/epochs and batch-size >=2 are required")
    if a.max_steps<0 or a.limit<0 or a.temperature<=0 or a.replay_tolerance<=0 or a.adapter_alpha<=0 or not 0<=a.ema_decay<1:
        p.error("Invalid loss, limit, max-steps or EMA configuration")
    if torch.device(a.device).type=="cpu" and a.precision!="fp32": p.error("Use --precision fp32 on CPU")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    out = Path(a.out); out.mkdir(parents=True,exist_ok=True)
    existing = ("head_init.pt","last_init.pt") if a.phase=="init" else ("refined.pt","last_refine.pt")
    if not a.resume and any((out/name).exists() for name in existing):
        raise FileExistsError("Use a fresh output directory or explicit --resume")
    (out/f"config_{a.phase}.json").write_text(json.dumps(vars(a),indent=2),encoding="utf-8")
    (run_init if a.phase=="init" else run_refine)(a,torch.device(a.device))


if __name__=="__main__": main()
