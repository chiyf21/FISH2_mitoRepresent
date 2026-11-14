"""Refine Balanced1152D for 50 epochs, starting from a completed NNA20 run."""
import argparse
import fcntl
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from train import core, load_models, write, append, rng_state, restore, save_epoch, sha


def load_data(config):
    baseline = Path(config["baseline_run"])
    rows = json.loads((baseline / "cohort.json").read_text())
    for row in rows:
        row["spacing_xyz_nm"] = config["spacing_xyz_nm"]
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    if len(rows) != 10678 or counts != {"train": 7357, "val": 1706, "test": 1615}:
        raise ValueError(f"Unexpected baseline cohort: {len(rows)}, {counts}")
    if [r["cohort_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Baseline cohort ordering changed")
    points = np.load(config["pointclouds"], mmap_mode="r", allow_pickle=False)
    if points.shape != (10678, 2, 2048, 3):
        raise ValueError(f"Unexpected point cache: {points.shape}")
    return rows, points


def calibrate(model, config, train_ids, probe, device):
    # Reuse the baseline's canonical NNA20 calibration, before its first refine update.
    path = Path(config["baseline_run"]) / "calibration_epoch00_probe.npz"
    with np.load(path) as data:
        if not np.array_equal(data["ids"], probe):
            raise ValueError("Baseline calibration IDs do not match the fixed probe")
        keep = np.isin(data["ids"], train_ids)
        with torch.no_grad():
            for layer in core.LAYERS:
                variances = []
                for modality, prefix in (("point", "p"), ("image", "i")):
                    x = np.asarray(data[f"{modality}_{layer}"][keep], np.float64)
                    mean = x.mean(0)
                    variance = np.mean(np.sum((x - mean) ** 2, axis=1))
                    getattr(model.balance, f"mu_{prefix}_{layer}").copy_(
                        torch.tensor(mean, device=device, dtype=torch.float32))
                    variances.append(variance)
                getattr(model.balance, f"v_{layer}").copy_(
                    torch.tensor(variances, device=device, dtype=torch.float32))
            model.balance.updates.fill_(1)


def forbid_fusion(module, inputs):
    raise RuntimeError("The training model's Fusion MLP must never run")


def groups(model):
    return {
        "point_cls": [model.point_cls],
        "image_cls": [model.image.cls_token],
        "point_last4": [p for b in model.point.blocks.blocks[-4:] for p in b.parameters()],
        "image_last4": [p for b in model.image.blocks[-4:] for p in b.parameters()],
    }


def references(model):
    result = {"point_cls": model.point_cls, "image_cls": model.image.cls_token}
    for name, blocks in (("point", model.point.blocks.blocks), ("image", model.image.blocks)):
        for layer in range(8, 12):
            result[f"{name}_block{layer + 1}"] = next(blocks[layer].parameters())
    return result


def compare_losses(reference, captured, ids, parts):
    comparison = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for layer in core.LAYERS:
            old = reference(captured[layer].float())
            _, terms = core.vicreg(old)
            infonce, _ = core.filtered_infonce(old, ids, topk=20, temperature=0.2)
            terms["infonce"] = infonce
            comparison[str(layer)] = {
                key: {"mlp384": float(terms[key]),
                      "balanced1152": float(parts[f"{layer}/{key}"])}
                for key in ("invariance", "variance", "covariance", "infonce")
            }
    mean = {}
    for key in ("invariance", "variance", "covariance", "infonce"):
        old = statistics.mean(comparison[str(l)][key]["mlp384"] for l in core.LAYERS)
        new = statistics.mean(comparison[str(l)][key]["balanced1152"] for l in core.LAYERS)
        mean[key] = {"mlp384": old, "balanced1152": new,
                     "ratio": new / old if old else None}
    return dict(scope="same first batch, same raw CLS and PairBalance output; "
                      "a separate frozen, no-grad reference head is used only for comparison",
                per_layer=comparison, layer_mean=mean,
                covariance_matrix_fp32_bytes={"mlp384": 384 * 384 * 4,
                                               "balanced1152": 1152 * 1152 * 4})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-batches", help="comma-separated original batch numbers")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not cfg.get("without_fusion_head"):
        raise ValueError("This trainer requires without_fusion_head=true")
    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != cfg["world_size"] or world != 4:
        raise ValueError("Keep the baseline world size of four for exact sampler/RNG restoration")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    timeout = float(cfg.get("distributed_timeout_seconds", 600))
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    core.ROOT = out
    (out / "rank_status").mkdir(exist_ok=True)
    def rank_stage(stage, epoch=0, batch=0):
        write(out / "rank_status" / f"rank{rank}.json",
              dict(rank=rank, stage=stage, epoch=epoch, batch=batch, time=time.time()))
    scale = float(cfg.get("main_covariance_scale", 1.))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("main_covariance_scale must be positive and finite")
    expected_weights = dict(invariance=25., variance=5., covariance=.2*scale,
                            infonce=1., point_variance=1.25, point_covariance=.05,
                            image_variance=1.25, image_covariance=.05)
    if cfg["loss_weights"] != expected_weights:
        raise ValueError("Config effective loss coefficients disagree with the trainer")
    lock = None
    if rank == 0:
        lock = (out / "train.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "last.pt").exists() and not args.resume:
        raise FileExistsError("Existing run requires --resume")
    if args.resume and not (out / "last.pt").exists():
        raise FileNotFoundError("Resume requested without last.pt")
    if not args.smoke and not args.resume:
        smoke = json.loads(Path(cfg["smoke_result"]).read_text())
        if (smoke.get("status") != "smoke_passed" or
                smoke.get("main_covariance_scale") != scale or
                smoke.get("source_checkpoint") != cfg["init_pretrain"]):
            raise ValueError("Production requires a successful smoke test")
        if cfg.get("post_smoke_gradient_result"):
            check = json.loads(Path(cfg["post_smoke_gradient_result"]).read_text())
            if (check.get("status") != "diagnostic_complete" or
                    check.get("main_covariance_scale") != scale or
                    check.get("optimizer_steps") != 0 or not check.get("parameters_unchanged")):
                raise ValueError("Production requires the corrected post-smoke gradient check")
            acceptance = json.loads(Path(cfg["validation_acceptance"]).read_text())
            if acceptance.get("status") != "accepted":
                raise ValueError("Corrected validation has not been accepted")
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rows, points = load_data(cfg)
    tr = [j for j, r in enumerate(rows) if r["split"] == "train"]
    va = [j for j, r in enumerate(rows) if r["split"] == "val"]
    rng = np.random.default_rng(seed)
    probe = sorted(rng.choice(tr, 256, False).tolist() + rng.choice(va, 256, False).tolist())
    point, grouping, image = load_models(cfg, device)
    model = core.System(point, image, without_fusion_head=True,
                        main_covariance_scale=scale).to(device)
    checkpoint_path = out / "last.pt" if args.resume else Path(cfg["init_pretrain"])
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.resume:
        if ck["stage"] != "refine" or not ck["config"].get("without_fusion_head"):
            raise ValueError("Resume requires this ablation's refinement checkpoint")
        if bool(ck["config"].get("smoke", False)) != args.smoke:
            raise ValueError("Smoke checkpoints cannot initialize production")
        for key in ("seed", "world_size", "global_batch", "lr_blocks", "lr_cls_heads",
                    "pointclouds", "baseline_run", "init_pretrain", "spacing_xyz_nm",
                    "main_covariance_scale"):
            if ck["config"][key] != cfg[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        state = ck["model"]
    else:
        if (ck["stage"], ck["epoch"], ck["step"]) != ("pretrain", 20, 2300):
            raise ValueError("Initialization must be the baseline's completed NNA20")
        state = {k: v for k, v in ck["model"].items() if not k.startswith("point_nna.")}
    if ck["world_size"] != world or len(ck["rng_per_rank"]) != world:
        raise ValueError("Checkpoint RNG world size mismatch")
    model.load_state_dict(state, strict=True)
    model.phase = "refine"
    model.fusion.requires_grad_(False)
    image.cls_token.requires_grad_(True)
    for block in image.blocks[-4:]:
        block.requires_grad_(True)
    train_groups = groups(model)
    expected = {id(p) for values in train_groups.values() for p in values}
    if {id(p) for p in model.parameters() if p.requires_grad} != expected:
        raise ValueError("Unexpected trainable parameter outside last4 blocks and CLS tokens")
    enc = train_groups["point_last4"] + train_groups["image_last4"]
    cls = train_groups["point_cls"] + train_groups["image_cls"]
    opt = torch.optim.AdamW(
        [dict(params=enc, lr=1e-5, base_lr=1e-5),
         dict(params=cls, lr=2e-4, base_lr=2e-4)],
        weight_decay=1e-4, betas=(.9, .95))
    if args.resume:
        opt.load_state_dict(ck["optimizer"])
    restore(ck["rng_per_rank"][rank], device)
    if not args.resume:
        calibrate(model, cfg, tr, probe, device)
    ck_stage, ck_epoch, ck_step = ck["stage"], ck["epoch"], ck["step"]
    step = ck["step"]
    first_epoch = ck["epoch"] + 1 if args.resume else 1
    del state, ck
    optimized = {id(p) for group in opt.param_groups for p in group["params"]}
    assert not any(id(p) in optimized for p in model.fusion.parameters())
    fusion_guard = model.fusion.register_forward_pre_hook(forbid_fusion)
    fusion_before = {k: v.detach().cpu().clone() for k, v in model.fusion.state_dict().items()}
    if rank == 0:
        write(out / "config.resolved.json", dict(cfg, smoke=args.smoke))
        write(out / "cohort.json", rows)
        write(out / "monitor_ids.json", probe)
        write(out / "initialization_check.json", dict(
            status="PASS", source=str(checkpoint_path), strict_model_load=True,
            discarded_state_prefix=None if args.resume else "point_nna.",
            baseline_nna_retrained=False, mae_retrained=False,
            fusion_forward_guard=True, fusion_trainable=False, fusion_in_optimizer=False,
            representation="Balanced1152D", main_covariance_scale=scale,
            source_stage=ck_stage, source_epoch=ck_epoch, source_optimizer_step=ck_step,
            fresh_fusion_optimizer=not args.resume, trainable_parameters={
                k: sum(p.numel() for p in v) for k, v in train_groups.items()},
            trainable_names=[n for n, p in model.named_parameters() if p.requires_grad]))
        write(out / "status.json", dict(status="initializing", phase="refine", epoch=first_epoch-1))
    sampler = DistributedSampler(tr, num_replicas=world, rank=rank, shuffle=True,
                                 seed=seed, drop_last=True)
    loader = DataLoader(tr, batch_size=16, sampler=sampler, drop_last=False, num_workers=0)
    smoke_batches = (sorted(set(int(x) for x in args.smoke_batches.split(",")))
                     if args.smoke_batches else cfg.get("smoke_batches", [1, 2]))
    if args.smoke and (len(smoke_batches)<2 or min(smoke_batches)<1 or
                       max(smoke_batches)>len(loader)):
        raise ValueError("Invalid smoke batch selection")
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    if not args.smoke and not args.resume:
        core.monitor(model, rows, probe, points, grouping, device, rank, world, 0, "refine")
        save_epoch(model, opt, "refine", 0, step, out, rank, world, device,
                   dict(cfg, smoke=False))
    smoke_checks = []
    torch.cuda.reset_peak_memory_stats(device)
    epoch = first_epoch - 1
    for epoch in range(first_epoch, 51):
        stop = torch.tensor(int((out / "STOP_AFTER_EPOCH").exists()), device=device)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            if rank == 0:
                write(out / "paused.json", dict(stage="refine", last_completed_epoch=epoch-1,
                                               reason="STOP_AFTER_EPOCH", time=time.time()))
            dist.barrier()
            dist.destroy_process_group()
            return
        sampler.set_epoch(epoch + 1000)
        ddp.train()
        began = time.monotonic()
        total, count = 0., 0
        for batchno, positions in enumerate(loader, 1):
            if args.smoke and batchno not in smoke_batches:
                continue
            step_began = time.monotonic()
            rr = [rows[int(j)] for j in positions]
            ids = torch.tensor([r["cohort_index"] for r in rr], device=device)
            rank_stage("prepare", epoch, batchno)
            inputs = core.prepare(rr, points, point, grouping, image, device, epoch + 1000,
                                  True, point_only=False)
            prepare_seconds = time.monotonic()-step_began
            rank_stage("forward", epoch, batchno)
            frac = (epoch - 1) + (batchno - 1) / len(loader)
            factor = min(1., frac + 1/len(loader)) * (.5 + .5*math.cos(math.pi*frac/50))
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * factor
            opt.zero_grad(set_to_none=True)
            captured = {}
            capture_hook = None
            reference = None
            if args.smoke and batchno == 1:
                with torch.random.fork_rng(devices=[device.index]):
                    reference = core.MLPFusion().to(device).eval().requires_grad_(False)
                reference.load_state_dict(model.fusion.state_dict(), strict=True)
                capture_hook = model.balance.register_forward_hook(
                    lambda module, values, output: captured.__setitem__(values[2], output[0].detach()))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts, stats = ddp(inputs, ids)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            if capture_hook is not None:
                capture_hook.remove()
                comparison = compare_losses(reference, captured, ids, parts)
                if rank == 0:
                    write(out / "loss_scale_comparison.json", comparison)
                del reference, captured
            rank_stage("backward", epoch, batchno)
            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            group_grad_norms = {
                key: float(torch.stack([p.grad.detach().float().square().sum()
                                       for p in values if p.grad is not None]).sum().sqrt())
                for key, values in train_groups.items()} if rank == 0 else None
            grad = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(grad):
                raise RuntimeError("Nonfinite gradient")
            if args.smoke:
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
                    raise RuntimeError("Missing or nonfinite trainable gradient")
                if not all(p.grad is None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen parameter received a gradient")
                gradient_sums = {
                    key: float(sum(p.grad.abs().sum() for p in values))
                    for key, values in train_groups.items()}
                assert all(x > 0 for x in gradient_sums.values())
                reps = references(model)
                before = {k: p.detach().clone() for k, p in reps.items()}
            rank_stage("optimizer", epoch, batchno)
            opt.step()
            step += 1
            rank_stage("updated", epoch, batchno)
            total += float(loss.detach()) * len(rr)
            count += len(rr)
            if args.smoke:
                deltas = {k: float((p.detach() - before[k]).abs().max()) for k, p in reps.items()}
                assert all(x > 0 for x in deltas.values()), deltas
                assert all(torch.equal(fusion_before[k], v.detach().cpu())
                           for k, v in model.fusion.state_dict().items())
                smoke_checks.append(dict(
                    rank=rank, batch=batchno, loss=float(loss), grad_norm_before_clip=float(grad),
                    finite_trainable_gradients=True, frozen_gradients_absent=True,
                    gradient_sums=gradient_sums, optimizer_max_deltas=deltas,
                    fusion_forward_calls=0, fusion_gradients_absent=True,
                    fusion_unchanged=True, fusion_in_optimizer=False))
            if rank == 0:
                record = dict(
                    phase="refine", epoch=epoch, batch=batchno, batches_per_epoch=len(loader),
                    optimizer_step=step, refinement_optimizer_step=step-2300,
                    loss=float(loss), parts={k: float(v) for k, v in parts.items()},
                    grad_norm=float(grad), elapsed_epoch_sec=time.monotonic()-began,
                    step_seconds=time.monotonic()-step_began, prepare_seconds=prepare_seconds,
                    time=time.time(), main_covariance_scale=scale,
                    clipped=bool(grad>1.), clip_coefficient=min(1.,1./(float(grad)+1e-6)),
                    gradient_norms_by_group=group_grad_norms,
                    weighted_loss_components={
                        k: float(torch.stack([parts[f"{l}/{k}"] for l in core.LAYERS]).mean())*w
                        for k, w in dict(invariance=25., variance=5., covariance=.2*scale,
                                         infonce=1., point_regularizer=1.,
                                         image_regularizer=1.).items()},
                    learning_rates=[g["lr"] for g in opt.param_groups],
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2,
                    balance={k: dict(point_variance=float(s[0]), image_variance=float(s[1]),
                                     point_gain=float(s[2]), image_gain=float(s[3]))
                             for k, s in stats.items()})
                append(out / "step_metrics.jsonl", record)
                write(out / "status.json", dict(status="training", **record))
            if args.smoke and batchno == smoke_batches[-1]:
                break
        training_seconds = time.monotonic()-began
        if not args.smoke:
            core.monitor(model, rows, probe, points, grouping, device, rank, world, epoch, "refine")
        save_epoch(model, opt, "refine", epoch, step, out, rank, world, device,
                   dict(cfg, smoke=args.smoke), smoke=args.smoke)
        if rank == 0:
            path = out / "checkpoints" / f"refine_epoch{epoch:03d}.pt"
            write(out / "checkpoint_manifests" / f"refine_epoch{epoch:03d}.json",
                  dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                       stage="refine", epoch=epoch, step=step, representation="Balanced1152D"))
            summary = dict(status="epoch_complete", phase="refine", epoch=epoch, epochs=50,
                           optimizer_step=step, mean_loss=total/count,
                           training_seconds=training_seconds,
                           elapsed_epoch_sec=time.monotonic()-began, time=time.time())
            append(out / "epoch_timing.jsonl", summary)
            write(out / "status.json", summary)
        dist.barrier()
        if args.smoke:
            saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"], strict=True)
            opt.load_state_dict(saved["optimizer"])
            restore(saved["rng_per_rank"][rank], device)
            now = rng_state(device)
            expected_rng = saved["rng_per_rank"][rank]
            assert torch.equal(now["torch"], expected_rng["torch"].cpu())
            assert torch.equal(now["cuda"], expected_rng["cuda"].cpu())
            assert now["python"] == expected_rng["python"]
            assert now["numpy"][0] == expected_rng["numpy"][0]
            assert np.array_equal(now["numpy"][1], expected_rng["numpy"][1])
            assert now["numpy"][2:] == expected_rng["numpy"][2:]
            assert saved["stage"] == "refine" and saved["step"] == 2300+len(smoke_batches)
            assert all(float(state["step"]) == len(smoke_batches) for state in opt.state.values())
            payload = dict(rank=rank, checks=smoke_checks, checkpoint_restore="PASS",
                           peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                           peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
            gathered = [None]*world
            dist.all_gather_object(gathered, payload)
            if rank == 0:
                write(out / "smoke_checks.json", gathered)
            break
    if rank == 0:
        write(out / "complete.json", dict(status="smoke_passed" if args.smoke else "training_complete",
                                         stage="refine", epoch=epoch, steps=step,
                                         representation="Balanced1152D", main_covariance_scale=scale,
                                         source_checkpoint=cfg["init_pretrain"],
                                         smoke_batches=smoke_batches if args.smoke else None,
                                         time=time.time()))
    fusion_guard.remove()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
