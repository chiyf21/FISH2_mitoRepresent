"""Train and resume Full326 fusion with per-view VICReg and negative-only BCE."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from full326_per_view_train import step
from train import core, load_models, restore, rng_state, write


def balanced_ranks(ids, cost, world):
    groups = [[] for _ in range(world)]
    totals = [0] * world
    for pos in sorted(range(len(ids)), key=lambda i: (-cost[int(ids[i])], i)):
        group = min((g for g in range(world) if len(groups[g]) < len(ids) // world),
                    key=lambda g: (totals[g], g))
        groups[group].append(pos)
        totals[group] += cost[int(ids[pos])]
    return [ids[sorted(group)] for group in groups]


def crop_costs(root, rows, needed, rank):
    p = root / "em_crop_voxels.json"
    costs = None
    if rank == 0:
        if p.exists():
            costs = json.loads(p.read_text())
        else:
            start = time.monotonic()
            costs = {int(i): int(np.load(rows[i]["em_path"], mmap_mode="r", allow_pickle=False).size)
                     for i in sorted(needed)}
            write(p, costs)
            print(f"Read {len(costs)} crop sizes in {time.monotonic()-start:.1f}s", flush=True)
        costs = {int(k): v for k, v in costs.items()}
        if set(costs) != needed or any(v <= 0 for v in costs.values()):
            raise ValueError("Crop-size cache does not match the fixed split")
    holder = [costs]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def truncate_incomplete(path, key, boundary):
    if path.exists():
        lines = path.read_text().splitlines()
        keep = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row[key] <= boundary:
                keep.append(line)
        if len(keep) != len(lines):
            tmp = path.with_suffix(".recovered")
            tmp.write_text("".join(line + "\n" for line in keep))
            tmp.replace(path)
        return [json.loads(line) for line in keep]
    return []


def learning_rate_factor(updates, steps_per_epoch, base_epochs, extension):
    progress = updates / steps_per_epoch
    if extension is not None:
        anchor = extension["anchor_epoch"]
        end = extension["end_epoch"]
        progress = anchor + (progress - anchor) * (base_epochs - anchor) / (end - anchor)
    warmup = min(1., (updates + 1) / steps_per_epoch)
    return warmup * (.5 + .5 * math.cos(math.pi * progress / base_epochs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int, help="Absolute final epoch of this invocation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-schedule-to", type=int,
                        help="Stretch the remaining cosine decay from the resumed epoch to this epoch")
    parser.add_argument("--stop-at-unix", type=float, help="Stop at an epoch boundary near this Unix time")
    a = parser.parse_args()
    root = a.config.resolve().parent
    cfg = json.loads(a.config.read_text())
    if a.epochs < 1 or (a.extend_schedule_to is not None and not a.resume):
        parser.error("A positive epoch is required, and extending the schedule requires --resume")
    base_epochs = cfg["schedule_epochs"]
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    world = dist.get_world_size()
    assert world == 4 and cfg["global_batch"] == 512
    if rank == 0:
        lock = (root / "train.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dist.barrier()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", local)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    run = Path(cfg["point_run"])
    rows = [dict(row, cohort_index=i, point_row=i)
            for i, row in enumerate(json.loads((run / "data/instances.json").read_text()))]
    points = np.load(run / "data/pointclouds.npy", mmap_mode="r", allow_pickle=False)
    split = json.loads((root / "split.json").read_text())
    cfg_split = json.loads((run / "split.json").read_text())
    if split["train"] != cfg_split["train"] or set(split["validation"]) != set(cfg_split["val"]):
        raise ValueError("Split differs from the Full326 pretraining baseline")
    train = np.array(split["train"])
    val = np.array(split["validation_batches"]).ravel().reshape(-1, cfg["global_batch"])
    monitor = json.loads((root / "validation_probe_ids.json").read_text())
    if not set(val.ravel()).union(monitor) <= set(split["validation"]):
        raise ValueError("Validation output must use the held-out validation split")
    excluded_train = set(cfg.get("exclude_train_indices", []))
    if not excluded_train.issubset(set(train.tolist())):
        raise ValueError("Crop exclusions must belong to the training split")
    steps_per_epoch = (len(train) - len(excluded_train)) // cfg["global_batch"]
    if steps_per_epoch == 0:
        raise ValueError("The filtered training split must contain a complete batch")
    costs = crop_costs(root, rows, set(train.tolist()) | set(val.ravel().tolist()), rank)
    point, grouping, image = load_models(cfg, device)
    model = core.System(point, image).to(device)
    model.phase = "refine"
    model.fusion.requires_grad_(True)
    model.image.cls_token.requires_grad_(True)
    for b in model.image.blocks[-4:]:
        b.requires_grad_(True)
    enc = [p for b in list(point.blocks.blocks[-4:])+list(image.blocks[-4:]) for p in b.parameters()]
    heads = [model.point_cls, model.image.cls_token] + list(model.fusion.parameters())
    peaks = cfg["lr_peak"]
    opt = torch.optim.AdamW([dict(params=enc, lr=peaks[0]), dict(params=heads, lr=peaks[1])],
                           weight_decay=1e-4, betas=(.9, .95))
    first = updates = 0
    extension = None
    last_epoch_seconds = 0.
    last = root / "last.pt"
    if a.resume:
        saved = sorted((root / "checkpoints").glob("refine_epoch*.pt"))
        if not saved:
            raise FileNotFoundError("No finished epoch checkpoint to resume")
        state = torch.load(saved[-1], map_location="cpu", mmap=True, weights_only=False)
        if state["stage"] != "refine" or state["settings"] != cfg or state["world_size"] != world:
            raise ValueError("Checkpoint and training settings must match")
        first = state["epoch"] + 1
        updates = state["step"]
        if not first <= a.epochs or updates != state["epoch"] * steps_per_epoch:
            raise ValueError("Resume target must advance from a complete epoch")
        extension = state.get("schedule_extension")
        last_epoch_seconds = state["epoch_summary"]["seconds"]
        if a.extend_schedule_to is not None:
            if extension is None:
                anchor = state["epoch"]
                if not 0 < anchor < base_epochs < a.extend_schedule_to:
                    raise ValueError("Extend from a completed epoch before the original schedule ends")
                extension = dict(anchor_epoch=anchor, end_epoch=a.extend_schedule_to)
            elif extension["end_epoch"] != a.extend_schedule_to:
                raise ValueError("Resume must use the same extended schedule")
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        restore(state["rng_per_rank"][rank], device)
        if rank == 0:
            history = truncate_incomplete(root/"epochs.jsonl", "epoch", first-1)
            truncate_incomplete(root/"steps.jsonl", "updates", updates)
            truncate_incomplete(root/"epoch_metrics.jsonl", "epoch", first-1)
            if not history or history[-1]["epoch"] < first-1:
                core.append(root/"epochs.jsonl", state["epoch_summary"])
            row = dict(epoch=first-1, step=updates, optimizer_entries=len(opt.state), time=time.time(),
                       restored_rng_ranks=len(state["rng_per_rank"]), next_epoch=first,
                       schedule_extension=extension)
            core.append(root/"resume_events.jsonl", row)
            print("Restored", json.dumps(row), flush=True)
        del state
    else:
        if last.exists() or (root/"epochs.jsonl").exists():
            raise FileExistsError("Existing training output requires --resume")
        state = torch.load(cfg["point_cls_init_checkpoint"], map_location="cpu", mmap=True, weights_only=False)
        if state["epoch"] != 20 or state["config"]["point_checkpoint"] != cfg["point_checkpoint"] or len(state["point_last_blocks"]) != 4:
            raise ValueError("Expected the matching E20 Point CLS and four tuned encoder blocks")
        for b, weights in zip(point.blocks.blocks[-4:], state["point_last_blocks"]):
            b.load_state_dict(weights, strict=True)
        with torch.no_grad():
            model.point_cls.copy_(state["point_cls"])
        del state
    if a.epochs > (extension["end_epoch"] if extension is not None else base_epochs):
        parser.error("The final epoch must fit the learning rate schedule")
    core.ROOT = root
    negative = cfg["negative_bce"]
    if rank == 0:
        (root/"complete.json").unlink(missing_ok=True)
        (root/"paused.json").unlink(missing_ok=True)
    dist.barrier()
    for epoch in range(first, a.epochs + 1):
        if a.stop_at_unix is not None:
            stop = torch.tensor(rank == 0 and time.time() + last_epoch_seconds >= a.stop_at_unix,
                                device=device, dtype=torch.uint8)
            dist.broadcast(stop, src=0)
            if stop.item():
                if rank == 0:
                    write(root/"paused.json", dict(reason="wall_time_budget", epoch=epoch-1,
                                                   updates=updates, checkpoint=str(last),
                                                   stopped_unix=time.time(), budget_end_unix=a.stop_at_unix))
                dist.destroy_process_group()
                return
        started = time.monotonic()
        if epoch:
            model.train()
            order = np.random.default_rng(seed+epoch).permutation(train)
            if excluded_train:
                order = order[~np.isin(order, sorted(excluded_train))]
            stats = []
            for i, start in enumerate(range(0, len(order)-cfg["global_batch"]+1, cfg["global_batch"]), 1):
                indices = balanced_ranks(order[start:start+cfg["global_batch"]], costs, world)[rank]
                opt.zero_grad(set_to_none=True)
                factor = learning_rate_factor(updates, steps_per_epoch, base_epochs, extension)
                for g, peak in zip(opt.param_groups, peaks):
                    g["lr"] = peak * factor
                began = time.monotonic()
                scores = step(model, rows, indices, points, grouping, device, 1000+epoch,
                              cfg["weights"], rank, opt, cfg["micro"], negative=negative)
                updates += 1
                stats.append(scores)
                if rank == 0:
                    row = dict(epoch=epoch, batch=i, updates=updates, lr=[x["lr"] for x in opt.param_groups],
                               step_seconds=time.monotonic()-began, time=time.time(), **scores)
                    core.append(root/"steps.jsonl", row)
                    if i == 1 or i % 10 == 0:
                        print(json.dumps(row), flush=True)
        model.eval()
        val_stats = [step(model, rows, balanced_ranks(ids, costs, world)[rank], points, grouping,
                          device, 6000, cfg["weights"], rank, None, cfg["micro"], negative=negative) for ids in val]
        core.monitor(model, rows, monitor, points, grouping, device, rank, world, epoch, "refine")
        rng = [None] * world
        dist.all_gather_object(rng, rng_state(device))
        if rank == 0:
            def avg(seq):
                return {k: float(np.mean([r[k] for r in seq])) for k in seq[0]}
            row = dict(epoch=epoch, updates=updates, validation=avg(val_stats), seconds=time.monotonic()-started)
            if epoch:
                row["train"] = avg(stats)
            path = root/"checkpoints"/f"refine_epoch{epoch:03d}.pt"
            if path.exists():
                raise FileExistsError(path)
            tmp = path.with_suffix(".tmp")
            torch.save(dict(stage="refine", epoch=epoch, step=updates, world_size=world, config=cfg,
                            settings=cfg, model=model.state_dict(), optimizer=opt.state_dict(),
                            rng_per_rank=rng, epoch_summary=row, schedule_extension=extension), tmp)
            tmp.replace(path)
            link = root/"last.new"
            link.unlink(missing_ok=True)
            os.link(path, link)
            link.replace(last)
            core.append(root/"epochs.jsonl", row)
            print(json.dumps(row), flush=True)
        dist.barrier()
        last_epoch_seconds = time.monotonic() - started
    if rank == 0:
        write(root/"complete.json", dict(epochs=a.epochs, updates=updates, checkpoint=str(last)))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
