"""Single-GPU masked point modeling trainer for mitochondria caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from torch.utils.data import DataLoader

from pointmae import (
    PointCloudCache,
    PointMAE,
    augmented_point_view,
    count_parameters,
    model_config_from_yaml,
    nt_xent_loss,
    seed_everything,
    split_by_neuron,
    split_stratified_by_group,
)


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    amp: bool,
    train: bool,
    grad_clip: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    view_keep_ratio: float,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for batch in loader:
            xyz = batch["xyz"].to(device, non_blocking=True)
            attrs = batch["attrs"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                result = model(xyz, attrs, return_loss=True)
                if contrastive_weight > 0:
                    view_xyz, view_attrs = augmented_point_view(xyz, attrs, model.attr_names, view_keep_ratio)
                    z1 = model.encode(xyz, attrs)
                    z2 = model.encode(view_xyz, view_attrs)
                    contrastive = nt_xent_loss(z1, z2, contrastive_temperature)
                else:
                    contrastive = result["loss"].new_zeros(())
                loss = result["loss"] + contrastive_weight * contrastive
            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            bs = xyz.shape[0]
            count += bs
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * bs
            totals["reconstruction_loss"] = totals.get("reconstruction_loss", 0.0) + float(result["loss"].detach()) * bs
            totals["contrastive_loss"] = totals.get("contrastive_loss", 0.0) + float(contrastive.detach()) * bs
            for name, value in result.items():
                if name == "loss":
                    continue
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * bs
    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed_everything(int(cfg.get("seed", 0)))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    features = cfg["data"]["feature_names"]
    group = zarr.open_group(args.cache, mode="r")
    group_ids = np.asarray(group["neuron_id"])
    split_mode = cfg["data"].get("split_mode", "neuron")
    split_fn = {"neuron": split_by_neuron, "source_stratified_instance": split_stratified_by_group}.get(split_mode)
    if split_fn is None:
        raise ValueError(f"Unknown data.split_mode: {split_mode}")
    split = split_fn(group_ids, int(cfg.get("split_seed", 0)), float(cfg["data"].get("val_fraction", 0.1)), float(cfg["data"].get("test_fraction", 0.1)))
    train_ds = PointCloudCache(args.cache, split["train"], features)
    val_ds = PointCloudCache(args.cache, split["val"], features)
    test_ds = PointCloudCache(args.cache, split["test"], features)
    train_cfg = cfg["train"]
    kwargs = {"batch_size": int(train_cfg["batch_size"]), "num_workers": int(train_cfg.get("num_workers", 0)), "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **kwargs)

    model_cfg = model_config_from_yaml(cfg)
    model = PointMAE(model_cfg, features).to(device)
    contrastive_weight = float(train_cfg.get("contrastive_weight", 0.1))
    contrastive_temperature = float(train_cfg.get("contrastive_temperature", 0.1))
    view_keep_ratio = float(train_cfg.get("view_keep_ratio", 0.9))
    total, trainable = count_parameters(model)
    print(f"device={device}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.05)))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    source_names = {}
    records_path = Path(args.cache) / "records.jsonl"
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            source_names[int(group_ids[int(record["cache_index"])])] = str(record["neuron_id"])
    split_summary = {
        name: {
            "n_instances": int(len(indices)),
            "sources": {
                source_names.get(int(group_id), str(int(group_id))): int(np.sum(group_ids[indices] == group_id))
                for group_id in np.unique(group_ids[indices])
            },
        }
        for name, indices in split.items()
    }
    (out / "split.json").write_text(json.dumps({"mode": split_mode, "splits": split_summary}, indent=2), encoding="utf-8")
    print(json.dumps({"split_mode": split_mode, "splits": split_summary}, sort_keys=True), flush=True)
    start_epoch = 0
    best = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        saved_train = state.get("config", {}).get("train", {})
        saved_objective = (
            float(saved_train.get("contrastive_weight", 0.0)),
            float(saved_train.get("contrastive_temperature", 0.1)),
            float(saved_train.get("view_keep_ratio", 1.0)),
        )
        current_objective = (contrastive_weight, contrastive_temperature, view_keep_ratio)
        best = float(state.get("best", best)) if saved_objective == current_objective else float("inf")

    epochs = int(args.epochs or train_cfg["epochs"])
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(start_epoch, epochs):
            train_stats = run_epoch(model, train_loader, optimizer, scaler, device, amp, True, float(train_cfg.get("grad_clip", 1.0)), contrastive_weight, contrastive_temperature, view_keep_ratio)
            val_stats = run_epoch(model, val_loader, optimizer, scaler, device, amp, False, 0.0, contrastive_weight, contrastive_temperature, view_keep_ratio)
            row = {"epoch": epoch, "train": train_stats, "val": val_stats}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "best": best, "config": cfg}
            torch.save(state, out / "last.pt")
            if val_stats["loss"] < best:
                best = val_stats["loss"]
                state["best"] = best
                torch.save(state, out / "best.pt")

    best_state = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    test_stats = run_epoch(model, test_loader, optimizer, scaler, device, amp, False, 0.0, contrastive_weight, contrastive_temperature, view_keep_ratio)
    test_by_source = {}
    for group_id in np.unique(group_ids[split["test"]]):
        indices = split["test"][group_ids[split["test"]] == group_id]
        source_loader = DataLoader(PointCloudCache(args.cache, indices, features), shuffle=False, drop_last=False, **kwargs)
        name = source_names.get(int(group_id), str(int(group_id)))
        test_by_source[name] = run_epoch(model, source_loader, optimizer, scaler, device, amp, False, 0.0, contrastive_weight, contrastive_temperature, view_keep_ratio)
    (out / "test_metrics.json").write_text(json.dumps({"best_epoch": int(best_state["epoch"]), "overall": test_stats, "by_source": test_by_source}, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_stats, "test_by_source": test_by_source}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
