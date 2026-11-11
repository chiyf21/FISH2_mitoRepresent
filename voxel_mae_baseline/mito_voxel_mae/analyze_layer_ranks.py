#!/usr/bin/env python3
"""Measure instance- and token-level effective rank through VoxelMAE."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mito_voxel_mae.data import MitoDataset, collate_one, to_device
from mito_voxel_mae.model import VoxelMAE, fps_groups


def rank_stats(cov: np.ndarray) -> dict[str, float | int]:
    cov = (cov + cov.T) * 0.5
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = float(eig.sum())
    if total <= 1e-30:
        return {"effective_rank": 0.0, "participation_ratio": 0.0, "numerical_rank_1e-8": 0, "total_variance": total}
    p = eig / total
    p = p[p > 0]
    er = float(np.exp(-(p * np.log(p)).sum()))
    pr = float(total * total / np.square(eig).sum())
    return {
        "effective_rank": er,
        "participation_ratio": pr,
        "numerical_rank_1e-8": int((eig > eig.max() * 1e-8).sum()),
        "total_variance": total,
    }


def matrix_rank_stats(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    cov = np.cov(x, rowvar=False, ddof=1)
    out = rank_stats(cov)
    out["mean_norm"] = float(np.linalg.norm(x, axis=1).mean())
    out["mean_channel_std"] = float(x.std(axis=0, ddof=1).mean())
    unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    out["effective_rank_l2"] = rank_stats(np.cov(unit, rowvar=False, ddof=1))["effective_rank"]
    return out


def encode_trace(model: VoxelMAE, b: dict[str, torch.Tensor]):
    """Mirror VoxelMAE.encode while retaining the intermediate activations."""
    fmap = model.unet(torch.cat((b["image"], b["observed"], b["image_valid"]), 1))
    xyz, t, valid = b["points_r"], b["points_t"], b["points_valid"]
    grid = b["points_grid"].unsqueeze(2).unsqueeze(2)
    feat = F.grid_sample(fmap, grid, mode="bilinear", align_corners=False, padding_mode="zeros")[:, :, :, 0, 0].transpose(1, 2)

    center_idx = fps_groups(xyz, valid, model.groups)
    batch = torch.arange(xyz.shape[0], device=xyz.device)[:, None]
    centers = xyz[batch, center_idx]
    d = torch.cdist(centers.float(), xyz.float()).masked_fill(~valid[:, None], float("inf"))
    idx = d.topk(min(model.k, xyz.shape[1]), largest=False).indices
    bi = torch.arange(xyz.shape[0], device=xyz.device)[:, None, None]
    local = xyz[bi, idx] - centers[:, :, None]
    local_t = t[bi, idx] - t[batch, center_idx][:, :, None]
    patch_input = torch.cat((local, local_t, feat[bi, idx]), -1)
    point_values = model.point(patch_input)
    neighbor_valid = valid[bi, idx]
    patch_tokens = point_values.masked_fill(~neighbor_valid[..., None], float("-inf")).amax(2)
    transformer_input = patch_tokens + model.pos(centers) + model.scale(b["log_size"])[:, None]

    h = transformer_input
    layers = []
    for layer in model.encoder.layers:
        h = layer(h)
        layers.append(h)
    if model.encoder.norm is not None:
        h = model.encoder.norm(h)
        layers[-1] = h
    mean_pool = h.mean(1)
    z = model.bottleneck(mean_pool)
    return {
        "unet_support": feat[valid],
        "patch_tokens": patch_tokens,
        "transformer_input": transformer_input,
        "layers": layers,
        "mean_pool": mean_pool,
        "bottleneck": z,
    }


def add_token_moments(acc: dict, name: str, values: torch.Tensor) -> None:
    values = values.reshape(-1, values.shape[-1]).double()
    if values.shape[0] == 0:
        raise ValueError(f"No valid features for {name}")
    mean = values.mean(0)
    second = values.T @ values / values.shape[0]
    if name not in acc:
        acc[name] = {"sum_mean": torch.zeros_like(mean), "sum_second": torch.zeros_like(second), "n_instances": 0, "n_tokens": 0}
    acc[name]["sum_mean"] += mean
    acc[name]["sum_second"] += second
    acc[name]["n_instances"] += 1
    acc[name]["n_tokens"] += int(values.shape[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--saved-embeddings", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Optional small smoke test")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = VoxelMAE(**state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"])

    ds = MitoDataset(a.cache, "all", queries=1, views=1, augmentations=False, masking=False, seed=42, limit=a.limit)
    saved = np.load(a.saved_embeddings)
    n = len(ds)
    if not np.array_equal(saved["record_id"][:n], np.asarray([r["record_id"] for r in ds.records])):
        raise ValueError("Saved embedding order does not match the selected manifest records")
    vector_names = ["unet_support_mean", "patch_token_mean", "transformer_input_mean"]
    vector_names += [f"transformer_layer_{i}_mean" for i in range(1, len(model.encoder.layers) + 1)]
    vector_names += ["mean_pool", "bottleneck"]
    vectors = {name: np.empty((n, 24 if name == "unet_support_mean" else model.dim), dtype=np.float32) for name in vector_names}
    body_ids = [""] * n
    record_ids = [""] * n
    token_acc: dict = {}
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    diff_count = 0
    saved_z = saved["embedding"]

    with torch.inference_mode():
        for i in range(n):
            sample = ds[i]
            b = to_device(collate_one([sample]), device)
            out = encode_trace(model, b)
            u = out["unet_support"]
            add_token_moments(token_acc, "U-Net sampled support features", u)
            add_token_moments(token_acc, "patch tokens before position/scale", out["patch_tokens"])
            add_token_moments(token_acc, "transformer input after position/scale", out["transformer_input"])

            vectors["unet_support_mean"][i] = u.mean(0).cpu().numpy()
            vectors["patch_token_mean"][i] = out["patch_tokens"].mean(1)[0].cpu().numpy()
            vectors["transformer_input_mean"][i] = out["transformer_input"].mean(1)[0].cpu().numpy()
            for li, h in enumerate(out["layers"], 1):
                add_token_moments(token_acc, f"Transformer layer {li} tokens", h)
                vectors[f"transformer_layer_{li}_mean"][i] = h.mean(1)[0].cpu().numpy()
            vectors["mean_pool"][i] = out["mean_pool"][0].cpu().numpy()
            vectors["bottleneck"][i] = out["bottleneck"][0].cpu().numpy()

            delta = (out["bottleneck"][0] - torch.as_tensor(saved_z[i], device=device)).abs()
            max_abs_diff = max(max_abs_diff, float(delta.max()))
            sum_abs_diff += float(delta.sum())
            diff_count += delta.numel()
            body_ids[i] = str(ds.records[i]["body_id"])
            record_ids[i] = str(sample["record_id"])
            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"processed {i + 1}/{n}", flush=True)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "stage_vectors.npz", record_id=np.asarray(record_ids), body_id=np.asarray(body_ids), **vectors)

    instance_rows = []
    for name, x in vectors.items():
        instance_rows.append({"stage": name, "readout": "one vector per mitochondrion", "n_instances": len(x), "dim": x.shape[1], **matrix_rank_stats(x)})
    with (out_dir / "instance_level_rank.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(instance_rows[0]))
        w.writeheader(); w.writerows(instance_rows)

    token_rows = []
    for name, acc in token_acc.items():
        mean = acc["sum_mean"] / acc["n_instances"]
        cov = acc["sum_second"] / acc["n_instances"] - torch.outer(mean, mean)
        stats = rank_stats(cov.cpu().numpy())
        token_rows.append({"stage": name, "readout": "token/support-point population; instances equally weighted", "n_instances": acc["n_instances"], "n_tokens_total": acc["n_tokens"], "dim": int(mean.numel()), **stats})
    with (out_dir / "token_level_rank.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(token_rows[0]))
        w.writeheader(); w.writerows(token_rows)

    checks = {
        "n_instances": n,
        "checkpoint": str(Path(a.checkpoint)),
        "inference": "eval + no augmentation + masking disabled, matching the saved embedding export",
        "embedding_match_max_abs": max_abs_diff,
        "embedding_match_mean_abs": sum_abs_diff / diff_count if diff_count else None,
        "metric": "entropy effective rank = exp(Shannon entropy of centered covariance eigenvalue spectrum); participation ratio also reported",
    }
    (out_dir / "rank_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    report = [
        "# Voxel-MAE 逐阶段有效秩诊断",
        "",
        f"- 实例数：{n:,}。推理使用 eval、无增强、关闭 masking，与已导出的 embedding 模式一致。",
        "- U-Net 阶段统计的是 U-Net feature map 在 mask/halo 支持点上的采样特征（24 维）；patch token 是最多 32 个邻近支持点经 MLP + masked max 后、加入位置/尺度编码前的 256 维 token。",
        "- Transformer 输入统计包含 position 与 log-size 编码；随后报告 6 层各自 token 输出。mean_pool 是第 6 层 token 的平均，bottleneck 是 256 维投影输出。",
        "- 主表 `instance_level_rank.csv`：每个线粒体先按模型读出方式形成一个固定维向量，再跨实例计算中心化协方差有效秩；patch/各层 token 的分析读出使用对应 token 的实例内均值。此处的均值是用于秩比较的诊断 readout。",
        "- 辅表 `token_level_rank.csv`：在支持点/64 个 token 层面计算通道协方差，先对每个实例等权，再合并实例内和实例间变化；它与跨实例单向量 rank 回答的问题不同。",
        "- `mean_pool` 与 `transformer_layer_6_mean` 应数值一致；最终 bottleneck 与原有 `embeddings.npz` 的一致性误差见 `rank_checks.json`。",
        "- 有效秩采用中心化协方差谱的 entropy effective rank；同时给 participation ratio、相对阈值 1e-8 数值秩。实例向量表还给 L2 单位化后的 entropy rank 作参考。",
        "",
        "## 跨实例样本级有效秩",
        "",
        "见 `instance_level_rank.csv`。",
        "",
        "## 支持点/token 通道有效秩",
        "",
        "见 `token_level_rank.csv`。",
        "",
        f"Checkpoint：`{checks['checkpoint']}`。",
    ]
    (out_dir / "layer_rank_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
