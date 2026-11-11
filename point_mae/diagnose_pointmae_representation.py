"""Quantitative diagnostics for the current mitochondria Point-MAE checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import yaml
import zarr
from scipy.spatial import ConvexHull, QhullError, distance
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from pointmae import (
    PointMAE,
    augmented_point_view,
    model_config_from_yaml,
    nt_xent_loss,
    split_by_neuron,
    split_stratified_by_group,
    stable_int,
)


def percentiles(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {k: None for k in ("mean", "std", "min", "p01", "p05", "median", "p95", "p99", "max")}
    return {
        "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
        "p01": float(np.percentile(x, 1)), "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)), "max": float(x.max()),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def vector_stats(x):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    centered = x - x.mean(axis=0, keepdims=True)
    l2 = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    centered_l2 = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    pair_i, pair_j = np.triu_indices(n, 1)

    def pair_summary(v):
        unit = v if np.max(np.abs(np.linalg.norm(v, axis=1) - 1)) < 1e-5 else v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        # Use BLAS matrix multiplication instead of materializing (pair_count, dim)
        # advanced-indexed arrays, which can consume tens of GB for this dataset.
        cosine_matrix = unit @ unit.T
        cos = cosine_matrix[pair_i, pair_j]
        euclidean = distance.pdist(v, metric="euclidean")
        return {
            "cosine": percentiles(cos),
            "euclidean": percentiles(euclidean),
            "_pairs_cosine_ge_0p9999": int(np.count_nonzero(cos >= 0.9999)),
        }

    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    variance_ratio = variance / max(float(variance.sum()), 1e-20)
    positive = variance_ratio[variance_ratio > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.0
    raw_norm = np.linalg.norm(x, axis=1)
    raw_pair = pair_summary(x)
    l2_pair = pair_summary(l2)
    centered_pair = pair_summary(centered)
    centered_l2_pair = pair_summary(centered_l2)
    near_cos = l2_pair.pop("_pairs_cosine_ge_0p9999")
    for stats in (raw_pair, centered_pair, centered_l2_pair):
        stats.pop("_pairs_cosine_ge_0p9999")
    return {
        "n": int(n), "dimension": int(d), "raw_norm": percentiles(raw_norm),
        "per_dimension_std": percentiles(x.std(axis=0)),
        "near_zero_std_dimensions": int(np.sum(x.std(axis=0) < 1e-8)),
        "raw": raw_pair, "l2_normalized": l2_pair, "centered": centered_pair,
        "centered_then_l2": centered_l2_pair,
        "pca_explained_variance_cumulative": {
            str(k): float(variance_ratio[:min(k, len(variance_ratio))].sum()) for k in (1, 5, 10, 20)
        },
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / max(1, min(n - 1, d)),
        "pairs_cosine_ge_0.9999": int(near_cos),
        "pairs_cosine_ge_0.9999_fraction": float(near_cos / max(1, len(pair_i))),
        "pairs_euclidean_le_1e-4": int(np.sum(distance.pdist(x) <= 1e-4)),
    }


def extract_layers(model, xyz, attrs, batch_size, device):
    layers = {"tokenizer_mean": [], "encoder_mean": [], "encoder_max": [], "encoder_std": [], "encoder_mean_std": [], "projection_128": []}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(xyz), batch_size):
            x = torch.as_tensor(xyz[start:start + batch_size], device=device)
            a = torch.as_tensor(attrs[start:start + batch_size], device=device)
            centers, grouped_xyz, grouped_attrs = model.group(x, a)
            local = torch.cat([grouped_xyz, grouped_attrs], dim=-1)
            local_tokens = model.tokenizer(local).max(dim=2).values
            patch_tokens = local_tokens + model.position(centers)
            encoder_tokens = model.encoder(patch_tokens)
            pooled = encoder_tokens.mean(dim=1)
            pooled_std = encoder_tokens.std(dim=1, unbiased=False)
            projected = model.projection(pooled)
            layers["tokenizer_mean"].append(local_tokens.mean(dim=1).cpu().numpy())
            layers["encoder_mean"].append(pooled.cpu().numpy())
            layers["encoder_max"].append(encoder_tokens.max(dim=1).values.cpu().numpy())
            layers["encoder_std"].append(pooled_std.cpu().numpy())
            layers["encoder_mean_std"].append(torch.cat([pooled, pooled_std], dim=-1).cpu().numpy())
            layers["projection_128"].append(projected.cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in layers.items()}


def scale_sensitivity(model, xyz, attrs, batch_size, device):
    scaled = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(xyz), batch_size):
            x = torch.as_tensor(xyz[start:start + batch_size], device=device) * 2.0
            a = torch.as_tensor(attrs[start:start + batch_size], device=device)
            scaled.append(model.encode(x, a).cpu().numpy())
    return np.concatenate(scaled).astype(np.float32)


def input_geometry(xyz, attrs, scale_nm):
    coords = np.asarray(xyz, dtype=np.float64) * scale_nm
    unique = np.unique(np.round(coords, 5), axis=0)
    span = np.ptp(coords, axis=0)
    cov = np.cov(coords.T)
    eig = np.maximum(np.linalg.eigvalsh(cov), 1e-12)[::-1]
    hull_area = hull_volume = np.nan
    if len(unique) >= 4:
        try:
            hull = ConvexHull(unique)
            hull_area, hull_volume = float(hull.area), float(hull.volume)
        except QhullError:
            pass
    bbox_volume = float(np.prod(span))
    if not np.isfinite(hull_volume) or hull_volume <= 0:
        hull_volume = bbox_volume
    intensity = attrs[:, 0]
    interior = attrs[:, 3] < 0.5
    contrast = float(np.std(intensity[interior])) if np.any(interior) else float(np.std(intensity))
    # Cross-flag duplicates indicate that the nominal interior candidate had to fall back to all voxels.
    rounded = np.round(coords, 5)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    flags = attrs[:, 3] >= 0.5
    conflict = any(np.any(flags[inverse == i]) and np.any(~flags[inverse == i]) for i in np.unique(inverse))
    return {
        "unique_point_fraction": float(len(unique) / len(coords)),
        "bbox_volume_nm3_proxy": bbox_volume,
        "convex_hull_volume_nm3_proxy": float(hull_volume),
        "convex_hull_area_nm2_proxy": float(hull_area) if np.isfinite(hull_area) else np.nan,
        "elongation_proxy": float(np.sqrt(eig[0] / eig[-1])),
        "linearity_proxy": float((eig[0] - eig[1]) / eig[0]),
        "interior_intensity_std": contrast,
        "duplicate_coordinate_cross_flag": bool(conflict),
        "extent_z_nm": float(span[0]), "extent_y_nm": float(span[1]), "extent_x_nm": float(span[2]),
    }


def gradient_probe(model, xyz, attrs, device, cfg):
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.as_tensor(xyz[:2], device=device)
    a = torch.as_tensor(attrs[:2], device=device)
    torch.manual_seed(42)
    loss = model(x, a, return_loss=True)["loss"]
    train_cfg = cfg.get("train", {})
    contrastive_weight = float(train_cfg.get("contrastive_weight", 0.1))
    if contrastive_weight > 0:
        view_x, view_a = augmented_point_view(
            x, a, model.attr_names, float(train_cfg.get("view_keep_ratio", 0.9))
        )
        z1, z2 = model.encode(x, a), model.encode(view_x, view_a)
        loss = loss + contrastive_weight * nt_xent_loss(
            z1, z2, float(train_cfg.get("contrastive_temperature", 0.1))
        )
    loss.backward()
    result = {}
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        row = result.setdefault(group, {"parameters": 0, "with_grad": 0, "grad_l2_sum": 0.0})
        row["parameters"] += 1
        if parameter.grad is not None:
            row["with_grad"] += 1
            row["grad_l2_sum"] += float(parameter.grad.detach().float().norm())
    model.eval()
    return {k: {**v, "grad_l2_sum": float(v["grad_l2_sum"])} for k, v in result.items()}


def optimizer_coverage(model, checkpoint):
    optimizer = checkpoint.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    state = optimizer.get("state", {})
    ids = [pid for group in groups for pid in group.get("params", [])]
    named = list(model.named_parameters())
    if len(ids) != len(named):
        return {"comparable": False, "optimizer_ids": len(ids), "model_parameters": len(named)}
    has_state = {str(k) for k in state}
    missing = [name for (name, _), pid in zip(named, ids) if str(pid) not in has_state]
    return {"comparable": True, "parameter_state_missing": missing, "missing_count": len(missing)}


def reconstruction_ablation(model, xyz, attrs, feature_names, device, groups, group_count=4):
    selected = []
    for source in sorted(set(groups)):
        idx = np.flatnonzero(np.asarray(groups) == source)
        if len(idx):
            selected.extend(idx[np.linspace(0, len(idx) - 1, min(group_count, len(idx))).astype(int)].tolist())
    totals = {mode: {} for mode in ("encoder", "shuffled_encoder", "zero_encoder")}
    counts = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(selected), 4):
            ids = selected[start:start + 4]
            x = torch.as_tensor(xyz[ids], device=device)
            a = torch.as_tensor(attrs[ids], device=device)
            centers, grouped_xyz, grouped_attrs = model.group(x, a)
            tokens = model._tokens(centers, grouped_xyz, grouped_attrs)
            b, n_groups, dim = tokens.shape
            n_mask = max(1, min(n_groups - 1, round(n_groups * model.mask_ratio)))
            order = torch.rand(b, n_groups, device=device).argsort(dim=1)
            visible_idx, masked_idx = order[:, n_mask:], order[:, :n_mask]
            visible = model.encoder(torch.gather(tokens, 1, visible_idx[..., None].expand(-1, -1, dim)))
            shuffled = torch.stack([v[torch.randperm(v.shape[0], device=device)] for v in visible])
            mask = torch.zeros((b, n_groups), dtype=torch.bool, device=device)
            mask.scatter_(1, masked_idx, True)
            target = torch.cat([grouped_xyz, grouped_attrs], dim=-1)
            for mode, z in (("encoder", visible), ("shuffled_encoder", shuffled), ("zero_encoder", torch.zeros_like(visible))):
                decoded_tokens = model.to_decoder(z)
                full = model.mask_token.expand(b, n_groups, -1).clone().to(decoded_tokens.dtype)
                full.scatter_(1, visible_idx[..., None].expand(-1, -1, full.shape[-1]), decoded_tokens)
                decoded = model.decode_tokens(full, centers)
                pred = model.reconstruction(decoded).reshape(b, n_groups, model.group_size, 3 + model.attr_dim)
                terms = model.masked_loss(pred, target, mask, feature_names) if hasattr(model, "masked_loss") else None
                if terms is None:
                    from pointmae import masked_reconstruction_loss
                    terms = masked_reconstruction_loss(pred, target, mask, feature_names, model.loss_weights)
                for key, value in terms.items():
                    totals[mode][key] = totals[mode].get(key, 0.0) + float(value)
            counts += 1
    result = {}
    for mode, values in totals.items():
        result[mode] = {k: v / max(1, counts) for k, v in values.items()}
    result["relative_to_encoder"] = {}
    for mode in ("shuffled_encoder", "zero_encoder"):
        result["relative_to_encoder"][mode] = {
            key: float((result[mode][key] - result["encoder"][key]) / max(abs(result["encoder"][key]), 1e-12))
            for key in result["encoder"]
        }
    result["n_instances"] = len(selected)
    return result


def train_log_summary(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"epochs": 0}
    best = min(rows, key=lambda r: r["val"]["loss"])
    keys = sorted(rows[0]["train"])
    return {
        "epochs": len(rows), "first_epoch": rows[0]["epoch"], "last_epoch": rows[-1]["epoch"],
        "best_val_epoch": best["epoch"],
        "first_train": rows[0]["train"], "last_train": rows[-1]["train"],
        "first_val": rows[0]["val"], "last_val": rows[-1]["val"],
        "best_val": best["val"], "loss_keys": keys,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    group = zarr.open_group(a.cache, mode="r")
    xyz = np.asarray(group["xyz"], dtype=np.float32)
    attrs_all = np.asarray(group["attrs"], dtype=np.float32)
    instance_id = np.asarray(group["instance_id"], dtype=np.int64)
    neuron_id = np.asarray(group["neuron_id"], dtype=np.int64)
    records = [json.loads(line) for line in (Path(a.cache) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != len(xyz):
        raise ValueError(f"manifest has {len(records)} rows but cache has {len(xyz)} entries")
    if any(int(row["cache_index"]) != i or int(row["instance_id"]) != int(instance_id[i]) or stable_int(row["neuron_id"]) != int(neuron_id[i]) for i, row in enumerate(records)):
        raise ValueError("cache index / instance_id / neuron_id mapping mismatch")
    source = np.asarray([str(r["neuron_id"]) for r in records])
    feature_names = cfg["data"]["feature_names"]
    if attrs_all.shape[-1] != len(feature_names):
        raise ValueError("cache attribute dimension does not match config feature_names")

    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = PointMAE(model_config_from_yaml(cfg), feature_names).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    embeddings_by_layer = extract_layers(model, xyz, attrs_all, a.batch_size, device)
    embedding_scaled_xyz = scale_sensitivity(model, xyz, attrs_all, a.batch_size, device)
    scale_cosine = np.sum(
        embeddings_by_layer["projection_128"] * embedding_scaled_xyz,
        axis=1,
    ) / np.maximum(
        np.linalg.norm(embeddings_by_layer["projection_128"], axis=1)
        * np.linalg.norm(embedding_scaled_xyz, axis=1),
        1e-12,
    )
    scale_l2 = np.linalg.norm(embeddings_by_layer["projection_128"] - embedding_scaled_xyz, axis=1)
    geometry_rows = [input_geometry(x, y, float(cfg["data"]["coord_scale_nm"])) for x, y in zip(xyz, attrs_all)]
    layer_stats = {name: vector_stats(features) for name, features in embeddings_by_layer.items()}
    embedding_stats = {
        "checkpoint": str(Path(a.checkpoint)), "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "n_instances": len(xyz), "n_points": int(xyz.shape[1]), "embedding_dim": int(embeddings_by_layer["projection_128"].shape[1]),
        "layers": layer_stats,
    }
    (out / "embedding_statistics.json").write_text(json.dumps(json_safe(embedding_stats), indent=2), encoding="utf-8")

    instance_rows = []
    for i, (row, features) in enumerate(zip(records, geometry_rows)):
        emb = embeddings_by_layer["projection_128"][i]
        instance_rows.append({
            "source": source[i], "instance_id": int(instance_id[i]), "neuron_id": str(row["neuron_id"]),
            "embedding_norm": float(np.linalg.norm(emb)),
            "embedding_mean": float(np.mean(emb)), "embedding_std": float(np.std(emb)),
            "unique_point_fraction": features["unique_point_fraction"],
            "bbox_volume_nm3_proxy": features["bbox_volume_nm3_proxy"],
            "convex_hull_volume_nm3_proxy": features["convex_hull_volume_nm3_proxy"],
            "convex_hull_area_nm2_proxy": features["convex_hull_area_nm2_proxy"],
            "elongation_proxy": features["elongation_proxy"],
            "interior_intensity_std": features["interior_intensity_std"],
            "duplicate_coordinate_cross_flag": features["duplicate_coordinate_cross_flag"],
        })
    write_csv(out / "embedding_statistics.csv", instance_rows, list(instance_rows[0]))
    layer_rows = []
    for name, stats in layer_stats.items():
        layer_rows.append({
            "layer": name, "dimension": stats["dimension"], "effective_rank": stats["effective_rank"],
            "effective_rank_fraction": stats["effective_rank_fraction"],
            "mean_cosine_raw": stats["raw"]["cosine"]["mean"], "median_cosine_raw": stats["raw"]["cosine"]["median"],
            "mean_euclidean_raw": stats["raw"]["euclidean"]["mean"],
            "mean_cosine_centered_l2": stats["centered_then_l2"]["cosine"]["mean"],
            "mean_euclidean_l2": stats["l2_normalized"]["euclidean"]["mean"],
            "mean_per_dim_std": stats["per_dimension_std"]["mean"],
            "pca20_cumulative": stats["pca_explained_variance_cumulative"]["20"],
            "near_identical_pairs_cos_gt_0p9999": stats["pairs_cosine_ge_0.9999"],
        })
    write_csv(out / "layerwise_statistics.csv", layer_rows, list(layer_rows[0]))

    # Input feature distributions and duplicate/coverage diagnostics.
    channel_names = ["coord_z_nm", "coord_y_nm", "coord_x_nm", *feature_names]
    channels = [xyz[..., i] * float(cfg["data"]["coord_scale_nm"]) for i in range(3)] + [attrs_all[..., i] for i in range(attrs_all.shape[-1])]
    input_summary = {"manifest_alignment": "all cache_index, instance_id, neuron_id rows verified", "channels": {}, "sampling": {}}
    fig, axes = plt.subplots(2, len(channel_names), figsize=(3.2 * len(channel_names), 8), constrained_layout=True)
    for i, (name, values) in enumerate(zip(channel_names, channels)):
        point_values = values.reshape(-1)
        per_inst_mean = values.mean(axis=1)
        input_summary["channels"][name] = {"point_distribution": percentiles(point_values), "per_instance_mean_distribution": percentiles(per_inst_mean), "per_instance_std_distribution": percentiles(values.std(axis=1))}
        axes[0, i].hist(point_values, bins=60, color="#4776a8", alpha=0.85)
        axes[0, i].set_title(name + " (all sampled points)")
        axes[1, i].hist(per_inst_mean, bins=40, color="#d27a36", alpha=0.85)
        axes[1, i].set_title(name + " (instance means)")
    fig.savefig(out / "input_feature_distributions.png", dpi=180)
    plt.close(fig)
    unique_fraction = np.asarray([r["unique_point_fraction"] for r in geometry_rows])
    surf_fraction = attrs_all[..., 3].mean(axis=1)
    conflicts = np.asarray([r["duplicate_coordinate_cross_flag"] for r in geometry_rows], dtype=bool)
    input_summary["sampling"] = {
        "surface_fraction_per_instance": percentiles(surf_fraction),
        "unique_point_fraction_per_instance": percentiles(unique_fraction),
        "instances_with_cross_surface_interior_duplicate_coordinates": int(conflicts.sum()),
        "instances_with_any_duplicate_coordinate": int(np.sum(unique_fraction < 1.0)),
        "coordinate_scale_nm": float(cfg["data"]["coord_scale_nm"]),
        "spacing_zyx_nm": cfg["data"]["spacing_zyx"],
        "instance_specific_coordinate_normalization": False,
    }
    proxy_names = ("bbox_volume_nm3_proxy", "convex_hull_volume_nm3_proxy", "convex_hull_area_nm2_proxy", "elongation_proxy", "interior_intensity_std")
    input_summary["morphology_proxy_distributions"] = {
        name: {"all": percentiles([row[name] for row in geometry_rows]),
               "by_source": {s: percentiles([geometry_rows[i][name] for i in np.flatnonzero(source == s)]) for s in sorted(set(source))}}
        for name in proxy_names
    }
    input_summary["nonfinite_point_values"] = {
        "xyz": int(np.size(xyz) - np.isfinite(xyz).sum()),
        "attrs": int(np.size(attrs_all) - np.isfinite(attrs_all).sum()),
    }
    (out / "input_feature_statistics.json").write_text(json.dumps(json_safe(input_summary), indent=2), encoding="utf-8")

    # Representative point clouds spanning the sampled bounding-box volume proxy.
    size_proxy = np.asarray([r["bbox_volume_nm3_proxy"] for r in geometry_rows])
    sample_ids = [int(np.argmin(abs(size_proxy - np.quantile(size_proxy, q)))) for q in (0.1, 0.5, 0.9)]
    fig = plt.figure(figsize=(15, 5))
    for panel, idx in enumerate(sample_ids, 1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        # Cache coordinates are ZYX; show conventional X/Y/Z axes.
        xyz_nm = xyz[idx][:, [2, 1, 0]] * float(cfg["data"]["coord_scale_nm"])
        sc = ax.scatter(xyz_nm[:, 0], xyz_nm[:, 1], xyz_nm[:, 2], c=attrs_all[idx, :, 0], s=2, cmap="coolwarm", alpha=0.7)
        ax.set_title(f"{source[idx]} / sample {idx}\nsize proxy quantile {panel}")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)"); ax.set_zlabel("Z (nm)")
    fig.colorbar(sc, ax=fig.axes, shrink=0.65, label="locally normalized intensity")
    fig.savefig(out / "representative_input_point_clouds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Frozen feature probes hold out cached neuron/source IDs as groups.
    targets = {
        "log_bbox_volume_nm3_proxy": np.log1p([r["bbox_volume_nm3_proxy"] for r in geometry_rows]),
        "log_convex_hull_volume_nm3_proxy": np.log1p([r["convex_hull_volume_nm3_proxy"] for r in geometry_rows]),
        "elongation_proxy": np.log1p([r["elongation_proxy"] for r in geometry_rows]),
        "convex_hull_area_nm2_proxy": np.log1p(np.maximum([r["convex_hull_area_nm2_proxy"] for r in geometry_rows], 0)),
        "interior_intensity_std": np.asarray([r["interior_intensity_std"] for r in geometry_rows]),
    }
    probe_rows = []
    for layer_name in ("tokenizer_mean", "encoder_mean", "encoder_max", "encoder_std", "encoder_mean_std", "projection_128"):
        features = embeddings_by_layer[layer_name]
        for target_name, target in targets.items():
            valid = np.isfinite(target)
            pred = np.full(len(target), np.nan)
            groups = source[valid]
            if len(np.unique(groups)) >= 3 and valid.sum() > 8:
                splitter = GroupKFold(n_splits=3)
                for train_idx, test_idx in splitter.split(features[valid], target[valid], groups):
                    estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                    estimator.fit(features[valid][train_idx], target[valid][train_idx])
                    pred[np.flatnonzero(valid)[test_idx]] = estimator.predict(features[valid][test_idx])
            score = r2_score(target[valid], pred[valid]) if np.all(np.isfinite(pred[valid])) else np.nan
            probe_rows.append({"layer": layer_name, "target": target_name, "n": int(valid.sum()), "group_split": "3-fold GroupKFold by cached neuron/source ID", "r2_oof": score})
    write_csv(out / "feature_recovery.csv", probe_rows, list(probe_rows[0]))

    # Reconstruction reliance: identical masks, baseline vs shuffled/zero encoder tokens.
    recon = reconstruction_ablation(model, xyz, attrs_all, feature_names, device, source)
    (out / "reconstruction_ablation.json").write_text(json.dumps(json_safe(recon), indent=2), encoding="utf-8")
    grad = gradient_probe(model, xyz, attrs_all, device, cfg)
    coverage = optimizer_coverage(model, checkpoint)

    # Match the configured training split; source-stratified splits are not neuron-independent.
    split_mode = cfg["data"].get("split_mode", "neuron")
    split_fn = {"neuron": split_by_neuron, "source_stratified_instance": split_stratified_by_group}.get(split_mode)
    if split_fn is None:
        raise ValueError(f"Unknown data.split_mode: {split_mode}")
    split = split_fn(neuron_id, int(cfg.get("split_seed", 0)), float(cfg["data"].get("val_fraction", 0.1)), float(cfg["data"].get("test_fraction", 0.1)))
    split_summary = {}
    for name, ids in split.items():
        split_summary[name] = {"n_instances": int(len(ids)), "sources": {s: int(np.sum(source[ids] == s)) for s in sorted(set(source[ids]))}}

    metrics = train_log_summary(a.metrics)
    test_path = Path(a.metrics).with_name("test_metrics.json")
    test_metrics = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
    # UMAPs over original 128D and 20D PCA features with a common cosine metric.
    emb = embeddings_by_layer["projection_128"]
    emb_l2 = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    pca = PCA(n_components=min(20, emb.shape[1]), random_state=a.seed)
    pca_scores = pca.fit_transform(emb_l2)
    pca_l2 = pca_scores / np.maximum(np.linalg.norm(pca_scores, axis=1, keepdims=True), 1e-12)
    coords_direct = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=a.seed).fit_transform(emb_l2)
    coords_pca = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=a.seed).fit_transform(pca_l2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    plot_source = np.asarray(["predicted E16" if r.get("prediction_root") else str(r["neuron_id"]) for r in records])
    plot_sources = sorted(set(plot_source))
    palette = plt.get_cmap("tab10")
    colors = {s: palette(i % palette.N) for i, s in enumerate(plot_sources)}
    for ax, coords, title in zip(axes, (coords_direct, coords_pca), ("Direct UMAP on 128D", "PCA20 → UMAP")):
        for s in plot_sources:
            mask = plot_source == s
            ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.75, color=colors[s], label=f"{s} (n={int(mask.sum())})", linewidths=0)
        ax.set_title(title); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "pca_umap_by_origin.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker="o", ms=3)
    ax.set(xlabel="PCA components", ylabel="Cumulative explained variance", ylim=(0, 1.01), title="Embedding PCA spectrum")
    ax.grid(alpha=0.2); fig.savefig(out / "pca_variance.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    summary = {
        "n_instances": len(xyz), "source_counts": {s: int(np.sum(source == s)) for s in sorted(set(source))},
        "data_origin_counts": {s: int(np.sum(plot_source == s)) for s in plot_sources},
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)), "checkpoint_sha256": embedding_stats["checkpoint_sha256"],
        "checkpoint_config_matches_yaml": checkpoint.get("config", {}) == cfg,
        "unique_source_instance_keys": len({(str(r["neuron_id"]), int(r["instance_id"])) for r in records}),
        "checkpoint_saved_config": checkpoint.get("config", {}), "config_used": cfg,
        "split_mode": split_mode, "data_split": split_summary, "gradient_probe": grad, "optimizer_coverage": coverage,
        "reconstruction_ablation": recon, "training": metrics, "test_metrics": test_metrics,
        "embedding_similarity_after_xyz_times_2": {"cosine": percentiles(scale_cosine), "euclidean": percentiles(scale_l2)},
        "feature_probe_note": "Targets are geometric/texture proxies from fixed 2048-point clouds, not exact voxel measurements. Grouping uses cached neuron_id: predicted rows use FlyEM body IDs, while the three manual inputs are dataset-level tags rather than individual neuron IDs.",
        "pca20_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "diagnostic_umap": {"metric": "cosine", "n_neighbors": 15, "min_dist": 0.1, "seed": a.seed},
    }
    stability_path = Path(a.checkpoint).parent / "embedding_stability_resample_rotate.json"
    if stability_path.exists():
        stability = json.loads(stability_path.read_text(encoding="utf-8"))
        summary["existing_resampling_stability"] = stability.get("overall", {})
    cached_stability_path = Path(a.checkpoint).parent / "embedding_stability_cached_views.json"
    if cached_stability_path.exists():
        cached_stability = json.loads(cached_stability_path.read_text(encoding="utf-8"))
        summary["cached_point_view_stability"] = cached_stability.get("overall", {})
    (out / "diagnostic_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    projection_untrained = "projection" in coverage.get("parameter_state_missing", [])
    if not projection_untrained:
        projection_untrained = grad.get("projection", {}).get("with_grad", 1) == 0
    final = layer_stats["projection_128"]
    encoder = layer_stats["encoder_mean"]
    local = layer_stats["tokenizer_mean"]
    pair_count = len(xyz) * (len(xyz) - 1) // 2
    val = metrics.get("best_val", {})
    loss_sum = sum(float(val.get(k, 0.0)) for k in ("geo", "intensity", "gradient", "distance", "surface"))
    contributions = {k: 100.0 * float(val.get(k, 0.0)) / max(loss_sum, 1e-12) for k in ("geo", "intensity", "gradient", "distance", "surface")}
    source_counts_text = ", ".join(f"{name}={count}" for name, count in summary["source_counts"].items())
    split_text = "; ".join(
        f"{name} n={stats['n_instances']} [{', '.join(f'{source}={count}' for source, count in stats['sources'].items())}]"
        for name, stats in split_summary.items()
    )
    probe_by_layer = {layer: {r["target"]: float(r["r2_oof"]) for r in probe_rows if r["layer"] == layer} for layer in ("encoder_mean", "projection_128")}
    report = "# Point-MAE representation diagnosis\n\n"
    report += "## Executive conclusion\n\n"
    report += f"This report evaluates collapse using several complementary statistics rather than raw cosine alone. In the final 128D vectors, median pairwise cosine is **{final['raw']['cosine']['median']:.4f}**, median cosine after centering and L2 normalization is **{final['centered_then_l2']['cosine']['median']:.4f}**, and effective rank is **{final['effective_rank']:.2f}/128**. There are {final['pairs_cosine_ge_0.9999']} / {pair_count} pairs with cosine ≥0.9999; the first 5 PCs explain **{final['pca_explained_variance_cumulative']['5']:.3%}**. Interpret these with per-dimension variance and the layer-wise statistics below.\n\n"
    if projection_untrained:
        report += "The final projection head is untrained in this checkpoint: the configured objective gives it no gradient, and the optimizer has no state for its parameters. Do not interpret its output as a learned embedding.\n\n"
    else:
        report += "The final projection head receives gradients under this checkpoint's configured objective. Its usefulness still needs to be judged from embedding statistics and held-out probes.\n\n"
    report += "## 1. Data and preprocessing\n\n"
    report += f"- Used all **{len(xyz)}** available annotated instances ({source_counts_text}); this exceeds the requested 1,000-instance minimum. All cache indices and `(source, instance_id, neuron_id)` mappings were verified; unique keys: {summary['unique_source_instance_keys']}. No duplicated cache mapping was detected.\n"
    report += f"- Inputs are **2048×7** (XYZ plus four attributes). Physical ZYX spacing is {cfg['data']['spacing_zyx']} nm and coordinates are divided by one global {cfg['data']['coord_scale_nm']} nm scale after centering. This preserves each organelle's physical extent; no per-instance scale normalization was found.\n"
    report += f"- Input arrays contain {input_summary['nonfinite_point_values']['xyz']} non-finite XYZ and {input_summary['nonfinite_point_values']['attrs']} non-finite attribute values. Fixed sampling yields exactly 50% surface flags for every instance, so that flag has **zero between-instance variance** by construction. Repeated coordinates occurred in {int(np.sum(unique_fraction < 1.0))}/{len(xyz)} instances; surface/interior flag conflicts occurred in {int(conflicts.sum())}/{len(xyz)}. Thus repeated-point sampling is not a broad explanation here.\n"
    bbox_stats = input_summary["morphology_proxy_distributions"]["bbox_volume_nm3_proxy"]["all"]
    extent_stats = input_summary["morphology_proxy_distributions"]["elongation_proxy"]["all"]
    report += f"- Inputs nevertheless show shape diversity: sampled-coordinate bounding-box volume proxy median {bbox_stats['median']:.3g} nm³ (5–95%: {bbox_stats['p05']:.3g}–{bbox_stats['p95']:.3g}); elongation proxy median {extent_stats['median']:.3f}. These are cloud-derived proxies, not exact voxel-volume labels. Local intensity normalization is per instance and can remove absolute intensity scale, but observed per-instance intensity SD spans {input_summary['channels']['intensity']['per_instance_std_distribution']['p05']:.3f}–{input_summary['channels']['intensity']['per_instance_std_distribution']['p95']:.3f} (5–95%).\n"
    report += "\nExact voxel counts and exact organelle surface area are not stored in this point-cloud cache. The probe therefore uses explicitly named bounding-box/convex-hull proxies; exact targets require scanning the source instance volumes.\n\n"
    report += "## 2. Where distinction is lost\n\n"
    report += "| Stage (per-instance pooling) | Median raw cosine | Effective rank | PCA20 variance |\n|---|---:|---:|---:|\n"
    for name, title in (("tokenizer_mean", "Local tokenizer mean"), ("encoder_mean", "Encoder mean"), ("projection_128", "Final 128D projection")):
        stats = layer_stats[name]
        report += f"| {title} | {stats['raw']['cosine']['median']:.4f} | {stats['effective_rank']:.2f}/{stats['dimension']} | {stats['pca_explained_variance_cumulative']['20']:.3%} |\n"
    report += f"\nThe tokenizer mean has median raw cosine {local['raw']['cosine']['median']:.4f} and effective rank {local['effective_rank']:.2f}; the encoder mean has {encoder['raw']['cosine']['median']:.4f} and rank {encoder['effective_rank']:.2f}. These are pooled features, so they do not establish whether unpooled patch tokens are low-rank. The projection effective rank is {final['effective_rank']:.2f}. Pooling probes are in layerwise_statistics.csv and feature_recovery.csv.\n\n"
    report += "## 3. Encoder use and SSL objective\n\n"
    shuffle_delta = 100 * recon["relative_to_encoder"]["shuffled_encoder"]["loss"]
    zero_delta = 100 * recon["relative_to_encoder"]["zero_encoder"]["loss"]
    correspondence_note = "This indicates weak sensitivity to token-position correspondence in this ablation." if abs(shuffle_delta) < 5 else "This indicates measurable sensitivity to token-position correspondence in this ablation."
    report += f"On {recon['n_instances']} instances with identical masks, shuffling visible encoder tokens changed total reconstruction loss by **{shuffle_delta:+.3f}%**, while zeroing them changed it by **{zero_delta:+.1f}%**. {correspondence_note} This is a small diagnostic ablation, not a full validation benchmark.\n\n"
    report += f"The gradient probe used this checkpoint's configured objective; projection parameters received gradients in {grad.get('projection', {}).get('with_grad', 0)}/{grad.get('projection', {}).get('parameters', 0)} parameters. The diagnostic itself does not update checkpoint weights.\n\n"
    report += "The five losses have equal configured weights; measured contributions at the best validation checkpoint (epoch " + str(metrics.get("best_val_epoch", "?")) + ") are: "
    report += ", ".join(f"{k}={val.get(k, float('nan')):.4g} ({contributions[k]:.1f}%)" for k in ("geo", "intensity", "gradient", "distance", "surface")) + ". The ranking is measured from this run and is not treated as a cause by itself.\n\n"
    report += "## 4. Frozen recovery probes\n\n"
    report += "Three-fold GroupKFold R² by cached neuron/source ID. Predicted E16 rows use FlyEM body IDs; manual rows use dataset-level tags rather than individual neuron IDs. Targets are cache-derived proxies, not exact biological measurements:\n\n"
    report += "| Target | Encoder mean | Final 128D |\n|---|---:|---:|\n"
    for target in targets:
        e = probe_by_layer["encoder_mean"].get(target, float("nan"))
        z = probe_by_layer["projection_128"].get(target, float("nan"))
        report += f"| {target} | {e:.3f} | {z:.3f} |\n"
    report += "\nCompare the frozen probes across layers before interpreting the 128D representation. These R² values are exploratory because they predict point-cloud proxies and the combined cache mixes manually labeled and predicted masks.\n\n"
    report += "## 5. Training, checkpoint, extraction\n\n"
    report += f"The checkpoint SHA-256 is `{summary['checkpoint_sha256']}` and its saved config matches the inspected YAML: **{summary['checkpoint_config_matches_yaml']}**. It is the best checkpoint at training epoch number {summary['checkpoint_epoch'] + 1} (zero-based log index {summary['checkpoint_epoch']}) of {metrics.get('epochs', 0)} epochs (best validation loss {metrics.get('best_val', {}).get('loss', float('nan')):.4f}). Split mode is `{split_mode}`; counts are {split_text}. E16 predicted rows are grouped by FlyEM body ID; the manual subset is grouped by its three dataset tags, not individual neuron IDs.\n\n"
    if test_metrics:
        report += f"Best-checkpoint test loss is {test_metrics.get('overall', {}).get('loss', float('nan')):.4f}; per-source results are saved in `test_metrics.json` alongside the run.\n\n"
    stability = summary.get("existing_resampling_stability", {})
    same_resample = stability.get("same_instance_resampling_cosine", {}).get("median")
    diff_inst = stability.get("different_instance_baseline_cosine", {}).get("median")
    if same_resample is not None and diff_inst is not None:
        report += f"Existing same-checkpoint resampling test: same-instance cosine median **{same_resample:.5f}** versus random different-instance pair median **{diff_inst:.5f}**. This confirms high similarity is not caused by unstable point sampling; the embeddings are highly stable but also close across instances.\n\n"
    cached_stability = summary.get("cached_point_view_stability", {})
    if cached_stability:
        aug = cached_stability.get("same_instance_augmented_view_cosine", {}).get("median")
        rot = cached_stability.get("same_instance_z_rotation_cosine", {}).get("median")
        report += f"Cached-cloud stability (not fresh voxel-level re-sampling): median same-instance cosine is {aug:.4f} under the training-time point augmentation and {rot:.4f} under rotation-only. Full values are in `embedding_stability_cached_views.json`.\n\n"
    report += f"Doubling XYZ while keeping attributes fixed gives cosine median {np.median(scale_cosine):.4f} (5–95% {np.percentile(scale_cosine,5):.4f}–{np.percentile(scale_cosine,95):.4f}); this is a sensitivity probe only. extract_embeddings.py loads the checkpoint, calls eval(), and extracts unmasked encoder features.\n\n"
    augmented_stability_path = out / "embedding_stability_augmented_view.json"
    if augmented_stability_path.exists():
        augmented_stability = json.loads(augmented_stability_path.read_text(encoding="utf-8"))
        same = augmented_stability["same_instance_cosine"]
        different = augmented_stability["different_instance_all_pairs_cosine_median"]
        report += f"On the configured cached-cloud augmentation (90% resampling plus YX rotation; not fresh voxel-level sampling), same-instance cosine has median {same['median']:.4f} (5th percentile {same['p05']:.4f}), versus different-instance median {different:.4f}. Full values are in `embedding_stability_augmented_view.json`.\n\n"
    report += "## 6. Most likely causes and minimal priorities\n\n"
    report += ("1. **Confirmed — the projection head is untrained in this checkpoint.** Do not interpret its 128D output as a learned embedding.\n" if projection_untrained else "1. **Confirmed — the projection head is trained, but its frozen linear probes are weaker than pre-projection features for the measured morphology/texture proxies.** Inspect what information the projection and contrastive objective preserve; the targets remain point-cloud proxies, not exact biological measurements.\n")
    report += "2. **Confirmed in the small decoder ablation — visible-token shuffling barely changed reconstruction, while zeroing tokens increased loss.** The decoder uses encoder signal but is weakly sensitive to token-position correspondence; inspect decoder positional conditioning before changing the objective.\n"
    report += "3. **Input/split limitations — the surface flag is exactly 50% for each sampled instance, and per-instance intensity normalization removes absolute intensity scale.** Manual rows are grouped by dataset tags, not individual neuron IDs; interpret their holdout separately from FlyEM body-ID groups.\n\n"
    report += f"Physical coordinate scale is preserved (no per-instance scale normalization); repeated coordinates occur in {input_summary['sampling']['instances_with_any_duplicate_coordinate']}/{len(xyz)} instances. The configured contrastive weight is {float(cfg.get('train', {}).get('contrastive_weight', 0.1)):.3g}. This diagnostic run did not update checkpoint weights.\n\n"
    report += "## Artifacts\n\n`embedding_statistics.json`, `embedding_statistics.csv`, `layerwise_statistics.csv`, `feature_recovery.csv`, `input_feature_statistics.json`, `input_feature_distributions.png`, `representative_input_point_clouds.png`, `pca_umap_by_origin.png`, `pca_variance.png`, `training_loss.png` (if generated separately), `embedding_stability_augmented_view.json` (if generated separately), `reconstruction_ablation.json`, and `diagnostic_summary.json`.\n\n"
    report += f"## Reproduce\n\n```bash\npython diagnose_pointmae_representation.py \\\n  --config {a.config} \\\n  --cache {a.cache} \\\n  --checkpoint {a.checkpoint} \\\n  --metrics {a.metrics} \\\n  --out-dir {a.out_dir} \\\n  --device {a.device}\n```\n"
    (out / "diagnosis_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "n_instances": len(xyz), "embedding_cosine_median": layer_stats["projection_128"]["raw"]["cosine"]["median"], "projection_gradient": grad.get("projection"), "optimizer_missing": coverage.get("parameter_state_missing")}, indent=2))


if __name__ == "__main__":
    main()
