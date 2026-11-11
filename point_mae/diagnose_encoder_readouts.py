"""Layer/readout diagnostics for the original 128-D Point-MAE checkpoint.

Read-only with respect to the training run; all new outputs go to --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pointmae import PointMAE, augmented_point_view, model_config_from_yaml


PROXIES = {
    "log_bbox_volume": ("bbox_volume_nm3_proxy", "log"),
    "log_convex_hull_volume": ("convex_hull_volume_nm3_proxy", "log"),
    "log_convex_hull_area": ("convex_hull_area_nm2_proxy", "log"),
    "log_elongation": ("elongation_proxy", "log"),
    "interior_intensity_std": ("interior_intensity_std", "identity"),
}


def effective_rank(cov):
    eigen = np.maximum(np.linalg.eigvalsh(cov), 0)
    total = eigen.sum()
    if total <= 0:
        return 0.0
    p = eigen[eigen > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def embedding_stats(features):
    x = np.asarray(features, dtype=np.float64)
    cov = np.cov(x, rowvar=False)
    return {
        "effective_rank": effective_rank(cov),
        "mean_dim_std": float(x.std(axis=0).mean()),
        "median_norm": float(np.median(np.linalg.norm(x, axis=1))),
    }


def neighbor_stability(reference, alternate, k, device):
    x = F.normalize(torch.as_tensor(reference, device=device, dtype=torch.float32), dim=1)
    y = F.normalize(torch.as_tensor(alternate, device=device, dtype=torch.float32), dim=1)
    n = len(x)
    same_index = torch.arange(n, device=device)
    within_x = x @ x.T
    within_y = y @ y.T
    within_x.fill_diagonal_(-float("inf"))
    within_y.fill_diagonal_(-float("inf"))
    top_x = within_x.topk(k, dim=1).indices
    top_y = within_y.topk(k, dim=1).indices
    overlap = (top_x[:, :, None] == top_y[:, None, :]).any(dim=2).float().mean()
    cross = x @ y.T
    retrieval = (cross.argmax(dim=1) == same_index).float().mean()
    paired_cos = cross[same_index, same_index]
    return {
        "knn_overlap_at_k": float(overlap),
        "paired_view_top1_retrieval": float(retrieval),
        "paired_view_cosine_median": float(paired_cos.median()),
    }


def linear_probes(features, targets, groups):
    rows = []
    for name, target in targets.items():
        valid = np.isfinite(target)
        pred = np.full(len(target), np.nan)
        X, y, g = features[valid], target[valid], groups[valid]
        for train, test in GroupKFold(n_splits=3).split(X, y, g):
            fit = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            fit.fit(X[train], y[train])
            pred[np.flatnonzero(valid)[test]] = fit.predict(X[test])
        rows.append({"proxy": name, "n": int(valid.sum()), "r2_oof": float(r2_score(target[valid], pred[valid]))})
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proxy-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_cfg = checkpoint.get("config", cfg)
    model = PointMAE(model_config_from_yaml(training_cfg), training_cfg["data"]["feature_names"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    if len(model.encoder.layers) != 10:
        raise ValueError(f"Expected 10 encoder blocks, got {len(model.encoder.layers)}")

    cache = zarr.open_group(args.cache, mode="r")
    n = min(len(cache["xyz"]), args.max_instances or len(cache["xyz"]))
    xyz = np.asarray(cache["xyz"][:n], dtype=np.float32)
    attrs = np.asarray(cache["attrs"][:n], dtype=np.float32)
    instance_id = np.asarray(cache["instance_id"][:n])
    with Path(args.proxy_csv).open(newline="", encoding="utf-8") as file:
        proxy_rows = list(csv.DictReader(file))[:n]
    if len(proxy_rows) != n or any(int(row["instance_id"]) != int(instance_id[i]) for i, row in enumerate(proxy_rows)):
        raise ValueError("Proxy rows do not align with the point-cloud cache")
    groups = np.asarray([row["neuron_id"] for row in proxy_rows])
    if len(set(groups)) < 3:
        raise ValueError("Need at least three distinct neuron/source groups for probing")
    targets = {}
    for name, (column, transform) in PROXIES.items():
        value = np.asarray([float(row[column]) for row in proxy_rows], dtype=np.float64)
        targets[name] = np.log1p(np.maximum(value, 0)) if transform == "log" else value

    original = {f"block_{i}": [] for i in range(11)}
    augmented = {f"block_{i}": [] for i in range(11)}
    original_max = []
    augmented_max = []
    projection = []
    augmented_projection = []
    token_sums = [torch.zeros(384, device=device, dtype=torch.float64) for _ in range(11)]
    token_outer = [torch.zeros((384, 384), device=device, dtype=torch.float64) for _ in range(11)]
    token_count = 0

    def run_tokens(x, a, capture_tokens=False):
        centers, grouped_xyz, grouped_attrs = model.group(x, a)
        tokens = model._tokens(centers, grouped_xyz, grouped_attrs)
        means, maxima = [], []
        for layer in [None, *model.encoder.layers]:
            if layer is not None:
                tokens = layer(tokens)
            means.append(tokens.mean(dim=1).cpu().numpy())
            maxima.append(tokens.max(dim=1).values.cpu().numpy())
            if capture_tokens:
                index = len(means) - 1
                flat = tokens.reshape(-1, tokens.shape[-1]).to(torch.float64)
                token_sums[index].add_(flat.sum(dim=0))
                token_outer[index].add_(flat.T @ flat)
        return means, maxima

    with torch.inference_mode():
        # Actual forward-call trace: MAE uses visible tokens; both NT-Xent views use all tokens.
        observed = []
        handle = model.encoder.register_forward_pre_hook(lambda _module, inputs: observed.append(list(inputs[0].shape)))
        x = torch.as_tensor(xyz[:2], device=device)
        a = torch.as_tensor(attrs[:2], device=device)
        model(x, a, return_loss=True)
        view_x, view_a = augmented_point_view(x, a, model.attr_names, 0.9)
        model.encode(x, a)
        model.encode(view_x, view_a)
        handle.remove()
        if [shape[1] for shape in observed] != [26, 64, 64]:
            raise AssertionError(f"Unexpected encoder patch counts: {observed}")

        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            x = torch.as_tensor(xyz[start:stop], device=device)
            a = torch.as_tensor(attrs[start:stop], device=device)
            base_mean, base_max = run_tokens(x, a, capture_tokens=True)
            token_count += len(x) * model.num_groups
            view_x, view_a = augmented_point_view(x, a, model.attr_names, 0.9)
            view_mean, view_max = run_tokens(view_x, view_a)
            for i in range(11):
                original[f"block_{i}"].append(base_mean[i])
                augmented[f"block_{i}"].append(view_mean[i])
            original_max.append(base_max[-1])
            augmented_max.append(view_max[-1])
            projection.append(model.projection(torch.as_tensor(base_mean[-1], device=device)).cpu().numpy())
            augmented_projection.append(model.projection(torch.as_tensor(view_mean[-1], device=device)).cpu().numpy())
            if start % (args.batch_size * 32) == 0:
                print(f"encoded {stop}/{n}", flush=True)

    original = {name: np.concatenate(value).astype(np.float32) for name, value in original.items()}
    augmented = {name: np.concatenate(value).astype(np.float32) for name, value in augmented.items()}
    original_max = np.concatenate(original_max).astype(np.float32)
    augmented_max = np.concatenate(augmented_max).astype(np.float32)
    projection = np.concatenate(projection).astype(np.float32)
    augmented_projection = np.concatenate(augmented_projection).astype(np.float32)
    mean_max = np.concatenate([original["block_10"], original_max], axis=1)
    view_mean_max = np.concatenate([augmented["block_10"], augmented_max], axis=1)
    readouts = {
        "block_10_mean": (original["block_10"], augmented["block_10"]),
        "block_10_max": (original_max, augmented_max),
        "block_10_mean_plus_max": (mean_max, view_mean_max),
        "trained_projection_128": (projection, augmented_projection),
    }

    layer_rows = []
    probe_rows = []
    for i in range(11):
        name = f"block_{i}"
        feat = original[name]
        stats = embedding_stats(feat)
        token_mean = token_sums[i] / token_count
        token_cov = token_outer[i] / token_count - torch.outer(token_mean, token_mean)
        token_cov_np = token_cov.cpu().numpy()
        token_rank = effective_rank(token_cov_np)
        between_cov = np.cov(feat.astype(np.float64), rowvar=False) * ((n - 1) / n)
        within_cov = token_cov_np - between_cov
        within_fraction = float(np.trace(within_cov) / np.trace(token_cov_np))
        within_rank = effective_rank(within_cov)
        stability = neighbor_stability(feat, augmented[name], min(15, n - 1), device)
        layer_rows.append({"layer": name, "tokens_per_instance": 64, **stats,
                           "unpooled_token_effective_rank": token_rank,
                           "within_instance_token_variance_fraction": within_fraction,
                           "within_instance_token_effective_rank": within_rank, **stability})
        probe_rows.extend({"layer": name, **row} for row in linear_probes(feat, targets, groups))
        print(f"analyzed {name}: pooled rank={stats['effective_rank']:.2f}, token rank={token_rank:.2f}", flush=True)

    readout_rows = []
    for name, (feat, alt) in readouts.items():
        stats = embedding_stats(feat)
        stability = neighbor_stability(feat, alt, min(15, n - 1), device) if alt is not None else {}
        readout_rows.append({"readout": name, "dimension": feat.shape[1], **stats, **stability})
        probe_rows.extend({"layer": name, **row} for row in linear_probes(feat, targets, groups))
    write_csv(output / "layerwise_statistics.csv", layer_rows)
    write_csv(output / "readout_statistics.csv", readout_rows)
    write_csv(output / "linear_probes.csv", probe_rows)
    trace = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "n_instances": n,
        "unique_neuron_or_source_groups": len(set(groups)),
        "seed": args.seed,
        "mask_ratio": model.mask_ratio,
        "encoder_input_shapes_mae_then_ntxent_view1_view2": observed,
        "mae_visible_patches": observed[0][1],
        "ntxent_patches_each_view": [observed[1][1], observed[2][1]],
        "proxy_note": "Proxies computed earlier from the same 2048 sampled points, not true full-volume metadata.",
        "probe_split": "3-fold GroupKFold by cached neuron/source ID; StandardScaler + Ridge(alpha=10), OOF R2",
        "nearest_neighbor": "cosine, k=15 or n-1, original vs 90% resampled + YX rotated view",
    }
    (output / "view_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(json.dumps({"saved_to": str(output), "trace": observed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
