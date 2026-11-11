from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .backbone import load_adapter
from .data_utils import load_global_views, load_sample, make_dataset
from .features import extract_features
from .neighbor_consensus import cosine_knn
from .residual_adapter import ResidualEmbeddingAdapter
from .training_step import precision_context


REPRESENTATIONS = ("block6_mean", "final")
TARGET_NAMES = (
    "log10_volume_um3", "physical_elongation", "log10_surface_area_um2",
    "sphericity", "intensity_mean", "intensity_std", "intensity_variance",
    "gradient_magnitude_mean", "gradient_orientation_coherence",
)


def embedding_stats(values):
    x = np.asarray(values, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular ** 2
    prob = variance / max(variance.sum(), 1e-30)
    nonzero = prob > 0
    effective_rank = float(np.exp(-(prob[nonzero] * np.log(prob[nonzero])).sum())) if nonzero.any() else 0.0
    participation = float(variance.sum() ** 2 / max((variance ** 2).sum(), 1e-30))
    explained = variance / max(variance.sum(), 1e-30)
    unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sims = unit @ unit.T
    upper = sims[np.triu_indices(len(x), k=1)]
    return {"n": int(len(x)), "dim": int(x.shape[1]),
            "mean_dimension_std": float(x.std(axis=0).mean()),
            "mean_norm": float(np.linalg.norm(x, axis=1).mean()),
            "effective_rank": effective_rank, "participation_ratio": participation,
            "pca_cumulative_variance_top5": float(explained[:5].sum()),
            "pca_cumulative_variance_top10": float(explained[:10].sum()),
            "pca_cumulative_variance_top20": float(explained[:20].sum()),
            "mean_pairwise_cosine": float(upper.mean()) if len(upper) else None,
            "median_pairwise_cosine": float(np.median(upper)) if len(upper) else None}


def target_features(ds, record_index):
    record = ds.records[record_index]
    folder = ds.root / record["relative_dir"]
    mask = np.load(folder / "mask.npy", mmap_mode="r").astype(bool)
    valid = np.load(folder / "valid.npy", mmap_mode="r").astype(bool)
    foreground = mask & valid
    spacing_um = np.asarray(record["spacing_zyx_nm"], dtype=np.float64) / 1000.0
    volume_um3 = float(foreground.sum() * np.prod(spacing_um))
    # Signed differences: uint8 would turn a falling edge (-1) into 255.
    padded = np.pad(foreground.astype(np.int8), 1)
    face_areas = (spacing_um[1] * spacing_um[2],
                  spacing_um[0] * spacing_um[2],
                  spacing_um[0] * spacing_um[1])
    surface_um2 = sum(float(np.abs(np.diff(padded, axis=axis)).sum()) * face_areas[axis]
                      for axis in range(3))
    coords = np.argwhere(foreground).astype(np.float64) * spacing_um[None]
    if len(coords) > 1:
        eigenvalues = np.maximum(np.linalg.eigvalsh(np.cov(coords, rowvar=False)), 1e-12)
        elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    else:
        elongation = 1.0
    sphericity = (float(np.pi ** (1.0 / 3.0) * (6.0 * volume_um3) ** (2.0 / 3.0) / surface_um2)
                  if volume_um3 > 0 and surface_um2 > 0 else 0.0)
    # Texture probes target raw EM, independent of augmentation / z-score.
    image = np.asarray(np.load(folder / "em.npy", mmap_mode="r"), dtype=np.float32)
    vals = image[foreground]
    if len(vals):
        intensity_mean = float(vals.mean())
        intensity_std = float(vals.std())
        intensity_var = float(vals.var())
        gradients = [np.gradient(image, spacing_um[a], axis=a) if image.shape[a]>1
                     else np.zeros_like(image) for a in range(3)]
        grad = np.stack(gradients, axis=0)[:, foreground]
        magnitude = np.linalg.norm(grad, axis=0)
        gradient_mean = float(magnitude.mean())
        nonzero = magnitude > 1e-10
        coherence = (float(np.linalg.norm((grad[:, nonzero] / magnitude[nonzero]).mean(axis=1)))
                     if nonzero.any() else 0.0)
    else:
        intensity_mean = intensity_std = intensity_var = gradient_mean = coherence = 0.0
    return np.asarray([
        np.log10(max(volume_um3, 1e-12)), elongation,
        np.log10(max(surface_um2, 1e-12)), sphericity, intensity_mean,
        intensity_std, intensity_var, gradient_mean, coherence,
    ], dtype=np.float64)


@torch.no_grad()
def embed_dataset(adapter, model, ds, device, representations=REPRESENTATIONS,
                  two_global_views=False, view_ds=None, precision="fp32",
                  embedding_adapter=None):
    model.eval()
    vectors = {name: [] for name in representations}
    paired = {name: [] for name in representations} if two_global_views else None
    targets, record_ids, body_ids = [], [], []
    n = len(ds.records)
    if ds.augmentations or ds.masking:
        raise ValueError("Primary embeddings require full visibility and no augmentation/noise")
    if two_global_views:
        if view_ds is None or [r['record_id'] for r in view_ds.records] != [r['record_id'] for r in ds.records]:
            raise ValueError("Stability dataset must have exactly the primary cohort/order")
    for i in range(n):
        with precision_context(device, precision):
            feature = extract_features(adapter, model, load_sample(ds, i, device))
            if embedding_adapter is not None:
                feature["final"] = embedding_adapter(feature["final"])
        for name in representations:
            vectors[name].append(feature[name][0].float().cpu().numpy())
        if two_global_views:
            with precision_context(device, precision):
                views = [extract_features(adapter, model, batch)
                         for batch in load_global_views(view_ds, i, device)]
                if embedding_adapter is not None:
                    views = [{**view, "final": embedding_adapter(view["final"])} for view in views]
            for name in representations:
                paired[name].append(np.stack([view[name][0].float().cpu().numpy() for view in views]))
        targets.append(target_features(ds, i))
        record_ids.append(str(ds.records[i]["record_id"]))
        body_ids.append(str(ds.records[i]["body_id"]))
        if (i + 1) % 64 == 0 or i + 1 == n:
            print(json.dumps({"phase": "evaluation_embedding", "done": i + 1, "total": n}), flush=True)
    result = {"features": {name: np.stack(rows) for name, rows in vectors.items()},
              "targets": np.stack(targets), "record_id": np.asarray(record_ids),
              "body_id": np.asarray(body_ids)}
    if two_global_views:
        result["paired_views"] = {name: np.stack(rows) for name, rows in paired.items()}
    return result


def r2_probe(train_x, train_y, test_x, test_y, alpha=10.0):
    mu, sd = train_x.mean(axis=0), train_x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x, xt = (train_x - mu) / sd, (test_x - mu) / sd
    ym, ys = train_y.mean(axis=0), train_y.std(axis=0)
    ys[ys < 1e-8] = 1.0
    y = (train_y - ym) / ys
    coef = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    pred = (xt @ coef) * ys + ym
    denom = ((test_y - test_y.mean(0)) ** 2).sum(0)
    numer = ((test_y - pred) ** 2).sum(0)
    return [None if d < 1e-12 else float(1.0 - n / d) for n, d in zip(numer, denom)]


def gini(values):
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, len(x) + 1) - len(x) - 1) @ x / (len(x) * x.sum()))


def neighbor_metrics(reference, candidate, body_ids, k=15):
    ref_nn, cand_nn = cosine_knn(reference, k), cosine_knn(candidate, k)
    if not len(ref_nn):
        return {"mean_knn_jaccard": None, "same_body_top1": None,
                "hubness_max_top15": None, "hubness_gini_top15": None,
                "candidate_neighbors": cand_nn}
    jac = [len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
           for a, b in zip(ref_nn, cand_nn)]
    counts = np.bincount(cand_nn.ravel(), minlength=len(candidate))
    return {"mean_knn_jaccard": float(np.mean(jac)),
            "same_body_top1": float(np.mean(np.asarray(body_ids)[cand_nn[:, 0]] == np.asarray(body_ids))),
            "hubness_max_top15": int(counts.max()), "hubness_gini_top15": gini(counts),
            "candidate_neighbors": cand_nn}


def paired_view_stability(paired_features):
    result = {}
    for name, values in paired_features.items():
        unit = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
        sims = np.sum(unit[:, 0] * unit[:, 1], axis=-1)
        result[name] = {"n_instances": int(len(sims)), "mean_cosine": float(sims.mean()),
                        "std_cosine": float(sims.std()), "p10_cosine": float(np.quantile(sims, 0.1)),
                        "p50_cosine": float(np.quantile(sims, 0.5)),
                        "p90_cosine": float(np.quantile(sims, 0.9))}
    return result


def write_neighbors(path, ids, bodies, features, neighbor_indices, k=5):
    rows = []
    for method, vectors in features.items():
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        neighbors = neighbor_indices[method]
        for i in range(len(ids)):
            for rank, j in enumerate(neighbors[i, :k], 1):
                rows.append((method, ids[i], bodies[i], rank, ids[j], bodies[j],
                             float(unit[i] @ unit[j])))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "anchor_record_id", "anchor_body_id", "rank",
                         "neighbor_record_id", "neighbor_body_id", "cosine"))
        writer.writerows(rows)


def plot_nearest_neighbor_shapes(path, ds, record_ids, body_ids, nn_indices, methods):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    anchors, seen = [], set()
    for i, body in enumerate(body_ids):
        if body not in seen:
            seen.add(body)
            anchors.append(i)
        if len(anchors) == 3:
            break
    if not anchors:
        return False
    fig = plt.figure(figsize=(13, 3.8 * len(anchors)))
    columns = len(methods) + 1
    axes = [[fig.add_subplot(len(anchors), columns, row * columns + col + 1, projection="3d")
             for col in range(columns)] for row in range(len(anchors))]
    rng = np.random.default_rng(42)
    display_name = {"pretrained": "Pretrained", "refined_online": "Online", "refined_ema": "EMA"}
    colors = {"Anchor": "#555555", "pretrained": "#6a3d9a",
              "refined_online": "#1b9e77", "refined_ema": "#d95f02"}

    def short_id(value):
        return "…" + "__".join(str(value).split("__")[-2:])

    for row, anchor in enumerate(anchors):
        indices = [anchor] + [int(nn_indices[method][anchor, 0]) for method in methods]
        for col, index in enumerate(indices):
            ax = axes[row][col]
            record = ds.records[index]
            folder = ds.root / record["relative_dir"]
            mask = np.load(folder / "mask.npy", mmap_mode="r").astype(bool)
            coords = np.argwhere(mask)
            if len(coords) > 1800:
                coords = coords[rng.choice(len(coords), 1800, replace=False)]
            spacing = np.asarray(record["spacing_zyx_nm"], dtype=np.float64) / 1000.0
            physical = (coords - (np.asarray(mask.shape) - 1) / 2) * spacing[None]
            ax.scatter(physical[:, 2], physical[:, 1], physical[:, 0],
                       s=1.5, alpha=0.7, color=colors["Anchor"] if col == 0 else colors[methods[col - 1]])
            extent = np.maximum(np.ptp(physical, axis=0)[::-1], 1e-3)
            ax.set_box_aspect(extent)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if col == 0:
                ax.set_title(f"Anchor\n{short_id(record_ids[index])}", fontsize=9)
            else:
                method = methods[col - 1]
                ax.set_title(f"{display_name[method]} Top-1\n{short_id(record_ids[index])}", fontsize=9)
    fig.suptitle("Held-out mitochondria: anchor and nearest-neighbor masks", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained and refined Voxel-MAE representations")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--refined-checkpoint", required=True)
    parser.add_argument("--backbone-adapter", default=f"{__package__}.v5_adapter:V5VoxelMAEAdapter")
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS),
                        help="Comma-separated global feature names returned by the adapter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-limit", type=int, default=1024,
                        help="Body-balanced training subset for frozen linear probes; 0 uses all train instances")
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="Held-out test instances; 0 uses the current cache's full test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision",choices=("fp32","bf16"),
                        default="bf16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--slice-microbatch",type=int,default=8)
    parser.add_argument("--tile-batch",type=int,default=2)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"{out / 'metrics.json'} already exists; use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    adapter = load_adapter(args.backbone_adapter)
    representations = tuple(name.strip() for name in args.representations.split(",") if name.strip())
    if not representations or len(set(representations)) != len(representations):
        raise ValueError("Evaluation representations must be non-empty and unique")
    base, _, _ = adapter.load_model(args.base_checkpoint, device)
    refined, refined_state, _ = adapter.load_model(args.refined_checkpoint, device)
    if refined_state.get("backbone_adapter", adapter.identifier) != adapter.identifier:
        raise ValueError("Refined checkpoint and evaluator use different backbone adapters")
    ema = None
    refined_embedding_adapter = None
    ema_embedding_adapter = None
    if refined_state.get("embedding_adapter") is not None:
        refined_embedding_adapter = ResidualEmbeddingAdapter(
            **refined_state["embedding_adapter_config"]).to(device)
        refined_embedding_adapter.load_state_dict(refined_state["embedding_adapter"])
        refined_embedding_adapter.eval()
    if refined_state.get("ema_model") is not None:
        ema = copy.deepcopy(refined).eval()
        ema.load_state_dict(refined_state["ema_model"], strict=True)
        if refined_state.get("ema_embedding_adapter") is not None:
            ema_embedding_adapter = ResidualEmbeddingAdapter(
                **refined_state["embedding_adapter_config"]).to(device)
            ema_embedding_adapter.load_state_dict(refined_state["ema_embedding_adapter"])
            ema_embedding_adapter.eval()
    for model in (base,refined,ema):
        if model is not None:
            model.slice_microbatch=args.slice_microbatch
            model.tile_batch=args.tile_batch

    train_ds = make_dataset(args.cache, "train", args.train_limit, args.seed)
    test_ds = make_dataset(args.cache, "test", args.eval_limit, args.seed + 1)
    stability_ds = make_dataset(args.cache, "test", args.eval_limit, args.seed + 1, train_views=True)
    train_bodies={str(r['body_id']) for r in train_ds.records}
    if train_bodies & {str(r['body_id']) for r in test_ds.records}:
        raise ValueError("Linear probes require disjoint train/test neuron bodies")
    same_checkpoint = Path(args.base_checkpoint).resolve() == Path(args.refined_checkpoint).resolve()
    models = {"pretrained": (base, None)}
    if not same_checkpoint or refined_embedding_adapter is not None:
        models["refined_online"] = (refined, refined_embedding_adapter)
    if ema is not None:
        models["refined_ema"] = (ema, ema_embedding_adapter)
    train_payloads, test_payloads = {}, {}
    for name, (model, model_adapter) in models.items():
        print(json.dumps({"phase": "probe_train_embeddings", "model": name}), flush=True)
        train_payloads[name] = embed_dataset(adapter, model, train_ds, device, representations,
                                             precision=args.precision,
                                             embedding_adapter=model_adapter)
        print(json.dumps({"phase": "heldout_test_embeddings", "model": name}), flush=True)
        test_payloads[name] = embed_dataset(adapter, model, test_ds, device, representations,
                                            two_global_views=True, view_ds=stability_ds,
                                            precision=args.precision,embedding_adapter=model_adapter)

    result = {
        "seed": args.seed,
        "precision":args.precision,
        "n_train_probe": len(train_ds.records),
        "n_test_eval": len(test_ds.records),
        "test_body_count": len({str(r["body_id"]) for r in test_ds.records}),
        "test_split_record_ids": [str(r["record_id"]) for r in test_ds.records],
        "head_initialization": refined_state.get("head_layers", []),
        "primary_embedding_protocol": "full visibility, no augmentation/noise, model.eval(), full mask conditioned",
        "texture_target_protocol": "raw cached EM inside mask & valid; gradient units raw intensity/um",
        "surface_area_protocol": "exposed voxel faces with physical spacing; not smoothed mesh area",
        "bottleneck_refined": refined_state.get("bottleneck_refined", "final" in refined_state.get("head_layers", [])),
        "refinement_positive": "official-style uniform random single sample from Top-20",
        "views": "two global views (dataset view 0 and mild XY-rotation view 2); local views not used",
        "feature_statistics": {}, "same_instance_view_stability": {},
        "feature_recovery_r2": {}, "size_only_baseline_r2": {},
        "size_controlled_texture_r2": {}, "knn_stability": {},
    }
    ids = test_payloads["pretrained"]["record_id"]
    bodies = test_payloads["pretrained"]["body_id"]
    targets_train = train_payloads["pretrained"]["targets"]
    targets_test = test_payloads["pretrained"]["targets"]
    result["size_only_baseline_r2"] = dict(zip(
        TARGET_NAMES,
        r2_probe(targets_train[:, :1], targets_train, targets_test[:, :1], targets_test, alpha=10.0),
    ))
    neighbor_indices = {name: {} for name in models}
    arrays = {"record_id": ids, "body_id": bodies, "targets": targets_test}
    for rep in representations:
        result["feature_statistics"][rep] = {}
        result["same_instance_view_stability"][rep] = {}
        result["feature_recovery_r2"][rep] = {}
        result["size_controlled_texture_r2"][rep] = {}
        result["knn_stability"][rep] = {}
        for name in models:
            test_x = test_payloads[name]["features"][rep]
            train_x = train_payloads[name]["features"][rep]
            stats = embedding_stats(test_x)
            if name != "pretrained":
                a = test_x / np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), 1e-12)
                b = test_payloads["pretrained"]["features"][rep]
                b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
                stats["mean_cosine_to_pretrained"] = float(np.mean(np.sum(a * b, axis=1)))
            result["feature_statistics"][rep][name] = stats
            paired = test_payloads[name]["paired_views"][rep]
            result["same_instance_view_stability"][rep][name] = paired_view_stability({rep: paired})[rep]
            r2 = r2_probe(train_x, train_payloads[name]["targets"], test_x,
                          test_payloads[name]["targets"])
            result["feature_recovery_r2"][rep][name] = dict(zip(TARGET_NAMES, r2))
            train_targets = train_payloads[name]["targets"]
            test_targets = test_payloads[name]["targets"]
            combined_train = np.concatenate((train_targets[:, :1], train_x), axis=1)
            combined_test = np.concatenate((test_targets[:, :1], test_x), axis=1)
            combined_r2 = r2_probe(combined_train, train_targets, combined_test, test_targets, alpha=10.0)
            result["size_controlled_texture_r2"][rep][name] = {
                target: {"r2_size_plus_embedding": combined_r2[i],
                         "delta_r2_over_size_only": (
                             combined_r2[i] - result["size_only_baseline_r2"][target]
                             if combined_r2[i] is not None and result["size_only_baseline_r2"][target] is not None
                             else None)}
                for i, target in enumerate(TARGET_NAMES) if i >= 4
            }
            unit = test_x / np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), 1e-12)
            arrays[f"{name}_{rep}"] = test_x
            arrays[f"{name}_{rep}_view0_view2"] = paired
            neighbors = cosine_knn(unit, 15)
            neighbor_indices[name][rep] = neighbors
        base_x = test_payloads["pretrained"]["features"][rep]
        for name in models:
            if name == "pretrained":
                continue
            values = neighbor_metrics(base_x, test_payloads[name]["features"][rep], bodies, k=15)
            neighbor_indices[name][rep] = values.pop("candidate_neighbors")
            result["knn_stability"][rep][f"pretrained_vs_{name}"] = values

    np.savez_compressed(out / "embeddings_test_heldout.npz", **arrays)
    with (out / "feature_recovery.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("representation", "model", "target", "test_r2"))
        for rep, methods in result["feature_recovery_r2"].items():
            for method, values in methods.items():
                writer.writerows((rep, method, target, score) for target, score in values.items())
    with (out / "size_only_baseline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("predictor", "target", "test_r2"))
        writer.writerows(("log10_volume_um3", target, score)
                         for target, score in result["size_only_baseline_r2"].items())

    shape_representation = "final" if "final" in representations else representations[-1]
    nn_for_plot = {name: neighbor_indices[name][shape_representation] for name in models}
    all_nn_features = {name: test_payloads[name]["features"][shape_representation] for name in models}
    write_neighbors(out / "nearest_neighbors_top15.csv", ids, bodies, all_nn_features,
                    nn_for_plot, k=15)
    plot_ok = plot_nearest_neighbor_shapes(out / "nearest_neighbor_shapes_readable.png", test_ds,
                                          ids, bodies, nn_for_plot, list(models))
    result["nearest_neighbor_visualization"] = {
        "file": "nearest_neighbor_shapes_readable.png" if plot_ok else None,
        "csv": "nearest_neighbors_top15.csv", "plot_created": plot_ok,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report = [
        "# MIM-Refiner R0 v2：held-out 评价", "",
        f"- Seed: {args.seed}; 线性 probe 训练样本: {len(train_ds.records)}（body-balanced train split）。",
        f"- Held-out 测试: {len(test_ds.records)} instances / {result['test_body_count']} bodies；使用既有 test split。",
        f"- 下游特征由 `{adapter.identifier}` adapter 导出：{', '.join(representations)}；不使用 ID Projector 输出。",
        "- 主 embedding 来自全可见、无增强、无噪声的完整实例；完整实例 mask 参与输入和支持点构造。",
        f"- Bottleneck 是否直接接受 Refiner 梯度：{result['bottleneck_refined']}；B4/B5/B6 默认只训练瓶颈前主干。",
        "- 纹理 proxy 从原始缓存 EM 计算；表面积为物理体素暴露面估计。",
        "- 同一实例稳定性对比固定两个 global views（view 0 与轻微 XY 旋转的 view 2）；本轮没有 local views。",
        "- 连续 proxy、KNN、ER/PR 都是初步表征诊断，不等同于证明生物学亚型可分。", "",
        "## Effective rank / Participation ratio", "",
        "| 表征 | 模型 | ER | PR | 平均两两 cosine |", "|---|---|---:|---:|---:|",
    ]
    for rep, methods in result["feature_statistics"].items():
        for name, stats in methods.items():
            report.append(f"| {rep} | {name} | {stats['effective_rank']:.2f} | {stats['participation_ratio']:.2f} | {stats['mean_pairwise_cosine']:.4f} |")
    report += ["", "## Same-instance View Stability", "",
               "每个值是同一 mitochondrion 的 view 0 与 view 2 embedding cosine。", "",
               "| 表征 | 模型 | N | 平均 cosine | SD | P10 |", "|---|---|---:|---:|---:|---:|"]
    for rep, methods in result["same_instance_view_stability"].items():
        for name, stats in methods.items():
            report.append(f"| {rep} | {name} | {stats['n_instances']} | {stats['mean_cosine']:.4f} | {stats['std_cosine']:.4f} | {stats['p10_cosine']:.4f} |")
    report += ["", "## 线性可读性（held-out R²）", "",
               "详见 `feature_recovery.csv`；`metrics.json` 另含 size+embedding 对纹理目标的 R² 及相对 size-only 的增量。size-only baseline 使用同一 alpha=10。", "",
               "## 邻域保持、Hubness 与最近邻形态", "",
               "每个模型相对预训练模型比较 15NN Jaccard；同体 Top-1 比例和 hubness 统计见 `metrics.json`。",
               f"最近邻形态拼图使用 `{shape_representation}` 特征；明细见 `nearest_neighbors_top15.csv`，形态拼图见 `nearest_neighbor_shapes_readable.png`。", "",
               "## 实验边界", "",
               "两个 global views、Voxel-MAE 多层输入、Head 宽度和可选 Encoder EMA 都是适配设置；报告本身不证明训练充分。",
               "尚未复现官方 local-view loss、可选 Momentum Projector、多随机种子训练稳定性或 Soft NN。", ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote evaluation artifacts to {out}", flush=True)


if __name__ == "__main__":
    main()
