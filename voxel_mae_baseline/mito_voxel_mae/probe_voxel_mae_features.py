"""Extract basic mask/EM features and evaluate frozen Voxel MAE latents."""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


def voxel_features(record):
    root = Path(record["relative_dir"])
    em = np.load(root / "em.npy", mmap_mode="r")
    mask = np.load(root / "mask.npy", mmap_mode="r").astype(bool)
    n = int(mask.sum())
    if n != int(record["foreground_voxels"]):
        raise ValueError(f"mask count mismatch for {record['record_id']}")
    spacing = np.asarray(record["spacing_zyx_nm"], dtype=np.float64)
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    face_counts = []
    for axis in range(3):
        plus = [slice(1, -1)] * 3
        minus = [slice(1, -1)] * 3
        plus[axis] = slice(2, None)
        minus[axis] = slice(None, -2)
        face_counts.append(int(np.count_nonzero(center & ~padded[tuple(plus)]) +
                               np.count_nonzero(center & ~padded[tuple(minus)])))
    face_areas_nm2 = [spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1]]
    area_um2 = float(np.dot(face_counts, face_areas_nm2) / 1e6)
    values = np.asarray(em[mask], dtype=np.float64)
    volume_um3 = float(n * np.prod(spacing) / 1e9)
    return {"record_id": record["record_id"], "body_id": record["body_id"],
            "split": record["split"], "cell_type": record.get("cell_type", "NA"),
            "foreground_voxels": n, "volume_um3": volume_um3,
            "surface_area_um2": area_um2, "em_mean_raw": float(values.mean()),
            "em_variance_raw": float(values.var())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    data = np.load(args.embeddings)
    x = np.asarray(data["embedding"], dtype=np.float64)
    ids = np.asarray(data["record_id"]).astype(str)
    manifest = json.loads(Path(args.manifest).read_text())
    by_id = {r["record_id"]: r for r in manifest["records"]}
    if len(by_id) != len(manifest["records"]) or set(ids) != set(by_id):
        raise ValueError("Embedding IDs and manifest records do not match exactly")
    records = [by_id[rid] for rid in ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(voxel_features, records), 1):
            rows.append(row)
            if i % 1000 == 0:
                print(json.dumps({"features_extracted": i, "total": len(records)}), flush=True)
    table = {r["record_id"]: r for r in rows}
    split = np.asarray([table[rid]["split"] for rid in ids])
    y_raw = {
        "log10_volume_um3": np.log10([table[rid]["volume_um3"] for rid in ids]),
        "log10_surface_area_um2": np.log10([table[rid]["surface_area_um2"] for rid in ids]),
        "em_mean_raw": np.asarray([table[rid]["em_mean_raw"] for rid in ids]),
        "log10_em_variance_plus1": np.log10(1 + np.asarray([table[rid]["em_variance_raw"] for rid in ids])),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feature_fields = list(rows[0])
    with (out / "feature_targets.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fields)
        writer.writeheader(); writer.writerows(rows)

    train, val, test = (np.flatnonzero(split == name) for name in ("train", "val", "test"))
    scaler = StandardScaler().fit(x[train])
    xs = scaler.transform(x)
    alphas = np.logspace(-3, 4, 15)
    results = []
    predictions = {}
    for name, y in y_raw.items():
        candidates = []
        for alpha in alphas:
            model = Ridge(alpha=float(alpha)).fit(xs[train], y[train])
            candidates.append((r2_score(y[val], model.predict(xs[val])), float(alpha), model))
        _, alpha, model = max(candidates, key=lambda v: v[0])
        pred = model.predict(xs)
        predictions[name] = (y, pred)
        for part, indices in (("train", train), ("validation", val), ("test", test)):
            baseline = np.full(len(indices), y[train].mean())
            results.append({"feature": name, "split": part, "n": int(len(indices)), "alpha": alpha,
                            "r2": float(r2_score(y[indices], pred[indices])),
                            "mae": float(mean_absolute_error(y[indices], pred[indices])),
                            "mean_baseline_r2": float(r2_score(y[indices], baseline)),
                            "spearman": float(spearmanr(y[indices], pred[indices]).statistic)})

    # Control for scale supplied directly to the Encoder as log_size.
    log_size = np.log(np.asarray([
        np.asarray(r["tight_shape_zyx"], dtype=np.float64) * np.asarray(r["spacing_zyx_nm"], dtype=np.float64) / 1000
        for r in records
    ]))
    size_scaler = StandardScaler().fit(log_size[train])
    size_x = size_scaler.transform(log_size)
    scale_baselines = []
    for name in ("log10_volume_um3", "log10_surface_area_um2"):
        y = y_raw[name]
        candidates = []
        for alpha in alphas:
            model = Ridge(alpha=float(alpha)).fit(size_x[train], y[train])
            candidates.append((r2_score(y[val], model.predict(size_x[val])), float(alpha), model))
        _, alpha, model = max(candidates, key=lambda v: v[0])
        pred = model.predict(size_x[test])
        scale_baselines.append({"feature": name, "split": "test", "n": int(len(test)), "alpha": alpha,
                                "r2": float(r2_score(y[test], pred)),
                                "mae": float(mean_absolute_error(y[test], pred)),
                                "spearman": float(spearmanr(y[test], pred).statistic)})

    with (out / "linear_probe_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    with (out / "explicit_log_size_baseline.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scale_baselines[0]))
        writer.writeheader(); writer.writerows(scale_baselines)
    features = list(y_raw)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    test_results = {r["feature"]: r for r in results if r["split"] == "test"}
    for ax, name in zip(axes.flat, features):
        y, pred = predictions[name]
        ax.scatter(y[test], pred[test], s=10, alpha=.55, linewidths=0, rasterized=True)
        low = min(float(y[test].min()), float(pred[test].min()))
        high = max(float(y[test].max()), float(pred[test].max()))
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name} | test R²={test_results[name]['r2']:.3f}")
        ax.set_xlabel("Observed (transformed scale)"); ax.set_ylabel("Predicted")
        ax.grid(alpha=.2)
    fig.suptitle("Frozen Voxel MAE embedding: neuron-held-out linear probe")
    fig.savefig(out / "linear_probe_test_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = {
        "embedding_file": args.embeddings, "n_instances": len(ids), "embedding_dim": int(x.shape[1]),
        "split_counts": {name: int(np.sum(split == name)) for name in ("train", "validation", "test")},
        "split_unit": "body/neuron; inherited from the prepared manifest",
        "targets": {
            "log10_volume_um3": "log10(mask voxel count × voxel volume in µm³)",
            "log10_surface_area_um2": "log10(6-neighbor exposed voxel-face area using anisotropic spacing, in µm²)",
            "em_mean_raw": "mean original uint8 EM intensity over mask voxels",
            "log10_em_variance_plus1": "log10(1 + variance of original uint8 EM intensity over mask voxels)",
        },
        "selection": "Ridge alpha selected by validation R²; test split used once for final evaluation",
        "metrics": results,
        "explicit_log_size_only_test_baseline": scale_baselines,
        "caveats": [
            "Surface area is a voxel-face estimate and depends on the voxel grid.",
            "log_size is explicitly given to the Encoder and is also the size-head target, so size-related readout is not independent evidence that size was inferred from EM alone.",
            "EM input is per-instance z-scored after clipping; raw absolute mean and variance are therefore partly removed by preprocessing.",
            "UMAP/PCA visualization is not used in this regression; the probe uses the original 256-D latent.",
        ],
    }
    (out / "linear_probe_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Voxel MAE embedding 基础特征线性回归", "",
             f"样本数：{len(ids)}；latent 维度：{x.shape[1]}；按 neuron/body 划分。",
             "训练集拟合 Ridge，验证集选择 alpha，测试集仅用于最终报告。", "",
             "| 特征 | split | n | alpha | R² | MAE（变换后尺度） | 均值基线 R² | Spearman |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['feature']} | {r['split']} | {r['n']} | {r['alpha']:.4g} | {r['r2']:.4f} | {r['mae']:.4f} | {r['mean_baseline_r2']:.4f} | {r['spearman']:.4f} |")
    lines += ["", "## 标签定义", "",
              "volume 和 surface area 使用 log10；EM mean 使用原始 uint8 均值；EM variance 使用 log10(1+variance)。表面积由各向异性 voxel mask 的 6 邻接暴露面计数估算。", "",
              "## 只用显式尺度输入的对照", "",
              "| 特征 | 测试集 R² | MAE（变换后尺度） | Spearman |",
              "|---|---:|---:|---:|"]
    for r in scale_baselines:
        lines.append(f"| {r['feature']} | {r['r2']:.4f} | {r['mae']:.4f} | {r['spearman']:.4f} |")
    lines += ["", 
              "## 解读限制", "",
              "Encoder 输入已提供 log physical size，size loss 也直接预测该值，因此体积/表面积 probe 会受显式尺度输入影响。raw EM 强度在输入前进行每实例 z-score，所以 raw mean/variance 的可读性可能被预处理主动削弱。表面积是 voxel-face 估计，不是平滑连续表面的精确测量。"]
    (out / "linear_probe_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"n": len(ids), "out": str(out), "test": [r for r in results if r["split"] == "test"]}, indent=2))


if __name__ == "__main__":
    main()
