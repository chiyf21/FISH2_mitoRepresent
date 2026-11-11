"""Assemble the Chinese Phase-1 report from saved evaluation and training artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERIMENTS = {
    "E0 Pretrained": ("baseline/expA_pretrained", "e0_expA.log"),
    "E1 R0 B4/B5/B6": ("final_head/e1_b4_b5_b6/evaluation", "e1_stage_b.log"),
    "E2-A Bottleneck head": ("final_head/e2a_b4_b5_final/evaluation", "e2a_stage_b.log"),
    "E2-B Frozen + adapter": ("residual_adapter/e2b_frozen_adapter/evaluation", "e2b_stage_b.log"),
}
TRAIN_LOGS = {
    "E1 R0 B4/B5/B6": ("e1_stage_a.log", "e1_stage_b.log"),
    "E2-A Bottleneck head": ("e2a_stage_a.log", "e2a_stage_b.log"),
    "E2-B Frozen + adapter": ("e2b_stage_a.log", "e2b_stage_b.log"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def log_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    metrics = {}
    for name, (relative, _) in EXPERIMENTS.items():
        path = root / relative / "metrics.json"
        if path.exists():
            metrics[name] = read_json(path)

    lines = [
        "# Point-MAE v5 → MIM-Refiner Phase 1 探索性实验报告", "",
        "## 范围与数据", "",
        "本报告只总结一次短 pilot，不代表多 seed、完整训练或生物学亚型验证。",
        "主初始化为 expA (`occ_intensity`) 的 best.pt；expB (`occ_lowfreq`) 未混入 E1/E2 对照。",
        "实际缓存 split：train 15,508 instances / 148 bodies；val 1,953 / 18 bodies；",
        "held-out test 2,486 / 18 bodies。所有方法复用同一完整 test split；线性 probe 使用 seed 42 的 512 个 body-balanced train instances。",
        "特征推理为 full visibility、无增强/噪声、eval 模式；mask 参与输入和 support 构造。两视图稳定性对比 view 0 与轻微 XY 旋转 view 2。",
        "proxy 包含物理体积/表面积、elongation、sphericity、原始 EM intensity 与梯度统计；表面积是暴露体素面估算，不是平滑 mesh 面积。", "",
        "## 实际训练配置", "",
        "| 方案 | Heads / 最终输出 | Stage A updates | Stage B updates | Stage B loss（首→末） | 累计 step runtime | 峰值显存 | 冻结策略 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, (stage_a, stage_b) in TRAIN_LOGS.items():
        rows_a = log_rows(root / "logs" / stage_a)
        rows_b = log_rows(root / "logs" / stage_b)
        active_a = [r for r in rows_a if r.get("phase") == "head_init"]
        active_b = [r for r in rows_b if r.get("phase") == "encoder_refine"]
        updates_a = max((r.get("optimizer_updates", 0) for r in active_a), default=0)
        updates_b = max((r.get("optimizer_updates", 0) for r in active_b), default=0)
        runtime_b = sum(r.get("runtime_s", 0.) for r in active_b)
        peak_b = max((r.get("gpu_peak_allocated_mb", 0.) for r in active_b), default=0.)
        loss_range = (f"{active_b[0].get('loss', float('nan')):.3f} → {active_b[-1].get('loss', float('nan')):.3f}"
                      if active_b else "—")
        labels = {
            "E1 R0 B4/B5/B6": ("B4/B5/B6；Encoder downstream 主干", "仅 MAE bottleneck 冻结"),
            "E2-A Bottleneck head": ("B4/B5/final；最终 bottleneck 直接受监督", "仅 MAE bottleneck 冻结策略按 R0 配置"),
            "E2-B Frozen + adapter": ("B4/B5/final + residual adapter；导出 z′", "Encoder 全冻结；adapter/head 可训练"),
        }[name]
        lines.append(f"| {name} | {labels[0]} | {updates_a} | {updates_b} | {loss_range} | {runtime_b/60:.1f} min | {peak_b/1024:.1f} GiB | {labels[1]} |")
    lines += [
        "", "所有 run：seed 42、B=8 instances × 2 global views、bf16、true contrastive batch、Stage A hard Top-1、Stage B official-style random single positive from Top-20、固定 queue；各组独立初始化。计划 Stage A≈50 / Stage B≈100；Stage A 首批队列为空，故 50 steps 对应 49 个优化更新。E3 Soft-NN、E4 consensus loss、E5 aggregator 未做完整训练。", "",
        "## 最终 256D 表征主要指标", "",
        "| 方法 | Effective rank | PR | mean pair cosine | 同实例双视图 cosine | 15NN vs pretrained Jaccard |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, obj in metrics.items():
        stats = obj.get("feature_statistics", {}).get("final", {})
        for model_key in ("pretrained", "refined_online", "refined_ema"):
            if model_key not in stats:
                continue
            s = stats[model_key]
            stability = obj.get("same_instance_view_stability", {}).get("final", {}).get(model_key, {})
            knn_key = f"pretrained_vs_{model_key}"
            knn = obj.get("knn_stability", {}).get("final", {}).get(knn_key, {})
            label = name if model_key == "pretrained" else f"{name} {model_key.removeprefix('refined_')}"
            lines.append(f"| {label} | {fmt(s.get('effective_rank'),1)} | {fmt(s.get('participation_ratio'),1)} | {fmt(s.get('mean_pairwise_cosine'))} | {fmt(stability.get('mean_cosine'))} | {fmt(knn.get('mean_knn_jaccard'))} |")
    lines += ["", "## Texture 可读性（size-controlled ΔR²）", "",
              "ΔR² = size + embedding 相对 size-only 的测试集增量；只作为短 pilot 的可读性 proxy。", "",
              "| 方法 / checkpoint | intensity mean | intensity std | gradient mean | orientation coherence |", "|---|---:|---:|---:|---:|"]
    targets = ("intensity_mean", "intensity_std", "gradient_magnitude_mean", "gradient_orientation_coherence")
    for name, obj in metrics.items():
        for rep_name in ("final",):
            for model_key, values in obj.get("size_controlled_texture_r2", {}).get(rep_name, {}).items():
                if model_key not in ("pretrained", "refined_online", "refined_ema"):
                    continue
                label = name if model_key == "pretrained" else f"{name} {model_key.removeprefix('refined_')}"
                scores = [values.get(target, {}).get("delta_r2_over_size_only") for target in targets]
                lines.append("| " + label + " | " + " | ".join(fmt(v) for v in scores) + " |")
    lines += ["", "## Block 层诊断", "",
              "各方法 block4/block5/block6_mean 的完整统计、PCA 累计解释方差、9 项线性 proxy R²、same-instance view stability、KNN/hubness 详见各自 evaluation/metrics.json 和 feature_recovery.csv。",
              "E0 还导出全部 held-out embeddings 与 nearest-neighbor Top-15 列表、三维 mask 近邻图。", "",
              "## 邻居选择 / 架构改进", "",
              "E4 仅运行 B4/B5/B6 held-out 邻居重叠诊断（如 `cross_layer_neighbors.json` 已生成）；不训练 consensus loss。E3 Soft-NN 没有在本轮训练，需在后续单独固定候选邻域后测试 Top-1 / Uniform Top-20 / softmax Top-20 极限和采样分布。E5 attention/multi-query aggregator 未实现为训练方案，也未运行。", "",
              "## 解释边界与建议", "",
              "短训练中的 NNA loss 下降只说明优化在工作，不是 embedding 变好的证据。重点比较 held-out 的 view stability、形态/纹理 proxy、size-controlled ΔR²、15NN 保持和 hubness；单次 seed、512 训练实例和 100 updates 不足以定输赢。若某方案改善这些指标且没有明显损害纹理/几何 probe，再增加训练预算与 seed；不要仅按 effective rank 或 UMAP 分离度选模型。", ""]
    consensus_path = root / "diagnostics" / "cross_layer_neighbors.json"
    if consensus_path.exists():
        neighbor = read_json(consensus_path)["neighbor_consensus"]
        top1, jac = neighbor["top1_agreement"], neighbor["knn_jaccard"]
        common = neighbor["consensus"]["all_layers_top20"]
        insertion = (
            "## E4 邻居重叠诊断（非训练结果）\n\n"
            f"Top-1 一致率：B4/B5={top1['block4|block5']:.3f}，"
            f"B4/B6={top1['block4|block6_mean']:.3f}，"
            f"B5/B6={top1['block5|block6_mean']:.3f}。\n\n"
            f"Top-20 Jaccard：B4/B5={jac['block4|block5|k20']:.3f}，"
            f"B4/B6={jac['block4|block6_mean|k20']:.3f}，"
            f"B5/B6={jac['block5|block6_mean|k20']:.3f}；"
            f"三层共同 Top-20 平均 {common['mean_shared_neighbors']:.2f} 个，"
            f"无共同近邻比例 {common['fraction_with_no_shared_neighbor']:.3f}。\n\n"
            "这表示层间候选近邻高度重叠，不代表这些近邻已被证明在生物学上正确。\n\n")
        lines.insert(lines.index("## 解释边界与建议"), insertion)
    out = root / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
