"""Validate and report paired fixed-LR results; never fit classifiers or encoders."""
import argparse
from pathlib import Path
import shutil

import pandas as pd

from .common import TASKS, dump, read, sha, table
from .evaluate import metrics



def plot_comparison(result, output):
    """Presentation only: plot the saved paired confusion counts without refitting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    for task, config in TASKS.items():
        path = output / f"comparison_{task}_confusion.png"
        if path.exists():
            raise FileExistsError(path)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout="constrained")
        for column, (name, title) in enumerate([
                ("Fusion50", "Fusion50"),
                ("NoHead50_calibrated", "No-Head50 + covariance calibration")]):
            for row, level in enumerate(["per_mito", "per_group"]):
                score = result[name][task]["embedding"][level]
                counts = np.asarray(score["confusion"])
                normalized = np.asarray(score["confusion_normalized"])
                ax = axes[row, column]
                ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
                for i in range(len(counts)):
                    for j in range(len(counts)):
                        ax.text(j, i, f"{counts[i,j]}\n{normalized[i,j]:.1%}",
                                ha="center", va="center", fontsize=10 if len(counts)>2 else 16,
                                color="white" if normalized[i,j]>.55 else "black")
                ax.set_xticks(range(len(counts)), config["classes"])
                ax.set_yticks(range(len(counts)), config["classes"])
                ax.set_xlabel("Predicted class")
                ax.set_ylabel("True class")
                label = "Single mitochondrion" if row == 0 else (
                    "Body aggregate" if task == "cell_type" else "True-compartment aggregate")
                ax.set_title(f"{title}\n{label}: BA={score['balanced_accuracy']:.4f}, Macro-F1={score['macro_f1']:.4f}",
                             fontsize=11)
        aggregation = "Body-level aggregation" if task == "cell_type" else "Body x true-compartment aggregation (oracle)"
        fig.suptitle(f"Fish2 Reviewed81 | {task} | fixed LR | body-disjoint five folds\n{aggregation}",
                     fontsize=15)
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--nohead", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = {"Fusion50": args.baseline.resolve(), "NoHead50_calibrated": args.nohead.resolve()}
    runs = {name: read(path / "run.json") for name, path in paths.items()}
    reference = runs["Fusion50"]
    for name, run in runs.items():
        assert (paths[name] / "_SUCCESS.json").exists() and run["status"] == "complete"
        assert run["classifier"] == "lr" and set(run["tasks"]) == set(TASKS)
        assert run["feature_dimensions"]["embedding"] == 1152
        for field in ["assets", "asset_manifest_sha256", "protocol", "classifier_parameters",
                      "aggregation", "python", "numpy", "sklearn"]:
            assert run[field] == reference[field], field
    result = {name: read(path / "metrics.json") for name, path in paths.items()}
    rows = []
    for task, config in TASKS.items():
        source = Path(reference["assets"]) / f"{task}_instances.csv"
        for name, path in paths.items():
            assert sha(path / task / "instances.csv") == sha(source)
            assert set(result[name][task]) == {"embedding"}
            frame = table(path / task / "embedding_instance_oof.csv")
            frozen = table(source)
            assert frame.record_id.is_unique and len(frame) == config["n"]
            for column in frozen.columns:
                assert frame[column].equals(frozen[column]), (task, column)
            grouped = table(path / task / "embedding_group_oof.csv")
            assert len(grouped) == config["groups"]
            for level, oof in [("per_mito", frame), ("per_group", grouped)]:
                assert metrics(oof.label, oof.prediction, config["classes"]) == result[name][task]["embedding"][level]
            models = read(path / task / "embedding_training.json")
            assert len(models) == (30 if task == "compartment" else 5)
            assert max(max(m["n_iter"]) for m in models) < 3000
        for level in ["per_mito", "per_group"]:
            old = result["Fusion50"][task]["embedding"][level]
            new = result["NoHead50_calibrated"][task]["embedding"][level]
            rows.append(dict(task=task, level=level,
                baseline_ba=old["balanced_accuracy"], baseline_macro_f1=old["macro_f1"],
                nohead_ba=new["balanced_accuracy"], nohead_macro_f1=new["macro_f1"],
                delta_ba_pp=100*(new["balanced_accuracy"]-old["balanced_accuracy"]),
                delta_macro_f1_pp=100*(new["macro_f1"]-old["macro_f1"])))
    export = Path(runs["NoHead50_calibrated"]["embedding"]["matrix"]).parent
    provenance = read(export / "complete.json")
    assert provenance["status"] == "complete" and provenance["shape"] == [5568, 1152]
    assert provenance["qa"]["coverage"] == dict(instances=5568, legacy=4462, added=1106,
                                               bodies=81, each_instance_written_once=True)
    args.output.mkdir(parents=True, exist_ok=False)
    identities = {}
    for name, run in runs.items():
        checkpoint = run["embedding"]["checkpoint"]["path"]
        identities[name] = dict(run=str(paths[name]), representation=run["embedding"]["representation"],
            checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint),
            descriptor=run["embedding_descriptor"], matrix=run["embedding"]["matrix"],
            matrix_sha256=sha(run["embedding"]["matrix"]),
            instance_map=run["embedding"]["instance_map"])
        for task in TASKS:
            shutil.copy2(paths[name]/task/"confusion.png", args.output/f"{name}_{task}_confusion.png")
    plot_comparison(result, args.output)
    dump(args.output / "comparison.json", dict(identities=identities, differences=rows,
        full_metrics=result, export_validation=provenance,
        protocol_asset_sha256=reference["asset_manifest_sha256"],
        classifier=reference["classifier_parameters"], aggregation=reference["aggregation"],
        checks="identical frozen instances, labels, folds and classifier; OOF metrics recomputed",
        interpretation="No Head + calibrated main covariance; changes cannot be attributed solely to Head removal"))
    pd.DataFrame(rows).to_csv(args.output / "metrics_comparison.csv", index=False)
    lines = [
        "# Fish2 No-Head50：Reviewed81 固定 LR 正式评价", "",
        "全部任务成功完成。此次为**无 Head＋主 Covariance 系数校准**的联合消融，性能差异不能单独归因于移除 Head。", "",
        "导出覆盖 reviewed81_v1 全部 5,568 实例、81 个神经元：4,462 个旧实例＋1,106 个新增实例；",
        "严格 record_id 映射，每行恰写一次，float32 的 L12 Balanced1152D，无 NaN/Inf。未用旧 10,678 队列替代。",
        "两个来源各 8 个实例重复推理逐位一致；相同预处理复现的原 Fusion50 向量与既有基准逐位一致。",
        "沿用训练时的 core.prepare/System、原点云缓存、填洞 mask、view0/原 cohort_index 种子、bf16 Transformer 和冻结 PairBalance EMA。",
        "Fusion MLP 全程被绕过；无梯度、零优化器更新。", "",
        "固定五折按 body 隔离，seed=20261113；LR 为 C=1/L2/lbfgs/balanced/max_iter=3000，StandardScaler 仅拟合训练折。",
        "cell_type：5,568 实例、81 bodies、六类 pooled LR；按 body 聚合。",
        "compartment：5,059 个 QC=ok 的 axon/dendrite 实例、81 bodies、按神经元类型训练 LR；161 个 body×真实区室组（oracle 聚合）。",
        "两项均按组内平均 log10(clip(p,1e-15,1)) 后 argmax 聚合。两模型数据、标签、折、分类器和预处理一致，无参数搜索。", "",
        "BA、Macro-F1 为百分数；Δ 为百分点（No-Head − Fusion50）。", "",
        "|任务|粒度|Fusion50 BA|Fusion50 Macro-F1|No-Head BA|No-Head Macro-F1|ΔBA|ΔMacro-F1|",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        label = "单线粒体" if r["level"] == "per_mito" else ("神经元" if r["task"] == "cell_type" else "真实区室聚合")
        lines.append(f'|{r["task"]}|{label}|{100*r["baseline_ba"]:.4f}|{100*r["baseline_macro_f1"]:.4f}|{100*r["nohead_ba"]:.4f}|{100*r["nohead_macro_f1"]:.4f}|{r["delta_ba_pp"]:+.4f}|{r["delta_macro_f1_pp"]:+.4f}|')
    lines += ["", "## 身份与结果路径", "",
        "主 Covariance：原基准 0.2 → 本模型 0.0004；Point/Image CLS Covariance 仍各为 0.05。",
        "其他系数不变：主 Invariance=25、Variance=5、InfoNCE=1，Point/Image CLS Variance 各 1.25。", ""]
    code = chr(96)
    for name, identity in identities.items():
        lines += [f"**{name}**", ""]
        lines += [f"- {k}: {code}{v}{code}" for k, v in identity.items()]
        lines += [""]
    lines += [
        f"冻结协议 assets.json SHA256：{code}{reference['asset_manifest_sha256']}{code}。", "",
        f"导出耗时（含加载/验证/文件 hash）：{provenance['elapsed_seconds']:.2f} 秒；GPU 分配峰值 {provenance['peak_allocated_mib']/1024:.2f} GiB。",
        f"导出代码提交：{code}{provenance['code_commit']}{code}。评价代码提交：{code}{reference['code']['commit']}{code}。",
        "完整未舍入指标、每类 precision/recall、原始/归一化混淆矩阵在 comparison.json；逐实例和聚合 OOF、每折训练记录保存在各评价目录。",
        "", "## 混淆矩阵", "", "行是真实类别，列是预测类别；以下均为计数。", ""]
    for task, config in TASKS.items():
        for level in ["per_mito", "per_group"]:
            for name in paths:
                cm = result[name][task]["embedding"][level]["confusion"]
                lines += [f"### {task} / {level} / {name}", "",
                          "|真实 / 预测|"+"|".join(config["classes"])+"|",
                          "|---|"+"|".join(["---:"]*len(config["classes"]))+"|"]
                lines += ["|"+label+"|"+"|".join(map(str, row))+"|" for label, row in zip(config["classes"], cm)]
                lines += [""]
    lines += ["配对混淆矩阵图（上：实例；下：聚合；左：Fusion50；右：No-Head）：", ""]
    for task in TASKS:
        lines.append(f"- [{task}](comparison_{task}_confusion.png)")
    lines += ["", "各模型原始评价图：", ""]
    for name in paths:
        for task in TASKS:
            f = f"{name}_{task}_confusion.png"
            lines.append(f"- [{name} / {task}]({f})")
    (args.output / "report_zh.md").write_text("\n".join(lines)+"\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("REPORT", args.output / "report_zh.md")


if __name__ == "__main__":
    main()
