"""Aggregate held-out predictions, paired body bootstrap, and confusion figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment import CLASSES, SEED, dump, metrics, sha, table


def scores(c):
    c = np.asarray(c, dtype=float)
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    support, predicted = c.sum(-1), c.sum(-2)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=support + predicted != 0)
    return np.stack([rec.mean(-1), f1.mean(-1)], axis=-1)


def body_matrices(frame, bodies):
    matrices = []
    for body in bodies:
        subset = frame.loc[frame.body_id.eq(body)]
        cm = np.zeros((2, 2), dtype=int)
        np.add.at(cm, (subset.label.to_numpy(dtype=int), subset.prediction.to_numpy(dtype=int)), 1)
        matrices.append(cm)
    return np.stack(matrices)


def interval(values):
    return np.quantile(values, [0.025, 0.975], axis=0).tolist()


def per_body_ba(c):
    true = c.sum(-1)
    recalls = np.divide(np.diagonal(c, axis1=-2, axis2=-1), true,
                        out=np.zeros_like(true, dtype=float), where=true != 0)
    return recalls.sum(-1) / (true > 0).sum(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["fish2", "flyem"])
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()
    dest = args.output / "report"
    dest.mkdir(exist_ok=True)
    summary, fold_rows, paired, all_results = [], [], [], {}
    rng = np.random.default_rng(SEED)
    for dataset in args.datasets:
        data = args.output / dataset
        provenance = json.loads((data / "data_provenance.json").read_text())
        manifest = table(data / "instances.csv")
        frames = {}
        all_results[dataset] = dict(data=provenance, metrics={})
        for feature in ["handcrafted", "embedding", "dummy"]:
            completed = json.loads((data / feature / "complete.json").read_text())
            for classifier, result in completed.items():
                frame = table(data / feature / classifier / "predictions.csv")
                assert len(frame) == len(manifest) and frame.record_id.is_unique
                assert set(frame.record_id) == set(manifest.record_id)
                check = frame.merge(manifest, on="record_id", validate="one_to_one", suffixes=("", "_original"))
                for col in ["body_id", "label", "partition", "outer_fold"]:
                    assert (check[col].astype(str) == check[col + "_original"].astype(str)).all()
                frames[feature, classifier] = frame
                all_results[dataset]["metrics"][feature + "/" + classifier] = result
                for i, fold in enumerate(result["outer_folds"]):
                    fold_rows.append(dict(dataset=dataset, feature=feature, classifier=classifier, fold=i,
                                          train_instances=fold["train_instances"], test_instances=fold["test_instances"],
                                          train_bodies=fold["train_bodies"], test_bodies=fold["test_bodies"],
                                          selected_parameters=json.dumps(fold["parameters"]), **fold["metrics"]))
        for evaluation in ["holdout", "development_oof"]:
            bodies = sorted(manifest.loc[manifest.partition.eq("holdout" if evaluation == "holdout" else "development"), "body_id"].unique())
            weights = rng.multinomial(len(bodies), np.full(len(bodies), 1 / len(bodies)), size=args.bootstrap)
            cubes = {}
            for (feature, classifier), frame in frames.items():
                f = frame.loc[frame.evaluation.eq(evaluation)].sort_values("record_id")
                cube = body_matrices(f, bodies)
                cubes[feature, classifier] = cube
                draws = scores(np.einsum("bi,ijk->bjk", weights, cube))
                lo, hi = interval(draws)
                point = metrics(f.label, f.prediction)
                row = dict(dataset=dataset, feature=feature, classifier=classifier, evaluation=evaluation,
                           instances=len(f), bodies=len(bodies), balanced_accuracy=point["balanced_accuracy"],
                           macro_f1=point["macro_f1"], ba_ci_low=lo[0], ba_ci_high=hi[0],
                           f1_ci_low=lo[1], f1_ci_high=hi[1],
                           precision_axon=point["precision"]["axon"], recall_axon=point["recall"]["axon"],
                           precision_dendrite=point["precision"]["dendrite"], recall_dendrite=point["recall"]["dendrite"])
                summary.append(row)
                prefix = dest / (dataset + "_" + evaluation + "_" + feature + "_" + classifier)
                pd.DataFrame(point["confusion"], index=CLASSES, columns=CLASSES).to_csv(str(prefix) + "_counts.csv")
                pd.DataFrame(point["confusion_normalized"], index=CLASSES, columns=CLASSES).to_csv(str(prefix) + "_normalized.csv")
            for classifier in ["logistic", "random_forest"]:
                h, e = cubes["handcrafted", classifier], cubes["embedding", classifier]
                difference = scores(e.sum(0)) - scores(h.sum(0))
                draws = scores(np.einsum("bi,ijk->bjk", weights, e)) - scores(np.einsum("bi,ijk->bjk", weights, h))
                lo, hi = interval(draws)
                by_body = per_body_ba(e) - per_body_ba(h)
                body_draws = weights @ by_body / len(bodies)
                blo, bhi = interval(body_draws)
                per = pd.DataFrame(dict(body_id=bodies, instances=h.sum((1, 2)), axon=h[:, 0].sum(1), dendrite=h[:, 1].sum(1),
                                        handcrafted_ba=per_body_ba(h), embedding_ba=per_body_ba(e), delta_ba=by_body))
                per.to_csv(dest / (dataset + "_" + evaluation + "_" + classifier + "_paired_by_body.csv"), index=False)
                paired.append(dict(dataset=dataset, classifier=classifier, evaluation=evaluation, bodies=len(bodies),
                                   delta_balanced_accuracy=difference[0], ba_ci_low=lo[0], ba_ci_high=hi[0],
                                   delta_macro_f1=difference[1], f1_ci_low=lo[1], f1_ci_high=hi[1],
                                   mean_body_delta_ba=float(by_body.mean()), body_delta_ci_low=blo, body_delta_ci_high=bhi))
            fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
            for i, classifier in enumerate(["logistic", "random_forest"]):
                for j, feature in enumerate(["handcrafted", "embedding"]):
                    cm = cubes[feature, classifier].sum(0)
                    norm = cm / cm.sum(1, keepdims=True)
                    ax = axes[i, j]
                    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
                    for a in range(2):
                        for b in range(2):
                            ax.text(b, a, f"{cm[a,b]}\n{norm[a,b]:.1%}", ha="center", va="center",
                                    color="white" if norm[a,b] > 0.55 else "black", fontsize=13)
                    ax.set_xticks([0, 1], CLASSES)
                    ax.set_yticks([0, 1], CLASSES)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title(feature + " / " + classifier.replace("_", " "))
            fig.suptitle(f"{dataset.upper()} / {evaluation.replace('_', ' ')} / n={sum(cubes['handcrafted','logistic'].sum((1,2)))}, bodies={len(bodies)}")
            fig.savefig(dest / (dataset + "_" + evaluation + "_confusion.png"), dpi=180)
            plt.close(fig)
    s = pd.DataFrame(summary)
    p = pd.DataFrame(paired)
    s.to_csv(dest / "results_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(dest / "fold_results.csv", index=False)
    p.to_csv(dest / "paired_differences.csv", index=False)
    dump(dest / "all_results.json", all_results)
    dump(dest / "uncertainty_protocol.json", dict(
        seed=SEED, resamples=args.bootstrap, unit="body", method="paired cluster percentile bootstrap, identical body multiplicities for both feature sets",
        conditional_scope="Fixed fitted models/predictions; does not include encoder retraining, probe refitting, model selection, or label uncertainty.",
        per_body_ba="Mean recall over classes actually present in that body; pooled BA is a separate metric."))
    lines = [
        "# Fish2 × FlyEM 单线粒体 compartment classification",
        "",
        "主结果为保留的独立body测试集；五折外层交叉验证仅在development bodies中开展，每个训练折用三折内层分组CV选择超参数。最终外测模型也仅在development中调参。两种特征共享全部实例与划分。",
        "",
        "| 数据集 | Handcrafted 22维 | Learned 384维 |",
        "|---|---|---|",
    ]
    def fmt(dataset, feature, evaluation="holdout"):
        sub = s[(s.dataset == dataset) & (s.feature == feature) & (s.evaluation == evaluation)]
        parts = []
        for c, label in [("logistic", "LR"), ("random_forest", "RF")]:
            row = sub.loc[sub.classifier.eq(c)].iloc[0]
            parts.append(f"{label}: BA={row.balanced_accuracy:.4f}, Macro-F1={row.macro_f1:.4f}")
        return "; ".join(parts)
    for dataset in args.datasets:
        lines.append(f"| {dataset} | {fmt(dataset, 'handcrafted')} | {fmt(dataset, 'embedding')} |")
    lines += ["", "## 样本和独立性", ""]
    for dataset in args.datasets:
        prov = all_results[dataset]["data"]
        parts = prov["partition_summary"]
        lines.append(f"- {dataset}: 总计{prov['shape_handcrafted'][0]}实例；development {parts['development']['instances']}实例/{parts['development']['bodies']} bodies，独立外测{parts['holdout']['instances']}实例/{parts['holdout']['bodies']} bodies。标签数：{json.dumps(parts, ensure_ascii=False)}")
    lines += ["", "## 五折development结果", "", "| 数据集 | Handcrafted（折均值±SD） | Learned（折均值±SD） |", "|---|---|---|"]
    for dataset in args.datasets:
        cells = []
        for feature in ["handcrafted", "embedding"]:
            result = all_results[dataset]["metrics"]
            cells.append("; ".join(
                f"{label}: BA={result[feature+'/'+c]['outer_mean']['balanced_accuracy']:.4f}±{result[feature+'/'+c]['outer_sd']['balanced_accuracy']:.4f}, F1={result[feature+'/'+c]['outer_mean']['macro_f1']:.4f}±{result[feature+'/'+c]['outer_sd']['macro_f1']:.4f}"
                for c, label in [("logistic", "LR"), ("random_forest", "RF")]))
        lines.append(f"| {dataset} | {cells[0]} | {cells[1]} |")
    lines += ["", "## 配对差异：Embedding − Handcrafted", "", "下列为独立外测的总体BA差，按body配对重采样5000次的95%百分位区间；包含0时，不宜据此断言性能有稳定差异。", ""]
    for row in paired:
        if row["evaluation"] == "holdout":
            lines.append(f"- {row['dataset']} / {row['classifier']}: ΔBA={row['delta_balanced_accuracy']:+.4f} [{row['ba_ci_low']:+.4f}, {row['ba_ci_high']:+.4f}]；ΔMacro-F1={row['delta_macro_f1']:+.4f} [{row['f1_ci_low']:+.4f}, {row['f1_ci_high']:+.4f}]；body等权平均ΔBA={row['mean_body_delta_ba']:+.4f}。")
    lines += ["", "## 外测每类Precision / Recall", "", "| 数据集 | 特征 | 分类器 | axon P/R | dendrite P/R |", "|---|---|---|---|---|"]
    for row in summary:
        if row["evaluation"] == "holdout" and row["feature"] != "dummy":
            lines.append(f"| {row['dataset']} | {row['feature']} | {row['classifier']} | {row['precision_axon']:.4f} / {row['recall_axon']:.4f} | {row['precision_dendrite']:.4f} / {row['recall_dendrite']:.4f} |")
    lines += ["", "## 当前证据支持的结论", ""]
    for row in paired:
        if row["evaluation"] == "holdout":
            verdict = "该外测中embedding的BA优势区间为正" if row["ba_ci_low"] > 0 else ("该外测中手工特征的BA优势区间为正" if row["ba_ci_high"] < 0 else "该外测未能确认两种特征有稳定BA差异")
            lines.append(f"- {row['dataset']} / {row['classifier']}：{verdict}；这是当前body样本与固定模型下的条件证据。")
    lines += ["", "## Dummy", ""]
    for row in summary:
        if row["feature"] == "dummy" and row["evaluation"] == "holdout":
            lines.append(f"- {row['dataset']}: BA={row['balanced_accuracy']:.4f}, Macro-F1={row['macro_f1']:.4f}。")
    lines += [
        "", "## 证据与限制", "",
        "- Fish2使用已导出的Fusion50 epoch050；FlyEM固定使用refine_both_cls_ln epoch033中间checkpoint，不继续训练、不自动换版本。FlyEM为正式canonical导出，未使用512实例监控向量替代。",
        "- 只使用axon/dendrite；Fish2要求compartment_qc=ok，FlyEM仅用作者exact-center匹配标签。Fish2人工审核源文件/revision追溯仍不完整。",
        "- 手工22维包含沿神经元骨架长度，以及PCA主轴长度、惯性、对称性和截面等量，不能都称为纯mask形态。未将标签、ID、cell_type、投影距离或QC字段作为输入。",
        "- 数据集单位不同，分别拟合。自监督训练配置也不同；跨数据集分数差异不能解释为某个单因素的因果效应。",
        "- 分类划分按body完全隔离，所有标准化仅拟合训练子集，RF不标准化。没有重新使用外测标签调参。",
        "- 保留已有encoder测试body可减少暴露混淆，但并不证明整个冻结encoder从未接触测试body；历史图像预训练暴露仍是限制。development OOF尤其不能称为对完全未见body的整个预训练系统评价。",
        "- 评价单位是单个线粒体，不与Science论文的神经突段概率聚合成绩直接比较；RF为本次实例级模型。",
        "- bootstrap仅描述固定模型在现有测试body上的条件不确定性，不包含重新训练、选模或标签不确定性；FlyEM独立外测仅11个body，区间应谨慎解读。",
        "", "## 产物入口", "",
        "- results_summary.csv：每类precision/recall、BA/F1及body bootstrap区间；fold_results.csv：每折成绩与超参数。",
        "- paired_differences.csv与*_paired_by_body.csv：总体配对区间与逐body差异。",
        "- *_counts.csv、*_normalized.csv、*_confusion.png：混淆数量及按真实类别归一化。",
        "- ../<dataset>/instances.csv、splits.json、data_provenance.json：身份、标签、冻结划分与文件哈希；各feature目录保留模型、调参轨迹、预测和运行环境。",
        "- ../flyem_embeddings/{fused_12.npy,instance_map.csv,provenance.json,complete.json}：正式向量及固定checkpoint身份。",
        "",
    ]
    lines += ["", "## 固定资产和版本", ""]
    for dataset in args.datasets:
        prov = all_results[dataset]["data"]
        lines.append(f"- {dataset} checkpoint: "+str(prov["checkpoint"]))
        for path, info in prov["source_files"].items():
            lines.append(f"  - {path}；SHA256={info['sha256']}")
    export_record = args.output / "flyem_embeddings/complete.json"
    if export_record.exists():
        details = json.loads(export_record.read_text())
        lines.append("- FlyEM full export: "+json.dumps(details, ensure_ascii=False))
        lines.append("- 导出期间发现其他进程进入同卡后暂停自身推理，随后恢复到独占空闲卡；恢复保留了已验证前缀，完整来源历史和进程记录保留在输出目录。")
    for dataset in args.datasets:
        lines += [f"## {dataset} 混淆矩阵", "", f"![{dataset} independent holdout]({dataset}_holdout_confusion.png)", ""]
    (dest / "report_zh.md").write_text("\n".join(lines))
    files = [x for x in sorted(dest.iterdir()) if x.is_file() and x.name != "artifact_hashes.json"]
    dump(dest / "artifact_hashes.json", {x.name: sha(x) for x in files})
    print(s.to_string(index=False))
    print(p.to_string(index=False))


if __name__ == "__main__":
    main()
