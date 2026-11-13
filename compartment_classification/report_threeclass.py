"""Three-class report using the existing saved-prediction and body-bootstrap helpers."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import experiment as protocol
import summarize as previous
from experiment import dump, metrics, sha, table, SEED

CLASSES = ["axon", "dendrite", "connecting cable"]
protocol.CLASSES = CLASSES
previous.CLASSES = CLASSES


def class_scores(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    truth, predicted = cm.sum(-1), cm.sum(-2)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, truth, out=np.zeros_like(tp), where=truth != 0)
    f1 = np.divide(2 * tp, truth + predicted, out=np.zeros_like(tp), where=truth + predicted != 0)
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()
    data, dest = args.output / "flyem", args.output / "report"
    dest.mkdir(exist_ok=True)
    manifest = table(data / "instances.csv")
    provenance = json.loads((data / "data_provenance.json").read_text())
    assert provenance["classes"] == CLASSES and len(manifest) == 6400
    frames, completed, fold_rows = {}, {}, []
    for feature in ["handcrafted", "embedding", "dummy"]:
        completed[feature] = json.loads((data / feature / "complete.json").read_text())
        for classifier, result in completed[feature].items():
            frame = table(data / feature / classifier / "predictions.csv")
            assert frame.record_id.is_unique and set(frame.record_id) == set(manifest.record_id)
            for i, fold in enumerate(result["outer_folds"]):
                fold_rows.append(dict(feature=feature, classifier=classifier, fold=i,
                     balanced_accuracy=fold["metrics"]["balanced_accuracy"], macro_f1=fold["metrics"]["macro_f1"],
                     cable_precision=fold["metrics"]["precision"]["connecting cable"],
                     cable_recall=fold["metrics"]["recall"]["connecting cable"],
                     selected_parameters=json.dumps(fold["parameters"]), train_bodies=fold["train_bodies"],
                     test_bodies=fold["test_bodies"], test_instances=fold["test_instances"]))
            frames[feature, classifier] = frame
    rng = np.random.default_rng(SEED)
    summary, paired, rejected = [], [], {}
    for evaluation, partition in [("holdout", "holdout"), ("development_oof", "development")]:
        actual = manifest.loc[manifest.partition.eq(partition)]
        bodies = sorted(actual.body_id.unique())
        truth = np.array([[sum(actual.body_id.eq(b) & actual.label.eq(k)) for k in range(3)] for b in bodies])
        weights = rng.multinomial(len(bodies), np.full(len(bodies), 1 / len(bodies)), size=args.bootstrap)
        invalid = (weights @ truth == 0).any(1)
        count = 0
        while invalid.any():
            count += int(invalid.sum())
            weights[invalid] = rng.multinomial(len(bodies), np.full(len(bodies), 1 / len(bodies)), size=int(invalid.sum()))
            invalid = (weights @ truth == 0).any(1)
        rejected[evaluation] = count
        cubes = {}
        for (feature, classifier), frame in frames.items():
            subset = frame.loc[frame.evaluation.eq(evaluation)].sort_values("record_id")
            expected = actual.sort_values("record_id")
            assert subset.record_id.tolist() == expected.record_id.tolist()
            assert subset.body_id.tolist() == expected.body_id.tolist() and subset.label.tolist() == expected.label.tolist()
            cube = previous.body_matrices(subset, bodies)
            cubes[feature, classifier] = cube
            total = cube.sum(0)
            point = metrics(subset.label, subset.prediction)
            np.testing.assert_allclose(previous.scores(total), [point["balanced_accuracy"], point["macro_f1"]])
            draws = np.einsum("bi,ijk->bjk", weights, cube)
            draws_scores = previous.scores(draws)
            lo, hi = previous.interval(draws_scores)
            precision, recall, f1 = class_scores(total)
            cprec, crec, cf1 = class_scores(draws)
            row = dict(feature=feature, classifier=classifier, evaluation=evaluation, instances=len(subset), bodies=len(bodies),
                       balanced_accuracy=point["balanced_accuracy"], macro_f1=point["macro_f1"],
                       ba_ci_low=lo[0], ba_ci_high=hi[0], f1_ci_low=lo[1], f1_ci_high=hi[1])
            for k, name in enumerate(CLASSES):
                key = name.replace(" ", "_")
                row.update({key + "_precision": precision[k], key + "_recall": recall[k], key + "_f1": f1[k]})
            for key, values in [("cable_precision", cprec[:, 2]), ("cable_recall", crec[:, 2]), ("cable_f1", cf1[:, 2])]:
                row[key + "_ci_low"], row[key + "_ci_high"] = previous.interval(values)
            summary.append(row)
            for suffix, matrix in [("counts", total), ("normalized", total / total.sum(1, keepdims=True))]:
                pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(dest / f"{evaluation}_{feature}_{classifier}_{suffix}.csv")
        for classifier in ["logistic", "random_forest"]:
            h, e = cubes["handcrafted", classifier], cubes["embedding", classifier]
            hs = np.einsum("bi,ijk->bjk", weights, h)
            es = np.einsum("bi,ijk->bjk", weights, e)
            difference = previous.scores(e.sum(0)) - previous.scores(h.sum(0))
            lo, hi = previous.interval(previous.scores(es) - previous.scores(hs))
            row = dict(classifier=classifier, evaluation=evaluation, delta_balanced_accuracy=difference[0],
                       ba_ci_low=lo[0], ba_ci_high=hi[0], delta_macro_f1=difference[1],
                       f1_ci_low=lo[1], f1_ci_high=hi[1], bodies=len(bodies))
            hp, hr, hf = class_scores(h.sum(0))
            ep, er, ef = class_scores(e.sum(0))
            hpb, hrb, hfb = class_scores(hs)
            epb, erb, efb = class_scores(es)
            for name, point, draws in [("cable_precision", ep[2] - hp[2], epb[:, 2] - hpb[:, 2]),
                                       ("cable_recall", er[2] - hr[2], erb[:, 2] - hrb[:, 2]),
                                       ("cable_f1", ef[2] - hf[2], efb[:, 2] - hfb[:, 2])]:
                row["delta_" + name] = point
                row[name + "_ci_low"], row[name + "_ci_high"] = previous.interval(draws)
            paired.append(row)
            delta_body = previous.per_body_ba(e) - previous.per_body_ba(h)
            pd.DataFrame(dict(body_id=bodies, instances=h.sum((1, 2)), cable_instances=h[:, 2].sum(1),
                         handcrafted_ba=previous.per_body_ba(h), embedding_ba=previous.per_body_ba(e),
                         delta_ba=delta_body)).to_csv(dest / f"{evaluation}_{classifier}_paired_by_body.csv", index=False)
    s, p = pd.DataFrame(summary), pd.DataFrame(paired)
    s.to_csv(dest / "results_summary.csv", index=False)
    p.to_csv(dest / "paired_differences.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(dest / "fold_results.csv", index=False)
    dump(dest / "all_results.json", dict(provenance=provenance, completed=completed))
    dump(dest / "uncertainty_protocol.json", dict(seed=SEED, resamples=args.bootstrap, unit="body",
         method="paired percentile cluster bootstrap; same body multiplicities for both features",
         rejected_missing_true_class_draws=rejected, conditioning="retain all three true classes in each draw",
         scope="Fixed fitted models and labels; not encoder/probe retraining or label uncertainty",
         cable_holdout_support="94 instances from 6 of the 11 holdout bodies"))
    lines = ["# FlyEM 三分类：axon / dendrite / connecting cable", "",
             "直接复用固定epoch033全量向量和原二分类的body分区、全部外层/内层body折；仅新增cable实例，归入所属body原有分区。分类器与超参数网格不变，未训练或导出编码器。", "",
             "共6400实例/96 bodies，全部作者published_mito_exact_center标签；不含最近骨架节点迁移标签。",
             "development：5718实例/85 bodies（axon 2808、dendrite 2012、cable 898）；独立holdout：682实例/11 bodies（axon 340、dendrite 248、cable 94）。holdout的cable仅来自6个body。", "",
             "## 独立holdout主结果", "", "| 特征 | 分类器 | BA | Macro-F1 | Cable Precision | Cable Recall | Cable F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in s[s.evaluation.eq("holdout")].iterrows():
        lines.append(f"| {r.feature} | {r.classifier} | {r.balanced_accuracy:.4f} | {r.macro_f1:.4f} | {r.connecting_cable_precision:.4f} | {r.connecting_cable_recall:.4f} | {r.connecting_cable_f1:.4f} |")
    lines += ["", "## 配对差异：Embedding − Handcrafted", "",
              "5000次按body配对重采样的95%百分位区间；重点结论使用独立holdout，OOF另列。", "",
              "| 分类器 | ΔBA [95% CI] | ΔMacro-F1 [95% CI] | ΔCable Recall [95% CI] | ΔCable F1 [95% CI] |",
              "|---|---|---|---|---|"]
    for _, r in p[p.evaluation.eq("holdout")].iterrows():
        lines.append(f"| {r.classifier} | {r.delta_balanced_accuracy:+.4f} [{r.ba_ci_low:+.4f}, {r.ba_ci_high:+.4f}] | {r.delta_macro_f1:+.4f} [{r.f1_ci_low:+.4f}, {r.f1_ci_high:+.4f}] | {r.delta_cable_recall:+.4f} [{r.cable_recall_ci_low:+.4f}, {r.cable_recall_ci_high:+.4f}] | {r.delta_cable_f1:+.4f} [{r.cable_f1_ci_low:+.4f}, {r.cable_f1_ci_high:+.4f}] |")
    lines += ["", "## Cable错误去向", ""]
    for feature in ["handcrafted", "embedding"]:
        for classifier in ["logistic", "random_forest"]:
            cm = completed[feature][classifier]["holdout"]["metrics"]["confusion"]
            lines.append(f"- {feature}/{classifier}：94条cable中，预测为axon {cm[2][0]}、dendrite {cm[2][1]}、cable {cm[2][2]}。")
    lines += ["", "## development OOF", "", "| 特征 | 分类器 | BA | Macro-F1 | Cable Precision | Cable Recall | Cable F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in s[s.evaluation.eq("development_oof")].iterrows():
        lines.append(f"| {r.feature} | {r.classifier} | {r.balanced_accuracy:.4f} | {r.macro_f1:.4f} | {r.connecting_cable_precision:.4f} | {r.connecting_cable_recall:.4f} | {r.connecting_cable_f1:.4f} |")
    lines += ["", "## 对问题的回答", ""]
    for _, r in p[p.evaluation.eq("holdout")].iterrows():
        overall = "BA差异区间为正，支持本外测上的embedding优势" if r.ba_ci_low > 0 else ("BA差异区间为负，支持本外测上的手工特征优势" if r.ba_ci_high < 0 else "BA差异区间包含0，尚不能确认整体优势")
        cable = "cable recall差异区间为正" if r.cable_recall_ci_low > 0 else ("cable recall差异区间为负" if r.cable_recall_ci_high < 0 else "cable recall差异区间包含0")
        lines.append(f"- {r.classifier}：{overall}；{cable}。是否更易区分cable还应结合precision、F1和错误去向，不只看recall点估计。")
    lines += ["", "## 检查与限制", "",
              "- 已核实实例一对一匹配、三类在所有分区/内外折中覆盖、body隔离、训练子集内标准化与调参；两种特征共用6400实例和所有冻结折。未使用ID、cell_type、标签或分区元数据作为输入。",
              "- 22维手工特征含骨架长度及PCA几何量，不全是纯mask形态；本次RF和LR仍为单线粒体分类，不做聚合或按神经元类型拆分。",
              "- cable外测仅94实例、6个有cable的body，区间受body异质性影响较大。bootstrap仅包含固定预测下的body抽样不确定性，不含重训练、选模或标签不确定性。",
              "- 下游body隔离并不证明整个既有encoder从未暴露于测试body，沿用二分类报告的历史图像预训练限制。",
              "- 逐类Precision/Recall/F1、每折成绩、配对区间、混淆数量/行归一化矩阵均保存在本report目录；源资产、哈希、原划分身份见../flyem/data_provenance.json，全部类别覆盖见../flyem/class_coverage.csv。",
              "", "## Holdout 混淆矩阵", "", "![FlyEM holdout](holdout/flyem_holdout_confusion.png)",
              "", "## Development OOF 混淆矩阵", "", "![FlyEM development OOF](development_oof/flyem_development_oof_confusion.png)", ""]
    (dest / "report_zh.md").write_text("\n".join(lines))
    print(s.to_string(index=False))
    print(p.to_string(index=False))


if __name__ == "__main__":
    main()
