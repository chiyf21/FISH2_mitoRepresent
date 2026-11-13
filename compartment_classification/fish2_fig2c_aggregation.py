"""Fish2 binary adapter for the existing Figure-2C-style RF aggregation."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

from experiment import FEATURES, SEED, dump, sha
from fig2c_aggregation import EPS, score, run_oof_rf, aggregate_probabilities, plot_four_conditions

CLASSES = ["axon", "dendrite"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = args.output
    assert out.is_dir() and not any(out.iterdir()), "Use a new empty output directory"
    start = time.time()
    hp = args.dataset_root / "fish2_handcraft_81neurons_22_with_compartment.csv"
    ep = args.dataset_root / "fish2_model_embedding_81neurons_384.csv"
    h, e = [pd.read_csv(p, dtype={"record_id": str, "body_id": str, "source_run_id": str}) for p in [hp, ep]]
    assert len(h) == len(e) == 5568 and h.record_id.is_unique and e.record_id.is_unique
    df = h.merge(e, on="record_id", how="inner", validate="one_to_one", suffixes=("", "_emb"))
    assert len(df) == 5568
    for field in ["body_id", "cell_type", "compartment", "compartment_qc", "source_run_id", "global_instance"]:
        assert df[field].astype(str).tolist() == df[field + "_emb"].astype(str).tolist(), field
    df = df.loc[df.compartment_qc.eq("ok") & df.compartment.isin(CLASSES)].sort_values("record_id").reset_index(drop=True)
    assert len(df) == 5059 and df.body_id.nunique() == 81
    xs = {"handcrafted": df[FEATURES].to_numpy(dtype=np.float64),
          "embedding": df[[f"embedding_{i:03d}" for i in range(384)]].to_numpy(dtype=np.float64)}
    assert xs["handcrafted"].shape == (5059, 22) and xs["embedding"].shape == (5059, 384)
    assert all(np.isfinite(x).all() for x in xs.values())
    y = df.compartment.map({"axon": 0, "dendrite": 1}).to_numpy(dtype=int)
    assert df.groupby("body_id").cell_type.nunique().eq(1).all()
    bodies = df[["body_id", "cell_type"]].drop_duplicates().sort_values(["cell_type", "body_id"]).reset_index(drop=True)
    type_counts = bodies.cell_type.value_counts().sort_index()
    assert len(type_counts) == 6 and type_counts.min() >= 5, "A type cannot support five body folds"
    bodies["fold"] = -1
    for fold, (_, test) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(bodies)), bodies.cell_type)):
        bodies.loc[test, "fold"] = fold
    df["fold"] = df.body_id.map(dict(zip(bodies.body_id, bodies.fold)))
    manifest = df[["record_id", "body_id", "cell_type", "compartment", "source_run_id", "fold"]].copy()
    manifest["label"] = y
    manifest.to_csv(out / "instances.csv", index=False)
    bodies.to_csv(out / "body_folds.csv", index=False)
    coverage = manifest.groupby(["cell_type", "compartment"]).agg(instances=("record_id", "size"), bodies=("body_id", "nunique")).reset_index()
    coverage.to_csv(out / "coverage.csv", index=False)
    groups = manifest.groupby(["body_id", "compartment"], sort=True).agg(cell_type=("cell_type", "first"),
        fold=("fold", "first"), label=("label", "first"), n_mito=("record_id", "size")).reset_index()
    assert len(groups) == 161 and groups.n_mito.sum() == 5059
    groups.to_csv(out / "groups.csv", index=False)
    sizes = groups.n_mito
    stats = dict(instances=5059, bodies=81, groups=161, groups_by_class=groups.compartment.value_counts().to_dict(),
                 min=int(sizes.min()), q25=float(sizes.quantile(.25)), median=float(sizes.median()),
                 q75=float(sizes.quantile(.75)), max=int(sizes.max()), mean=float(sizes.mean()))
    dump(out / "group_size_distribution.json", stats)
    sources = [args.dataset_root.parent / name / "complete.json" for name in ["fusion50_all16428_20261112", "fusion50_fish2_14_20261112"]]
    exports = [json.loads(p.read_text()) for p in sources]
    assert all(r["checkpoint"].endswith("/refine_epoch050.pt") and r["dimension"] == 384 for r in exports)
    assert len({r["checkpoint"] for r in exports}) == 1 and len({r["checkpoint_sha256"] for r in exports}) == 1
    provenance = dict(dataset="Fish2", labels=CLASSES, qc="ok", seed=SEED, type_body_counts=type_counts.to_dict(),
        checkpoint=exports[0]["checkpoint"], checkpoint_sha256_recorded=exports[0]["checkpoint_sha256"],
        source_exports=exports, files={str(p.resolve()): sha(p) for p in [hp, ep] + sources}, features=FEATURES,
        reference_commit="099eabcbbb4b78019914f10d45b6723a501311d6",
        rf=dict(n_estimators=100, min_weight_fraction_leaf=.01, max_features="sqrt",
                sample_weight="inverse training compartment instance count", tune=False),
        aggregation="body_id + true compartment; argmax(mean(log10(max(p,1e-15)))) (oracle grouping)",
        split="Shared five body folds stratified by neuron type on all 81 bodies; no inner tuning or holdout",
        python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        source_hashes={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("fig2c_aggregation.py")]})
    dump(out / "provenance.json", provenance)
    results, training, numeric, fold_scores = {}, [], [], []
    paired = groups.copy()
    for mode in ["A_by_type", "B_pooled"]:
        for feature, x in xs.items():
            name = mode + "_" + feature
            probability, records = run_oof_rf(x, y, df, mode, args.workers, CLASSES, name)
            training.extend(records)
            instances = manifest.copy()
            for k, label in enumerate(CLASSES):
                instances["p_" + label] = probability[:, k]
            instances["prediction"] = probability.argmax(1)
            instances.to_csv(out / (name + "_instance_oof.csv"), index=False)
            prediction, literal, all_inf, zeros = aggregate_probabilities(df, groups, probability)
            paired[name + "_prediction"] = [CLASSES[i] for i in prediction]
            paired[name + "_correct"] = prediction == groups.label.to_numpy()
            paired[name + "_unclipped_prediction"] = [CLASSES[i] for i in literal]
            results[name] = score(groups.label, prediction, CLASSES)
            numeric.append(dict(condition=name, zero_probability_entries=zeros, literal_all_negative_infinity_groups=all_inf,
                clipped_vs_unclipped_disagreements=int(np.sum(prediction != literal)), literal_metrics=score(groups.label, literal, CLASSES)))
            for fold in range(5):
                mask = groups.fold.eq(fold)
                fold_scores.append(dict(condition=name, fold=fold, groups=int(mask.sum()),
                                        **score(groups.loc[mask, "label"], prediction[mask], CLASSES)))
            pd.DataFrame(results[name]["confusion"], index=CLASSES, columns=CLASSES).to_csv(out / (name + "_confusion_counts.csv"))
            pd.DataFrame(results[name]["confusion_normalized"], index=CLASSES, columns=CLASSES).to_csv(out / (name + "_confusion_normalized.csv"))
            print(json.dumps(dict(condition=name, **results[name])), flush=True)
    paired.to_csv(out / "paired_group_predictions.csv", index=False)
    comparisons = []
    for first, second in [("A_by_type_handcrafted", "A_by_type_embedding"), ("B_pooled_handcrafted", "B_pooled_embedding"),
                          ("A_by_type_handcrafted", "B_pooled_handcrafted"), ("A_by_type_embedding", "B_pooled_embedding")]:
        a, b = paired[first + "_correct"], paired[second + "_correct"]
        comparisons.append(dict(first=first, second=second,
            delta_ba=results[second]["balanced_accuracy"] - results[first]["balanced_accuracy"],
            delta_f1=results[second]["macro_f1"] - results[first]["macro_f1"],
            second_only_correct=int((~a & b).sum()), first_only_correct=int((a & ~b).sum()),
            both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum())))
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame([dict(condition=n, balanced_accuracy=r["balanced_accuracy"], macro_f1=r["macro_f1"],
                       **{"recall_" + c: r["recall"][c] for c in CLASSES}) for n, r in results.items()]).to_csv(out / "summary.csv", index=False)
    dump(out / "metrics.json", results)
    dump(out / "training_records.json", training)
    dump(out / "numerical_audit.json", numeric)
    plot_four_conditions(results, CLASSES,
        "Fish2 | 5-fold body OOF | oracle compartment grouping\n5059 mitochondria | 81 bodies | 161 groups",
        out / "four_conditions_confusion.png", by_type_label="neuron type")
    lines = ["# Fish2 二分类 oracle 聚合", "", "| 训练方式 | Handcrafted 22D：BA / Macro-F1 | Embedding 384D：BA / Macro-F1 |", "|---|---|---|"]
    for mode, title in [("A_by_type", "A：按神经元类型分别训练"), ("B_pooled", "B：混合类型训练")]:
        h, e = results[mode + "_handcrafted"], results[mode + "_embedding"]
        lines.append(f"| {title} | {h['balanced_accuracy']:.4f} / {h['macro_f1']:.4f} | {e['balanced_accuracy']:.4f} / {e['macro_f1']:.4f} |")
    lines += ["", "| 条件 | axon Recall | dendrite Recall |", "|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['recall']['axon']:.4f} | {r['recall']['dendrite']:.4f} |")
    lines += ["", "类型body数量："+json.dumps(type_counts.to_dict())+"。全部支持五折；训练缺类模型数："+str(sum(bool(r["missing_classes"]) for r in training))+
              "，测试缺类模型数："+str(sum(bool(r["missing_test_classes"]) for r in training))+"。未跳过任何实例或实际存在的区室组。",
              "", "组大小分布："+json.dumps(stats, ensure_ascii=False),
              "", "四组共享同一套全81个body的固定五折（按神经元类型分层，seed=20261113）。原二分类外层五折仅覆盖58个development bodies、另23个body外测，因此本次重新生成覆盖全体的OOF划分。A按已知类型路由，训练集中不混入其他类型；B混合训练，cell_type不作为数值输入。",
              "复用FlyEM的RF及聚合实现：100树、min_weight_fraction_leaf=0.01、训练类别计数倒数权重；无调参、不标准化、不补缺失值（所有数值有限）。聚合为body_id+真实compartment的oracle grouping，平均log10概率后argmax；log(0)下界固定1e-15。",
              "", "数值保护改变原未截断式预测的组数："+json.dumps({r["condition"]:r["clipped_vs_unclipped_disagreements"] for r in numeric}),
              "", "差异（后者−前者）："]
    for r in comparisons:
        lines.append(f"- {r['second']} − {r['first']}：ΔBA={r['delta_ba']:+.4f}，ΔMacro-F1={r['delta_f1']:+.4f}；后者/前者独有正确组={r['second_only_correct']}/{r['first_only_correct']}。")
    lines += ["", "边界：评价单位是已知真实区室的聚合组，不代表自动发现区室，不能直接与单线粒体或FlyEM三分类成绩比较。Fish2 reviewed标签仍缺完整人工审核来源追溯；手工特征包含神经元骨架长度和PCA几何；既有编码器的预训练暴露未重新控制。本次结论限于固定五折下游OOF。未运行Logistic Regression、三分类或编码器。",
              "", "文件：summary.csv含每类Recall；paired_group_predictions.csv含四组配对预测；groups.csv含每组大小；body_folds.csv、instances.csv记录固定划分；training_records.json记录每个RF的训练范围与类别覆盖；provenance.json记录epoch050输入身份与哈希。",
              "", "![四组混淆矩阵](four_conditions_confusion.png)", ""]
    (out / "report_zh.md").write_text("\n".join(lines))
    dump(out / "complete.json", dict(status="complete", dataset="Fish2", instances=5059, bodies=81, groups=161,
        fitted_rf=len(training), elapsed_seconds=time.time() - start, code_commit=provenance["code_commit"],
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name != "complete.json"}))


if __name__ == "__main__":
    main()
