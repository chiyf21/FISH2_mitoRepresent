"""Verify and summarize the fixed three-representation Cell Type LR benchmark."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .common import BASE, ROOT, dump, read, sha, table
from .evaluate import metrics


def compare(actual, expected):
    for key in ["balanced_accuracy", "macro_f1"]:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual["confusion"], expected["confusion"])


def checked(directory, task, feature):
    if not (directory / "_SUCCESS.json").is_file():
        raise ValueError("Incomplete run: " + str(directory))
    record = read(directory / "run.json")
    expected = table(ROOT / f"assets/reviewed81_v1/{task}_instances.csv")
    frame = table(directory / task / f"{feature}_instance_oof.csv")
    pd.testing.assert_frame_equal(frame[expected.columns], expected)
    scores = read(directory / task / "metrics.json")[feature]
    classes = record["protocol"][task]["classes"]
    p = frame[["p_" + c for c in classes]].to_numpy()
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_array_equal(p.argmax(1), frame.prediction)
    compare(metrics(frame.label, frame.prediction, classes), scores["per_mito"])
    grouped = table(directory / task / f"{feature}_group_oof.csv")
    compare(metrics(grouped.label, grouped.prediction, classes), scores["per_group"])
    if task == "cell_type":
        assert len(frame) == 5568 and grouped.body_id.is_unique and len(grouped) == 81
        training = read(directory / task / f"{feature}_training.json")
        assert len(training) == 5 and all(row["scope"] == "all_types" for row in training)
        if record["classifier"] == "lr":
            assert all(max(row["n_iter"]) < 3000 for row in training)
    return scores, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["balanced-lr", "fused-lr", "fused-rf", "rf-reference", "compartment-lr", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    configurations = [
        ("Handcrafted22D", "handcrafted", args.balanced_lr, args.rf_reference),
        ("Balanced1152D", "embedding", args.balanced_lr, args.rf_reference),
        ("Fused384D", "embedding", args.fused_lr, args.fused_rf)]
    rows, iterations, run_records = [], [], {}
    for representation, feature, lr_path, rf_path in configurations:
        lr, lr_record = checked(lr_path, "cell_type", feature)
        rf, rf_record = checked(rf_path, "cell_type", feature)
        assert lr_record["asset_manifest_sha256"] == rf_record["asset_manifest_sha256"]
        assert lr_record["classifier"] == "lr" and rf_record["classifier"] == "rf"
        if feature == "embedding":
            assert lr_record["embedding"]["sha256"] == rf_record["embedding"]["sha256"]
            assert lr_record["embedding"]["checkpoint"] == rf_record["embedding"]["checkpoint"]
            assert lr_record["embedding"]["shape"][1] == (1152 if representation == "Balanced1152D" else 384)
        params = lr_record["classifier_parameters"]
        assert params["C"] == 1 and params["solver"] == "lbfgs"
        assert params["max_iter"] == 3000 and params["class_weight"] == "balanced"
        training = read(lr_path / "cell_type" / f"{feature}_training.json")
        iterations += [n for row in training for n in row["n_iter"]]
        for level in ["per_mito", "per_group"]:
            a, b = lr[level], rf[level]
            rows.append(dict(representation=representation, level=level,
                rf_ba=b["balanced_accuracy"], rf_macro_f1=b["macro_f1"],
                lr_ba=a["balanced_accuracy"], lr_macro_f1=a["macro_f1"],
                delta_ba=a["balanced_accuracy"]-b["balanced_accuracy"],
                delta_macro_f1=a["macro_f1"]-b["macro_f1"]))
        run_records[representation] = dict(lr=str(lr_path), rf=str(rf_path),
            lr_code=lr_record["code"]["commit"], rf_code=rf_record["code"]["commit"],
            checkpoint=lr_record["embedding"]["checkpoint"],
            descriptor=lr_record["embedding_descriptor"],
            run_sha256=sha(lr_path / "run.json"), rf_run_sha256=sha(rf_path / "run.json"))
    # RF numerical regression against the existing colleague's frozen protocol.
    old_rf = read(BASE / "point-MAE/runs/cell_type_fig2c_rf_20261113/results.json")["reviewed_81"]
    for representation, feature, lr_path, rf_path in configurations:
        previous = {"Handcrafted22D":"handcrafted_22", "Balanced1152D":"balanced_pair_12", "Fused384D":"fused_12"}[representation]
        old = next(v for v in old_rf["results"] if v["representation"] == previous)
        compare(read(rf_path / "cell_type/metrics.json")[feature]["per_group"], old)
    # Compartment LR still reproduces its existing fixed benchmark.
    comp, _ = checked(args.compartment_lr, "compartment", "embedding")
    old_comp = read(BASE / "compartment_classification/balanced_lr_20261113_v1/fish2/metrics.json")
    compare(comp["per_mito"], old_comp["per-mitochondrion/balanced"])
    compare(comp["per_group"], old_comp["per-compartment/balanced"])
    pd.DataFrame(rows).to_csv(out / "lr_vs_rf.csv", index=False)
    dump(out / "run_index.json", dict(owner="codex_benchmark", runs=run_records,
        compartment_lr_regression=str(args.compartment_lr),
        instances=5568, bodies=81, folds=5, seed=20261113,
        all_lr_models_converged=True, lr_models=15,
        min_iterations=min(iterations), max_iterations=max(iterations),
        rf_body_regression="all 3 representations reproduce historical RF",
        compartment_lr_regression_status="matches prior scores and confusion counts"))
    lines = ["# Fish2 Cell Type：固定LR基准", "",
        "reviewed81_v1：5568实例、81 bodies、6类。三种表示严格复用冻结五折body-disjoint划分（seed=20261113），不调参、不选最佳checkpoint。",
        "每折全部训练类型混合拟合一个多分类LR；训练折StandardScaler，L2、C=1、lbfgs、class_weight=balanced、max_iter=3000、random_state=SEED+fold。实例OOF后按body平均log10(clip(p,1e-15,1))取argmax。",
        "RF采用原100树/叶权重0.01/逆训练类别计数sample_weight协议，不标准化。两分类器共享实例、标签、fold及聚合定义。",
        "",
        "| 表示 | 评价 | RF BA | RF Macro-F1 | LR BA | LR Macro-F1 | Δ BA | Δ Macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        level = "实例" if row["level"] == "per_mito" else "body聚合"
        lines.append(f"| {row['representation']} | {level} | {row['rf_ba']:.4f} | {row['rf_macro_f1']:.4f} | "
            f"{row['lr_ba']:.4f} | {row['lr_macro_f1']:.4f} | {row['delta_ba']:+.4f} | {row['delta_macro_f1']:+.4f} |")
    lines += ["", "Δ=LR−RF，0–1分数绝对差值。没有进行统计显著性检验。",
        f"15次LR拟合全部收敛，迭代次数{min(iterations)}–{max(iterations)}；7项单元测试通过。Compartment LR与三组Cell Type RF历史回归通过：BA/Macro-F1容差1e-12，混淆矩阵计数完全一致。", "",
        "Balanced与Fused来自同一Fish2 Fusion50 refine_epoch050.pt；仅复用已注册向量和实例映射，没有Encoder推理。Handcrafted使用固定22列。"]
    for representation, info in run_records.items():
        lines += ["", f"**{representation}**", "", f"- LR运行：{info['lr']}", f"- RF运行：{info['rf']}",
                  f"- LR代码提交：{info['lr_code']}", f"- 向量descriptor：{info['descriptor']}"]
    lines += ["", "RF对照来源：Handcrafted/Balanced复用既有已核验OOF；Fused补算同协议RF OOF以补齐实例指标，body混淆矩阵/BA/Macro-F1复现原历史结果。旧RF运行和冻结资产均未覆盖。",
        "本固定基准中，Balanced与Fused的LR body级BA和Macro-F1均高于对应RF；因此建议将这套固定LR作为当前learned embedding的线性评价方法，并保留RF非线性对照。Handcrafted没有同样的BA提升，不能泛化为所有表示或所有新checkpoint上LR都更优。没有进行超参数搜索或模型选择。",
        "每个运行均保存owner、checkpoint/embedding身份与hash、冻结实例/fold、实例及body OOF、混淆矩阵、指标、训练记录和完整日志。"]
    (out / "README_zh.md").write_text("\n".join(lines) + "\n")
    dump(out / "_SUCCESS.json", dict(status="complete"))
    print(pd.DataFrame(rows).to_string(index=False))
    print("CONVERGED", len(iterations), min(iterations), max(iterations))
    print("REPORT", out)


if __name__ == "__main__":
    main()
