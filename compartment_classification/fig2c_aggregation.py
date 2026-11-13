"""Lightweight Figure-2C-style three-class RF with shared body OOF folds."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from experiment import FEATURES, SEED, dump, sha

CLASSES = ["axon", "dendrite", "connecting cable"]
EPS = 1e-15
AUTHOR_URL = "https://github.com/ClarkLabCode/MitochondrialMorphologyPosition"
NOTEBOOK = "morphology classifier - LC neurons.ipynb"


def score(y, prediction, classes=CLASSES):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y, prediction, labels=labels)
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
                recall=dict(zip(classes, recall_score(y, prediction, labels=labels, average=None, zero_division=0).tolist())),
                confusion=cm.tolist(),
                confusion_normalized=np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True) > 0).tolist())


def run_oof_rf(x, y, df, mode, workers, classes, condition):
    training = []
    probability = np.full((len(df), len(classes)), np.nan)
    seen = np.zeros(len(df), dtype=int)
    for fold in range(5):
        scopes = sorted(df.cell_type.unique()) if mode == "A_by_type" else ["all"]
        for scope in scopes:
            domain = df.cell_type.eq(scope).to_numpy() if scope != "all" else np.ones(len(df), dtype=bool)
            train = np.flatnonzero((df.fold.to_numpy() != fold) & domain)
            test = np.flatnonzero((df.fold.to_numpy() == fold) & domain)
            assert len(train) and len(test)
            assert not set(df.body_id.iloc[train]) & set(df.body_id.iloc[test])
            if scope != "all":
                assert set(df.cell_type.iloc[train]) == {scope} and set(df.cell_type.iloc[test]) == {scope}
            counts = np.bincount(y[train], minlength=len(classes))
            weights = 1.0 / counts[y[train]]
            totals = np.bincount(y[train], weights=weights, minlength=len(classes))
            np.testing.assert_allclose(totals[counts > 0], 1.0)
            model = RandomForestClassifier(n_estimators=100, min_weight_fraction_leaf=.01,
                random_state=SEED + fold, n_jobs=workers, class_weight=None, max_features="sqrt")
            model.fit(x[train], y[train], sample_weight=weights)
            probability[np.ix_(test, model.classes_.astype(int))] = model.predict_proba(x[test])
            missing = np.setdiff1d(list(range(len(classes))), model.classes_)
            if len(missing):
                probability[np.ix_(test, missing)] = 0.0
            seen[test] += 1
            training.append(dict(condition=condition, fold=fold, scope=scope, train_instances=len(train), test_instances=len(test),
                                 train_bodies=df.body_id.iloc[train].nunique(), test_bodies=df.body_id.iloc[test].nunique(),
                                 class_counts=counts.tolist(), missing_classes=[classes[i] for i in missing],
                                 test_class_counts=np.bincount(y[test], minlength=len(classes)).tolist(),
                                 missing_test_classes=[classes[i] for i in range(len(classes)) if not np.any(y[test] == i)]))
    assert np.all(seen == 1) and np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(1), 1.0)
    return probability, training


def aggregate_probabilities(df, groups, probability):
    group_predictions, literal_predictions, all_inf, zeros = [], [], 0, int((probability == 0).sum())
    for _, group in groups.iterrows():
        mask = df.body_id.eq(group.body_id).to_numpy() & df.compartment.eq(group.compartment).to_numpy()
        p = probability[mask]
        log_mean = np.log10(np.clip(p, EPS, 1)).mean(0)
        with np.errstate(divide="ignore"):
            raw_mean = np.log10(p).mean(0)
        all_inf += int(np.isneginf(raw_mean).all())
        group_predictions.append(int(log_mean.argmax()))
        literal_predictions.append(int(raw_mean.argmax()))
    prediction = np.array(group_predictions)
    return prediction, np.array(literal_predictions), all_inf, zeros


def plot_four_conditions(results, classes, title, path, by_type_label="LC type"):
    """Identical figure style for binary and three-class oracle aggregation."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.subplots_adjust(left=.15, right=.98, bottom=.10, top=.83, wspace=.5, hspace=.58)
        ticks = list(range(len(classes)))
        labels = [c.replace("connecting cable", "connecting\ncable") for c in classes]
        for ax, (name, result) in zip(axes.flat, results.items()):
            cm, norm = np.array(result["confusion"]), np.array(result["confusion_normalized"])
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in ticks:
                for j in ticks:
                    ax.text(j, i, f"{cm[i,j]}\n{norm[i,j]:.1%}", ha="center", va="center",
                            color="white" if norm[i,j] > .55 else "black", fontsize=12)
            ax.set_xticks(ticks, labels)
            ax.set_yticks(ticks, labels)
            ax.set_xlabel("Predicted compartment")
            ax.set_ylabel("True compartment")
            label = name.replace("A_by_type", "A: By " + by_type_label).replace("B_pooled", "B: Pooled").replace("_handcrafted", " / Handcrafted 22D").replace("_embedding", " / Embedding 384D")
            ax.set_title(label + f"\nBA={result['balanced_accuracy']:.3f}; Macro-F1={result['macro_f1']:.3f}", fontsize=11)
        fig.suptitle(title, fontsize=15, y=.97)
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--previous", type=Path, required=True)
    ap.add_argument("--handcrafted", type=Path, required=True)
    ap.add_argument("--author-repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = args.output
    assert out.is_dir() and not any(out.iterdir()), "Use a new empty output directory"
    started = time.time()
    hp = args.handcrafted
    ep = args.previous / "flyem_embeddings/fused_12.npy"
    mp = ep.parent / "instance_map.csv"
    export = json.loads((ep.parent / "provenance.json").read_text())
    complete = json.loads((ep.parent / "complete.json").read_text())
    assert export["epoch"] == 33 and "refine_both_cls_ln/checkpoints/refine_epoch033.pt" in export["checkpoint"]
    assert complete["status"] == "complete" and complete["shape"] == [9731, 384]
    assert sha(ep) == complete["fused_12_sha256"] and sha(mp) == complete["instance_map_sha256"]
    h = pd.read_csv(hp, dtype={"record_id": str, "body_id": str, "source_mito_label": str})
    m = pd.read_csv(mp, dtype={"record_id": str, "body_id": str})
    assert len(h) == 6400 and h.record_id.is_unique and h.source_mito_label.is_unique
    assert set(h.compartment) == set(CLASSES)
    assert len(m) == 9731 and m.record_id.is_unique and m.embedding_row.tolist() == list(range(9731))
    df = h.merge(m, on=["body_id", "record_id"], validate="one_to_one", how="inner").sort_values("record_id").reset_index(drop=True)
    assert len(df) == 6400 and df.body_id.nunique() == 96
    assert df.groupby("body_id").cell_type.nunique().eq(1).all()
    columns = ["length along skeleton" if c == "length_along_skeleton_nm" else c for c in FEATURES]
    xs = {"handcrafted": df[columns].to_numpy(dtype=np.float64),
          "embedding": np.array(np.load(ep, mmap_mode="r")[df.embedding_row.to_numpy(dtype=int)], dtype=np.float64)}
    assert xs["handcrafted"].shape == (6400, 22) and xs["embedding"].shape == (6400, 384)
    assert all(np.isfinite(x).all() for x in xs.values())
    y = df.compartment.map(dict(zip(CLASSES, range(3)))).to_numpy(dtype=int)
    bodies = df[["body_id", "cell_type"]].drop_duplicates().sort_values(["cell_type", "body_id"]).reset_index(drop=True)
    assert bodies.cell_type.nunique() == 6 and bodies.cell_type.value_counts().min() >= 5
    bodies["fold"] = -1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, test) in enumerate(cv.split(np.zeros(len(bodies)), bodies.cell_type)):
        bodies.loc[test, "fold"] = fold
    body_fold = dict(zip(bodies.body_id, bodies.fold))
    df["fold"] = df.body_id.map(body_fold)
    assert df.groupby("body_id").fold.nunique().eq(1).all()
    manifest = df[["record_id", "body_id", "cell_type", "compartment", "embedding_row", "fold"]].copy()
    manifest["label"] = y
    manifest.to_csv(out / "instances.csv", index=False)
    bodies.to_csv(out / "body_folds.csv", index=False)
    groups = manifest.groupby(["body_id", "compartment"], sort=True).agg(
        cell_type=("cell_type", "first"), fold=("fold", "first"), label=("label", "first"),
        n_mito=("record_id", "size")).reset_index()
    assert len(groups) == 212 and groups.n_mito.sum() == 6400
    groups.to_csv(out / "groups.csv", index=False)
    sizes = groups.n_mito
    size_stats = dict(groups=len(groups), bodies=96, mitochondria=6400,
                      group_class_counts=groups.compartment.value_counts().to_dict(),
                      min=int(sizes.min()), q25=float(sizes.quantile(.25)), median=float(sizes.median()),
                      q75=float(sizes.quantile(.75)), max=int(sizes.max()), mean=float(sizes.mean()))
    dump(out / "group_size_distribution.json", size_stats)
    coverage = manifest.groupby(["cell_type", "compartment"]).agg(instances=("record_id", "size"), bodies=("body_id", "nunique")).reset_index()
    coverage.to_csv(out / "coverage.csv", index=False)
    notebook = subprocess.check_output(["git", "-C", str(args.author_repo), "show", "upstream/main:" + NOTEBOOK])
    ref_commit = subprocess.check_output(["git", "-C", str(args.author_repo), "rev-parse", "upstream/main"]).decode().strip()
    provenance = dict(seed=SEED, classes=CLASSES, probability_floor=EPS,
        RF=dict(n_estimators=100, min_weight_fraction_leaf=.01, max_features="sqrt", bootstrap=True,
                class_weight=None, sample_weight="1 / count of true compartment within this model's training rows"),
        aggregation="argmax of within-group mean log10(max(p,1e-15)); body_id + true compartment (oracle grouping)",
        splits="Five folds on all 96 bodies, stratified by LC type; same assignments for all four conditions",
        old_split_reason="Existing outer folds cover only 85 development bodies and reserve 11 holdout bodies; not full-cohort per-type OOF.",
        preprocessing="No scaling/imputation for RF; all features finite. Original notebook used full-cohort z-scores.",
        reference=dict(repository=AUTHOR_URL, notebook=NOTEBOOK, commit=ref_commit,
                       sha256=hashlib.sha256(notebook).hexdigest(), saved_run_LC_types=21),
        checkpoint=export["checkpoint"], checkpoint_sha256=export["checkpoint_sha256"],
        files={str(p.resolve()): sha(p) for p in [hp, ep, mp, ep.parent / "provenance.json"]},
        features=columns, python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        source_sha256=sha(__file__), started_unix=started)
    dump(out / "provenance.json", provenance)
    results, training, fold_scores, numeric = {}, [], [], []
    paired = groups.copy()
    for mode in ["A_by_type", "B_pooled"]:
        for feature, x in xs.items():
            name = mode + "_" + feature
            probability, fitted = run_oof_rf(x, y, df, mode, args.workers, CLASSES, name)
            training.extend(fitted)
            instances = manifest.copy()
            for k, label in enumerate(CLASSES):
                instances["p_" + label.replace(" ", "_")] = probability[:, k]
            instances["prediction"] = probability.argmax(1)
            instances.to_csv(out / (name + "_instance_oof.csv"), index=False)
            prediction, literal_predictions, all_inf, zeros = aggregate_probabilities(df, groups, probability)
            paired[name + "_prediction"] = [CLASSES[k] for k in prediction]
            paired[name + "_correct"] = prediction == groups.label.to_numpy()
            paired[name + "_unclipped_prediction"] = [CLASSES[k] for k in literal_predictions]
            results[name] = score(groups.label, prediction)
            numeric.append(dict(condition=name, zero_probability_entries=zeros, literal_all_negative_infinity_groups=all_inf,
                                clipped_vs_unclipped_disagreements=int(np.sum(prediction != literal_predictions)),
                                literal_unclipped_metrics=score(groups.label, literal_predictions)))
            for fold in range(5):
                mask = groups.fold.eq(fold)
                fold_scores.append(dict(condition=name, fold=fold, groups=int(mask.sum()), **score(groups.loc[mask, "label"], prediction[mask])))
            cm = np.array(results[name]["confusion"])
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / (name + "_confusion_counts.csv"))
            pd.DataFrame(results[name]["confusion_normalized"], index=CLASSES, columns=CLASSES).to_csv(out / (name + "_confusion_normalized.csv"))
            print(json.dumps(dict(condition=name, **results[name])), flush=True)
    paired.to_csv(out / "paired_group_predictions.csv", index=False)
    comparisons = []
    pairs = [("A_by_type_handcrafted", "A_by_type_embedding"), ("B_pooled_handcrafted", "B_pooled_embedding"),
             ("A_by_type_handcrafted", "B_pooled_handcrafted"), ("A_by_type_embedding", "B_pooled_embedding")]
    for first, second in pairs:
        a, b = paired[first + "_correct"], paired[second + "_correct"]
        comparisons.append(dict(first=first, second=second, delta_ba=results[second]["balanced_accuracy"] - results[first]["balanced_accuracy"],
                                delta_f1=results[second]["macro_f1"] - results[first]["macro_f1"],
                                second_only_correct=int((~a & b).sum()), first_only_correct=int((a & ~b).sum()),
                                both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum())))
    dump(out / "metrics.json", results)
    dump(out / "training_records.json", training)
    dump(out / "numerical_audit.json", numeric)
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(out / "fold_metrics.csv", index=False)
    rows = [dict(condition=name, balanced_accuracy=v["balanced_accuracy"], macro_f1=v["macro_f1"],
                 **{"recall_" + k.replace(" ", "_"): r for k, r in v["recall"].items()}) for name, v in results.items()]
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    plot_four_conditions(results, CLASSES,
        "FlyEM | 5-fold body OOF | oracle compartment grouping\n6400 mitochondria | 96 bodies | 212 groups",
        out / "four_conditions_confusion.png")
    lines = ["# FlyEM Figure 2C 风格三分类聚合", "", "| 训练方式 | Handcrafted 22D：BA / Macro-F1 | Embedding 384D：BA / Macro-F1 |", "|---|---|---|"]
    for mode, title in [("A_by_type", "A：按LC类型分别训练"), ("B_pooled", "B：混合类型训练")]:
        h, e = results[mode + "_handcrafted"], results[mode + "_embedding"]
        lines.append(f"| {title} | {h['balanced_accuracy']:.4f} / {h['macro_f1']:.4f} | {e['balanced_accuracy']:.4f} / {e['macro_f1']:.4f} |")
    lines += ["", "6,400实例、96 bodies、6种LC；全部四组使用相同五折与212个真实区室组（axon 96、dendrite 58、connecting cable 58）。",
              "旧五折仅覆盖85个development bodies，另11个body作为外测，不能直接覆盖本次全体OOF；因此按LC类型在body表上分层生成新五折，seed=20261113。无内层调参或额外holdout。",
              "", "RF固定100树、min_weight_fraction_leaf=0.01；训练权重为各区室训练实例数的倒数，不使用额外class_weight。RF不做标准化，避免原notebook的全样本zscore。模型不输入ID、cell_type或标签；A仅用cell_type路由到对应模型。",
              "聚合为body_id + true compartment的oracle grouping，组内平均log10概率后argmax；仅对log(0)使用1e-15下界。它不代表自动发现未知区室。",
              "", "组内线粒体数量分布："+json.dumps(size_stats, ensure_ascii=False), "", "## 主要比较", ""]
    for row in comparisons:
        lines.append(f"- {row['second']} − {row['first']}: ΔBA={row['delta_ba']:+.4f}，ΔMacro-F1={row['delta_f1']:+.4f}；后者独有正确/前者独有正确={row['second_only_correct']}/{row['first_only_correct']}。")
    best = max(results, key=lambda k: results[k]["macro_f1"])
    lines += ["", f"最高Macro-F1为{best}：{results[best]['macro_f1']:.4f}，与用户给出的论文0.90相差{results[best]['macro_f1'] - .9:+.4f}。",
              "该结果仅针对本次6类LC的作者exact-center子集。参考notebook保存的运行输出覆盖21种LC，样本覆盖、固定折、RF随机种子及数值保护不同，因此不是严格同数据复现。一次固定OOF划分也不能证明普遍优越性。",
              "", "## 必要核验", "", f"- 共拟合{len(training)}个RF；缺失训练类别的模型数："+str(sum(bool(r["missing_classes"]) for r in training))+"。未跳过任何类型、fold、实例或已存在区室组。",
              "- 四组均逐实例仅一次OOF预测；训练/测试body零交叉；A的训练集严格只包含对应LC类型；反比权重在各训练类上的总权重均为1。",
              "- 数值审计："+json.dumps(numeric, ensure_ascii=False),
              "- 手工22维包含沿神经元骨架长度及PCA几何量，并非全为纯mask形态。比较单位是区室聚合组，不能与此前单线粒体成绩直接比较。",
              "- 使用现成epoch033 embedding，未运行编码器、未训练其他分类器；原始数据及既有实验产物未修改。",
              "", "## 文件", "", "paired_group_predictions.csv：四组逐组配对预测；groups.csv：每组大小；body_folds.csv、instances.csv：固定划分及实例映射；summary.csv：每类Recall；*_confusion_*.csv：数量/行归一化混淆；provenance.json：输入哈希与版本；training_records.json：每个RF训练范围与缺失类记录。",
              "", f"作者参考：{AUTHOR_URL}/blob/{ref_commit}/morphology%20classifier%20-%20LC%20neurons.ipynb", "",
              "![四组聚合混淆矩阵](four_conditions_confusion.png)", ""]
    (out / "report_zh.md").write_text("\n".join(lines))
    dump(out / "complete.json", dict(status="complete", instances=6400, bodies=96, groups=212, fitted_rf=len(training),
        elapsed_seconds=time.time()-started, code_commit=provenance["code_commit"],
        output_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "complete.json"}))


if __name__ == "__main__":
    main()
