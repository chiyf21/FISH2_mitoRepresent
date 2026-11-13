"""Fixed-parameter, by-type LR control for the saved balanced RF experiment."""
import argparse
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment import SEED, dump, sha
from balanced_confusion import table, aligned_handcrafted, fish_balanced, fly_balanced, plot
from fig2c_aggregation import score, aggregate_probabilities


def run_lr(x, y, manifest, classes, condition):
    x = np.asarray(x, dtype=np.float64)
    probabilities = np.full((len(y), len(classes)), np.nan)
    seen = np.zeros(len(y), dtype=int)
    records = []
    for fold in range(5):
        for scope in sorted(manifest.cell_type.unique()):
            domain = manifest.cell_type.eq(scope).to_numpy()
            train = np.flatnonzero(domain & manifest.fold.ne(fold).to_numpy())
            test = np.flatnonzero(domain & manifest.fold.eq(fold).to_numpy())
            assert len(train) and len(test)
            assert not set(manifest.body_id.iloc[train]) & set(manifest.body_id.iloc[test])
            assert set(manifest.cell_type.iloc[train]) == set(manifest.cell_type.iloc[test]) == {scope}
            counts = np.bincount(y[train], minlength=len(classes))
            assert np.all(counts > 0), "The original RF type-fold should cover every training class"
            # sklearn 1.9 expresses pure L2 as l1_ratio=0; lbfgs is multinomial for FlyEM.
            model = make_pipeline(StandardScaler(), LogisticRegression(
                l1_ratio=0.0, C=1.0, solver="lbfgs", class_weight="balanced",
                max_iter=3000, random_state=SEED + fold))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x[train], y[train])
            lr = model.named_steps["logisticregression"]
            scaler = model.named_steps["standardscaler"]
            assert int(scaler.n_samples_seen_) == len(train)
            assert lr.classes_.tolist() == list(range(len(classes)))
            probabilities[test] = model.predict_proba(x[test])
            seen[test] += 1
            records.append(dict(condition=condition, fold=fold, neuron_type=scope,
                                train_instances=len(train), test_instances=len(test),
                                train_bodies=manifest.body_id.iloc[train].nunique(),
                                test_bodies=manifest.body_id.iloc[test].nunique(),
                                training_class_counts=counts.tolist(), n_iter=lr.n_iter_.tolist(),
                                convergence_warning=False, scaler_fit_samples=int(scaler.n_samples_seen_)))
        print(f"{condition}: fold {fold + 1}/5 complete", flush=True)
    assert np.all(seen == 1) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(1), 1.0)
    return probabilities, records


def dataset_run(rf_root, out, base, slug, dataset):
    reference = rf_root / slug
    rf_provenance = json.loads((reference / "provenance.json").read_text())
    rf_metrics = json.loads((reference / "metrics.json").read_text())
    # Verify the fixed RF inputs, without recomputing features or reading model weights.
    for filename, expected_sha in rf_provenance["files"].items():
        path = Path(filename)
        if path.suffix != ".py":
            assert sha(path) == expected_sha, "RF input changed: " + filename
    manifest = table(reference / "instances.csv")
    bodies = table(reference / "body_folds.csv")
    groups = table(reference / "groups.csv")
    classes = rf_provenance["classes"]
    assert manifest.record_id.is_unique and bodies.body_id.is_unique
    assert manifest.groupby("body_id").fold.nunique().eq(1).all()
    assert manifest.groupby("body_id").cell_type.nunique().eq(1).all()
    assert manifest.fold.tolist() == manifest.body_id.map(bodies.set_index("body_id").fold).tolist()
    assert set(manifest.fold) == set(range(5))
    assert manifest.compartment.map(dict(zip(classes, range(len(classes))))).tolist() == manifest.label.tolist()
    assert len(manifest) == rf_provenance["instances"] and len(groups) == rf_provenance["groups"]
    assert manifest.body_id.nunique() == rf_provenance["bodies"]
    assert groups.n_mito.sum() == len(manifest)
    hand_path = next(iter(rf_provenance["files"]))
    selected, handcrafted, columns = aligned_handcrafted(hand_path, manifest, classes, slug == "fish2")
    balanced, mapping, files, checkpoint = (
        fish_balanced(base, selected) if slug == "fish2" else fly_balanced(base, manifest))
    assert checkpoint == rf_provenance["checkpoint"]
    assert balanced.shape == (len(manifest), 1152) and np.isfinite(balanced).all()
    current_map = mapping.set_index("record_id").loc[manifest.record_id].reset_index()
    previous_map = table(reference / "balanced_instance_map.csv")
    pd.testing.assert_frame_equal(current_map[previous_map.columns], previous_map, check_dtype=False)
    directory = out / slug
    directory.mkdir()
    for name, frame in [("instances", manifest), ("body_folds", bodies), ("groups", groups),
                        ("balanced_instance_map", current_map)]:
        frame.to_csv(directory / f"{name}.csv", index=False)
    y = manifest.label.to_numpy(dtype=int)
    results, training, comparisons = {}, [], []
    paired = groups.copy()
    for feature, x in [("handcrafted", handcrafted), ("balanced", balanced)]:
        probability, records = run_lr(x, y, manifest, classes, dataset + "/" + feature)
        training.extend(records)
        predictions = manifest.copy()
        for i, label in enumerate(classes):
            predictions["p_" + label] = probability[:, i]
        predictions["prediction"] = probability.argmax(1)
        predictions.to_csv(directory / f"{feature}_instance_oof.csv", index=False)
        group_prediction = aggregate_probabilities(manifest, groups, probability)[0]
        paired[feature + "_prediction"] = [classes[i] for i in group_prediction]
        for task, truth, prediction in [
            ("per-mitochondrion", y, probability.argmax(1)),
            ("per-compartment", groups.label, group_prediction)]:
            key = task + "/" + feature
            result = score(truth, prediction, classes)
            results[key] = result
            comparisons.append(dict(dataset=dataset, task=task, feature=feature,
                lr_ba=result["balanced_accuracy"], lr_macro_f1=result["macro_f1"],
                rf_ba=rf_metrics[key]["balanced_accuracy"], rf_macro_f1=rf_metrics[key]["macro_f1"],
                delta_ba=result["balanced_accuracy"] - rf_metrics[key]["balanced_accuracy"],
                delta_macro_f1=result["macro_f1"] - rf_metrics[key]["macro_f1"]))
    paired.to_csv(directory / "group_predictions.csv", index=False)
    dump(directory / "metrics.json", results)
    dump(directory / "training_records.json", training)
    provenance = dict(rf_reference=str(reference), checkpoint=checkpoint, classes=classes,
                      instances=len(manifest), bodies=manifest.body_id.nunique(), groups=len(groups),
                      handcrafted_features=columns, balanced_dimension=1152,
                      rf_inputs_sha256=sha(reference / "provenance.json"),
                      files={str(reference / name): sha(reference / name) for name in [
                          "instances.csv", "body_folds.csv", "groups.csv", "balanced_instance_map.csv", "metrics.json"]})
    dump(directory / "provenance.json", provenance)
    plot(results, dataset, classes, provenance["instances"], provenance["bodies"], provenance["groups"],
         out / f"{slug}_balanced_lr_2x2_confusion.png", classifier="LR")
    print(json.dumps({"dataset": dataset, "comparison": comparisons}), flush=True)
    return comparisons, training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    base = args.rf_results.parent.parent
    comparisons, records = [], []
    for slug, dataset in [("fish2", "Fish2"), ("flyem", "FlyEM")]:
        rows, training = dataset_run(args.rf_results, args.output, base, slug, dataset)
        comparisons.extend(rows)
        records.extend(training)
    assert len(comparisons) == 8 and len(records) == 120
    pd.DataFrame(comparisons).to_csv(args.output / "lr_vs_rf.csv", index=False)
    iterations = [n for row in records for n in row["n_iter"]]
    metadata = dict(rf_reference=str(args.rf_results),
                    model=dict(regularization="L2", l1_ratio=0.0, solver="lbfgs", C=1.0,
                               class_weight="balanced", max_iter=3000, tol=1e-4),
                    scaling="StandardScaler fit separately on training instances of each type-fold",
                    split="Unchanged RF instance, label, neuron type, and five body-fold manifests",
                    aggregation="Unchanged body_id + true compartment oracle grouping; mean log probability, floor=1e-15",
                    all_models_converged=True, fitted_models=len(records),
                    n_iter_min=min(iterations), n_iter_max=max(iterations),
                    python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                    code_sha256=sha(Path(__file__)), elapsed_seconds=time.time() - started)
    dump(args.output / "run.json", metadata)
    lines = ["# Balanced LR 对照", "",
             "原 RF 的实例、标签、类型、body 五折和 oracle 聚合全部复用；每个 type-fold 内单独拟合 StandardScaler。LR 使用 L2、lbfgs、C=1、class_weight=balanced、max_iter=3000；FlyEM 为 multinomial LR，无调参。",
             "",
             "Fish2：5,059 实例 / 81 bodies / 161 groups；FlyEM：6,400 实例 / 96 bodies / 212 groups。Handcrafted 22D，Balanced 1152D。",
             "",
             "| 数据集 | 评价 | 特征 | LR BA | LR Macro-F1 | Δ BA | Δ Macro-F1 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['dataset']} | {row['task']} | {row['feature']} | "
                     f"{row['lr_ba']:.4f} | {row['lr_macro_f1']:.4f} | "
                     f"{row['delta_ba']:+.4f} | {row['delta_macro_f1']:+.4f} |")
    lines += ["", "Δ = LR − 对应 RF，均为 0–1 分数的绝对差值。",
              f"120 个 type-fold 模型全部收敛，迭代次数 {min(iterations)}–{max(iterations)}，无 ConvergenceWarning。",
              "", f"RF 参考目录：`{args.rf_results}`。", f"运行代码提交：`{metadata['code_commit']}`。"]
    (args.output / "README_zh.md").write_text("\n".join(lines) + "\n")
    dump(args.output / "complete.json", dict(status="complete", elapsed_seconds=time.time() - started))


if __name__ == "__main__":
    main()
