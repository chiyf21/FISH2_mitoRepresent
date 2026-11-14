"""FlyEM LR Cable aggregation diagnostics, fixed Ridge probes, and one-column ablation."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment import dump, sha
from balanced_confusion import table, aligned_handcrafted, fly_balanced
from balanced_lr import run_lr
from fig2c_aggregation import score, aggregate_probabilities, EPS


def group_diagnostics(manifest, groups, probabilities, classes):
    indices = manifest.groupby(["body_id", "compartment"], sort=False).indices
    group_rows, instance_rows = [], []
    for feature, probability in probabilities.items():
        logs = np.log(np.clip(probability, EPS, 1))
        assert (probability > 0).all(), "Raw LR log probabilities must be finite"
        raw_logs = np.log(probability)
        for g in groups.itertuples(index=False):
            idx = indices[(g.body_id, g.compartment)]
            assert len(idx) == g.n_mito
            y = int(g.label)
            others = [c for c in range(len(classes)) if c != y]
            mean = logs[idx].mean(0)
            raw_mean = raw_logs[idx].mean(0)
            competitor = others[int(np.argmax(mean[others]))]
            margins = logs[idx, y] - logs[idx, competitor]
            negative_mass = float(-margins[margins < 0].sum())
            order = np.argsort(margins, kind="stable")
            k10 = int(np.ceil(len(idx) * .1))
            row = dict(feature=feature, body_id=g.body_id, compartment=g.compartment,
                cell_type=g.cell_type, fold=int(g.fold), n_mito=len(idx),
                label=y, prediction=int(mean.argmax()), predicted_class=classes[int(mean.argmax())],
                correct=int(mean.argmax()) == y,
                instance_correct_fraction=float((probability[idx].argmax(1) == y).mean()),
                competitor=classes[competitor], mean_log_margin=float(mean[y] - mean[competitor]),
                raw_mean_log_margin=float(raw_mean[y] - max(raw_mean[c] for c in others)),
                raw_prediction=int(raw_mean.argmax()), below_floor_entries=int((probability[idx] < EPS).sum()),
                negative_fraction=float((margins < 0).mean()), median_instance_margin=float(np.median(margins)),
                min_instance_margin=float(margins.min()), negative_mass=negative_mass,
                worst1_negative_mass_share=float(max(-margins[order[0]], 0) / negative_mass) if negative_mass else 0.0,
                worst10pct_negative_mass_share=float(np.maximum(-margins[order[:k10]], 0).sum() / negative_mass) if negative_mass else 0.0)
            for c, name in enumerate(classes):
                row["mean_log_p_" + name] = float(mean[c])
            if g.compartment == "connecting cable":
                for name, k in [("drop1", 1), ("drop10pct", k10)]:
                    row[name + "_removed"] = k
                    row[name + "_valid"] = len(idx) > k
                    if len(idx) > k:
                        trimmed = logs[idx[order[k:]]].mean(0)
                        row[name + "_prediction"] = int(trimmed.argmax())
                        row[name + "_correct"] = int(trimmed.argmax()) == y
                        row[name + "_margin"] = float(trimmed[y] - max(trimmed[c] for c in others))
                    else:
                        row[name + "_prediction"] = None
                        row[name + "_correct"] = None
                        row[name + "_margin"] = None
                ranks = np.empty(len(idx), dtype=int)
                ranks[order] = np.arange(1, len(idx) + 1)
                for j, ii in enumerate(idx):
                    instance_rows.append(dict(feature=feature, body_id=g.body_id,
                        record_id=manifest.record_id.iloc[ii], cell_type=g.cell_type, fold=int(g.fold),
                        group_competitor=classes[competitor], n_mito=len(idx), worst_rank=int(ranks[j]),
                        predicted_class=classes[int(probability[ii].argmax())],
                        instance_correct=int(probability[ii].argmax()) == y,
                        margin=float(margins[j]), contribution_to_mean=float(margins[j] / len(idx)),
                        true_probability=float(probability[ii, y]),
                        competitor_probability=float(probability[ii, competitor])))
            group_rows.append(row)
    return pd.DataFrame(group_rows), pd.DataFrame(instance_rows)


def pairing_and_summaries(details, out):
    cable = details.loc[details.compartment.eq("connecting cable")].copy()
    keys = ["body_id", "compartment", "cell_type", "fold", "n_mito", "label"]
    a = cable.loc[cable.feature.eq("handcrafted")].drop(columns="feature")
    b = cable.loc[cable.feature.eq("balanced")].drop(columns="feature")
    paired = a.merge(b, on=keys, suffixes=("_handcrafted", "_balanced"), validate="one_to_one")
    assert len(paired) == 58 and paired.correct_handcrafted.sum() == 52 and paired.correct_balanced.sum() == 35
    paired["pair_status"] = np.select(
        [paired.correct_handcrafted & paired.correct_balanced,
         ~paired.correct_handcrafted & ~paired.correct_balanced,
         paired.correct_handcrafted & ~paired.correct_balanced],
        ["both_correct", "both_wrong", "handcrafted_only"], default="balanced_only")
    paired["size_bin"] = pd.cut(paired.n_mito, [0, 5, 10, 20, np.inf],
                               labels=["1-5", "6-10", "11-20", ">20"])
    paired.to_csv(out / "cable_paired_58.csv", index=False)
    summaries = {}
    for field in ["cell_type", "size_bin"]:
        rows = []
        for value, frame in paired.groupby(field, observed=True, sort=True):
            rows.append(dict(group=str(value), groups=len(frame),
                n_mito_min=int(frame.n_mito.min()), n_mito_max=int(frame.n_mito.max()),
                handcrafted_errors=int((~frame.correct_handcrafted).sum()),
                balanced_errors=int((~frame.correct_balanced).sum())))
        summary = pd.DataFrame(rows)
        summary.to_csv(out / ("cable_errors_by_" + field + ".csv"), index=False)
        summaries[field] = rows
    evidence = dict(pairing=paired.pair_status.value_counts().to_dict(), by=summaries)
    for feature in ["handcrafted", "balanced"]:
        frame = cable.loc[cable.feature.eq(feature)]
        wrong = frame.loc[~frame.correct]
        record = dict(groups=len(frame), wrong=len(wrong),
            wrong_majority_negative=int(wrong.negative_fraction.gt(.5).sum()),
            wrong_majority_instance_misclassified=int(wrong.instance_correct_fraction.lt(.5).sum()),
            wrong_median_negative_fraction=float(wrong.negative_fraction.median()),
            wrong_median_instance_accuracy=float(wrong.instance_correct_fraction.median()),
            wrong_median_margin=float(wrong.mean_log_margin.median()),
            wrong_median_worst1_negative_mass_share=float(wrong.worst1_negative_mass_share.median()),
            wrong_median_worst10pct_negative_mass_share=float(wrong.worst10pct_negative_mass_share.median()),
            correct_group_size_median=float(frame.loc[frame.correct, "n_mito"].median()),
            wrong_group_size_median=float(wrong.n_mito.median()))
        for name in ["drop1", "drop10pct"]:
            valid = wrong.loc[wrong[name + "_valid"]]
            record[name + "_wrong_groups_eligible"] = len(valid)
            record[name + "_wrong_groups_rescued"] = int(valid[name + "_correct"].sum())
            record[name + "_rescued_body_ids"] = valid.loc[valid[name + "_correct"].eq(True), "body_id"].tolist()
        evidence[feature] = record
    return paired, evidence


def ridge_probes(x, targets, manifest, columns, out):
    prediction = np.full(targets.shape, np.nan)
    folds, training = [], []
    x = np.asarray(x, dtype=np.float64)
    # One pooled map per fixed body fold; multi-output Ridge is 22 independent fits
    # with the same design and alpha. Raw targets, no target-dependent transforms.
    for fold in range(5):
        train = np.flatnonzero(manifest.fold.ne(fold))
        test = np.flatnonzero(manifest.fold.eq(fold))
        assert not set(manifest.body_id.iloc[train]) & set(manifest.body_id.iloc[test])
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="cholesky"))
        model.fit(x[train], targets[train])
        prediction[test] = model.predict(x[test])
        folds.append(r2_score(targets[test], prediction[test], multioutput="raw_values"))
        training.append(dict(fold=fold, train_instances=len(train), test_instances=len(test),
            train_bodies=manifest.body_id.iloc[train].nunique(), test_bodies=manifest.body_id.iloc[test].nunique()))
        print(f"Ridge fold {fold + 1}/5 complete", flush=True)
    assert np.isfinite(prediction).all()
    overall = r2_score(targets, prediction, multioutput="raw_values")
    result = pd.DataFrame(dict(feature=columns, oof_r2=overall))
    for fold, scores in enumerate(folds):
        result[f"fold_{fold}_r2"] = scores
    result.to_csv(out / "ridge_22_r2.csv", index=False)
    np.savez_compressed(out / "ridge_oof_predictions.npz", prediction=prediction,
                       target=targets, record_id=manifest.record_id.to_numpy(dtype=str),
                       body_id=manifest.body_id.to_numpy(dtype=str), fold=manifest.fold.to_numpy())
    dump(out / "ridge_protocol.json", dict(alpha=1.0, solver="cholesky",
        scope="one pooled regression per fixed body fold across the six LC types; no type metadata input",
        input_scaling="StandardScaler fit on training fold only; float64",
        target="22 raw author-table values (including existing log-sphericity SBI); no new transformations",
        weighting="equal per instance", tuning=False, pooled_oof_r2="1 - total OOF SSE / all-instance SST",
        training=training))
    return result


def ablation(x, manifest, groups, columns, classes, original_metrics, out):
    keep = [i for i, name in enumerate(columns) if name != "length along skeleton"]
    assert len(keep) == 21 and len(columns) == 22
    probability, training = run_lr(x[:, keep], manifest.label.to_numpy(dtype=int),
                                  manifest, classes, "FlyEM/handcrafted_without_skeleton_length")
    dump(out / "no_length_training_records.json", training)
    frame = manifest.copy()
    for c, name in enumerate(classes):
        frame["p_" + name] = probability[:, c]
    frame["prediction"] = probability.argmax(1)
    frame.to_csv(out / "no_length_instance_oof.csv", index=False)
    prediction = aggregate_probabilities(manifest, groups, probability)[0]
    new = dict(per_mito=score(manifest.label, probability.argmax(1), classes),
               per_compartment=score(groups.label, prediction, classes))
    dump(out / "no_length_metrics.json", new)
    g = groups.copy()
    g["prediction"] = prediction
    g["predicted_class"] = [classes[i] for i in prediction]
    g.to_csv(out / "no_length_group_predictions.csv", index=False)
    old = original_metrics["per-compartment/handcrafted"]
    cable = classes.index("connecting cable")
    summary = dict(removed_feature="length along skeleton", retained_features=[columns[i] for i in keep],
        original_ba=old["balanced_accuracy"], no_length_ba=new["per_compartment"]["balanced_accuracy"],
        delta_ba=new["per_compartment"]["balanced_accuracy"] - old["balanced_accuracy"],
        original_cable_recall=old["recall"]["connecting cable"],
        no_length_cable_recall=new["per_compartment"]["recall"]["connecting cable"],
        original_cable_correct=old["confusion"][cable][cable],
        no_length_cable_correct=new["per_compartment"]["confusion"][cable][cable],
        fitted_models=len(training), all_converged=True,
        maximum_iterations=max(v for row in training for v in row["n_iter"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = args.lr_results / "flyem"
    prov = json.loads((reference / "provenance.json").read_text())
    rf_reference = Path(prov["rf_reference"])
    rf_prov = json.loads((rf_reference / "provenance.json").read_text())
    for filename, expected in rf_prov["files"].items():
        if Path(filename).suffix != ".py":
            assert sha(filename) == expected, filename
    base = args.lr_results.parent.parent
    manifest = table(reference / "instances.csv")
    groups = table(reference / "groups.csv")
    classes = prov["classes"]
    assert len(manifest) == 6400 and manifest.body_id.nunique() == 96 and len(groups) == 212
    assert manifest.record_id.is_unique and manifest.groupby("body_id").fold.nunique().eq(1).all()
    assert manifest.groupby("body_id").cell_type.nunique().eq(1).all()
    hp = next(iter(rf_prov["files"]))
    selected, handcrafted, columns = aligned_handcrafted(hp, manifest, classes)
    balanced, mapping, sources, checkpoint = fly_balanced(base, manifest)
    assert checkpoint == prov["checkpoint"] and balanced.shape == (6400, 1152)
    original_metrics = json.loads((reference / "metrics.json").read_text())
    probabilities = {}
    for feature in ["handcrafted", "balanced"]:
        frame = table(reference / f"{feature}_instance_oof.csv")
        pd.testing.assert_frame_equal(frame[manifest.columns], manifest)
        probabilities[feature] = frame[["p_" + c for c in classes]].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(probabilities[feature].sum(1), 1)
        assert score(frame.label, frame.prediction, classes) == original_metrics["per-mitochondrion/" + feature]
        prediction = aggregate_probabilities(manifest, groups, probabilities[feature])[0]
        assert score(groups.label, prediction, classes) == original_metrics["per-compartment/" + feature]
    details, instances = group_diagnostics(manifest, groups, probabilities, classes)
    details.to_csv(args.output / "all_group_diagnostics.csv", index=False)
    instances.to_csv(args.output / "cable_instance_margins.csv", index=False)
    paired, evidence = pairing_and_summaries(details, args.output)
    equivalence = {}
    for feature in ["handcrafted", "balanced"]:
        d = details.loc[details.feature.eq(feature)]
        affected = d.loc[d.below_floor_entries.gt(0)]
        equivalence[feature] = dict(probabilities_below_floor=int((probabilities[feature] < EPS).sum()),
            groups_with_floor=int(len(affected)), raw_vs_clipped_prediction_disagreements=int((d.prediction != d.raw_prediction).sum()),
            cable_raw_vs_clipped_prediction_disagreements=int((d.loc[d.compartment.eq("connecting cable"), "prediction"] != d.loc[d.compartment.eq("connecting cable"), "raw_prediction"]).sum()))
    dump(args.output / "aggregation_evidence.json", evidence)
    dump(args.output / "linear_aggregation_equivalence.json", equivalence)
    ridge = ridge_probes(balanced, handcrafted, manifest, columns, args.output)
    removal = ablation(handcrafted, manifest, groups, columns, classes, original_metrics, args.output)
    dump(args.output / "no_length_comparison.json", removal)
    files = [reference / name for name in ["instances.csv", "groups.csv", "body_folds.csv",
              "handcrafted_instance_oof.csv", "balanced_instance_oof.csv", "metrics.json", "provenance.json"]] + [Path(hp)] + sources
    manifest.to_csv(args.output / "instances.csv", index=False)
    table(reference / "body_folds.csv").to_csv(args.output / "body_folds.csv", index=False)
    dump(args.output / "run.json", dict(reference=str(reference), checkpoint=checkpoint,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        code_sha256=sha(Path(__file__)), sklearn=sklearn.__version__, numpy=np.__version__,
        margin="natural-log p(true) - log p(group strongest competing class); mean margin is exact aggregation margin",
        diagnostics="remove minimum margin instance or ceil(10% * group n) to original group competitor; true-label-guided, not formal performance",
        classes=classes, input_sha256={str(p): sha(p) for p in files}))
    dump(args.output / "complete.json", dict(status="complete", instances=6400, bodies=96, groups=212, cable_groups=58))
    print(json.dumps(dict(evidence=evidence, equivalence=equivalence, no_length=removal,
        ridge=ridge[["feature", "oof_r2"]].to_dict("records")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
