"""Verify frozen identities, group isolation, train-only scaling, and saved predictions."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from experiment import dump, metrics, sha, table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["fish2", "flyem"])
    args = ap.parse_args()
    checks = []
    for dataset in args.datasets:
        data = args.output / dataset
        manifest = table(data / "instances.csv")
        y = manifest.label.to_numpy(dtype=int)
        groups = manifest.body_id.to_numpy(dtype=str)
        splits = json.loads((data / "splits.json").read_text())
        provenance = json.loads((data / "data_provenance.json").read_text())
        assert provenance["instance_manifest_sha256"] == sha(data / "instances.csv")
        assert provenance["splits_sha256"] == sha(data / "splits.json")
        assert manifest.record_id.is_unique
        held_bodies = set(groups[splits["holdout"]["test"]])
        dev_bodies = set(groups[splits["holdout"]["train"]])
        assert not held_bodies & dev_bodies
        assert len(splits["outer"]) == 5
        assert sorted(i for s in splits["outer"] for i in s["test"]) == sorted(splits["holdout"]["train"])
        for feature in ["handcrafted", "embedding", "dummy"]:
            x = np.zeros((len(y), 1)) if feature == "dummy" else np.load(data / (feature + ".npy"), allow_pickle=False)
            if feature != "dummy":
                assert x.shape[1] == (22 if feature == "handcrafted" else 384)
            identity = json.loads((data / feature / "fit_provenance.json").read_text())["identity"]
            assert identity["instance_sha256"] == sha(data / "instances.csv")
            assert identity["splits_sha256"] == sha(data / "splits.json")
            if feature != "dummy":
                assert identity["feature_sha256"] == sha(data / (feature + ".npy"))
            kinds = ["dummy"] if feature == "dummy" else ["logistic", "random_forest"]
            for classifier in kinds:
                for name, split in [(f"fold_{i}", s) for i, s in enumerate(splits["outer"])] + [("holdout", splits["holdout"])]:
                    train, test = split["train"], split["test"]
                    assert not set(groups[train]) & set(groups[test])
                    assert len(split["inner"]) == 3
                    for inner in split["inner"]:
                        assert not set(groups[inner["train"]]) & set(groups[inner["test"]])
                        assert set(inner["train"]) | set(inner["test"]) == set(train)
                        assert not (set(inner["train"]) | set(inner["test"])) & set(splits["holdout"]["test"])
                    folder = data / feature / classifier / name
                    estimator = joblib.load(folder / "model.joblib")
                    saved = np.load(folder / "predictions.npz", allow_pickle=False)
                    assert saved["indices"].tolist() == test
                    if classifier == "logistic":
                        scaler = estimator.steps[0][1]
                        np.testing.assert_allclose(scaler.mean_, x[train].mean(0), rtol=1e-10, atol=1e-10)
                        np.testing.assert_allclose(scaler.var_, x[train].var(0), rtol=1e-10, atol=1e-10)
                        assert estimator.steps[-1][1].n_features_in_ == x.shape[1]
                    elif classifier == "random_forest":
                        assert estimator.n_features_in_ == x.shape[1] and estimator.n_estimators == 300
                    prediction = estimator.predict(x[test])
                    np.testing.assert_array_equal(prediction, saved["prediction"])
                    np.testing.assert_allclose(estimator.predict_proba(x[test]), saved["probability"], rtol=1e-12, atol=1e-12)
                    result = json.loads((folder / "complete.json").read_text())
                    score = metrics(y[test], prediction)
                    assert score == result["metrics"]
                    assert result["parameters"] == max(result["tuning"], key=lambda v: v["validation_balanced_accuracy"])["parameters"]
                    checks.append(dict(dataset=dataset, feature=feature, classifier=classifier, split=name, status="PASS"))
    if "flyem" in args.datasets:
        export = args.output / "flyem_embeddings"
        complete = json.loads((export / "complete.json").read_text())
        config = json.loads((export / "provenance.json").read_text())
        assert config["epoch"] == 33 and "refine_both_cls_ln/checkpoints/refine_epoch033.pt" in config["checkpoint"]
        assert complete["reference_smoke_max_abs_delta"] == 0
        assert complete["fused_12_sha256"] == sha(export / "fused_12.npy")
        mapping = table(export / "instance_map.csv")
        assert mapping.record_id.is_unique and mapping.embedding_row.tolist() == list(range(9731))
        model_run = Path(config["config"]["point_run"]).parent
        cohort = json.loads((model_run / "refine_both_cls_ln/cohort.json").read_text())
        assert len(cohort) == len(mapping)
        for row, ref in zip(mapping.to_dict("records"), cohort):
            assert row["record_id"] == ref["key"][2] and row["body_id"] == ref["body_id"]
            assert row["cohort_index"] == ref["cohort_index"] and row["split"] == ref["split"]
            assert "/flyem_lc21_source_instances_20261110/results/" in ref["mask_path"]
    report = args.output / "report"
    report.mkdir(exist_ok=True)
    dump(report / "validation.json", dict(status="PASS", fitted_models_verified=len(checks), checks=checks,
                                         checks_include=["identity hashes", "outer/inner/holdout body isolation",
                                                         "train-only scaler means and variances", "saved-model predictions/probabilities",
                                                         "metrics recomputation", "inner-CV hyperparameter selection",
                                                         "fixed checkpoint and full-export/cohort mapping"]))
    print(json.dumps(dict(status="PASS", fitted_models_verified=len(checks))))


if __name__ == "__main__":
    main()
