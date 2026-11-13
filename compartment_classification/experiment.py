"""Nested body-disjoint compartment probes with a retained encoder holdout."""
import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261113
CLASSES = ["axon", "dendrite"]
FEATURES = ["size", "mito SA", "mito CA", "convex hull compactness"] + [
    f"PC{axis} {suffix}" for axis in (1, 2, 3)
    for suffix in ["Length", "inertia moment", "symmetry", "CA", "Circum"]
] + ["mito diameter", "length_along_skeleton_nm", "SBI"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def table(path):
    return pd.read_csv(path, dtype={"record_id": str, "body_id": str, "source_run_id": str,
                                   "source_mito_label": str})


def emit(**value):
    print(json.dumps(dict(time=time.time(), **value), ensure_ascii=False), flush=True)


def split_groups(indices, y, groups, n, seed):
    indices = np.asarray(indices, dtype=int)
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    result = []
    for a, b in cv.split(np.zeros(len(indices)), y[indices], groups[indices]):
        train, test = indices[a], indices[b]
        assert not set(groups[train]) & set(groups[test]), "Body leakage"
        assert set(y[train]) == {0, 1} and set(y[test]) == {0, 1}, "Missing class in split"
        result.append(dict(train=train.tolist(), test=test.tolist()))
    assert sorted(i for s in result for i in s["test"]) == sorted(indices.tolist())
    return result


def prepare(args):
    root, output = args.asset_root, args.output / args.dataset
    output.mkdir(parents=True, exist_ok=False)
    inputs = []
    if args.dataset == "fish2":
        base = root / "neuron_mito_dataset/embeddings/fish2_reviewed_81"
        hp = base / "fish2_handcraft_81neurons_22_with_compartment.csv"
        ep = base / "fish2_model_embedding_81neurons_384.csv"
        h, e = table(hp), table(ep)
        assert len(h) == len(e) == 5568
        for frame in [h, e]:
            assert frame.record_id.is_unique and frame[["record_id", "body_id", "compartment"]].notna().all().all()
        joined = h.merge(e, on="record_id", how="inner", validate="one_to_one", suffixes=("", "_emb"))
        assert len(joined) == 5568
        for field in ["body_id", "compartment", "compartment_qc", "cell_type", "source_run_id", "global_instance"]:
            assert (joined[field].astype(str) == joined[field + "_emb"].astype(str)).all(), field
        selected = joined.loc[joined.compartment.isin(CLASSES) & joined.compartment_qc.eq("ok")].copy()
        assert len(selected) == 5059 and selected.body_id.nunique() == 81
        cols = FEATURES
        embcols = [f"embedding_{i:03d}" for i in range(384)]
        model = root / "point-MAE/runs/mito_newmae_vicreg_reg02_cls005_20261105_01"
        source = json.loads((model / "cohort.json").read_text())
        bs = {}
        for row in source:
            body, split = str(row["body_id"]), row["split"]
            assert body not in bs or bs[body] == split
            bs[body] = split
        selected["encoder_split"] = selected.body_id.map(bs).fillna("new_body")
        selected["partition"] = np.where(selected.encoder_split.isin(["test", "new_body"]), "holdout", "development")
        provenance = dict(checkpoint=str(model / "checkpoints/refine_epoch050.pt"),
                          export_complete_records=[
                              json.loads((root / "neuron_mito_dataset/embeddings" / name / "complete.json").read_text())
                              for name in ["fusion50_all16428_20261112", "fusion50_fish2_14_20261112"]],
                          label_filter="axon/dendrite and compartment_qc == ok",
                          label_limit="Reviewed skeleton projection metadata exists, but review-source/revision provenance remains incomplete.")
        inputs = [hp, ep, model / "cohort.json"]
    else:
        hp = root / "flyem_lc21_source_instances_20261110/paper_features_lc21_v1/author_dryad/flyem_mito_author_features_exact.csv"
        ep = args.output / "flyem_embeddings/fused_12.npy"
        mp = ep.parent / "instance_map.csv"
        complete = json.loads((ep.parent / "complete.json").read_text())
        export = json.loads((ep.parent / "provenance.json").read_text())
        fixed = args.run_root / "refine_both_cls_ln/checkpoints/refine_epoch033.pt"
        assert export["epoch"] == 33 and export["checkpoint"] == str(fixed.resolve())
        assert complete["shape"] == [9731, 384] and complete["status"] == "complete"
        assert complete["fused_12_sha256"] == sha(ep) and complete["instance_map_sha256"] == sha(mp)
        h, m = table(hp), table(mp)
        assert len(h) == 6400 and h.record_id.is_unique and h.source_mito_label.is_unique
        assert len(m) == 9731 and m.record_id.is_unique
        assert m.embedding_row.tolist() == list(range(9731))
        joined = h.merge(m, on=["body_id", "record_id"], how="inner", validate="one_to_one")
        assert len(joined) == 6400, "Author exact set must be fully covered"
        selected = joined.loc[joined.compartment.isin(CLASSES)].copy()
        assert len(selected) == 5408 and selected.body_id.nunique() == 96
        cols = [c if c != "length_along_skeleton_nm" else "length along skeleton" for c in FEATURES]
        selected["encoder_split"] = selected["split"]
        selected["partition"] = np.where(selected.encoder_split.eq("test"), "holdout", "development")
        provenance = dict(checkpoint=str(fixed), checkpoint_sha256=export["checkpoint_sha256"],
                          embedding_provenance=export, label_filter="author exact-center matches; axon/dendrite",
                          label_limit="Author-published classes; nearest-node transferred labels excluded.")
        inputs = [hp, ep, mp, ep.parent / "provenance.json"]
    selected = selected.sort_values("record_id").reset_index(drop=True)
    xh = selected[cols].to_numpy(dtype=np.float64)
    if args.dataset == "fish2":
        xe = selected[embcols].to_numpy(dtype=np.float64)
    else:
        all_embeddings = np.load(ep, mmap_mode="r", allow_pickle=False)
        xe = np.array(all_embeddings[selected.embedding_row.to_numpy(dtype=int)], dtype=np.float64)
    assert xh.shape == (len(selected), 22) and xe.shape == (len(selected), 384)
    assert np.isfinite(xh).all() and np.isfinite(xe).all()
    y = selected.compartment.map({"axon": 0, "dendrite": 1}).to_numpy(dtype=int)
    groups = selected.body_id.to_numpy(dtype=str)
    dev = np.flatnonzero(selected.partition.eq("development"))
    holdout = np.flatnonzero(selected.partition.eq("holdout"))
    assert not set(groups[dev]) & set(groups[holdout])
    assert len(dev) + len(holdout) == len(y)
    outer = split_groups(dev, y, groups, 5, SEED)
    for i, fold in enumerate(outer):
        assert not set(groups[fold["train"]]) & set(groups[holdout])
        fold["inner"] = split_groups(fold["train"], y, groups, 3, SEED + 100 + i)
    final = dict(train=dev.tolist(), test=holdout.tolist(), inner=split_groups(dev, y, groups, 3, SEED + 999))
    manifest = selected[["record_id", "body_id", "compartment", "encoder_split", "partition"]].copy()
    manifest["label"] = y
    manifest["outer_fold"] = -1
    for i, fold in enumerate(outer):
        manifest.loc[fold["test"], "outer_fold"] = i
    assert manifest.record_id.is_unique
    manifest.to_csv(output / "instances.csv", index=False)
    np.save(output / "handcrafted.npy", xh)
    np.save(output / "embedding.npy", xe)
    dump(output / "splits.json", dict(seed=SEED, outer=outer, holdout=final,
                                     protocol="5 outer folds on development bodies; 3 inner group folds; retained independent body holdout"))
    summary = {part: dict(instances=int((selected.partition == part).sum()),
                          bodies=int(selected.loc[selected.partition == part, "body_id"].nunique()),
                          classes=selected.loc[selected.partition == part, "compartment"].value_counts().to_dict())
               for part in ["development", "holdout"]}
    dump(output / "data_provenance.json", dict(
        dataset=args.dataset, shape_handcrafted=list(xh.shape), shape_embedding=list(xe.shape),
        handcrafted_columns=cols, classes=CLASSES, partition_summary=summary, **provenance,
        source_files={str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in inputs},
        instance_manifest_sha256=sha(output / "instances.csv"), splits_sha256=sha(output / "splits.json"),
        seed=SEED, feature_policy="Only the explicitly listed 22 numeric features or 384 embedding dimensions; no metadata inputs",
        missing_values="No missing/nonfinite values; no imputation",
        created_unix=time.time()))
    emit(stage="prepared", dataset=args.dataset, partitions=summary)


def metrics(y, p):
    cm = confusion_matrix(y, p, labels=[0, 1])
    prec, rec, f1, support = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, p)), macro_f1=float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0)),
                precision=dict(zip(CLASSES, prec.tolist())), recall=dict(zip(CLASSES, rec.tolist())),
                support=dict(zip(CLASSES, support.tolist())), confusion=cm.tolist(),
                confusion_normalized=np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros((2, 2)), where=cm.sum(1, keepdims=True) != 0).tolist())


def candidates(kind):
    if kind == "logistic":
        return [dict(C=c) for c in [0.01, 0.1, 1.0, 10.0]]
    if kind == "random_forest":
        return [dict(min_samples_leaf=leaf, max_features=mf) for leaf, mf in itertools.product([5, 1], ["sqrt", 0.1])]
    return [dict(strategy="most_frequent")]


def classifier(kind, params, workers, seed):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(**params, class_weight="balanced", solver="lbfgs",
                                                                  max_iter=10000, tol=1e-6, random_state=seed))
    if kind == "random_forest":
        return RandomForestClassifier(**params, n_estimators=300, class_weight="balanced_subsample", n_jobs=workers,
                                      random_state=seed)
    return DummyClassifier(**params, random_state=seed)


def run_split(x, y, groups, split, kind, workers, seed, folder):
    folder.mkdir(exist_ok=True)
    if (folder / "complete.json").exists():
        data = np.load(folder / "predictions.npz", allow_pickle=False)
        assert data["indices"].tolist() == split["test"]
        return json.loads((folder / "complete.json").read_text()), data["prediction"], data["probability"]
    assert not set(groups[split["train"]]) & set(groups[split["test"]])
    traces = []
    for params in candidates(kind):
        vals = []
        for inner_id, inner in enumerate(split["inner"]):
            assert not set(groups[inner["train"]]) & set(groups[inner["test"]])
            assert set(inner["train"]) | set(inner["test"]) <= set(split["train"])
            estimator = classifier(kind, params, workers, seed + inner_id)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(x[inner["train"]], y[inner["train"]])
            p = estimator.predict(x[inner["test"]])
            vals.append(metrics(y[inner["test"]], p))
        traces.append(dict(parameters=params, validation_balanced_accuracy=float(np.mean([v["balanced_accuracy"] for v in vals])),
                           validation_folds=vals))
    # Stable candidate ordering resolves ties without looking at outer/holdout labels.
    best = max(range(len(traces)), key=lambda i: traces[i]["validation_balanced_accuracy"])
    params = traces[best]["parameters"]
    estimator = classifier(kind, params, workers, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(x[split["train"]], y[split["train"]])
    prediction = estimator.predict(x[split["test"]])
    probability = estimator.predict_proba(x[split["test"]])
    assert estimator.classes_.tolist() == [0, 1]
    score = metrics(y[split["test"]], prediction)
    result = dict(parameters=params, selected_inner_balanced_accuracy=traces[best]["validation_balanced_accuracy"],
                  metrics=score, train_instances=len(split["train"]), test_instances=len(split["test"]),
                  train_bodies=len(set(groups[split["train"]])), test_bodies=len(set(groups[split["test"]])),
                  tuning=traces)
    np.savez(folder / "predictions.npz", indices=np.array(split["test"]), prediction=prediction, probability=probability)
    joblib.dump(estimator, folder / "model.joblib", compress=3)
    dump(folder / "complete.json", result)
    return result, prediction, probability


def fit(args):
    data = args.output / args.dataset
    manifest = table(data / "instances.csv")
    splits = json.loads((data / "splits.json").read_text())
    y = manifest.label.to_numpy(dtype=int)
    groups = manifest.body_id.to_numpy(dtype=str)
    x = np.zeros((len(manifest), 1)) if args.feature == "dummy" else np.load(data / (args.feature + ".npy"), allow_pickle=False)
    assert len(x) == len(y) and np.isfinite(x).all()
    out = data / args.feature
    out.mkdir(exist_ok=True)
    identity = dict(instance_sha256=sha(data / "instances.csv"), splits_sha256=sha(data / "splits.json"),
                    feature_sha256=None if args.feature == "dummy" else sha(data / (args.feature + ".npy")))
    if (out / "fit_provenance.json").exists():
        old = json.loads((out / "fit_provenance.json").read_text())
        assert old["identity"] == identity
    else:
        dump(out / "fit_provenance.json", dict(
            identity=identity, python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
            script_sha256=sha(__file__), code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            workers=args.workers, seed=SEED, started_unix=time.time(), commands=os.sys.argv,
            logistic="StandardScaler fitted inside each training split; balanced L2 logistic, float64",
            random_forest="300 trees; balanced_subsample; no standardization; inner group CV selects leaf/max_features",
            grids={k: candidates(k) for k in ["logistic", "random_forest"]} ))
    kinds = ["dummy"] if args.feature == "dummy" else ["logistic", "random_forest"]
    results = {}
    for kind in kinds:
        kdir = out / kind
        kdir.mkdir(exist_ok=True)
        predictions = []
        folds = []
        for i, split in enumerate(splits["outer"]):
            result, p, prob = run_split(x, y, groups, split, kind, args.workers, SEED + i, kdir / f"fold_{i}")
            folds.append(result)
            f = manifest.iloc[split["test"]].copy()
            f["prediction"], f["p_axon"], f["p_dendrite"], f["evaluation"] = p, prob[:, 0], prob[:, 1], "development_oof"
            predictions.append(f)
            emit(stage="outer_fold_complete", dataset=args.dataset, feature=args.feature, classifier=kind,
                 fold=i, parameters=result["parameters"], metrics=result["metrics"])
        result, p, prob = run_split(x, y, groups, splits["holdout"], kind, args.workers, SEED + 999, kdir / "holdout")
        f = manifest.iloc[splits["holdout"]["test"]].copy()
        f["prediction"], f["p_axon"], f["p_dendrite"], f["evaluation"] = p, prob[:, 0], prob[:, 1], "holdout"
        predictions.append(f)
        allp = pd.concat(predictions).sort_values("record_id")
        assert len(allp) == len(manifest) and allp.record_id.is_unique
        allp.to_csv(kdir / "predictions.csv", index=False)
        oof = allp.loc[allp.evaluation.eq("development_oof")]
        results[kind] = dict(outer_folds=folds, outer_mean={metric: float(np.mean([s["metrics"][metric] for s in folds])) for metric in ["balanced_accuracy", "macro_f1"]},
                             outer_sd={metric: float(np.std([s["metrics"][metric] for s in folds], ddof=1)) for metric in ["balanced_accuracy", "macro_f1"]},
                             development_oof=metrics(oof.label, oof.prediction), holdout=result)
        emit(stage="classifier_complete", dataset=args.dataset, feature=args.feature, classifier=kind,
             oof=results[kind]["development_oof"], holdout=result["metrics"])
    dump(out / "complete.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "fit"])
    parser.add_argument("--dataset", choices=["fish2", "flyem"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--feature", choices=["handcrafted", "embedding", "dummy"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else fit)(args)


if __name__ == "__main__":
    main()
