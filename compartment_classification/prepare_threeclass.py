"""Add author exact connecting-cable rows to the original frozen body folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import FEATURES, SEED, dump, emit, sha, table

CLASSES = ["axon", "dendrite", "connecting cable"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary-results", type=Path, required=True)
    ap.add_argument("--author-directory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = args.output / "flyem"
    data.mkdir(parents=True, exist_ok=False)
    old_path = args.binary_results / "flyem/instances.csv"
    split_path = args.binary_results / "flyem/splits.json"
    old = table(old_path)
    old_splits = json.loads(split_path.read_text())
    old_provenance = json.loads((old_path.parent / "data_provenance.json").read_text())
    assert sha(old_path) == old_provenance["instance_manifest_sha256"]
    assert sha(split_path) == old_provenance["splits_sha256"]
    hp = args.author_directory / "flyem_mito_author_features_exact.csv"
    lp = args.author_directory / "paper22_author_compartments.csv"
    h, labels = table(hp), table(lp)
    assert len(h) == 6400 and h.record_id.is_unique and h.source_mito_label.is_unique
    assert h.compartment.value_counts().to_dict() == {"axon": 3148, "dendrite": 2260, "connecting cable": 992}
    labels = labels[["body_id", "record_id", "source_mito_label", "compartment_source", "compartment"]].rename(columns={"compartment": "author_compartment"})
    h = h.merge(labels, on=["body_id", "record_id", "source_mito_label"], validate="one_to_one")
    assert len(h) == 6400 and h.compartment_source.eq("published_mito_exact_center").all()
    assert h.compartment.eq(h.author_compartment).all()
    export_dir = args.binary_results / "flyem_embeddings"
    ep, mp = export_dir / "fused_12.npy", export_dir / "instance_map.csv"
    export = json.loads((export_dir / "provenance.json").read_text())
    complete = json.loads((export_dir / "complete.json").read_text())
    assert export["epoch"] == 33 and export["checkpoint"].endswith("/refine_both_cls_ln/checkpoints/refine_epoch033.pt")
    assert complete["shape"] == [9731, 384] and complete["status"] == "complete"
    assert sha(ep) == complete["fused_12_sha256"] and sha(mp) == complete["instance_map_sha256"]
    m = table(mp)
    assert m.record_id.is_unique and m.embedding_row.tolist() == list(range(9731))
    selected = h.merge(m, on=["body_id", "record_id"], validate="one_to_one").sort_values("record_id").reset_index(drop=True)
    assert len(selected) == 6400 and selected.body_id.nunique() == 96
    body_partitions = old[["body_id", "partition", "encoder_split"]].drop_duplicates()
    assert body_partitions.body_id.is_unique
    selected = selected.merge(body_partitions, on="body_id", validate="many_to_one", sort=False)
    assert len(selected) == 6400 and selected["split"].eq(selected.encoder_split).all()
    selected = selected.sort_values("record_id").reset_index(drop=True)
    columns = [c if c != "length_along_skeleton_nm" else "length along skeleton" for c in FEATURES]
    xh = selected[columns].to_numpy(dtype=np.float64)
    xe = np.array(np.load(ep, mmap_mode="r", allow_pickle=False)[selected.embedding_row.to_numpy(dtype=int)], dtype=np.float64)
    assert xh.shape == (6400, 22) and xe.shape == (6400, 384)
    assert np.isfinite(xh).all() and np.isfinite(xe).all()
    selected["label"] = selected.compartment.map(dict(zip(CLASSES, range(3))))
    assert selected.label.notna().all()
    groups = selected.body_id.to_numpy(dtype=str)
    y = selected.label.to_numpy(dtype=int)
    coverage = []

    def lift(split, level):
        result = {}
        for part in ["train", "test"]:
            original_bodies = set(old.iloc[split[part]].body_id)
            indices = np.flatnonzero(np.isin(groups, list(original_bodies)))
            assert set(groups[indices]) == original_bodies
            assert set(y[indices]) == {0, 1, 2}, (level, part)
            result[part] = indices.tolist()
            for k, name in enumerate(CLASSES):
                subset = indices[y[indices] == k]
                coverage.append(dict(level=level, part=part, compartment=name, instances=len(subset),
                                     bodies=len(set(groups[subset]))))
        assert not set(groups[result["train"]]) & set(groups[result["test"]])
        if "inner" in split:
            result["inner"] = [lift(inner, level + f"/inner_{i}") for i, inner in enumerate(split["inner"])]
            for inner in result["inner"]:
                assert set(inner["train"]) | set(inner["test"]) == set(result["train"])
        return result

    outer = [lift(fold, f"outer_{i}") for i, fold in enumerate(old_splits["outer"])]
    holdout = lift(old_splits["holdout"], "holdout")
    assert sorted(i for fold in outer for i in fold["test"]) == sorted(holdout["train"])
    assert selected.iloc[holdout["test"]].partition.eq("holdout").all()
    assert selected.iloc[holdout["train"]].partition.eq("development").all()
    manifest = selected[["record_id", "body_id", "compartment", "encoder_split", "partition", "label", "source_mito_label", "embedding_row"]].copy()
    manifest["outer_fold"] = -1
    for i, fold in enumerate(outer):
        manifest.loc[fold["test"], "outer_fold"] = i
    old_assignments = old[["record_id", "body_id", "partition", "outer_fold"]]
    check = manifest.merge(old_assignments, on=["record_id", "body_id"], validate="one_to_one", suffixes=("", "_binary"))
    assert len(check) == 5408
    assert check.partition.eq(check.partition_binary).all() and check.outer_fold.eq(check.outer_fold_binary).all()
    manifest.to_csv(data / "instances.csv", index=False)
    np.save(data / "handcrafted.npy", xh)
    np.save(data / "embedding.npy", xe)
    dump(data / "splits.json", dict(seed=SEED, outer=outer, holdout=holdout,
         protocol="Original binary body partitions AND all outer/inner body folds; cable rows inherit their body's memberships"))
    pd.DataFrame(coverage).to_csv(data / "class_coverage.csv", index=False)
    summary = {part: dict(instances=int((selected.partition == part).sum()),
               bodies=int(selected.loc[selected.partition == part, "body_id"].nunique()),
               classes=selected.loc[selected.partition == part, "compartment"].value_counts().to_dict(),
               class_bodies=selected.loc[selected.partition == part].groupby("compartment").body_id.nunique().to_dict())
               for part in ["development", "holdout"]}
    inputs = [hp, lp, ep, mp, export_dir / "provenance.json", old_path, split_path]
    dump(data / "data_provenance.json", dict(dataset="flyem", classes=CLASSES,
         shape_handcrafted=[6400, 22], shape_embedding=[6400, 384], handcrafted_columns=columns,
         partition_summary=summary, checkpoint=export["checkpoint"], checkpoint_sha256=export["checkpoint_sha256"],
         embedding_directory=str(export_dir.resolve()), embedding_provenance=export,
         label_filter="All 6400 author published_mito_exact_center matches; no skeleton-transferred labels",
         source_files={str(p.resolve()): dict(sha256=sha(p), bytes=p.stat().st_size) for p in inputs},
         instance_manifest_sha256=sha(data / "instances.csv"), splits_sha256=sha(data / "splits.json"),
         binary_instance_manifest_sha256=sha(old_path), binary_splits_sha256=sha(split_path),
         seed=SEED, feature_policy="Same 22 handcrafted or 384 embedding inputs; no metadata features",
         missing_values="All finite; no imputation", class_coverage_all_splits=True))
    emit(stage="threeclass_prepared", partitions=summary, inherited_body_folds=True)


if __name__ == "__main__":
    main()
