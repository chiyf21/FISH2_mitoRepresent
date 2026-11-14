"""Freeze the two existing protocols and assemble the already-exported reference vectors."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .common import BASE, FEATURES, SEED, TASKS, dump, read, sha, table, unique
from .inputs import register


def folds(frame):
    bodies = frame[["body_id", "cell_type"]].drop_duplicates()
    if not bodies.body_id.is_unique:
        raise ValueError("A body has multiple neuron types")
    bodies = bodies.sort_values(["cell_type", "body_id"]).reset_index(drop=True)
    if bodies.cell_type.value_counts().min() < 5:
        raise ValueError("A neuron type has fewer than five bodies")
    bodies["fold"] = -1
    for fold, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(
            np.zeros(len(bodies)), bodies.cell_type)):
        bodies.loc[test, "fold"] = fold
    return bodies


def prepare(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("Use a new asset directory; existing frozen assets are never overwritten")
    dataset = BASE / "neuron_mito_dataset/embeddings/fish2_reviewed_81"
    hp = dataset / "fish2_handcraft_81neurons_22_with_compartment.csv"
    ep = dataset / "fish2_model_embedding_81neurons_384.csv"
    h, e = table(hp), table(ep)
    unique(h); unique(e)
    if len(h) != 5568 or h.body_id.nunique() != 81:
        raise ValueError("Unexpected reviewed81 cohort")
    joined = h.merge(e, on="record_id", validate="one_to_one", suffixes=("", "_emb"))
    if len(joined) != 5568:
        raise ValueError("Handcrafted and reference fused do not cover the same instances")
    for name in ["body_id", "cell_type", "compartment", "compartment_qc", "source_run_id", "global_instance"]:
        if joined[name].astype(str).tolist() != joined[name + "_emb"].astype(str).tolist():
            raise ValueError("Source metadata disagreement: " + name)
    # Match the colleague's load_frame() order: numeric body then global instance.
    frame = h.assign(_body=h.body_id.astype(np.int64)).sort_values(
        ["_body", "global_instance"], kind="mergesort").drop(columns="_body").reset_index(drop=True)
    body_folds = folds(frame)
    out.mkdir(parents=True, exist_ok=False)
    instances = frame[["record_id", "body_id", "cell_type", "compartment", "compartment_qc",
                       "source_run_id", "global_instance"]].copy()
    instances["embedding_row"] = np.arange(len(instances))
    instances.to_csv(out / "instances.csv", index=False)
    np.save(out / "handcrafted_22.npy", frame[FEATURES].to_numpy(dtype=np.float64))
    body_folds.to_csv(out / "body_folds.csv", index=False)
    for task, config in TASKS.items():
        selected = instances.copy()
        if task == "compartment":
            selected = selected.loc[selected.compartment_qc.eq("ok") &
                selected.compartment.isin(config["classes"])].sort_values("record_id")
        selected = selected.reset_index(drop=True)
        selected["fold"] = selected.body_id.map(body_folds.set_index("body_id").fold)
        selected["label"] = selected[config["label"]].map(dict(zip(config["classes"], range(len(config["classes"])))))
        if len(selected) != config["n"] or selected.label.isna().any():
            raise ValueError("Target task coverage changed")
        selected.to_csv(out / f"{task}_instances.csv", index=False)
    runs = BASE / "point-MAE/runs"
    legacy_cohort = runs / "mito_newmae_vicreg_reg02_cls005_20261105_01/cohort.json"
    old_balanced = runs / "mito_embedding_dashboard_20261030_01/cache/features_b9a9cbbcd15a81c16c36/balanced_pair_12.npy"
    new_balanced = runs / "fish2_reviewed_81_protocol/fish2_14_layer12/balanced_pair_12.npy"
    new_map = BASE / "neuron_mito_dataset/embeddings/fusion50_fish2_14_20261112/instance_map.csv"
    old_rows = read(legacy_cohort)
    old_index = {(str(r["key"][0]), str(r["key"][1]), int(r["key"][2])): i for i, r in enumerate(old_rows)}
    nm = table(new_map); unique(nm)
    new_index = nm.set_index("record_id").embedding_row.to_dict()
    old, new = np.load(old_balanced, mmap_mode="r"), np.load(new_balanced, mmap_mode="r")
    balanced = np.empty((len(frame), 1152), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        if row.source_run_id == "e16_legacy_repair_reinfer_20261105":
            balanced[i] = old[old_index[(row.source_run_id, row.body_id, int(row.global_instance))]]
        else:
            balanced[i] = new[int(new_index[row.record_id])]
    if not np.isfinite(balanced).all():
        raise ValueError("Reference balanced is nonfinite")
    np.save(out / "canonical_balanced1152.npy", balanced)
    fused = e.set_index("record_id").loc[frame.record_id, [f"embedding_{i:03d}" for i in range(384)]].to_numpy(dtype=np.float32)
    np.save(out / "canonical_fused384.npy", fused)
    checkpoint = runs / "mito_newmae_vicreg_reg02_cls005_20261105_01/checkpoints/refine_epoch050.pt"
    for space in ["balanced1152", "fused384"]:
        register(out / f"canonical_{space}.npy", out / f"canonical_{space}.json",
                 "Fusion50_epoch050_L12_" + space, out / "instances.csv", checkpoint=checkpoint,
                 notes="Assembled from existing reviewed81 exports; no encoder inference")
    sources = [hp, ep, legacy_cohort, old_balanced, new_balanced, new_map]
    generated = [p for p in out.iterdir() if p.is_file()]
    dump(out / "assets.json", dict(schema="fish2.tasks.v1", seed=SEED, tasks=TASKS,
         handcrafted_columns=FEATURES, source_sha256={str(p): sha(p) for p in sources},
         frozen_sha256={p.name: sha(p) for p in generated}))
    return out


def validate(path):
    path = Path(path)
    config = read(path / "assets.json")
    if config["schema"] != "fish2.tasks.v1" or config["tasks"] != TASKS or config["seed"] != SEED or config["handcrafted_columns"] != FEATURES:
        raise ValueError("Unknown task schema")
    for name, expected in config["frozen_sha256"].items():
        if sha(path / name) != expected:
            raise ValueError("Frozen asset changed: " + name)
    return config
