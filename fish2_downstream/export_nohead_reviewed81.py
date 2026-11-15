"""Reviewed81 adapter around the frozen No-Head50 runtime (no encoder rewrite)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = Path("/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM")
RUNS = BASE / "point-MAE/runs"
RUN = RUNS / "mito_newmae_nohead_maincov_calibrated_20261114_01"
REFERENCE = RUNS / "mito_newmae_vicreg_reg02_cls005_20261105_01"
FISH = BASE / "neuron_mito_dataset/embeddings/fusion50_fish2_14_20261112"
INPUTS = BASE / "neuron_mito_dataset/handcraft_features/fish2_nov_reviewed_81_v1/model_input_manifest.csv"
ASSETS = BASE / "fish2_downstream/assets/reviewed81_v1"
RUNTIME = Path("/home/cyf/github_repos/mitochondria-pretraining-contrastive_worktrees/fish2_nohead_maincov_calibrated_20261114")
EXPECTED_CHECKPOINT_SHA = "898ca54903445f64f551ae1c1de6aff6ccc2220623ddaa6e9028364c1e81193c"
LEGACY = "e16_legacy_repair_reinfer_20261105"
ADDED = "e16_legacy_repair_reinfer_20261112_fish2_14"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path):
    return pd.read_csv(path, dtype={"record_id": str, "source_run_id": str, "body_id": str})


def reviewed_inputs(config):
    target = table(ASSETS / "instances.csv")
    assert len(target) == 5568 and target.record_id.is_unique and target.body_id.nunique() == 81
    assert target.source_run_id.value_counts().to_dict() == {LEGACY: 4462, ADDED: 1106}
    assert np.array_equal(target.embedding_row, np.arange(5568))
    cohort = json.loads((REFERENCE / "cohort.json").read_text())
    keys = {tuple([str(r["key"][0]), str(r["key"][1]), int(r["key"][2])]): r for r in cohort}
    assert len(keys) == len(cohort) == 10678
    mapping = table(FISH / "instance_map.csv")
    manifest = table(INPUTS)
    assert mapping.record_id.is_unique and manifest.record_id.is_unique
    new_map = mapping.set_index("record_id")
    manifest = manifest.set_index("record_id")
    old_points = np.load(config["pointclouds"], mmap_mode="r")
    new_points = np.load(FISH / "pointclouds.npy", mmap_mode="r")
    assert old_points.shape == (10678, 2, 2048, 3)
    assert new_points.shape == (1106, 2, 2048, 3)
    groups = {LEGACY: [], ADDED: []}
    input_rows = []
    for item in target.itertuples(index=False):
        key = [item.source_run_id, item.body_id, int(item.global_instance)]
        assert item.record_id == f"{key[0]}__{key[1]}__{key[2]:06d}"
        if item.source_run_id == LEGACY:
            row = dict(keys[tuple(key)])
        else:
            m = new_map.loc[item.record_id]
            source = manifest.loc[item.record_id]
            assert (str(m.source_run_id), str(m.body_id), int(m.global_instance)) == tuple(key)
            assert (str(source.source_run_id), str(source.body_id), int(source.global_instance)) == tuple(key)
            assert str(m.cell_type) == str(item.cell_type) == str(source.cell_type)
            row = dict(key=key, cohort_index=int(m.model_cohort_index),
                       point_row=int(m.embedding_row), normalization_radius_voxels=1.0,
                       em_path=str(source.em_path),
                       mask_path=str(FISH / "filled_masks" / f"{item.record_id}.npy"))
        row["spacing_xyz_nm"] = config["spacing_xyz_nm"]
        row["record_id"] = item.record_id
        row["embedding_row"] = int(item.embedding_row)
        assert row["key"] == key
        assert Path(row["em_path"]).is_file() and Path(row["mask_path"]).is_file()
        point_cache = old_points if item.source_run_id == LEGACY else new_points
        assert 0 <= row["point_row"] < len(point_cache)
        assert np.isfinite(point_cache[row["point_row"]]).all()
        groups[item.source_run_id].append(row)
        input_rows.append({k: row[k] for k in [
            "record_id", "embedding_row", "key", "cohort_index", "point_row",
            "normalization_radius_voxels", "spacing_xyz_nm", "em_path", "mask_path"]})
    for rows in groups.values():
        rows.sort(key=lambda r: r["point_row"])
        assert len({r["cohort_index"] for r in rows}) == len(rows)
        assert len({r["point_row"] for r in rows}) == len(rows)
    return target, groups, {LEGACY: old_points, ADDED: new_points}, input_rows


def runtime(config, checkpoint, device):
    # The frozen training worktree owns architecture, image handling and native forward.
    source_manifest = json.loads((RUN / "code_manifest.json").read_text())
    for relative, expected in source_manifest["sha256"].items():
        assert sha(RUNTIME / relative) == expected, relative
    sys.path.insert(0, str(RUNTIME / "fusion50/refine"))
    from train import core, load_models
    point, grouping, image = load_models(config, device)
    model = core.System(point, image, without_fusion_head=True,
                        main_covariance_scale=config["main_covariance_scale"]).to(device)
    assert checkpoint["stage"] == "refine" and checkpoint["epoch"] == 50
    model.load_state_dict(checkpoint["model"], strict=True)
    model.phase = "refine"
    model.eval().requires_grad_(False)
    grouping.eval().requires_grad_(False)
    def forbidden_fusion(*unused):
        raise AssertionError("Fusion MLP must never participate in Balanced1152 export")
    model.fusion.register_forward_pre_hook(forbidden_fusion)
    return core, model, grouping, source_manifest


def infer(core, model, grouping, rows, points, device):
    inputs = core.prepare(rows, points, model.point, grouping, model.image, device, 0, False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p, i, features, _ = model(inputs, views=1, loss=False)
    balanced, _ = model.balance(p[12], i[12], 12, False)
    assert torch.equal(features[12], balanced)
    value = features[12][:, 0].float().cpu().numpy()
    assert value.shape == (len(rows), 1152) and np.isfinite(value).all()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 8:
        parser.error("Use the reviewed reference inference batch size, 8")
    if os.environ.get("NUMPY_MADVISE_HUGEPAGE") != "0":
        raise RuntimeError("Set NUMPY_MADVISE_HUGEPAGE=0 before importing NumPy")
    # Same exclusive-free-card gate as the unified checkpoint entry, before CUDA allocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fish2_downstream.checkpoint import free_gpu
    from fish2_downstream import assets
    free_gpu(args.physical_gpu)
    assets.validate(ASSETS)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = json.loads((RUN / "config.json").read_text())
    assert args.checkpoint.resolve() == (RUN / "checkpoints/refine_epoch050.pt").resolve()
    assert sha(args.checkpoint) == EXPECTED_CHECKPOINT_SHA
    target, groups, points, rows_manifest = reviewed_inputs(config)
    dump(args.output / "input_rows.json", rows_manifest)
    print("MAPPING", {s: len(r) for s, r in groups.items()}, "total", len(target), flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    assert checkpoint["config"]["without_fusion_head"] is True
    assert checkpoint["config"]["main_covariance_scale"] == config["main_covariance_scale"] == .002
    assert checkpoint["config"]["loss_weights"] == config["loss_weights"]
    assert config["loss_weights"]["covariance"] == .0004
    assert config["loss_weights"]["point_covariance"] == config["loss_weights"]["image_covariance"] == .05
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261111)
    torch.cuda.manual_seed_all(20261111)
    device = torch.device("cuda:0")
    core, model, grouping, source_manifest = runtime(config, checkpoint, device)
    original_state = checkpoint["model"]
    model_before = {k: v.detach().cpu().clone() for k, v in model.balance.state_dict().items()}
    qa = {}
    with torch.inference_mode():
        for source, rows in groups.items():
            batch = rows[:8]
            a = infer(core, model, grouping, batch, points[source], device)
            b = infer(core, model, grouping, batch, points[source], device)
            assert np.array_equal(a, b), "Canonical inference must repeat bitwise"
            qa[source] = dict(record_ids=[r["record_id"] for r in batch],
                              shape=list(a.shape), deterministic_max_abs=float(np.max(abs(a-b))),
                              finite=True, cohort_indices=[r["cohort_index"] for r in batch])
        if args.smoke:
            # Only sixteen reference instances: check the exact runtime/preprocessing against
            # the existing canonical baseline; this does not create another model experiment.
            reference = torch.load(REFERENCE / "checkpoints/refine_epoch050.pt",
                                   map_location="cpu", mmap=True, weights_only=False)
            model.load_state_dict(reference["model"], strict=True)
            canonical = np.load(ASSETS / "canonical_balanced1152.npy", mmap_mode="r")
            for source, rows in groups.items():
                batch = rows[:8]
                a = infer(core, model, grouping, batch, points[source], device)
                b = np.asarray(canonical[[r["embedding_row"] for r in batch]])
                rel = float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-12))
                qa[source]["baseline_reference_relative_l2"] = rel
                qa[source]["baseline_reference_max_abs"] = float(np.max(abs(a-b)))
                assert rel < .025, "Preprocessing does not match existing reference"
            model.load_state_dict(original_state, strict=True)
            del reference
        else:
            temporary = args.output / "balanced_pair_12.partial.npy"
            result = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(5568, 1152))
            seen = np.zeros(5568, np.uint8)
            inference_started = time.monotonic()
            completed = 0
            for source, rows in groups.items():
                for start in range(0, len(rows), 8):
                    batch = rows[start:start+8]
                    value = infer(core, model, grouping, batch, points[source], device)
                    indices = [r["embedding_row"] for r in batch]
                    assert not seen[indices].any()
                    result[indices] = value
                    seen[indices] += 1
                    completed += len(batch)
                    if start % 128 == 0 or start+len(batch) == len(rows):
                        elapsed = time.monotonic() - inference_started
                        print(json.dumps(dict(source=source, done=completed, total=5568,
                            seconds=elapsed, instances_per_second=completed/elapsed,
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)), flush=True)
                        result.flush()
            assert np.all(seen == 1) and np.isfinite(result).all()
            result.flush()
            del result
            temporary.rename(args.output / "balanced_pair_12.npy")
            target.to_csv(args.output / "instance_map.csv", index=False)
            qa["coverage"] = dict(instances=5568, legacy=4462, added=1106, bodies=81,
                                  each_instance_written_once=True)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    for k, v in model.balance.state_dict().items():
        assert torch.equal(v.cpu(), model_before[k]), "Inference mutated PairBalance"
    files = [args.checkpoint, RUN / "config.json", ASSETS / "assets.json",
             ASSETS / "instances.csv", REFERENCE / "cohort.json",
             Path(config["pointclouds"]), FISH / "pointclouds.npy", FISH / "instance_map.csv",
             INPUTS, args.output / "input_rows.json", Path(__file__)]
    if not args.smoke:
        files += [args.output / "balanced_pair_12.npy", args.output / "instance_map.csv"]
    manifest = dict(status="smoke_passed" if args.smoke else "complete",
        representation="NoHead50_maincov0.0004_epoch050_L12_Balanced1152D",
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,
        model_config=str(RUN / "config.json"), task_assets=str(ASSETS),
        code_commit=subprocess.check_output(["git","-C",str(Path(__file__).resolve().parents[1]),
                                             "rev-parse","HEAD"],text=True).strip(),
        runtime_code=source_manifest, shape=[5568,1152], dtype="float32", batch_size=8,
        preprocessing=dict(point_view=0, spacing_xyz_nm=config["spacing_xyz_nm"],
            point_grouping="original cohort_index seed; native 64 groups of 32",
            image="original filled masks/minmax/native patch projection/local sincos",
            image_max_tokens=2048, augmentation=False, tf32=False,
            precision="float32 prepare; bf16 native transformer; float32 PairBalance",
            point_position="L12 raw block CLS384", image_position="L12 final-LN CLS768",
            pair_balance="checkpoint EMA buffers; frozen; no fitting on reviewed81"),
        no_fusion_mlp_forward=True, optimizer_updates=0, no_gradients=True,
        qa=qa, sha256={str(p):sha(p) for p in files},
        elapsed_seconds=time.monotonic()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        interpretation="Joint ablation: no Head plus main covariance 0.2 -> 0.0004; raw CLS coefficients unchanged")
    dump(args.output / "complete.json", manifest)
    print(manifest["status"], args.output, "seconds", manifest["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
