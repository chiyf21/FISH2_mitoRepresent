"""Evaluate the four saved E16 fuse-only heads using the original E16 probe."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .common import BASE, dump, read, sha, table

PROBE_REF = "8397cf7c18c2b2b2e1ff62c672186b0b5453be71"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    runs = BASE / "point-MAE/runs"
    head_run = runs / "e16_fuse_mlp_only10_diagnostics_20261114"
    probe_run = runs / "mito_representation_probe_20261111_1840_codex"
    cohort_path = runs / "mito_newmae_vicreg_reg02_cls005_20261105_01/cohort.json"
    rows = table(probe_run / "instances.csv")
    cohort = read(cohort_path)
    keys = [json.dumps(item["key"], ensure_ascii=False, separators=(",", ":")) for item in cohort]
    if keys != rows.stable_key.tolist() or not np.array_equal(rows.cohort_index, np.arange(len(rows))):
        raise ValueError("Probe and encoder cohort have different row identities")
    cache = head_run / "cache"
    for short, split in [("train", "train"), ("val", "val"), ("test", "test")]:
        cached = np.load(cache / f"{short}_ids.npy", allow_pickle=False)
        if not np.array_equal(cached, np.flatnonzero(rows.split == split)):
            raise ValueError(f"Saved head and source cohort disagree on {split}")
    pt, im = [
        np.load(cache / f"eval_{mod}.npy", mmap_mode="r", allow_pickle=False)
        for mod in ("p", "i")
    ]
    if pt.shape != (len(rows), 4, 384) or im.shape != (len(rows), 4, 768):
        raise ValueError("Unexpected cached feature dimensions")

    source = head_run / "pipeline.py"
    spec = importlib.util.spec_from_file_location("e16_fuse_only_pipeline", source)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    registry = []
    for balance in (True, False):
        for compress in (True, False):
            prefix = "balance" if balance else "unbalance"
            suffix = "compress384" if compress else "keep1152"
            condition = f"{prefix}_{suffix}"
            checkpoint = head_run / "conditions" / condition / "epoch10.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["epoch"] != 10 or state["condition"] != condition:
                raise ValueError(f"Wrong checkpoint for {condition}")
            head = (pipeline.core.MLPFusion() if compress else pipeline.NoCompress()).eval()
            calibrator = pipeline.core.PairBalance().eval()
            head.load_state_dict(state["fusion"])
            calibrator.load_state_dict(state["balance"])
            values = []
            with torch.inference_mode():
                for start in range(0, len(rows), 256):
                    stop = start + 256
                    p = torch.from_numpy(np.array(pt[start:stop, 3], dtype=np.float32))
                    i = torch.from_numpy(np.array(im[start:stop, 3], dtype=np.float32))
                    values.append(pipeline.fuse(head, calibrator, balance, p, i, 12, False).numpy())
            embedding = np.concatenate(values)
            if not np.isfinite(embedding).all():
                raise ValueError(f"Nonfinite embedding for {condition}")
            path = output / f"{condition}.npy"
            np.save(path, embedding)
            registry.append(dict(name=condition, path=str(path), dimension=embedding.shape[1],
                                 checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                                 embedding_sha256=sha(path), phase="fuse-only epoch 10; encoder frozen"))
    dump(output / "feature_registry.json", registry)
    shutil.copyfile(probe_run / "instances.csv", output / "instances.csv")
    protocol = read(probe_run / "protocol.json")
    protocol["regression_targets"] = {}
    dump(output / "protocol.json", protocol)
    runner = output / "probe_code"
    runner.mkdir()
    for f in ["common.py", "evaluate.py"]:
        contents = subprocess.check_output(["git", "show", f"{PROBE_REF}:representation_probe/{f}"])
        (runner / f).write_bytes(contents)

    # Compare the exact fold identities with the original saved probe.
    sys.path.insert(0, str(runner))
    import evaluate

    expected = read(probe_run / "final/folds.json")
    actual = evaluate.make_tasks(rows, protocol, smoke=False)
    if actual != {k: expected["tasks"][k] for k in actual}:
        raise ValueError("Evaluated classification folds differ from the original E16 probe")
    key_hash = hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()
    if expected["stable_keys_sha256"] != key_hash:
        raise ValueError("Stable key sequence differs from the original E16 probe")
    dump(output / "source_identity.json", dict(evaluator_revision=PROBE_REF,
         exact_classification_folds=True, sources={str(p): sha(p) for p in [
         source, cohort_path, cache / "eval_p.npy", cache / "eval_i.npy",
         probe_run / "protocol.json", probe_run / "final/folds.json"]}))
    evaluate.main(output, phase="classification", smoke=False, features="")


if __name__ == "__main__":
    main()
