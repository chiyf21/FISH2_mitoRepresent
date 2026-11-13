"""Export canonical L12 Fusion50 vectors; optional smoke checks preserve model forwards."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from train import core, load_models, load_points, records


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--indices", help="Smoke-only original cohort indices, preserving original IDs/seeds")
    parser.add_argument("--verify-repeat", action="store_true")
    parser.add_argument("--reference-smoke", type=Path)
    parser.add_argument("--expected-epoch", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = json.loads(args.config.read_text())
    all_rows = records(config)
    points = load_points(config, all_rows)
    selected = list(range(len(all_rows))) if not args.indices else [int(x) for x in args.indices.split(",")]
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(all_rows) for i in selected):
        raise ValueError("Invalid or duplicate cohort indices")
    rows = [all_rows[i] for i in selected]
    # Load the complete point cache before selecting rows; original cohort_index/point_row
    # are retained, so sampling, FPS seeds, and cached points are exactly canonical.
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    point, grouping, image = load_models(config, device)
    model = core.System(
        point, image,
        normalize_image_cls_regularizer=config.get("normalize_image_cls_regularizer", False),
        normalize_point_cls_regularizer=config.get("normalize_point_cls_regularizer", False),
    ).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["stage"] != "refine":
        raise ValueError("Use a fusion-stage checkpoint")
    if args.expected_epoch is not None and checkpoint["epoch"] != args.expected_epoch:
        raise ValueError("Wrong checkpoint epoch")
    model.load_state_dict(checkpoint["model"], strict=True)
    epoch, step = checkpoint["epoch"], checkpoint["step"]
    del checkpoint
    model.phase = "refine"
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    provenance = dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
        epoch=epoch, step=step, config=config, config_sha256=sha(args.config),
        cohort_size=len(all_rows), export_size=len(rows), selected_cohort_indices=selected if args.indices else None,
        point_input_manifest_sha256=sha(Path(config["point_run"]) / "data/instances.json"),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent).decode().strip(),
        source_sha256={name: sha(Path(__file__).parent / name) for name in ["export.py", "train.py", "code/core.py", "code/base.py"]},
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu=torch.cuda.get_device_name(0),
        batch_size=args.batch_size, dtype="float32", representation="canonical single-view L12 fused_12",
        started_unix=started,
    )
    write(args.output / "provenance.json", provenance)
    temporary = args.output / "fused_12.partial.npy"
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(rows), 384))
    repeat_max = 0.0

    def forward(batch):
        inputs = core.prepare(batch, points, point, grouping, image, device, 0, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, fused, _ = model(inputs, views=1, loss=False)
        return fused[12][:, 0].float().cpu().numpy()

    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            vectors = forward(batch)
            if vectors.shape != (len(batch), 384) or not np.isfinite(vectors).all():
                raise FloatingPointError(f"Invalid embeddings at row {start}")
            if args.verify_repeat:
                repeated = forward(batch)
                delta = float(np.max(np.abs(vectors - repeated)))
                repeat_max = max(repeat_max, delta)
                if not np.array_equal(vectors, repeated):
                    raise RuntimeError(f"Repeated canonical inference is not bitwise deterministic: {delta}")
            result[start:start + len(batch)] = vectors
            done = start + len(batch)
            if done % 128 == 0 or done == len(rows):
                result.flush()
                progress = dict(done=done, total=len(rows), elapsed_seconds=time.time() - started)
                write(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    result.flush()
    del result
    temporary.rename(args.output / "fused_12.npy")
    with (args.output / "instance_map.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["embedding_row", "source_run_id", "body_id", "record_id", "split", "cohort_index", "point_row"])
        writer.writerows((i, *r["key"], r["split"], r["cohort_index"], r["point_row"]) for i, r in enumerate(rows))
    reference_delta = None
    if args.reference_smoke:
        old = np.load(args.reference_smoke / "fused_12.npy", mmap_mode="r", allow_pickle=False)
        new = np.load(args.output / "fused_12.npy", mmap_mode="r", allow_pickle=False)
        row_lookup = {r["cohort_index"]: i for i, r in enumerate(rows)}
        with (args.reference_smoke / "instance_map.csv").open(newline="") as f:
            mapping = list(csv.DictReader(f))
        reference_delta = 0.0
        for j, r in enumerate(mapping):
            i = row_lookup[int(r["cohort_index"])]
            if rows[i]["key"][2] != r["record_id"] or rows[i]["body_id"] != r["body_id"]:
                raise ValueError("Smoke identity mismatch")
            reference_delta = max(reference_delta, float(np.max(np.abs(new[i] - old[j]))))
        if reference_delta != 0:
            raise RuntimeError(f"Full export differs from smoke: {reference_delta}")
    write(args.output / "complete.json", dict(
        status="complete", shape=[len(rows), 384], unique_record_ids=len({r["key"][2] for r in rows}),
        repeat_checked=args.verify_repeat, repeat_max_abs_delta=repeat_max,
        reference_smoke_max_abs_delta=reference_delta,
        fused_12_sha256=sha(args.output / "fused_12.npy"),
        instance_map_sha256=sha(args.output / "instance_map.csv"),
        elapsed_seconds=time.time() - started, finished_unix=time.time(),
    ))


if __name__ == "__main__":
    main()
