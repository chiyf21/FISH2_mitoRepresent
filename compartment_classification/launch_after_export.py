"""Continue this experiment as soon as its own full export completes."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--asset-root", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--python", required=True)
    args = ap.parse_args()
    work = Path(__file__).resolve().parents[1]
    record = json.loads((args.output / "flyem_export_process.json").read_text())
    while not (args.output / "flyem_embeddings/complete.json").exists():
        try:
            os.kill(record["pid"], 0)
            cmdline = Path(f"/proc/{record['pid']}/cmdline").read_bytes()
            if b"fusion50/refine/export.py" not in cmdline:
                raise RuntimeError("Exporter stopped without a completion record")
        except ProcessLookupError as exc:
            raise RuntimeError("Exporter stopped without a completion record") from exc
        time.sleep(10)
    complete = json.loads((args.output / "flyem_embeddings/complete.json").read_text())
    if complete["status"] != "complete" or complete["shape"] != [9731, 384] or complete["reference_smoke_max_abs_delta"] != 0:
        raise RuntimeError("Full export failed validation")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    prepare = [args.python, "-B", "-u", "compartment_classification/experiment.py", "prepare",
               "--dataset", "flyem", "--output", str(args.output), "--asset-root", str(args.asset_root),
               "--run-root", str(args.run_root)]
    with (args.output / "flyem_prepare.log").open("x") as log:
        subprocess.run(prepare, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    children = {}
    for feature in ["handcrafted", "embedding", "dummy"]:
        name = "flyem_" + feature
        cmd = ["nice", "-n", "5", args.python, "-B", "-u", "compartment_classification/experiment.py", "fit",
               "--dataset", "flyem", "--feature", feature, "--output", str(args.output), "--workers", "2"]
        with (args.output / (name + ".log")).open("x") as log:
            process = subprocess.Popen(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        write(args.output / (name + "_process.json"), dict(pid=process.pid, command=cmd, started=time.time()))
        children[name] = process
        print(json.dumps(dict(started=name, pid=process.pid)), flush=True)
    codes = {name: child.wait() for name, child in children.items()}
    write(args.output / "flyem_jobs_status.json", dict(exit_codes=codes, finished_unix=time.time()))
    if any(codes.values()):
        raise RuntimeError(f"Some classification jobs failed: {codes}")


if __name__ == "__main__":
    main()
