"""Explicit bridge to a model's own frozen exporter; no guessed architecture or forward."""
import os
import subprocess
from pathlib import Path

from .common import code_identity, dump, new_directory, read, sha
from .inputs import register


def free_gpu(index):
    if not index or not index.isdigit():
        raise ValueError("Execution requires an explicit physical --gpu index")
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                     "--format=csv,noheader,nounits"], text=True).splitlines()
    match = [line.split(",") for line in lines if line.split(",")[0].strip() == index]
    if len(match) != 1:
        raise ValueError("GPU index not found")
    _, uuid, memory = [s.strip() for s in match[0]]
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                        "--format=csv,noheader"], text=True).splitlines()
    if uuid in [s.strip() for s in occupied] or float(memory) > 256:
        raise RuntimeError("GPU is occupied; shared inference is not enabled")


def checkpoint_job(args):
    spec_path = Path(args.export_spec).resolve()
    spec = read(spec_path)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint.stat()
    if not isinstance(spec["argv"], list) or not all(isinstance(v, str) for v in spec["argv"]):
        raise ValueError("Exporter argv must be a string list; shell commands are not accepted")
    if not any("{checkpoint}" in value for value in spec["argv"]):
        raise ValueError("The exporter must explicitly receive {checkpoint}")
    cwd = Path(spec["cwd"]).resolve()
    if not cwd.is_dir():
        raise FileNotFoundError(cwd)
    job = new_directory(args.jobs_root, args.owner, args.name)
    export_dir = job / "vectors"  # The original exporter creates this new directory.
    values = dict(checkpoint=str(checkpoint), export_dir=str(export_dir), gpu="0")
    argv = [value.format_map(values) for value in spec["argv"]]
    plan = dict(owner=args.owner, checkpoint=str(checkpoint), export_spec=str(spec_path),
        spec_sha256=sha(spec_path), cwd=str(cwd), argv=argv, physical_gpu=args.gpu,
        code=code_identity(), executed=False, embedding_output=str(job / "embedding.json"))
    dump(job / "plan.json", plan)
    print(job / "plan.json", flush=True)
    if not args.execute:
        return
    free_gpu(args.gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with (job / "export.log").open("w") as log:
        subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    asset = register(export_dir / spec["matrix"], job / "embedding.json", spec["representation"],
        export_dir / spec["instance_map"], checkpoint=checkpoint,
        notes="Exact external exporter argv in plan.json; model forward not reimplemented")
    if asset["shape"][1] != int(spec["dimension"]):
        raise ValueError("Exporter dimension disagrees with its declared representation")
    plan["executed"] = True
    dump(job / "plan.json", plan)
    print("Registered:", job / "embedding.json")
