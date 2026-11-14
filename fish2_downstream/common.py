import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

BASE = Path("/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM")
ROOT = BASE / "fish2_downstream"
HERE = Path(__file__).resolve().parent
FEATURES = ["size", "mito SA", "mito CA", "convex hull compactness"] + [
    f"PC{axis} {suffix}" for axis in (1, 2, 3)
    for suffix in ["Length", "inertia moment", "symmetry", "CA", "Circum"]
] + ["mito diameter", "length_along_skeleton_nm", "SBI"]
SEED = 20261113
TASKS = {
    "compartment": dict(label="compartment", classes=["axon", "dendrite"],
                        scope="by_type", group=["body_id", "compartment"], n=5059, groups=161),
    "cell_type": dict(label="cell_type", classes=["CON", "GC", "MON1", "MON2", "PC", "VMN"],
                      scope="pooled", group=["body_id"], n=5568, groups=81),
}
FORBIDDEN = {"record_id", "body_id", "cell_type", "compartment", "compartment_qc",
             "source_run_id", "global_instance", "fold", "label", "embedding_row"}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def table(path):
    return pd.read_csv(path, dtype={"record_id": str, "body_id": str, "source_run_id": str})


def unique(frame):
    if "record_id" not in frame or frame.record_id.isna().any() or not frame.record_id.is_unique:
        raise ValueError("record_id must be present, nonmissing, and unique; body_id is not an instance ID")


def token(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("owner/name must use letters, numbers, dots, underscores or hyphens")
    return value


def new_directory(parent, owner, name):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(parent) / f"{stamp}_{token(owner)}_{token(name)}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def code_identity():
    return dict(commit=subprocess.check_output(
        ["git", "-C", str(HERE.parent), "rev-parse", "HEAD"], text=True).strip(),
        files={p.name: sha(p) for p in HERE.glob("*.py")})
