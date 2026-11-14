"""Explicit, identity-keyed embedding assets; never infer a space from its dimension."""
from pathlib import Path

import numpy as np
from .common import FORBIDDEN, dump, read, sha, table, unique


def register(matrix, output, representation, instance_map=None, prefix="embedding_",
             columns=None, checkpoint=None, model_config=None, notes=""):
    matrix, output = Path(matrix).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if matrix.suffix == ".csv":
        frame = table(matrix)
        unique(frame)
        columns = columns or [c for c in frame if c.startswith(prefix)]
        if not columns or FORBIDDEN.intersection(columns):
            raise ValueError("Specify embedding columns only; metadata/labels cannot be features")
        x = frame[columns].to_numpy(dtype=np.float64)
        mapping = frame[["record_id"] + (["body_id"] if "body_id" in frame else [])].copy()
        mapping["embedding_row"] = np.arange(len(frame))
    else:
        if matrix.suffix != ".npy" or instance_map is None:
            raise ValueError("Use CSV with explicit embedding columns, or NPY plus instance_map.csv")
        x = np.load(matrix, mmap_mode="r", allow_pickle=False)
        mapping = table(instance_map)
        unique(mapping)
        rows = mapping.embedding_row.to_numpy()
        if not np.issubdtype(rows.dtype, np.integer) or len(np.unique(rows)) != len(rows):
            raise ValueError("embedding_row must contain unique integers")
        if len(rows) == 0 or rows.min() < 0 or rows.max() >= len(x):
            raise ValueError("embedding_row is outside the matrix")
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a finite N x D matrix")
    sources = {str(matrix): sha(matrix)}
    if instance_map:
        sources[str(Path(instance_map).resolve())] = sha(instance_map)
    asset = dict(schema="fish2.embedding.v1", representation=representation,
        matrix=str(matrix), instance_map=str(Path(instance_map).resolve()) if instance_map else None,
        csv_columns=columns if matrix.suffix == ".csv" else None,
        shape=list(x.shape), mapped_instances=len(mapping), sha256=sources, notes=notes,
        checkpoint=None, model_config=str(Path(model_config).resolve()) if model_config else None)
    if checkpoint:
        p = Path(checkpoint).resolve()
        s = p.stat()
        asset["checkpoint"] = dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(output, asset)
    return asset


def load(descriptor):
    descriptor = Path(descriptor)
    asset = read(descriptor)
    if asset["schema"] != "fish2.embedding.v1":
        raise ValueError("Unknown embedding schema")
    for filename, expected in asset["sha256"].items():
        if sha(filename) != expected:
            raise ValueError("Embedding input changed: " + filename)
    path = Path(asset["matrix"])
    if path.suffix == ".csv":
        frame = table(path)
        mapping = frame[["record_id"] + (["body_id"] if "body_id" in frame else [])].copy()
        mapping["embedding_row"] = np.arange(len(frame))
        if FORBIDDEN.intersection(asset["csv_columns"]):
            raise ValueError("Metadata column in embedding")
        x = frame[asset["csv_columns"]].to_numpy(dtype=np.float64)
    else:
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        mapping = table(asset["instance_map"])
    unique(mapping)
    if list(x.shape) != asset["shape"] or not mapping.embedding_row.is_unique:
        raise ValueError("Matrix/map identity changed")
    return asset, x, mapping


def align(x, mapping, manifest):
    unique(manifest)
    index = mapping.set_index("record_id")
    missing = manifest.loc[~manifest.record_id.isin(index.index), "record_id"].tolist()
    if missing:
        raise ValueError(f"Missing {len(missing)} target instances, e.g. {missing[:3]}; no silent intersection")
    selected = index.loc[manifest.record_id]
    if "body_id" in selected:
        if selected.body_id.tolist() != manifest.body_id.tolist():
            raise ValueError("Embedding map body_id disagrees with frozen task manifest")
    out = np.asarray(x[selected.embedding_row.to_numpy(dtype=int)], dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite target embedding")
    return out
