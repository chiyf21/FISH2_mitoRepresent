"""Export the frozen Fusion50 L12 readout using per-modality and joint LayerNorm."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import layer_norm

from . import inputs
from .common import BASE, ROOT, dump, read, sha, table, unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = BASE / "point-MAE/runs"
    legacy_run = runs / "mito_newmae_vicreg_reg02_cls005_20261105_01"
    legacy_cache = runs / "mito_embedding_dashboard_20261030_01/cache/features_b9a9cbbcd15a81c16c36"
    followup_cache = runs / "fish2_reviewed_81_protocol/fish2_14_layer12"
    followup_map = BASE / "neuron_mito_dataset/embeddings/fusion50_fish2_14_20261112/instance_map.csv"
    reference_path = ROOT / "assets/reviewed81_v1/canonical_balanced1152.json"
    reference, balanced, rows = inputs.load(reference_path)
    rows = rows.sort_values("embedding_row", ignore_index=True)
    if not np.array_equal(rows.embedding_row, np.arange(len(balanced))):
        raise ValueError("Canonical instance map must cover the complete matrix")
    checkpoint = Path(reference["checkpoint"]["path"])
    followup_info = read(followup_cache / "complete.json")
    if Path(followup_info["checkpoint"]) != checkpoint:
        raise ValueError("Follow-up CLS cache used a different checkpoint")
    followup = table(followup_map)
    unique(followup)
    followup = followup.set_index("record_id")
    cohort_path = legacy_run / "cohort.json"
    cohort = read(cohort_path)
    legacy_index = {tuple(item["key"]): j for j, item in enumerate(cohort)}
    if len(legacy_index) != len(cohort):
        raise ValueError("Duplicate legacy instance keys")

    names = ("point_12.npy", "image_12.npy", "balanced_pair_12.npy")
    sources = [cohort_path, followup_map, followup_cache / "complete.json", reference_path]
    caches = {}
    for name, directory, count in [
        ("legacy", legacy_cache, len(cohort)),
        ("followup", followup_cache, followup_info["instances"]),
    ]:
        paths = [directory / filename for filename in names]
        sources.extend(paths)
        arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
        if [a.shape for a in arrays] != [(count, 384), (count, 768), (count, 1152)]:
            raise ValueError(f"Unexpected {name} cache shape")
        caches[name] = arrays
    point, image = np.empty((len(rows), 384), np.float32), np.empty((len(rows), 768), np.float32)
    for row in rows.itertuples(index=False):
        key = row.source_run_id, row.body_id, int(row.global_instance)
        if key in legacy_index:
            p, i, pair = caches["legacy"]
            j = legacy_index[key]
        else:
            match = followup.loc[row.record_id]
            if (match.source_run_id, match.body_id, int(match.global_instance)) != key:
                raise ValueError(f"Follow-up cache identity mismatch for {row.record_id}")
            p, i, pair = caches["followup"]
            j = int(match.embedding_row)
        dst = int(row.embedding_row)
        if not 0 <= j < len(p) or not np.array_equal(pair[j], balanced[dst]):
            raise ValueError(f"Cache row differs from the verified baseline for {row.record_id}")
        point[dst], image[dst] = p[j], i[j]

    torch.set_num_threads(2)
    with torch.inference_mode():
        joined = torch.cat((layer_norm(torch.from_numpy(point), (384,)),
                            layer_norm(torch.from_numpy(image), (768,))), dim=1)
        normalized = layer_norm(joined, (1152,))
    args.output.mkdir(parents=True, exist_ok=False)
    matrix = args.output / "norm_concat_norm.npy"
    np.save(matrix, normalized.numpy())
    inputs.register(matrix, args.output / "embedding.json", "Fusion50_epoch050_L12_LN_concat_LN",
                    Path(reference["instance_map"]), checkpoint=checkpoint,
                    notes="Frozen L12 CLS; affine-free LayerNorm (eps=1e-5) per modality, concat, affine-free joint LayerNorm")
    dump(args.output / "sources.json", dict(checkpoint=reference["checkpoint"],
         rows=len(rows), canonical_balanced_rows_equal=len(rows),
         max_change_from_joint_ln=float((normalized - joined).abs().max()),
         sources={str(path): sha(path) for path in sources}))


if __name__ == "__main__":
    main()
