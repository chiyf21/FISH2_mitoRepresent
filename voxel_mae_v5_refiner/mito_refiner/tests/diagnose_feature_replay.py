from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
from ..backbone import load_adapter
from ..data_utils import load_global_views, make_dataset
from ..features import extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-adapter",
                        default=f"{__package__.rsplit('.',1)[0]}.v5_adapter:V5VoxelMAEAdapter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-records", type=int, default=4,
                        help="Check this many records from the seeded first virtual batch")
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    adapter = load_adapter(args.backbone_adapter)
    model, _, _ = adapter.load_model(args.checkpoint, device)
    model.eval()
    ds = make_dataset(args.cache, "train", args.limit, seed=42, train_views=True)
    max_errors, relative_errors, cosine = None, None, None
    input_error = 0.0
    indices = np.random.default_rng(args.seed).permutation(len(ds.records))[:args.num_records]
    for index in indices:
        for batch in load_global_views(ds, index, device):
            cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
            replay = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cpu.items()}
            input_error = max(input_error, max(float((a-b).abs().max())
                for va,vb in zip(batch["volumes"], replay["volumes"]) for a,b in zip(va,vb)))
            with torch.no_grad():
                reference = extract_features(adapter, model, batch)
            differentiable = extract_features(adapter, model, replay)
            if max_errors is None:
                names = tuple(reference)
                max_errors = {name: 0.0 for name in names}
                relative_errors = {name: 0.0 for name in names}
                cosine = {name: [] for name in names}
            if set(reference) != set(max_errors) or set(differentiable) != set(max_errors):
                raise ValueError("Adapter feature schema changed between replay passes")
            for name in max_errors:
                a = reference[name].detach().float()
                b = differentiable[name].detach().float()
                diff = a - b
                max_errors[name] = max(max_errors[name], float(diff.abs().max()))
                relative_errors[name] = max(relative_errors[name],
                                            float(diff.norm() / a.norm().clamp_min(1e-12)))
                cosine[name].append(float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()))
    result = {
        "input_roundtrip_max_abs_error": input_error,
        "feature_max_abs_error": max_errors,
        "feature_max_relative_l2_error": relative_errors,
        "feature_mean_cosine": {name: sum(rows) / len(rows) for name, rows in cosine.items()},
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn": torch.backends.cudnn.allow_tf32},
        "forward_modes": "model.eval(); reference no_grad; replay grad-enabled",
        "record_indices": [int(index) for index in indices],
        "n_views": len(indices) * 2,
        "backbone_adapter": adapter.identifier,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
