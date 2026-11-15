"""Matched five-epoch head pilot on frozen E16 NNA20 two-view features."""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DistributedSampler

from .common import BASE, dump, read, sha, table
from .probe_e16_fuse_heads import run_classification_probe


class LinearFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_norm = nn.LayerNorm(1152)
        self.projection = nn.Linear(1152, 384)

    def forward(self, point, image, layer):
        point = F.layer_norm(point, (384,))
        # The cached Image L12 CLS is already after the original MAE final norm.
        if layer != 12:
            image = F.layer_norm(image, (768,))
        return self.projection(self.joint_norm(torch.cat((point, image), dim=-1)))


class ReferenceFusion(nn.Module):
    def __init__(self, core, initial):
        super().__init__()
        self.balance = core.PairBalance()
        self.head = core.MLPFusion()
        for prefix, module in [("balance.", self.balance), ("fusion.", self.head)]:
            module.load_state_dict({k[len(prefix):]: v for k, v in initial.items() if k.startswith(prefix)})

    def forward(self, point, image, layer):
        joined, _ = self.balance(point, image, layer, self.training)
        return self.head(joined)


def batches(train, epoch, seed):
    ranks = [DistributedSampler(train, num_replicas=4, rank=r, seed=seed, drop_last=True)
             for r in range(4)]
    for sampler in ranks:
        sampler.set_epoch(epoch + 1000)
    positions = np.asarray([list(sampler) for sampler in ranks])
    for start in range(0, positions.shape[1], 16):
        yield train[positions[:, start:start + 16].reshape(-1)]


def fit_step(model, p, i, ids, core):
    sequence = [model(p[:, j], i[:, j], layer) for j, layer in enumerate(core.LAYERS)]
    values = []
    for z in sequence:
        vicreg, _ = core.vicreg(z)
        nce, _ = core.filtered_infonce(z, ids, topk=20, temperature=0.2)
        values.append(vicreg + nce)
    return torch.stack(values).mean()


def export_l12(model, p, i):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([
            model(torch.from_numpy(np.array(p[start:start + 256, 3], np.float32)),
                  torch.from_numpy(np.array(i[start:start + 256, 3], np.float32)), 12).numpy()
            for start in range(0, len(p), 256)
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = BASE / "point-MAE/runs"
    baseline = runs / "mito_newmae_vicreg_reg02_cls005_20261105_01"
    head_run = runs / "e16_fuse_mlp_only10_diagnostics_20261114"
    probe = runs / "mito_representation_probe_20261111_1840_codex"
    rows = table(probe / "instances.csv")
    cohort = read(baseline / "cohort.json")
    keys = [json.dumps(row["key"], ensure_ascii=False, separators=(",", ":")) for row in cohort]
    if rows.stable_key.tolist() != keys or rows.cohort_index.tolist() != list(range(10678)):
        raise ValueError("Cached rows must match the original E16 feature protocol")
    for split, count in [("train", 7357), ("val", 1706), ("test", 1615)]:
        indices = np.load(head_run / f"cache/{split}_ids.npy", allow_pickle=False)
        if len(indices) != count or not np.array_equal(indices, np.flatnonzero(rows.split == split)):
            raise ValueError(f"Cached {split} indices differ from the E16 cohort")
    train = np.flatnonzero(rows.split == "train")
    heldout = np.flatnonzero(rows.split == "test")
    init = baseline / "checkpoints/refine_epoch000.pt"
    nna20 = baseline / "checkpoints/pretrain_epoch020.pt"
    reference = torch.load(init, map_location="cpu", mmap=True, weights_only=False)
    if (reference["stage"], reference["epoch"], reference["step"]) != ("refine", 0, 2300):
        raise ValueError("Expected the original pre-update refine state")
    if Path(read(head_run / "initialization_report.json")["checkpoint"]) != nna20:
        raise ValueError("The feature cache must come from the original NNA20 checkpoint")
    raw_paths = [head_run / "cache" / f"{kind}_{mod}.npy"
                 for kind in ("train_e01", "eval") for mod in ("p", "i")]
    pt, im, ep, ei = [np.load(path, mmap_mode="r", allow_pickle=False) for path in raw_paths]
    if [arr.shape for arr in (pt, im, ep, ei)] != [
        (10678, 4, 2, 384), (10678, 4, 2, 768), (10678, 4, 384), (10678, 4, 768),
    ]:
        raise ValueError("Unexpected cached feature shapes")

    source = head_run / "pipeline.py"
    spec = importlib.util.spec_from_file_location("e16_fusion_runtime", source)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    core = runtime.core
    torch.set_num_threads(4)
    seed = int(read(baseline / "config.json")["seed"])
    torch.manual_seed(seed)
    models = {"reference": ReferenceFusion(core, reference["model"]), "linear": LinearFusion()}
    del reference
    opts = {name: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, betas=(.9, .95))
            for name, model in models.items()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    config = dict(epochs=5, updates_per_epoch=115, schedule_epochs=50, batch=64, seed=seed,
                  source_scope="Frozen E16 NNA20 encoders and CLS; fixed cached views reused each epoch",
                  frozen_cls_regularizers="constant w.r.t. both heads; omitted from both loss logs",
                  loss="VICReg 25/5/0.2 + filtered InfoNCE top20, temp 0.2; mean over L9-L12",
                  linear="Point LN L9-L12, Image LN L9-L11 (L12 native norm), concat, affine LN, Linear 1152->384",
                  reference="Fusion50 refine_epoch000 MLP384 and PairBalance buffers, freshly optimized",
                  sources={str(path): sha(path) for path in [init, nna20, source, *raw_paths]})
    dump(output / "config.json", config)
    registry = []
    begin = time.monotonic()
    for epoch in range(6):
        losses = {name: [] for name in models}
        grad_norm = {name: [] for name in models}
        if epoch:
            for model in models.values():
                model.train()
            for batch, ids in enumerate(batches(train, epoch, seed)):
                fraction = epoch - 1 + batch / 115
                lr = 2e-4 * min(1., fraction + 1 / 115) * (.5 + .5 * math.cos(math.pi * fraction / 50))
                p, i = (torch.from_numpy(np.array(x[ids], np.float32)) for x in (pt, im))
                labels = torch.from_numpy(ids)
                for name, model in models.items():
                    opt = opts[name]
                    opt.param_groups[0]["lr"] = lr
                    opt.zero_grad(set_to_none=True)
                    loss = fit_step(model, p, i, labels, core)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite {name} objective")
                    opt.step()
                    losses[name].append(loss.item())
                    grad_norm[name].append(norm.item())
            if any(len(values) != 115 for values in losses.values()):
                raise ValueError("Sampler did not produce 115 matched updates")
        record = dict(epoch=epoch, updates=115 * epoch, metrics={})
        for name, model in models.items():
            ckpt = output / "checkpoints" / f"{name}_epoch{epoch:03d}.pt"
            torch.save(dict(epoch=epoch, steps=115 * epoch, model=model.state_dict(),
                            optimizer=opts[name].state_dict(), condition=name), ckpt)
            z = export_l12(model, ep, ei)
            if z.shape != (10678, 384) or not np.isfinite(z).all():
                raise ValueError(f"Invalid L12 features {name}/{epoch}")
            record["metrics"][name] = dict(loss=float(np.mean(losses[name])) if epoch else None,
                grad_norm=float(np.mean(grad_norm[name])) if epoch else None,
                **core.cov_metrics(z[heldout]))
            if epoch in (1, 3, 5):
                feature = f"{name}_epoch{epoch:03d}"
                path = output / f"{feature}.npy"
                np.save(path, z)
                registry.append(dict(name=feature, path=str(path), dimension=384, checkpoint=str(ckpt),
                                     checkpoint_sha256=sha(ckpt), embedding_sha256=sha(path),
                                     phase="cached NNA20 head pilot"))
        record["elapsed_seconds"] = time.monotonic() - begin
        with (output / "epoch_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        dump(output / "status.json", dict(status="head_training", **record))
        print(json.dumps(record), flush=True)
    dump(output / "feature_registry.json", registry)
    dump(output / "status.json", dict(status="classification", **record))
    run_classification_probe(output, probe, rows, [source, *raw_paths])
    dump(output / "status.json", dict(status="complete", **record))


if __name__ == "__main__":
    main()
