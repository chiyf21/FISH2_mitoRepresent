"""Two-batch gradient decomposition; never creates an optimizer or updates parameters."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DistributedSampler

from train import core, load_models, restore, write
from train_no_projection import load_data, groups, forbid_fusion

WEIGHTS = {
    "invariance": 25., "variance": 5., "covariance": .2, "infonce": 1.,
    "point_cls_variance": 1.25, "point_cls_covariance": .05,
    "image_cls_variance": 1.25, "image_cls_covariance": .05,
}


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def moments(z):
    a, b = z[:, 0].float(), z[:, 1].float()
    both = torch.cat((a, b), dim=0)
    variance = F.relu(1-torch.sqrt(both.var(dim=0, unbiased=False)+1e-4)).mean()
    centered = both-both.mean(0, keepdim=True)
    cov = centered.T@centered/(len(centered)-1)
    off = ~torch.eye(a.shape[-1], device=a.device, dtype=torch.bool)
    covariance = cov[off].square().sum()/a.shape[-1]
    return F.mse_loss(a, b), variance, covariance


def norm(grads, indices=None):
    selected = grads if indices is None else [grads[i] for i in indices]
    values = [g.float().square().sum() for g in selected if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def add_gradients(vectors):
    return [sum(nonempty) if (nonempty := [v[i] for v in vectors if v[i] is not None])
            else None for i in range(len(vectors[0]))]


def dot(a, b):
    values = [(x.float()*y.float()).sum() for x, y in zip(a, b)
              if x is not None and y is not None]
    return float(torch.stack(values).sum()) if values else 0.


def analyze(model, grouping, rows, points, positions, epoch, batch, device):
    selected = [rows[j] for j in positions]
    ids = torch.tensor(positions, device=device)
    inputs = core.prepare(selected, points, model.point, grouping, model.image,
                          device, epoch+1000, True, point_only=False)
    captured = {}
    raw = {}
    original_encode = model.encode

    def encode(*args, **kwargs):
        p, i = original_encode(*args, **kwargs)
        raw.update(point=p, image=i)
        return p, i

    model.encode = encode
    hook = model.balance.register_forward_hook(
        lambda module, values, output: captured.__setitem__(values[2], output[0]))
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        total, logged, _ = model(inputs, ids)
        terms = {k: [] for k in WEIGHTS}
        for layer in core.LAYERS:
            inv, var, cov = moments(captured[layer])
            infonce, _ = core.filtered_infonce(captured[layer], ids, topk=20, temperature=.2)
            for key, value in (("invariance", inv), ("variance", var),
                               ("covariance", cov), ("infonce", infonce)):
                terms[key].append(value)
                assert torch.allclose(value.detach(), logged[f"{layer}/{key}"],
                                      rtol=1e-6, atol=1e-6), (layer, key)
            for modality in ("point", "image"):
                _, var, cov = moments(raw[modality][layer])
                terms[f"{modality}_cls_variance"].append(var)
                terms[f"{modality}_cls_covariance"].append(cov)
                assert torch.allclose(var.detach(), logged[f"{layer}/{modality}_variance"],
                                      rtol=1e-6, atol=1e-6)
                assert torch.allclose(cov.detach(), logged[f"{layer}/{modality}_covariance"],
                                      rtol=1e-6, atol=1e-6)
        unweighted = {k: torch.stack(v).mean() for k, v in terms.items()}
        weighted = {k: v*WEIGHTS[k] for k, v in unweighted.items()}
        rebuilt = torch.stack(list(weighted.values())).sum()
    hook.remove()
    model.encode = original_encode
    assert torch.allclose(rebuilt.detach(), total.detach(), rtol=1e-6, atol=1e-3)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for n, p in named]
    lookup = {id(p): j for j, p in enumerate(parameters)}
    group_indices = {k: [lookup[id(p)] for p in values] for k, values in groups(model).items()}
    derivatives = {}
    term_report = {}
    for key, term in weighted.items():
        derivatives[key] = torch.autograd.grad(
            term, parameters, retain_graph=True, allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in derivatives[key])
        term_report[key] = dict(
            coefficient=WEIGHTS[key], raw_loss=float(unweighted[key].detach()),
            weighted_loss=float(term.detach()), gradient_l2=norm(derivatives[key]),
            gradient_l2_by_group={k: norm(derivatives[key], ix)
                                  for k, ix in group_indices.items()})
        print(json.dumps(dict(batch=batch, component=key, **term_report[key])), flush=True)
    true_total = torch.autograd.grad(total, parameters, allow_unused=True)
    full_norm = norm(true_total)
    assert all(g is None or torch.isfinite(g).all() for g in true_total)
    combined = {}
    for key, names in {
        "point_cls_regularizer": ["point_cls_variance", "point_cls_covariance"],
        "image_cls_regularizer": ["image_cls_variance", "image_cls_covariance"],
        "all_covariance_terms": ["covariance", "point_cls_covariance", "image_cls_covariance"],
        "non_covariance_terms": ["invariance", "variance", "infonce",
                                  "point_cls_variance", "image_cls_variance"],
        "all_except_main_covariance": [k for k in WEIGHTS if k != "covariance"],
    }.items():
        vector = add_gradients([derivatives[k] for k in names])
        combined[key] = dict(gradient_l2=norm(vector),
                             gradient_l2_by_group={k: norm(vector, ix)
                                                   for k, ix in group_indices.items()})
    summed = add_gradients(list(derivatives.values()))
    difference = [None if a is None and b is None
                  else (a if b is None else -b if a is None else a-b)
                  for a, b in zip(summed, true_total)]
    cov_norm = term_report["covariance"]["gradient_l2"]
    result = dict(
        epoch=epoch, batch=batch, instances=len(positions), parameters_unchanged=True,
        optimizer_steps=0, total_loss=float(total.detach()), reconstructed_loss=float(rebuilt.detach()),
        terms=term_report, combined=combined, total_gradient_l2=full_norm,
        total_gradient_l2_by_group={k: norm(true_total, ix) for k, ix in group_indices.items()},
        main_covariance_gradient_over_total=cov_norm/full_norm,
        main_covariance_gradient_over_remaining=cov_norm/combined["all_except_main_covariance"]["gradient_l2"],
        main_covariance_cosine_with_total=dot(derivatives["covariance"], true_total)/(cov_norm*full_norm),
        sum_components_relative_gradient_error=norm(difference)/full_norm,
        clip_threshold=1., would_clip=full_norm>1., clip_multiplier=min(1., 1./(full_norm+1e-6)),
        peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
    assert all(p.grad is None for p in model.parameters())
    print(json.dumps(dict(batch=batch, summary={k: v for k, v in result.items()
                                               if k not in ("terms", "combined")})), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not cfg.get("without_fusion_head"):
        raise ValueError("Use the no-head configuration")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows, points = load_data(cfg)
    point, grouping, image = load_models(cfg, device)
    model = core.System(point, image, without_fusion_head=True).to(device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert ck["stage"] == "refine" and ck["epoch"] == 0 and ck["step"] == 2300
    assert ck["config"]["without_fusion_head"] and not ck["config"].get("smoke")
    model.load_state_dict(ck["model"], strict=True)
    restore(ck["rng_per_rank"][0], device)
    model.phase = "refine"
    image.cls_token.requires_grad_(True)
    for block in image.blocks[-4:]:
        block.requires_grad_(True)
    model.fusion.requires_grad_(False)
    model.fusion.register_forward_pre_hook(forbid_fusion)
    expected = {id(p) for values in groups(model).values() for p in values}
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected
    del ck
    before = parameter_sha(model)
    tr = [j for j, row in enumerate(rows) if row["split"] == "train"]
    by_rank = []
    for rank in range(4):
        sampler = DistributedSampler(tr, num_replicas=4, rank=rank, shuffle=True,
                                     seed=cfg["seed"], drop_last=True)
        sampler.set_epoch(1001)
        by_rank.append([tr[k] for k in sampler])
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for batch in (1, 40):
        positions = [j for rank_ids in by_rank for j in rank_ids[(batch-1)*16:batch*16]]
        torch.cuda.reset_peak_memory_stats(device)
        result = analyze(model, grouping, rows, points, positions, 1, batch, device)
        results.append(result)
        write(args.output/f"batch{batch:03d}.json", result)
    after = parameter_sha(model)
    assert before == after
    write(args.output/"complete.json", dict(
        status="diagnostic_complete", checkpoint=str(args.checkpoint), checkpoint_epoch=0,
        checkpoint_step=2300, optimizer_steps=0, parameter_sha256_before=before,
        parameter_sha256_after=after, parameters_unchanged=True, results=results,
        scope="single-GPU global batch64; original four-rank sampler order and augmentations; "
              "two fixed batches at epoch000 parameters, not the lost step2340 state"))
    print("DIAGNOSTIC_COMPLETE_NO_UPDATES", flush=True)


if __name__ == "__main__":
    main()
