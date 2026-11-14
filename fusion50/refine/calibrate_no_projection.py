"""Matched MLP384/Balanced1152 loss-gradient comparison from NNA20; zero updates."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DistributedSampler

from train import core, load_models, restore, write
from train_no_projection import load_data, groups
from diagnose_no_projection import WEIGHTS, moments, norm, dot, add_gradients, parameter_sha


def cosine(a, b, indices=None):
    if indices is not None:
        a, b = [a[j] for j in indices], [b[j] for j in indices]
    denominator = norm(a)*norm(b)
    return max(-1., min(1., dot(a, b)/denominator)) if denominator else None


def difference(a, b):
    return [None if x is None and y is None else
            (x if y is None else -y if x is None else x-y) for x, y in zip(a, b)]


def measure(model, inputs, ids, no_head):
    model.without_fusion_head = no_head
    model.train()
    raw, balanced = {}, {}
    original_encode = model.encode

    def capture_encode(*args, **kwargs):
        p, i = original_encode(*args, **kwargs)
        raw.update(point=p, image=i)
        return p, i

    model.encode = capture_encode
    hook = model.balance.register_forward_hook(
        lambda module, values, output: balanced.__setitem__(values[2], output[0]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        total, logged, _ = model(inputs, ids)
        raw_terms = {k: [] for k in WEIGHTS}
        for layer in core.LAYERS:
            z = balanced[layer] if no_head else model.fusion(balanced[layer])
            inv, var, cov = moments(z)
            infonce, _ = core.filtered_infonce(z, ids, topk=20, temperature=.2)
            for key, value in (("invariance", inv), ("variance", var),
                               ("covariance", cov), ("infonce", infonce)):
                assert torch.allclose(value.detach(), logged[f"{layer}/{key}"],
                                      rtol=1e-6, atol=1e-6), (layer, key)
                raw_terms[key].append(value)
            for modality in ("point", "image"):
                _, var, cov = moments(raw[modality][layer])
                raw_terms[f"{modality}_cls_variance"].append(var)
                raw_terms[f"{modality}_cls_covariance"].append(cov)
        unweighted = {k: torch.stack(v).mean() for k, v in raw_terms.items()}
        terms = {k: v*WEIGHTS[k] for k, v in unweighted.items()}
    hook.remove()
    model.encode = original_encode
    rebuilt = torch.stack(list(terms.values())).sum()
    assert torch.allclose(rebuilt.detach(), total.detach(), rtol=1e-6, atol=1e-3)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for n, p in named]
    lookup = {id(p): j for j, p in enumerate(parameters)}
    indices = {k: [lookup[id(p)] for p in values] for k, values in groups(model).items()}
    indices["encoder_blocks_only"] = indices["point_last4"]+indices["image_last4"]
    indices["point_with_cls"] = indices["point_last4"]+indices["point_cls"]
    indices["image_with_cls"] = indices["image_last4"]+indices["image_cls"]
    vectors, results = {}, {}
    for key, loss in terms.items():
        vector = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in vector)
        vectors[key] = vector
        results[key] = dict(
            coefficient=WEIGHTS[key], raw_loss=float(unweighted[key].detach()),
            weighted_loss=float(loss.detach()), gradient_l2=norm(vector),
            gradient_l2_by_group={k: norm(vector, ix) for k, ix in indices.items()})
    vectors["total"] = torch.autograd.grad(total, parameters, allow_unused=True)
    results["total"] = dict(weighted_loss=float(total.detach()), gradient_l2=norm(vectors["total"]),
                            gradient_l2_by_group={k: norm(vectors["total"], ix)
                                                  for k, ix in indices.items()})
    for key in WEIGHTS:
        results[key]["cosine_with_total"] = cosine(vectors[key], vectors["total"])
        results[key]["cosine_with_total_by_group"] = {
            k: cosine(vectors[key], vectors["total"], ix) for k, ix in indices.items()}
    raw_sha = hashlib.sha256()
    for modality in ("point", "image"):
        for layer in core.LAYERS:
            raw_sha.update(raw[modality][layer].detach().cpu().numpy().tobytes())
    return dict(terms=results, raw_cls_sha256=raw_sha.hexdigest(),
                fusion_trainable=False, encoder_gradients_only=True), vectors, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows, points = load_data(cfg)
    point, grouping, image = load_models(cfg, device)
    model = core.System(point, image).to(device)
    source = Path(cfg["init_pretrain"])
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    assert (checkpoint["stage"], checkpoint["epoch"], checkpoint["step"]) == ("pretrain", 20, 2300)
    state = {k: v for k, v in checkpoint["model"].items() if not k.startswith("point_nna.")}
    model.load_state_dict(state, strict=True)
    restore(checkpoint["rng_per_rank"][0], device)
    del state, checkpoint
    model.phase = "refine"
    model.fusion.requires_grad_(False)  # Same encoder derivative; head is not an optimization target.
    image.cls_token.requires_grad_(True)
    for block in image.blocks[-4:]:
        block.requires_grad_(True)
    expected = {id(p) for values in groups(model).values() for p in values}
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected

    def guard(module, inputs):
        if model.without_fusion_head:
            raise RuntimeError("No-head path invoked Fusion MLP")
    model.fusion.register_forward_pre_hook(guard)
    before = parameter_sha(model)
    tr = [j for j, r in enumerate(rows) if r["split"] == "train"]
    by_rank = []
    for rank in range(4):
        sampler = DistributedSampler(tr, num_replicas=4, rank=rank, shuffle=True,
                                     seed=cfg["seed"], drop_last=True)
        sampler.set_epoch(1001)
        by_rank.append([tr[j] for j in sampler])
    args.output.mkdir(parents=True, exist_ok=False)
    comparisons = []
    for batch in (1, 40):
        positions = [j for rank_ids in by_rank for j in rank_ids[(batch-1)*16:batch*16]]
        selected = [rows[j] for j in positions]
        inputs = core.prepare(selected, points, point, grouping, image, device, 1001,
                              True, point_only=False)
        ids = torch.tensor(positions, device=device)
        old, old_grad, indices = measure(model, inputs, ids, False)
        new, new_grad, _ = measure(model, inputs, ids, True)
        assert old["raw_cls_sha256"] == new["raw_cls_sha256"]
        directions = {}
        for key in (*WEIGHTS, "total"):
            directions[key] = dict(
                cosine_mlp_vs_nohead=cosine(old_grad[key], new_grad[key]),
                cosine_by_group={k: cosine(old_grad[key], new_grad[key], ix)
                                 for k, ix in indices.items()},
                relative_gradient_difference=norm(difference(old_grad[key], new_grad[key]))
                                             /max(norm(old_grad[key]), 1e-30))
            if "_cls_" in key:
                assert directions[key]["relative_gradient_difference"] < .005, (batch, key, directions[key])
        alpha = .002
        adjusted_vectors = [
            tuple(None if g is None else g*alpha for g in vector) if key == "covariance" else vector
            for key, vector in new_grad.items() if key != "total"]
        adjusted = add_gradients(adjusted_vectors)
        projected = dict(
            scope="algebraic gradient projection for the single user-suggested candidate; not an optimizer update",
            main_covariance_scale=alpha, main_covariance_coefficient=.2*alpha,
            cls_covariance_coefficients=.05,
            total_gradient_l2=norm(adjusted),
            total_gradient_l2_by_group={k: norm(adjusted, ix) for k, ix in indices.items()},
            cosine_with_original_total=cosine(adjusted, old_grad["total"]),
            cosine_with_original_by_group={k: cosine(adjusted, old_grad["total"], ix)
                                           for k, ix in indices.items()},
            main_covariance_gradient_l2=new["terms"]["covariance"]["gradient_l2"]*alpha,
            raw_cls_regularizer_gradients_unchanged=True)
        result = dict(batch=batch, ids=positions, mlp384=old, balanced1152=new,
                      directions=directions, main_covariance_only_candidate=projected)
        write(args.output/f"batch{batch:03d}.json", result)
        print(json.dumps(dict(batch=batch, mlp384=old, balanced1152=new,
                              directions=directions, candidate=projected)), flush=True)
        comparisons.append(result)
        del old_grad, new_grad, adjusted_vectors, adjusted, inputs
    after = parameter_sha(model)
    assert before == after and all(p.grad is None for p in model.parameters())
    write(args.output/"complete.json", dict(
        status="comparison_passed", source_checkpoint=str(source), source_stage="pretrain",
        source_epoch=20, source_step=2300, fusion_optimizer_updates=0,
        optimizer_updates=0, parameters_sha256_before=before, parameters_sha256_after=after,
        identical_raw_cls_features=True, cls_gradient_relative_tolerance=.005,
        cls_regularizer_gradients_match_within_bf16_tolerance=True,
        target="both encoders last4 and both CLS; frozen MLP still differentiable with respect to its inputs",
        results=comparisons))
    print("COMPARISON_PASSED_ZERO_UPDATES", flush=True)


if __name__ == "__main__":
    main()
