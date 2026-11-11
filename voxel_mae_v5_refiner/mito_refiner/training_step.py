"""The same full-feature-batch NNA objective, with two encoder memory policies."""
from contextlib import nullcontext

import torch

from ..data import collate_batch, to_device
from .features import extract_features
from .nna import official_cross_view_nna_loss
from .virtual_batch import lookup_cross_view_positives, replay_feature_gradients


def precision_context(device, precision):
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"Unknown precision: {precision}")
    if precision == "bf16":
        if torch.device(device).type != "cuda":
            raise ValueError("bf16 training is supported on CUDA; use --precision fp32 for CPU checks")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def configure_encoder(model, names, activation_checkpointing=True, freeze_encoder=False):
    """Deterministic encoder forwards, explicit active parameter set.

    B4/B5/B6 heads do not train the bottleneck. Selecting 'final' does.
    Reconstruction heads are unused during this separate refinement stage.
    """
    model.eval()
    if freeze_encoder:
        model.requires_grad_(False)
        model.activation_checkpointing = bool(activation_checkpointing)
        return []
    for p in model.parameters():
        p.requires_grad_(True)
    for module_name in ("block", "size"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.requires_grad_(False)
    model.bottleneck.requires_grad_("final" in names)
    model.activation_checkpointing = bool(activation_checkpointing)
    return [p for p in model.parameters() if p.requires_grad]


def encoder_feature_batch(adapter, model, items, names, device, mode, precision):
    if mode == "true":
        batch = to_device(collate_batch(items), device)
        with precision_context(device, precision):
            feature = extract_features(adapter, model, batch)
        return {name: feature[name].float() for name in names}
    if mode != "replay":
        raise ValueError(f"Unknown encoder batch mode: {mode}")
    rows = {name: [] for name in names}
    with torch.no_grad():
        for item in items:
            batch = to_device(collate_batch([item]), device)
            with precision_context(device, precision):
                feature = extract_features(adapter, model, batch)
            for name in names:
                rows[name].append(feature[name].float().detach())
    return {name: torch.cat(values) for name, values in rows.items()}


def cross_view_heads(heads, features, queues, record_ids, temperature, step,
                     selection="official_random_topk", topk=20,
                     embedding_adapter=None):
    """BN runs once on all 2B rows; each view direction has BxB logits."""
    losses, keys, valid_counts, lookups, diagnostics = {}, {}, {}, {}, {}
    batch = len(record_ids)
    if batch < 2 or len(set(record_ids)) != batch:
        raise ValueError("A contrastive batch requires >=2 distinct instance IDs")
    for name, head in heads.heads.items():
        if features[name].shape[0] != batch * 2:
            raise ValueError("Expected exactly two instance-major views per record")
        values = features[name]
        if name == "final" and embedding_adapter is not None:
            values = embedding_adapter(values)
        pre, pred = head(values.float())
        pre, pred = pre.reshape(batch, 2, -1), pred.reshape(batch, 2, -1)
        post, valid, lookup = lookup_cross_view_positives(
            queues[name], pre, record_ids, topk=topk, selection=selection, current_step=step)
        loss, info = official_cross_view_nna_loss(pre, post, pred, temperature, valid)
        losses[name], keys[name] = loss, pre.detach()
        valid_counts[name], lookups[name], diagnostics[name] = int(valid.sum()), lookup, info
    return sum(losses.values()), losses, keys, valid_counts, lookups, diagnostics


def backward_refinement(adapter, model, heads, queues, items, record_ids,
                        device, mode="true", precision="fp32", temperature=.2,
                        step=0, replay_tolerance=2e-3, embedding_adapter=None):
    """Build gradients only; the caller performs one optimizer/EMA/queue update.

    Queue lookup always sees the pre-step bank. No automatic OOM fallback:
    retry explicitly with replay so execution mode stays reproducible.
    """
    names = tuple(heads.heads)
    reference = encoder_feature_batch(adapter, model, items, names, device, mode, precision)
    features = reference if mode == "true" else {
        name: value.detach().requires_grad_(True) for name, value in reference.items()}
    heads.train()
    total, losses, keys, counts, lookups, info = cross_view_heads(
        heads, features, queues, record_ids, temperature, step,
        embedding_adapter=embedding_adapter)
    errors = {}
    if sum(counts.values()):
        total.backward()
        if mode == "replay":
            gradients = {name: features[name].grad.detach().clone() for name in names}

            def recompute(item):
                batch = to_device(collate_batch([item]), device)
                with precision_context(device, precision):
                    values = extract_features(adapter, model, batch)
                return {name: values[name].float() for name in names}

            replay_feature_gradients(items, recompute, gradients, names, reference,
                                     error_report=errors)
            drift = max((v["max_relative_l2_error"] for v in errors.values()), default=0.)
            if drift > replay_tolerance:
                raise RuntimeError(f"Replay relative L2 drift {drift:.6g} > {replay_tolerance}; no optimizer step")
    return {
        "loss": float(total.detach()),
        "head_losses": {name: float(value.detach()) for name, value in losses.items()},
        "keys": keys, "valid_anchors": counts, "lookups": lookups,
        "nna_diagnostics": info, "feature_replay_by_layer": errors,
    }
