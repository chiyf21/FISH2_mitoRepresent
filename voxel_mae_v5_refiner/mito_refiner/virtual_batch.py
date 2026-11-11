from __future__ import annotations

import torch

from .nna import nna_loss, soft_nna_loss


@torch.no_grad()
def lookup_cross_view_positives(queue, preswap_keys, record_ids, topk=1,
                                selection="hard_top1", current_step=None):
    """Lookup detached queue positives for [B,V,D] keys without mutating the queue."""
    batch, views, dim = preswap_keys.shape
    positives = torch.zeros_like(preswap_keys)
    valid_rows = torch.ones(batch, dtype=torch.bool, device=preswap_keys.device)
    results = []
    for row, record_id in enumerate(record_ids):
        row_results = []
        for view in range(views):
            item = queue.lookup(preswap_keys[row, view], record_id, topk=topk,
                                selection=selection, current_step=current_step)
            if item["valid"]:
                positives[row, view].copy_(item["features"][0])
            else:
                valid_rows[row] = False
            row_results.append(item)
        results.append(row_results)
    return positives.detach(), valid_rows, results


def replay_feature_gradients(items, feature_fn, feature_gradients, names, reference_features=None,
                             error_report=None):
    """Replay batch1 encoder forwards with the exact gradients from a joint feature loss."""
    if not items:
        raise ValueError("Feature replay requires at least one cached input")
    max_abs_error = 0.0
    for index, item in enumerate(items):
        features = feature_fn(item)
        if reference_features is not None:
            for name in names:
                ref = reference_features[name][index:index + 1].to(features[name].device)
                difference = features[name].detach() - ref
                max_abs_error = max(max_abs_error, float(difference.abs().max()))
                if error_report is not None:
                    slot = error_report.setdefault(name, {"max_abs_error": 0.0,
                                                          "max_relative_l2_error": 0.0})
                    slot["max_abs_error"] = max(slot["max_abs_error"], float(difference.abs().max()))
                    relative = float(difference.norm() / ref.norm().clamp_min(1e-12))
                    slot["max_relative_l2_error"] = max(slot["max_relative_l2_error"], relative)
        outputs = [features[name] for name in names]
        gradients = [feature_gradients[name][index:index + 1].to(features[name].device)
                     for name in names]
        torch.autograd.backward(outputs, gradients)
    return max_abs_error


def layer_batch_loss(head, features, queue, record_ids, body_ids,
                     mode="hard_top1", topk=1, temperature=0.2, positive_temperature=0.05,
                     entropy_target=None):
    """One layer's loss and detached pre-update vectors for an atomic queue update."""
    projected, anchors = head(features)
    lookups = [queue.lookup(projected[i], record_ids[i], topk=topk, selection=mode)
               for i in range(len(record_ids))]
    valid = [i for i, result in enumerate(lookups) if result["valid"]]
    if not valid:
        return projected.sum() * 0.0, projected.detach(), 0, 0.0
    sub_ids = [record_ids[i] for i in valid]
    sub_anchor = anchors[valid]
    sub_projected = projected[valid]
    if mode == "soft_temperature":
        width = max(len(lookups[i]["features"]) for i in valid)
        candidates = projected.new_zeros((len(valid), width, projected.shape[-1]))
        scores = projected.new_full((len(valid), width), -torch.inf)
        mask = torch.zeros((len(valid), width), dtype=torch.bool, device=projected.device)
        for row, original in enumerate(valid):
            item = lookups[original]
            count = len(item["features"])
            candidates[row, :count] = item["features"]
            scores[row, :count] = item["scores"]
            mask[row, :count] = True
        loss, n = soft_nna_loss(sub_anchor, projected, candidates, scores, mask,
                                sub_ids, record_ids, temperature, positive_temperature, entropy_target)
    else:
        positives = torch.stack([lookups[i]["features"][0] for i in valid])
        loss, n = nna_loss(sub_anchor, projected, positives, sub_ids, record_ids, temperature)
    same_body = sum(lookups[i]["body_ids"][0] == str(body_ids[i]) for i in valid) / len(valid)
    return loss, projected.detach(), n, same_body


def true_batch_loss(heads, features, queues, record_ids, body_ids,
                    mode="hard_top1", topk=1, temperature=0.2, positive_temperature=0.05,
                    entropy_target=None):
    """True contrastive batch: heads see a real [B,D] feature batch."""
    losses, projected, counts, same_body = {}, {}, {}, {}
    total = None
    for name, head in heads.items():
        loss, keys, n, body_rate = layer_batch_loss(
            head, features[name], queues[name], record_ids, body_ids,
            mode, topk, temperature, positive_temperature,
            entropy_target,
        )
        losses[name], projected[name], counts[name], same_body[name] = loss, keys, n, body_rate
        total = loss if total is None else total + loss
    return total, losses, projected, counts, same_body


def virtual_anchor_loss(head, anchor_feature, pass1_keys, queue, record_id, body_id,
                        batch_record_ids, batch_body_ids, mode="hard_top1", topk=1,
                        temperature=0.2, positive_temperature=0.05, entropy_target=None):
    """Pass-2 anchor loss against pass-1 stop-gradient negatives and the old queue."""
    projected, anchor = head(anchor_feature)
    lookup = queue.lookup(projected[0], record_id, topk=topk, selection=mode)
    if not lookup["valid"]:
        return projected.sum() * 0.0, 0, 0.0
    if mode == "soft_temperature":
        candidates = lookup["features"][None]
        scores = lookup["scores"][None]
        mask = torch.ones_like(scores, dtype=torch.bool)
        loss, count = soft_nna_loss(anchor, pass1_keys, candidates, scores, mask,
                                    [record_id], batch_record_ids, temperature, positive_temperature,
                                    entropy_target)
    else:
        loss, count = nna_loss(anchor, pass1_keys, lookup["features"][:1],
                               [record_id], batch_record_ids, temperature)
    return loss, count, float(lookup["body_ids"][0] == str(body_id))
