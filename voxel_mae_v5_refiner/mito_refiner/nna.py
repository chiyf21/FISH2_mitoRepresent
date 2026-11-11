from __future__ import annotations

import torch
import torch.nn.functional as F


def scheduled_positive_temperature(start, end, step, total_steps, schedule="fixed"):
    if start <= 0 or end <= 0:
        raise ValueError("Positive temperatures must be greater than zero")
    if schedule == "fixed":
        return float(start)
    progress = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    if schedule == "linear":
        weight = progress
    elif schedule == "cosine":
        weight = 0.5 - 0.5 * torch.cos(torch.tensor(progress * torch.pi)).item()
    else:
        raise ValueError(f"Unknown temperature schedule: {schedule}")
    return float(start + (end - start) * weight)


def official_cross_view_logits(projected_preswap, projected_postswap, predicted, temperature=0.2):
    """Official NNCLR logits for every ordered pair of distinct global views."""
    if temperature <= 0:
        raise ValueError("NNA temperature must be positive")
    if projected_preswap.ndim != 3 or projected_postswap.shape != projected_preswap.shape:
        raise ValueError("preswap/postswap must have matching [B,V,D] shapes")
    if predicted.shape != projected_preswap.shape:
        raise ValueError("predicted must match the [B,V,D] key shape")
    batch, views, _ = predicted.shape
    if batch < 2 or views < 2:
        raise ValueError("Cross-view NNA needs at least two instances and two views")
    preswap = F.normalize(projected_preswap, dim=-1)
    postswap = F.normalize(projected_postswap.detach(), dim=-1)
    predicted = F.normalize(predicted, dim=-1)
    labels = torch.arange(batch, device=predicted.device)
    eye = torch.eye(batch, dtype=torch.bool, device=predicted.device)
    logits, directions = [], []
    for view_key in range(views):
        for view_student in range(views):
            if view_key == view_student:
                continue
            positive = postswap[:, view_key] @ predicted[:, view_student].T / temperature
            negative = preswap[:, view_key] @ predicted[:, view_student].T / temperature
            matrix = torch.where(eye, positive, negative)
            logits.append(matrix)
            directions.append((view_key, view_student))
    return logits, labels, directions


def official_cross_view_nna_loss(projected_preswap, projected_postswap, predicted,
                                 temperature=0.2, valid_rows=None):
    """Average official cross-view CE terms; keys are detached by the Head/Queue."""
    logits, labels, directions = official_cross_view_logits(
        projected_preswap, projected_postswap, predicted, temperature
    )
    batch = len(labels)
    if valid_rows is None:
        valid_rows = torch.ones(batch, dtype=torch.bool, device=labels.device)
    else:
        valid_rows = valid_rows.to(device=labels.device, dtype=torch.bool)
    if valid_rows.shape != (batch,):
        raise ValueError("valid_rows must have shape [B]")
    if not valid_rows.any():
        zero = predicted.sum() * 0.0
        return zero, {"positive_logit_mean": 0.0, "negative_logit_mean": 0.0,
                      "alignment": 0.0, "valid_rows": 0, "directions": len(directions)}
    losses, positive_values, negative_values, alignments = [], [], [], []
    for matrix in logits:
        selected = matrix[valid_rows]
        selected_labels = labels[valid_rows]
        losses.append(F.cross_entropy(selected, selected_labels))
        positive_values.append(matrix[labels, labels])
        off_diagonal = ~torch.eye(batch, dtype=torch.bool, device=labels.device)
        negative_values.append(matrix[off_diagonal])
        alignments.append(matrix.softmax(dim=-1)[labels, labels])
    loss = torch.stack(losses).mean()
    return loss, {
        "positive_logit_mean": float(torch.cat(positive_values).detach().mean()),
        "negative_logit_mean": float(torch.cat(negative_values).detach().mean()),
        "alignment": float(torch.cat(alignments).detach().mean()),
        "valid_rows": int(valid_rows.sum()),
        "directions": len(directions),
    }


def _positive_weights(scores, valid, temperature, entropy_target):
    safe = scores.masked_fill(~valid, -torch.inf)
    if entropy_target is None:
        return torch.softmax(safe / temperature, dim=1)
    weights = torch.zeros_like(scores)
    for row_index in range(scores.shape[0]):
        row = scores[row_index, valid[row_index]]
        if row.numel() == 0:
            continue
        if row.numel() == 1:
            weights[row_index, valid[row_index]] = 1.0
            continue
        target = min(max(float(entropy_target), 0.0), float(torch.log(torch.tensor(float(row.numel())))))
        if target <= 1e-7:
            row_weights = F.one_hot(row.argmax(), row.numel()).to(row.dtype)
        elif target >= float(torch.log(torch.tensor(float(row.numel())))) - 1e-7:
            row_weights = torch.full_like(row, 1.0 / row.numel())
        else:
            scale = max(float(row.detach().std()), 1e-3)
            lo, hi = scale * 1e-6, scale * 1e4
            for _ in range(32):
                mid = (lo + hi) * 0.5
                candidate = torch.softmax(row / mid, dim=0)
                entropy = -(candidate * candidate.clamp_min(1e-12).log()).sum()
                if float(entropy.detach()) < target:
                    lo = mid
                else:
                    hi = mid
            row_weights = torch.softmax(row / ((lo + hi) * 0.5), dim=0)
        weights[row_index, valid[row_index]] = row_weights
    return weights


def _negative_logits(anchors, keys, record_ids, key_record_ids, temperature):
    anchors = F.normalize(anchors, dim=-1)
    keys = F.normalize(keys.detach(), dim=-1)
    logits = anchors @ keys.T / temperature
    same_record = torch.tensor(
        [[a == b for b in key_record_ids] for a in record_ids],
        dtype=torch.bool,
        device=logits.device,
    )
    logits = logits.masked_fill(same_record, -torch.inf)
    valid = (~same_record).any(dim=1)
    return logits, valid


def nna_loss(anchors, keys, positives, record_ids, key_record_ids, temperature=0.2):
    """Hard NNA: queued positives are detached; current-batch negatives are stop-gradient."""
    if temperature <= 0:
        raise ValueError("NNA temperature must be positive")
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives.detach(), dim=-1)
    neg_logits, valid = _negative_logits(anchors, keys, record_ids, key_record_ids, temperature)
    pos_logits = (anchors * positives).sum(-1, keepdim=True) / temperature
    logits = torch.cat((pos_logits, neg_logits), dim=1)
    per_anchor = torch.logsumexp(logits, dim=1) - pos_logits[:, 0]
    if not valid.any():
        return anchors.sum() * 0.0, 0
    return per_anchor[valid].mean(), int(valid.sum())


def soft_nna_loss(anchors, keys, candidates, candidate_scores, candidate_mask,
                  record_ids, key_record_ids, temperature=0.2, positive_temperature=0.05,
                  entropy_target=None):
    """R1 interface: score-weighted candidate NNA losses; not used by the R0 run."""
    if temperature <= 0 or positive_temperature <= 0:
        raise ValueError("Both NNA temperatures must be positive")
    anchors = F.normalize(anchors, dim=-1)
    candidates = F.normalize(candidates.detach(), dim=-1)
    neg_logits, valid_neg = _negative_logits(anchors, keys, record_ids, key_record_ids, temperature)
    pos_logits = torch.einsum("bd,bkd->bk", anchors, candidates) / temperature
    all_logits = torch.cat((pos_logits.unsqueeze(-1), neg_logits.unsqueeze(1).expand(-1, candidates.shape[1], -1)), -1)
    candidate_loss = torch.logsumexp(all_logits, dim=-1) - pos_logits
    valid = candidate_mask.bool() & valid_neg[:, None]
    has_candidate = valid.any(dim=1)
    weights = _positive_weights(candidate_scores, valid, positive_temperature, entropy_target)
    per_anchor = (weights * candidate_loss.masked_fill(~valid, 0.0)).sum(dim=1)
    if not has_candidate.any():
        return anchors.sum() * 0.0, 0
    return per_anchor[has_candidate].mean(), int(has_candidate.sum())
