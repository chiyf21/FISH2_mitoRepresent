from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FeatureQueue(nn.Module):
    """Per-head FIFO of detached projector features and provenance metadata."""

    def __init__(self, capacity=2048, dim=256):
        super().__init__()
        if capacity < 1 or dim < 1:
            raise ValueError("Queue capacity and feature dimension must be positive")
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.register_buffer("features", torch.zeros(capacity, dim))
        self.register_buffer("steps", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("valid", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.record_ids = [None] * capacity
        self.body_ids = [None] * capacity

    def get_extra_state(self):
        return {"record_ids": self.record_ids, "body_ids": self.body_ids}

    def set_extra_state(self, state):
        self.record_ids = state["record_ids"]
        self.body_ids = state["body_ids"]

    @torch.no_grad()
    def enqueue(self, features, record_ids, body_ids, step):
        if len(features) != len(record_ids) or len(record_ids) != len(body_ids):
            raise ValueError("Features and queue metadata must have matching lengths")
        features = F.normalize(features.detach(), dim=-1)
        if len(features) > self.capacity:
            features = features[-self.capacity:]
            record_ids, body_ids = record_ids[-self.capacity:], body_ids[-self.capacity:]
        for feature, record_id, body_id in zip(features, record_ids, body_ids):
            slot = int(self.ptr.item())
            self.features[slot].copy_(feature)
            self.steps[slot] = int(step)
            self.valid[slot] = True
            self.record_ids[slot] = str(record_id)
            self.body_ids[slot] = str(body_id)
            self.ptr.fill_((slot + 1) % self.capacity)

    @torch.no_grad()
    def lookup(self, query, record_id, topk=1, selection="hard_top1", current_step=None):
        """Exclude every same-record entry; optional random Top-K mirrors official queue lookup."""
        slots = [i for i, queued_id in enumerate(self.record_ids)
                 if bool(self.valid[i]) and queued_id is not None and queued_id != str(record_id)]
        if not slots:
            return {"valid": False, "features": query.new_zeros((0, self.dim)),
                    "scores": query.new_zeros((0,)), "record_ids": [], "body_ids": [], "steps": []}
        bank = F.normalize(self.features[slots], dim=-1)
        scores = bank @ F.normalize(query.detach(), dim=-1)
        k = min(max(1, int(topk)), len(slots))
        values, top_positions = scores.topk(k, largest=True)
        if selection == "hard_top1":
            positions = top_positions[:1]
        elif selection == "official_random_topk":
            choice = torch.randint(len(top_positions), (1,), device=top_positions.device)
            positions = top_positions[choice]
        elif selection != "soft_temperature":
            raise ValueError(f"Unknown positive selection: {selection}")
        selected_slots = [slots[int(p)] for p in positions.tolist()]
        selected_scores = scores[positions]
        selected_score = float(selected_scores[0])
        rank = 1 + int((scores > selected_scores[0]).sum())
        slot = selected_slots[0]
        age = None if current_step is None else max(0, int(current_step) - int(self.steps[slot]))
        return {
            "valid": True,
            "features": self.features[selected_slots].detach(),
            "scores": selected_scores.detach(),
            "similarity": selected_score,
            "rank": rank,
            "age": age,
            "fill_ratio": float(self.valid.float().mean()),
            "record_ids": [self.record_ids[i] for i in selected_slots],
            "body_ids": [self.body_ids[i] for i in selected_slots],
            "steps": [int(self.steps[i]) for i in selected_slots],
        }
