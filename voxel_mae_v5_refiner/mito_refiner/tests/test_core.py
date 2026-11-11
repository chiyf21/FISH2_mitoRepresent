from __future__ import annotations

import copy
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from ..checkpoint import attach_resume_state, restore_resume_state
from ..features import extract_features
from ..heads import IDHead
from ..models import MultiLayerID
from ..nna import official_cross_view_logits, official_cross_view_nna_loss
from ..queue import FeatureQueue
from ..virtual_batch import lookup_cross_view_positives, replay_feature_gradients


def parity_official_cross_view_logits_loss():
    torch.manual_seed(11)
    batch, views, dim, temperature = 5, 2, 9, 0.17
    pre = torch.randn(batch, views, dim)
    post = torch.randn(batch, views, dim)
    pred = torch.randn(batch, views, dim)
    actual, labels, directions = official_cross_view_logits(pre, post, pred, temperature)
    pre_n, post_n, pred_n = (F.normalize(v, dim=-1) for v in (pre, post, pred))
    eye = torch.eye(batch, dtype=torch.bool)
    expected, expected_directions = [], []
    for i in range(views):
        for j in range(views):
            if i == j:
                continue
            positive = post_n[:, i] @ pred_n[:, j].T / temperature
            negative = pre_n[:, i] @ pred_n[:, j].T / temperature
            expected.append(torch.where(eye, positive, negative))
            expected_directions.append((i, j))
    expected_labels = torch.arange(batch)
    expected_loss = torch.stack([F.cross_entropy(x, expected_labels) for x in expected]).mean()
    actual_loss, info = official_cross_view_nna_loss(pre, post, pred, temperature)
    max_logits_error = max(float((a - b).abs().max()) for a, b in zip(actual, expected))
    max_loss_error = float((actual_loss - expected_loss).abs())
    assert directions == expected_directions and torch.equal(labels, expected_labels)
    assert max_logits_error < 1e-7 and max_loss_error < 1e-7
    assert info["directions"] == views * (views - 1)
    return {"max_logits_abs_error": max_logits_error, "max_loss_abs_error": max_loss_error,
            "cross_view_terms": len(actual)}


def detached_key_and_student_gradient_flow():
    torch.manual_seed(12)
    batch, views = 4, 2
    head = IDHead(input_dim=6, output_dim=8, hidden_dim=16).train()
    x = torch.randn(batch * views, 6, requires_grad=True)
    pre, predicted = head(x)
    predicted.retain_grad()
    post = torch.randn(batch, views, 8)
    loss, info = official_cross_view_nna_loss(
        pre.reshape(batch, views, -1), post,
        predicted.reshape(batch, views, -1), temperature=0.2,
    )
    loss.backward()
    assert not pre.requires_grad and pre.grad is None
    assert not post.requires_grad and predicted.grad is not None
    assert float(predicted.grad.norm()) > 0 and float(x.grad.norm()) > 0
    projector_grad = sum(float(p.grad.norm()) for p in head.projector.parameters() if p.grad is not None)
    predictor_grad = sum(float(p.grad.norm()) for p in head.predictor.parameters() if p.grad is not None)
    assert projector_grad > 0 and predictor_grad > 0
    return {"preswap_key_requires_grad": bool(pre.requires_grad),
            "postswap_key_requires_grad": bool(post.requires_grad),
            "predicted_student_grad_norm": float(predicted.grad.norm()),
            "projector_grad_norm_sum": projector_grad,
            "predictor_grad_norm_sum": predictor_grad,
            "encoder_feature_grad_norm": float(x.grad.norm()),
            "max_error": 0.0}


def queue_top1_top20_fifo_self_exclusion():
    torch.manual_seed(13)
    queue = FeatureQueue(capacity=32, dim=12)
    features = F.normalize(torch.randn(25, 12), dim=-1)
    records = [f"r{i}" for i in range(25)]
    bodies = [f"b{i % 4}" for i in range(25)]
    queue.enqueue(features, records, bodies, step=3)
    query = features[24]
    top1 = queue.lookup(query, "r24", topk=1, current_step=10)
    assert top1["valid"] and top1["record_ids"][0] != "r24"
    assert top1["rank"] == 1 and abs(top1["similarity"] - top1["scores"][0].item()) < 1e-6
    torch.manual_seed(131)
    random_results = [queue.lookup(query, "r24", topk=20,
                                  selection="official_random_topk", current_step=10)
                      for _ in range(40)]
    assert all(row["valid"] and len(row["features"]) == 1 for row in random_results)
    assert all(row["rank"] <= 20 for row in random_results)
    assert all("r24" not in row["record_ids"] for row in random_results)
    assert all(abs(row["similarity"] - row["scores"][0].item()) < 1e-6 for row in random_results)
    assert len({row["record_ids"][0] for row in random_results}) > 1

    fifo = FeatureQueue(capacity=5, dim=12)
    fifo.enqueue(features, records, bodies, step=7)
    assert fifo.record_ids == records[-5:]
    assert int(fifo.ptr) == 0 and int(fifo.valid.sum()) == 5
    restored = FeatureQueue(capacity=5, dim=12)
    restored.load_state_dict(copy.deepcopy(fifo.state_dict()))
    assert restored.record_ids == fifo.record_ids and int(restored.ptr) == int(fifo.ptr)
    assert torch.equal(restored.features, fifo.features)
    return {"top1_self_exclusion": True, "random_top20_draws": len(random_results),
            "random_top20_unique_neighbors": len({row["record_ids"][0] for row in random_results}),
            "fifo_kept_last_n": 5, "fifo_pointer": int(fifo.ptr), "max_error": 0.0}


def true_batch_vs_feature_replay():
    torch.manual_seed(14)
    batch, views, input_dim, feature_dim = 8, 2, 5, 6
    x = torch.randn(batch, views, input_dim)
    postswap = F.normalize(torch.randn(batch, views, 8), dim=-1)
    ids = [f"r{i}" for i in range(batch)]

    def encoder():
        return nn.Sequential(nn.Linear(input_dim, 8), nn.BatchNorm1d(8), nn.GELU(),
                             nn.Dropout(0.4), nn.Linear(8, feature_dim))

    enc_true, enc_replay = encoder().eval(), encoder().eval()
    enc_replay.load_state_dict(copy.deepcopy(enc_true.state_dict()))
    head_true = IDHead(feature_dim, 8, 16).train()
    head_replay = IDHead(feature_dim, 8, 16).train()
    head_replay.load_state_dict(copy.deepcopy(head_true.state_dict()))

    flat_x = x.reshape(batch * views, input_dim)
    true_features = enc_true(flat_x)
    key_true, pred_true = head_true(true_features)
    loss_true, _ = official_cross_view_nna_loss(
        key_true.reshape(batch, views, -1), postswap,
        pred_true.reshape(batch, views, -1), temperature=0.2,
    )
    loss_true.backward()

    items = [flat_x[i:i + 1].clone() for i in range(len(flat_x))]
    with torch.no_grad():
        cached = torch.cat([enc_replay(item) for item in items])
    leaf = cached.detach().requires_grad_(True)
    key_replay, pred_replay = head_replay(leaf)
    loss_replay, _ = official_cross_view_nna_loss(
        key_replay.reshape(batch, views, -1), postswap,
        pred_replay.reshape(batch, views, -1), temperature=0.2,
    )
    loss_replay.backward()
    leaf_gradient = leaf.grad.detach().clone()

    def replay_one(item):
        return {"block6_mean": enc_replay(item)}

    max_feature_drift = replay_feature_gradients(
        items, replay_one, {"block6_mean": leaf_gradient}, ("block6_mean",),
        {"block6_mean": cached.detach()},
    )
    errors = [float((a.grad - b.grad).abs().max())
              for a, b in zip(enc_true.parameters(), enc_replay.parameters())]
    errors += [float((a.grad - b.grad).abs().max())
               for a, b in zip(head_true.parameters(), head_replay.parameters())]
    max_grad_error = max(errors)
    max_loss_error = float((loss_true - loss_replay).detach().abs())

    opt_true = torch.optim.SGD(list(enc_true.parameters()) + list(head_true.parameters()), lr=0.01)
    opt_replay = torch.optim.SGD(list(enc_replay.parameters()) + list(head_replay.parameters()), lr=0.01)
    opt_true.step()
    opt_replay.step()
    max_update_error = max(float((a - b).detach().abs().max()) for a, b in zip(
        list(enc_true.parameters()) + list(head_true.parameters()),
        list(enc_replay.parameters()) + list(head_replay.parameters())))
    assert max_feature_drift < 1e-7
    # Feature replay sums per-example VJPs, while True Batch uses a single batched GEMM;
    # report small float32 reduction-order differences rather than requiring bitwise equality.
    assert max_loss_error < 1e-6 and max_grad_error < 1e-5 and max_update_error < 1e-6, (
        f"feature_drift={max_feature_drift:.3g}; loss_error={max_loss_error:.3g}; "
        f"gradient_error={max_grad_error:.3g}; update_error={max_update_error:.3g}"
    )
    return {"max_loss_abs_error": max_loss_error, "max_head_encoder_grad_abs_error": max_grad_error,
            "max_optimizer_update_abs_error": max_update_error,
            "max_replay_feature_drift": max_feature_drift,
            "encoder_dropout_mode": "eval; stochastic masks intentionally disabled for exact replay",
            "head_batchnorm_mode": "train on complete B*V batch", "contrastive_batch": [batch, views]}


def three_head_gradient_flow():
    torch.manual_seed(15)
    batch, views, dim = 4, 2, 16
    names = ("block4", "block5", "block6_mean")
    model = MultiLayerID(input_dim=dim, output_dim=8, hidden_dim=16, layers=names).train()
    features = {name: torch.randn(batch, views, dim, requires_grad=True) for name in names}
    record_ids = [f"query{i}" for i in range(batch)]
    body_ids = [f"body{i}" for i in range(batch)]
    queues = {name: FeatureQueue(capacity=24, dim=8) for name in names}
    for name in names:
        queues[name].enqueue(F.normalize(torch.randn(16, 8), dim=-1),
                             [f"bank{i}" for i in range(16)], [f"b{i % 3}" for i in range(16)], 1)
    total, counts, grad_norms = 0.0, {}, {}
    losses = []
    for name in names:
        key, pred = model.heads[name](features[name].reshape(batch * views, dim))
        key, pred = key.reshape(batch, views, -1), pred.reshape(batch, views, -1)
        pos, valid, _ = lookup_cross_view_positives(queues[name], key, record_ids, topk=20,
                                                     selection="official_random_topk", current_step=2)
        loss, _ = official_cross_view_nna_loss(key, pos, pred, valid_rows=valid)
        losses.append(loss)
        counts[name] = int(valid.sum())
    sum(losses).backward()
    for name in names:
        head = model.heads[name]
        projector_grad = sum(float(p.grad.norm()) for p in head.projector.parameters() if p.grad is not None)
        predictor_grad = sum(float(p.grad.norm()) for p in head.predictor.parameters() if p.grad is not None)
        grad_norms[name] = {"projector": projector_grad, "predictor": predictor_grad,
                            "encoder_feature": float(features[name].grad.norm())}
        assert counts[name] == batch
        assert min(projector_grad, predictor_grad, grad_norms[name]["encoder_feature"]) > 0
    return {"heads": list(names), "valid_rows": counts, "gradient_norms": grad_norms,
            "max_error": 0.0}


def checkpoint_optimizer_rng_resume():
    torch.manual_seed(16)
    np.random.seed(16)
    random.seed(16)
    model_a = nn.Linear(4, 2)
    optim_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    for _ in range(2):
        optim_a.zero_grad(set_to_none=True)
        model_a(torch.randn(3, 4)).square().mean().backward()
        optim_a.step()
    state = attach_resume_state({"model": copy.deepcopy(model_a.state_dict()), "step": 2}, optim_a)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resume.pt"
        torch.save(state, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        draws_a = (random.random(), np.random.rand(), torch.rand(4))
        optim_a.zero_grad(set_to_none=True)
        model_a(torch.randn(3, 4)).square().mean().backward()
        optim_a.step()
        expected = copy.deepcopy(model_a.state_dict())

        model_b = nn.Linear(4, 2)
        optim_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
        model_b.load_state_dict(saved["model"])
        restore_resume_state(saved, optim_b)
        draws_b = (random.random(), np.random.rand(), torch.rand(4))
        optim_b.zero_grad(set_to_none=True)
        model_b(torch.randn(3, 4)).square().mean().backward()
        optim_b.step()
    rng_error = max(abs(draws_a[0] - draws_b[0]), abs(draws_a[1] - draws_b[1]),
                    float((draws_a[2] - draws_b[2]).abs().max()))
    param_error = max(float((expected[k] - model_b.state_dict()[k]).abs().max()) for k in expected)
    assert rng_error == 0.0 and param_error < 1e-8
    return {"max_rng_replay_error": rng_error, "max_parameter_error_after_resume": param_error,
            "resume_step": 2}


def backbone_adapter_feature_contract():
    class ToyAdapter:
        identifier = "test.toy"

        def extract_features(self, model, batch):
            x = model(batch)
            return {"early": x[:, :5], "global": x}

    model = nn.Linear(7, 11)
    adapted = extract_features(ToyAdapter(), model, torch.randn(3, 7))
    assert {name: tuple(value.shape) for name, value in adapted.items()} == {
        "early": (3, 5), "global": (3, 11)
    }
    heads = MultiLayerID(input_dims={name: value.shape[-1] for name, value in adapted.items()},
                         output_dim=8, hidden_dim=16, layers=tuple(adapted))
    outputs = heads(adapted)
    assert all(pair[0].shape == (3, 8) and pair[1].shape == (3, 8)
               for pair in outputs.values())

    class BrokenAdapter:
        def extract_features(self, model, batch):
            return {"bad": torch.randn(3)}

    try:
        extract_features(BrokenAdapter(), model, torch.randn(3, 7))
    except ValueError as error:
        assert "[B,D]" in str(error)
    else:
        raise AssertionError("Adapter contract must reject non-global feature tensors")
    return {"adapter_id": ToyAdapter.identifier, "feature_dims": {k: v.shape[-1] for k, v in adapted.items()},
            "head_output_dim": 8, "malformed_output_rejected": True}


def run():
    tests = [
        ("official_cross_view_logits_loss_parity", parity_official_cross_view_logits_loss),
        ("detached_keys_and_student_gradient_flow", detached_key_and_student_gradient_flow),
        ("queue_top1_top20_fifo_self_exclusion", queue_top1_top20_fifo_self_exclusion),
        ("true_batch_vs_feature_gradient_replay", true_batch_vs_feature_replay),
        ("three_head_gradient_flow", three_head_gradient_flow),
        ("checkpoint_optimizer_rng_resume", checkpoint_optimizer_rng_resume),
        ("backbone_adapter_feature_contract", backbone_adapter_feature_contract),
    ]
    results = []
    for name, test in tests:
        try:
            detail = test()
            results.append({"test": name, "status": "PASS", **detail})
        except Exception as error:
            results.append({"test": name, "status": "FAIL", "error": repr(error)})
            for row in results:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            output = os.environ.get("MITO_REFINER_TEST_OUTPUT")
            if output:
                path = Path(output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            raise
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    output = os.environ.get("MITO_REFINER_TEST_OUTPUT")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Mito-Refiner R0 official-style P0 checks passed", flush=True)
    return results


if __name__ == "__main__":
    run()
