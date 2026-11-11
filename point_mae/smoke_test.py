"""Small runnable shape/gradient check; no dataset or checkpoint required."""

from __future__ import annotations

import torch

from pointmae import (
    PointMAE,
    augmented_point_view,
    count_parameters,
    nt_xent_loss,
    require_trained_projection,
    seed_everything,
    split_stratified_by_group,
)


def main() -> None:
    seed_everything(7)
    split_groups = torch.tensor([0] * 40 + [1] * 229 + [2] * 122).numpy()
    split = split_stratified_by_group(split_groups, seed=42)
    assert {name: len(ids) for name, ids in split.items()} == {"train": 313, "val": 39, "test": 39}
    for ids in split.values():
        assert set(split_groups[ids]) == {0, 1, 2}
    assert not set(split["train"]) & set(split["val"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["val"]) & set(split["test"])
    cfg = {
        "num_groups": 8,
        "group_size": 8,
        "encoder_dim": 64,
        "encoder_depth": 2,
        "encoder_heads": 4,
        "decoder_dim": 32,
        "decoder_depth": 1,
        "decoder_heads": 4,
        "mlp_ratio": 2,
        "mask_ratio": 0.5,
        "embedding_dim": 16,
    }
    model = PointMAE(cfg, ["intensity", "gradient", "distance", "surface"])
    xyz = torch.randn(2, 64, 3)
    attrs = torch.randn(2, 64, 4)
    result = model(xyz, attrs)
    assert result["loss"].isfinite().item()
    view_xyz, view_attrs = augmented_point_view(xyz, attrs, model.attr_names, 0.9)
    z1, z2 = model.encode(xyz, attrs), model.encode(view_xyz, view_attrs)
    contrastive = nt_xent_loss(z1, z2)
    loss = result["loss"] + 0.1 * contrastive
    assert contrastive.isfinite().item()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projection.parameters())
    assert model.decoder_position.weight.grad is not None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    require_trained_projection(model, {"optimizer": optimizer.state_dict()})
    decoder_input = torch.randn(2, 8, 32)
    centers = torch.randn(2, 8, 3)
    decoded = model.decode_tokens(decoder_input, centers)
    shuffled = model.decode_tokens(decoder_input[:, torch.randperm(8)], centers)
    assert not torch.allclose(decoded, shuffled)
    embedding = model.encode(xyz, attrs)
    assert embedding.shape == (2, 16)
    print({"parameters": count_parameters(model), "loss": float(loss.detach()), "contrastive": float(contrastive.detach()), "embedding": list(embedding.shape)})


if __name__ == "__main__":
    main()
