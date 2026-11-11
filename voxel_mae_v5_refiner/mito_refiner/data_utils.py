from __future__ import annotations

import random

from ..data import MitoDataset, collate_batch, to_device


def select_records(records, limit, seed=42):
    """Select a body-balanced subset without changing the source manifest."""
    if not limit or limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    by_body = {}
    for record in records:
        by_body.setdefault(str(record["body_id"]), []).append(record)
    bodies = sorted(by_body)
    for body in bodies:
        rng.shuffle(by_body[body])
    chosen = []
    while len(chosen) < limit:
        progressed = False
        for body in bodies:
            if by_body[body] and len(chosen) < limit:
                chosen.append(by_body[body].pop())
                progressed = True
        if not progressed:
            break
    return chosen


def make_dataset(cache, split, limit=0, seed=42, train_views=False):
    """Build v5 data with full visibility and the existing two global views.

    Refiner views deliberately keep MAE masking disabled. The MAE checkpoint
    remains responsible for learning the masked reconstruction task; the
    Refiner compares deterministic full-visibility/augmented instance views.
    """
    views = 4 if split == "train" and train_views else 1
    ds = MitoDataset(
        cache,
        split=split,
        queries=1,
        views=views,
        augmentations=bool(train_views),
        masking=False,
        seed=seed,
        limit=0,
        encoder_only=True,
    )
    ds.records = select_records(ds.records, limit, seed)
    ds.source_split = split
    if train_views:
        ds.split = "train"
        ds.augmentations = True
        ds.views = 4
    return ds


def load_sample(ds, index, device):
    item = ds[index]
    if item.get("skip", False):
        raise RuntimeError(
            f"v5 Refiner view was skipped for {item.get('record_id')}: {item.get('reason')}"
        )
    batch = collate_batch([item])
    if batch.get("empty", False):
        raise RuntimeError("v5 Refiner received an empty collated view")
    return to_device(batch, device)


def load_global_views(ds, record_index, device):
    """Return the previous Refiner convention: view kind 0 and mild kind 2."""
    n = len(ds.records)
    if ds.views < 3 or not ds.augmentations:
        raise ValueError("Two global views require the augmented four-view train dataset")
    return [load_sample(ds, record_index, device),
            load_sample(ds, 2 * n + record_index, device)]


def groups_of(indices, size):
    if size < 1:
        raise ValueError("Group size must be positive")
    groups = [list(indices[i:i + size]) for i in range(0, len(indices), size)]
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-2].extend(groups.pop())
    return groups


def load_paired_items(ds, indices):
    """Keep instance-major order [i0/v0,i0/v2,i1/v0,i1/v2,...].

    Failed views remove the whole pair; no zero-weight MAE filler enters BN/NNA.
    """
    if ds.views < 3 or not ds.augmentations:
        raise ValueError("Paired items require the global-view dataset")
    items, kept, skipped = [], [], []
    n = len(ds.records)
    for index in indices:
        pair = [ds[index], ds[2 * n + index]]
        if any(item.get("skip", False) for item in pair):
            skipped.append({"record_id":str(ds.records[index]["record_id"]),
                            "reasons":[s.get("reason") for s in pair if s.get("skip", False)]})
            continue
        if any(s["record_id"] != ds.records[index]["record_id"] for s in pair):
            raise ValueError("Global views must belong to the same instance")
        items.extend(pair)
        kept.append(index)
    return items, kept, skipped
