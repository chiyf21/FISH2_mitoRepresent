from __future__ import annotations

import numpy as np


def cosine_knn(features, k):
    x = np.asarray(features, dtype=np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    return np.argsort(-similarity, axis=1)[:, :min(k, max(0, len(x) - 1))]


def consensus_summary(layer_features, body_ids, ks=(5, 20)):
    """R2 diagnostic only: compare instance-ID neighborhoods, never average heads."""
    names = list(layer_features)
    neighbors = {name: {k: cosine_knn(layer_features[name], k) for k in ks} for name in names}
    result = {"layers": names, "n": len(body_ids), "top1_agreement": {}, "knn_jaccard": {},
              "mutual_knn_fraction": {}, "consensus": {}, "same_body_top1": {},
              "neighbor_support_across_layers": {}}
    for i, a in enumerate(names):
        top1 = cosine_knn(layer_features[a], 1).ravel()
        result["same_body_top1"][a] = float(np.mean(np.asarray(body_ids)[top1] == np.asarray(body_ids))) if len(top1) else 0.0
        for k in ks:
            knn = neighbors[a][k]
            reciprocal = [i in set(knn[j]) for i, row in enumerate(knn) for j in row]
            result["mutual_knn_fraction"][f"{a}|k{k}"] = float(np.mean(reciprocal)) if reciprocal else 0.0
        for b in names[i + 1:]:
            top_a, top_b = cosine_knn(layer_features[a], 1).ravel(), cosine_knn(layer_features[b], 1).ravel()
            result["top1_agreement"][f"{a}|{b}"] = float(np.mean(top_a == top_b))
            for k in ks:
                aa, bb = neighbors[a][k], neighbors[b][k]
                jac = [len(set(x) & set(y)) / max(1, len(set(x) | set(y))) for x, y in zip(aa, bb)]
                result["knn_jaccard"][f"{a}|{b}|k{k}"] = float(np.mean(jac)) if jac else 0.0
    if names:
        k = max(ks, default=1)
        if len(names) > 1:
            per_instance = []
            for row in range(len(body_ids)):
                counts = {}
                for name in names:
                    for neighbor in neighbors[name][k][row]:
                        counts[int(neighbor)] = counts.get(int(neighbor), 0) + 1
                per_instance.append(counts)
            support = [count for counts in per_instance for count in counts.values()]
            result["neighbor_support_across_layers"][f"all_layers_top{k}"] = {
                "fraction_of_unique_neighbors_seen_in_2plus_layers":
                    float(np.mean(np.asarray(support) >= 2)) if support else 0.0,
                "mean_number_of_repeated_neighbors_per_query":
                    float(np.mean([sum(v >= 2 for v in c.values()) for c in per_instance])) if per_instance else 0.0,
            }
        sets = [set(row) for row in neighbors[names[0]][k]]
        for name in names[1:]:
            sets = [left & set(right) for left, right in zip(sets, neighbors[name][k])]
        counts = np.asarray([len(x) for x in sets], dtype=np.int64)
        result["consensus"][f"all_layers_top{k}"] = {
            "mean_shared_neighbors": float(counts.mean()) if len(counts) else 0.0,
            "fraction_with_no_shared_neighbor": float(np.mean(counts == 0)) if len(counts) else 1.0,
            "fraction_with_at_least_one": float(np.mean(counts > 0)) if len(counts) else 0.0,
        }
    return result


def shared_neighbor_target_differences(layer_features, targets, target_names, k=20):
    names = list(layer_features)
    sets = [set(row) for row in cosine_knn(layer_features[names[0]], k)] if names else []
    for name in names[1:]:
        sets = [left & set(row) for left, row in zip(sets, cosine_knn(layer_features[name], k))]
    differences = [[] for _ in target_names]
    for anchor, neighbors in enumerate(sets):
        for neighbor in neighbors:
            for j, values in enumerate(differences):
                values.append(float(abs(targets[anchor, j] - targets[neighbor, j])))
    return {name: {"mean_absolute_difference": float(np.mean(values)) if values else None,
                   "n_pairs": len(values)} for name, values in zip(target_names, differences)}
