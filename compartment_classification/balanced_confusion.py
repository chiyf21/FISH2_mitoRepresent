"""Balanced-only replacement for the existing by-type RF compartment figures."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from experiment import FEATURES, SEED, dump, sha
from fig2c_aggregation import score, run_oof_rf, aggregate_probabilities

DEFAULT_BASE = Path("/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM")


def table(path):
    return pd.read_csv(path, dtype={"body_id": str, "record_id": str, "source_run_id": str})


def aligned_handcrafted(path, manifest, classes, fish=False):
    full = table(path)
    assert full.record_id.is_unique
    keep = full.compartment.isin(classes)
    if fish:
        keep &= full.compartment_qc.eq("ok")
    assert set(full.loc[keep, "record_id"]) == set(manifest.record_id)
    selected = full.set_index("record_id").loc[manifest.record_id].reset_index()
    for field in ["body_id", "cell_type", "compartment"]:
        assert selected[field].tolist() == manifest[field].tolist(), field
    columns = FEATURES if fish else [
        "length along skeleton" if c == "length_along_skeleton_nm" else c for c in FEATURES]
    x = selected[columns].to_numpy(dtype=np.float64)
    assert x.shape == (len(manifest), 22) and np.isfinite(x).all()
    return selected, x, columns


def fish_balanced(base, selected):
    runs = base / "point-MAE/runs"
    cache = runs / "mito_embedding_dashboard_20261030_01/cache/features_b9a9cbbcd15a81c16c36"
    protocol = runs / "fish2_reviewed_81_protocol"
    legacy_path = cache / "balanced_pair_12.npy"
    new_path = protocol / "fish2_14_layer12/balanced_pair_12.npy"
    cohort_path = runs / "mito_newmae_vicreg_reg02_cls005_20261105_01/cohort.json"
    map_path = base / "neuron_mito_dataset/embeddings/fusion50_fish2_14_20261112/instance_map.csv"
    cohort = json.loads(cohort_path.read_text())
    legacy_index = {
        (str(row["key"][0]), str(row["key"][1]), int(row["key"][2])): i
        for i, row in enumerate(cohort)}
    assert len(legacy_index) == len(cohort)
    mapping = table(map_path)
    assert mapping.record_id.is_unique
    new_index = dict(zip(mapping.record_id, mapping.embedding_row.astype(int)))
    old = np.load(legacy_path, mmap_mode="r")
    new = np.load(new_path, mmap_mode="r")
    assert old.shape == (len(cohort), 1152) and new.shape == (len(mapping), 1152)
    x = np.empty((len(selected), 1152), dtype=np.float32)
    rows = []
    legacy = "e16_legacy_repair_reinfer_20261105"
    for i, row in enumerate(selected.itertuples(index=False)):
        if row.source_run_id == legacy:
            index = legacy_index[(legacy, str(row.body_id), int(row.global_instance))]
            x[i] = old[index]
            path = legacy_path
        else:
            index = int(new_index[row.record_id])
            x[i] = new[index]
            mapped = mapping.loc[mapping.record_id.eq(row.record_id)].iloc[0]
            assert str(mapped.body_id) == str(row.body_id)
            path = new_path
        rows.append({"record_id": row.record_id, "balanced_file": str(path), "balanced_row": index})
    files = [legacy_path, new_path, cohort_path, map_path,
             protocol / "fish2_14_layer12/complete.json",
             protocol / "embedding_features.py"]
    checkpoint = str(runs / "mito_newmae_vicreg_reg02_cls005_20261105_01/checkpoints/refine_epoch050.pt")
    assert json.loads(files[4].read_text())["checkpoint"] == checkpoint
    return x, pd.DataFrame(rows), files, checkpoint


def fly_balanced(base, manifest):
    run = base / "point-MAE/runs/lc21_fusion50_128_20261112_153317"
    directory = run / "umap_epoch033_author6400"
    paths = [directory / f"vectors_{i}.npz" for i in range(2)]
    original = base / "compartment_classification/fish2_flyem_fixed33_20261113_v1/flyem_embeddings"
    old_map = table(original / "instance_map.csv")
    assert old_map.record_id.is_unique
    old_index = old_map.set_index(["body_id", "record_id"]).embedding_row.to_dict()
    old_fused = np.load(original / "fused_12.npy", mmap_mode="r")
    target = {(r.body_id, r.record_id): i for i, r in enumerate(manifest.itertuples(index=False))}
    seen, rows = set(), []
    x = np.empty((len(manifest), 1152), dtype=np.float32)
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            keys = [tuple(key) for key in z["keys"].tolist()]
            assert z["balanced_merge"].shape == (len(keys), 1152)
            # The sibling fused array is bit-identical to the fixed epoch033 export.
            # This validates the checkpoint/instance identity without loading any model.
            previous = [int(old_index[key]) for key in keys]
            np.testing.assert_array_equal(z["fused"], old_fused[previous])
            balanced = z["balanced_merge"]
            for j, key in enumerate(keys):
                assert key in target and key not in seen
                seen.add(key)
                x[target[key]] = balanced[j]
                rows.append({"record_id": key[1], "balanced_file": str(path),
                             "balanced_row": j, "balanced_array": "balanced_merge"})
    assert seen == set(target)
    checkpoint = str(run / "refine_both_cls_ln/checkpoints/refine_epoch033.pt")
    files = paths + [original / "instance_map.csv", original / "complete.json",
                     original / "provenance.json"]
    return x, pd.DataFrame(rows), files, checkpoint


def plot(results, dataset, classes, n, bodies, ngroups, path):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.subplots_adjust(left=.145, right=.975, bottom=.105, top=.82,
                            wspace=.47, hspace=.59)
        ticks = list(range(len(classes)))
        labels = [c.replace("connecting cable", "connecting\ncable") for c in classes]
        for row, task in enumerate(["per-mitochondrion", "per-compartment"]):
            for col, feature in enumerate(["handcrafted", "balanced"]):
                ax = axes[row, col]
                result = results[task + "/" + feature]
                cm = np.asarray(result["confusion"])
                norm = np.asarray(result["confusion_normalized"])
                assert cm.sum() == (n if row == 0 else ngroups)
                ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
                for i in ticks:
                    for j in ticks:
                        ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%}",
                                ha="center", va="center", fontsize=14,
                                color="white" if norm[i, j] > .55 else "black")
                ax.set_xticks(ticks, labels)
                ax.set_yticks(ticks, labels)
                ax.set_xlabel("Predicted compartment", labelpad=8)
                ax.set_ylabel("True compartment", labelpad=8)
                fname = "Handcrafted 22D" if col == 0 else "Balanced 1152D"
                count = n if row == 0 else ngroups
                ax.set_title(
                    f"{dataset} | {task}\n{fname} | n={count:,}\n"
                    f"BA={result['balanced_accuracy']:.3f} | Macro-F1={result['macro_f1']:.3f}",
                    fontsize=11, pad=13)
        fig.suptitle(f"{dataset} | RF by neuron type | 5-fold body OOF\n"
                     f"n={n:,} mitochondria | bodies={bodies} | groups={ngroups}",
                     fontsize=16, y=.97)
        fig.text(.5, .035, "Count + true-class normalized % | "
                 "Per-compartment: body + true compartment (oracle grouping)",
                 ha="center", fontsize=11)
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)


def run_dataset(base, out, dataset, workers):
    slug = dataset.lower()
    previous = base / f"compartment_classification/{slug}_fig2c_aggregation_20261113_v1"
    manifest_path = previous / "instances.csv"
    fold_path = previous / "body_folds.csv"
    group_path = previous / "groups.csv"
    manifest = table(manifest_path)
    folds, groups = table(fold_path), table(group_path)
    classes = ["axon", "dendrite"] + (["connecting cable"] if dataset == "FlyEM" else [])
    expected = (6400, 96, 212) if dataset == "FlyEM" else (5059, 81, 161)
    assert (len(manifest), manifest.body_id.nunique(), len(groups)) == expected
    assert manifest.record_id.is_unique and folds.body_id.is_unique
    assert set(manifest.fold) == set(range(5))
    assert manifest.groupby("body_id").fold.nunique().eq(1).all()
    assert manifest.groupby("body_id").cell_type.nunique().eq(1).all()
    assert manifest.fold.tolist() == manifest.body_id.map(folds.set_index("body_id").fold).tolist()
    assert manifest.compartment.map(dict(zip(classes, range(len(classes))))).tolist() == manifest.label.tolist()
    assert groups.n_mito.sum() == len(manifest)
    assert groups[["body_id", "compartment"]].duplicated().sum() == 0
    is_fish = dataset == "Fish2"
    hp = (base / "neuron_mito_dataset/embeddings/fish2_reviewed_81/fish2_handcraft_81neurons_22_with_compartment.csv"
          if is_fish else base / "flyem_lc21_source_instances_20261110/paper_features_lc21_v1/author_dryad/flyem_mito_author_features_exact.csv")
    selected, handcrafted, columns = aligned_handcrafted(hp, manifest, classes, is_fish)
    balanced, mapping, files, checkpoint = (
        fish_balanced(base, selected) if is_fish else fly_balanced(base, manifest))
    assert balanced.shape == (len(manifest), 1152) and np.isfinite(balanced).all()
    assert mapping.record_id.is_unique and set(mapping.record_id) == set(manifest.record_id)
    directory = out / slug
    directory.mkdir()
    manifest.to_csv(directory / "instances.csv", index=False)
    folds.to_csv(directory / "body_folds.csv", index=False)
    groups.to_csv(directory / "groups.csv", index=False)
    mapping.set_index("record_id").loc[manifest.record_id].reset_index().to_csv(
        directory / "balanced_instance_map.csv", index=False)
    results, training = {}, []
    paired = groups.copy()
    y = manifest.label.to_numpy(dtype=int)
    for feature, x in [("handcrafted", handcrafted), ("balanced", balanced)]:
        print(f"{dataset} {feature}: fitting 5-fold by-type RF {x.shape}", flush=True)
        probability, records = run_oof_rf(
            x, y, manifest, "A_by_type", workers, classes, slug + "_" + feature)
        training.extend(records)
        instance_predictions = manifest.copy()
        for i, name in enumerate(classes):
            instance_predictions["p_" + name] = probability[:, i]
        instance_predictions["prediction"] = probability.argmax(1)
        instance_predictions.to_csv(directory / f"{feature}_instance_oof.csv", index=False)
        group_prediction = aggregate_probabilities(manifest, groups, probability)[0]
        paired[feature + "_prediction"] = [classes[i] for i in group_prediction]
        results["per-mitochondrion/" + feature] = score(y, probability.argmax(1), classes)
        results["per-compartment/" + feature] = score(groups.label, group_prediction, classes)
    paired.to_csv(directory / "group_predictions.csv", index=False)
    missing = [r for r in training if r["missing_classes"]]
    # Preserve every instance; absent training classes have zero aligned probability.
    dump(directory / "training_records.json", training)
    files = [hp, manifest_path, fold_path, group_path] + files
    provenance = dict(dataset=dataset, classes=classes, handcrafted_features=columns,
                      balanced_dimension=1152, checkpoint=checkpoint,
                      instances=expected[0], bodies=expected[1], groups=expected[2],
                      balanced_sources=sorted(mapping.balanced_file.unique().tolist()),
                      files={str(p): sha(p) for p in files},
                      missing_training_classes=missing)
    dump(directory / "provenance.json", provenance)
    dump(directory / "metrics.json", results)
    plot(results, dataset, classes, *expected, out / f"{slug}_balanced_2x2_confusion.png")
    print(json.dumps({"dataset": dataset, "metrics": results}), flush=True)
    return provenance, results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    summaries = [run_dataset(args.base, args.output, name, args.workers)
                 for name in ["FlyEM", "Fish2"]]
    metadata = dict(seed=SEED, split="Original per-dataset fixed five body folds, reused unchanged",
                    training="Separate RF per neuron type; no pooled training",
                    rf=dict(n_estimators=100, min_weight_fraction_leaf=.01,
                            max_features="sqrt", sample_weight="1 / training class count"),
                    aggregation="body_id + true compartment; argmax(mean(log10(clip(p,1e-15,1))))",
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                    code_sha256=sha(Path(__file__)), python=platform.python_version(),
                    numpy=np.__version__, sklearn=sklearn.__version__,
                    elapsed_seconds=time.time() - start)
    dump(args.output / "run.json", metadata)
    lines = ["# Balanced 替换版混淆矩阵", "",
             "沿用各数据集原固定五折 body OOF、按神经元类型分别训练的论文式 RF（100 棵树、叶权重下限 0.01、训练类别计数倒数 sample_weight）。上下两行共用同一套折外概率；下行为 body_id + 真实 compartment 的 oracle grouping，取平均 log probability 后 argmax。",
             "",
             "实际 balanced 均为 **1152D**（point 384D + EM 768D，经方差 balance 后拼接）；384D 为 fused。本次直接使用现成 balanced，未降维、未训练或运行编码器。", ""]
    for provenance, results in summaries:
        dataset = provenance["dataset"]
        lines += [f"**{dataset}**：{provenance['instances']:,} 实例，{provenance['bodies']} bodies，{provenance['groups']} oracle groups。", "",
                  f"- Handcrafted：`{next(iter(provenance['files']))}`",
                  "- Balanced：" + "；".join("`" + p + "`" for p in provenance["balanced_sources"])
                  + ("，数组键 `balanced_merge`。" if dataset == "FlyEM" else "，按 cohort / instance_map 稳定身份合并。"),
                  f"- Checkpoint：`{provenance['checkpoint']}`。"]
        for task in ["per-mitochondrion", "per-compartment"]:
            scores = []
            for feature in ["handcrafted", "balanced"]:
                r = results[task + "/" + feature]
                scores.append(f"{feature} **{r['balanced_accuracy']:.4f} / {r['macro_f1']:.4f}**")
            lines.append(f"- {task}（BA / Macro-F1）：" + "；".join(scores) + "。")
        missing = provenance["missing_training_classes"]
        lines.append("- 缺训练类别：" + (str(len(missing)) + " 个 type-fold，保留测试实例并将缺失类概率置零。" if missing else "无；未删除任何目标实例。"))
        lines.append("")
    lines += ["FlyEM 保留作者 exact-center 三类标签；Fish2 仅 axon/dendrite 且 compartment_qc=ok，reviewed 标签仍有原有来源追溯限制。", "",
              f"代码分支：`codex/balanced-confusion-20261113`；运行提交：`{metadata['code_commit']}`。"]
    (args.output / "README_zh.md").write_text("\n".join(lines) + "\n")
    dump(args.output / "complete.json", {"status": "complete", "elapsed_seconds": time.time() - start})


if __name__ == "__main__":
    main()
