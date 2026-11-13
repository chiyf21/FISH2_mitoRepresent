"""Read two median-sized real instances and create a Chinese data showcase; no model calls."""
import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
from skimage.measure import marching_cubes

FEATURES = ["size", "mito SA", "mito CA", "convex hull compactness"] + [
    f"PC{i} {s}" for i in (1, 2, 3) for s in ["Length", "inertia moment", "symmetry", "CA", "Circum"]
] + ["mito diameter", "length_along_skeleton_nm", "SBI"]


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def one(path, key, value):
    found = [(i, r) for i, r in enumerate(rows(path)) if r[key] == str(value)]
    assert len(found) == 1, (path, key, value, len(found))
    return found[0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(value):
    if isinstance(value, (dict, list, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def mdtable(headers, data):
    esc = lambda v: text(v).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "|" + "|".join(["---"] * len(headers)) + "|"] +
                      ["| " + " | ".join(esc(v) for v in row) + " |" for row in data])


MEANINGS = {
    "record_id": "线粒体实例唯一标识；不是神经元ID",
    "body_id": "所属神经元ID", "bodyId": "此作者/神经元表中的所属神经元ID",
    "cell_type": "神经元类别代码；缩写展开未在本记录定义", "type": "神经元类别代码",
    "global_instance": "该body内的线粒体局部编号，不能跨body单独使用",
    "source_run_id": "实例来源批次/版本",
    "compartment": "该实例的区室类别", "compartment_qc": "区室映射质量标记，不是模型分类准确率",
    "compartment_review_revision": "区室审核修订号；完整审核来源仍有追溯限制",
    "skeleton_xyz_rescaled": "原表是否标记骨架XYZ重缩放",
    "skeleton_edge_type": "投影关联的骨架边类型", "skeleton_edge_id": "关联骨架边索引",
    "skeleton_edge_u_swc": "边端点u的SWC节点编号", "skeleton_edge_v_swc": "边端点v的SWC节点编号",
    "compartment_distance_nm": "区室骨架映射距离，nm；不能直接当作标注概率",
    "nearest_other_compartment_nm": "到其他区室的最近距离字段，nm；不是其他线粒体距离",
    "merged_from_instances": "合并/修补前实例编号列表；编号体系不同于最终global_instance",
    "voxel_count": "mask前景体素数", "source_voxel_count": "源记录的体素数",
    "bbox_xyz": "[xmin,ymin,zmin,xmax,ymax,zmax]，全局体素网格坐标",
    "array_axis_order": "NumPy轴顺序，ZYX表示先Z后Y再X",
    "mask_file": "实例目录内mask文件名", "em_file": "实例目录内EM文件名",
    "connectivity_merge": "合并时使用的3D连通邻域参数；26邻域含面、边、角接触",
    "boundary_repair_logit_threshold": "边界修补logit阈值；不直接解释成同数值概率",
    "boundary_repair_max_layers": "修补层数参数；null的执行语义未在本记录注明",
    "boundary_repair_roi_padding": "修补ROI扩展参数；本记录未注明单位",
    "mask_probability_threshold": "原分割概率阈值；当前mask已是二值结果，不保存概率体",
    "minimum_instance_voxels_before_repair": "修补前最小实例体素数阈值",
    "source_mito_label": "源mito-objects层的线粒体对象ID；不是body_id，也不是mask中的像素值",
    "center_xyz": "源记录中心/参考坐标，XYZ体素网格；不假定等于mask质心",
    "voxel_size_nm_xyz": "XYZ方向体素间距，nm",
    "source_mito_layer": "原始线粒体分割层URI", "source_em_layer": "原始EM图像层URI",
    "status": "此记录对象的状态字段；实例downloaded与神经元Traced语义不同",
    "author_row_id": "作者表中行ID，需结合body，不是线粒体全局ID",
    "compartment_source": "标签来源；本例必须是published_mito_exact_center",
    "compartment_id": "作者类别编码；本例3对应dendrite",
    "published_mito_class": "作者发布的该线粒体class编码",
    "nearest_node_class": "最近骨架节点class；本例只是对照，不用于替代exact标签",
    "nearest_node_id": "最近作者骨架节点编号", "nearest_node_distance_voxels": "最近节点距离，原字段单位为体素",
    "rowId": "作者表行ID", "class": "作者区室类别编码",
    "x": "作者表参考点X坐标", "y": "作者表参考点Y坐标", "z": "作者表参考点Z坐标",
    "instance": "神经元命名字段，不是global_instance",
    "cropped": "神经元层面的cropped标志，不能据此判断线粒体ROI是否截断",
    "pre": "神经元原始pre计数；具体接口口径本次未进一步确认",
    "post": "神经元原始post计数；不是该线粒体的计数",
    "somaLocation": "原始胞体位置及坐标参考信息；与本实例EM网格的关系未独立确认",
    "roi": "作者解剖ROI代码，保留原文", "mitoType": "作者线粒体类型/外观字段，保留原文",
    "skel CA": "神经元局部截面积，作者额外字段；未进入22D输入",
    "skel Circum": "神经元局部截面周长，作者额外字段；未进入22D输入",
    "branch_nodes_in": "作者骨架分支关联列表；列表内索引口径未独立确认",
    "edge_nodes": "作者骨架边关联记录，包含索引/长度量/up-down标记；不擅自补充单位",
    "voxel_size_xyz_nm": "XYZ方向体素间距，nm", "voxel_size_zyx_nm": "ZYX轴顺序的体素间距，nm",
    "centroid_x_voxel": "mask质心的全局X体素坐标", "centroid_y_voxel": "mask质心的全局Y体素坐标",
    "centroid_z_voxel": "mask质心的全局Z体素坐标",
    "skeleton_radius_nm": "关联神经元SWC节点的半径估计，nm；不是线粒体半径",
    "skeleton_section_method": "用最近SWC节点半径作圆形神经元截面的估计方法标记",
    "nearest_skeleton_node_id": "特征计算时的最近骨架节点编号；不必等于区室边端点",
    "nearest_node_x_global_vox": "关联骨架节点全局X体素坐标",
    "nearest_node_y_global_vox": "关联骨架节点全局Y体素坐标",
    "nearest_node_z_global_vox": "关联骨架节点全局Z体素坐标",
    "nearest_surface_distance_nm": "骨架映射记录中的最近表面距离，nm",
    "mapped_skeleton_node_count": "映射至该线粒体的骨架节点数量",
    "skeleton_endpoint_a_node_id": "覆盖路径端点a的SWC编号", "skeleton_endpoint_b_node_id": "覆盖路径端点b的SWC编号",
    "length_along_skeleton_um": "同一神经元骨架覆盖长度的µm表示；22D实际使用nm列",
    "swc_scale_applied": "计算时实际应用的SWC坐标缩放倍数",
    "swc_header_scale": "SWC头部记录的scale值；不等于本次实际应用倍数",
    "swc_coordinate_units": "所采用的SWC坐标单位/参考系文字标记",
    "surface_mapping_method": "将mask表面与骨架关联时的表面提取方法",
    "surface_point_count": "表面映射使用的点数", "qc": "特征/骨架映射QC标记",
    "swc_path": "关联神经元骨架SWC文件", "neuron_manifest_path": "关联神经元元数据manifest",
    "i_synapses_on": "作者突触关联索引列表；不能未经映射当作全局突触ID",
}


def metadata_table(record):
    return mdtable(["原始字段", "实际值", "解释"], [
        [f"`{k}`", f"`{text(v)}`", MEANINGS.get(k, "原始字段保留；具体含义或单位本次未确认")]
        for k, v in record.items()])


def feature_info(name, dataset):
    fish = dataset == "fish2"
    if name == "size":
        return ("前景体积" if fish else "源线粒体体素数"), ("nm³" if fish else "voxel，已与mask计数核对")
    if name == "mito SA":
        return "线粒体表面积估计", ("nm²，体素暴露面法" if fish else "作者网格面积单位u²，非直接nm²")
    if name == "mito CA":
        return ("PC1法向、过质心截面的面积；本例等于PC1 CA" if fish else "作者神经元局部优化截面平面中的线粒体面积，并非必为PC1截面"), ("nm²" if fish else "作者截面网格面积单位u²")
    if name == "convex hull compactness":
        return "实际为凸包表面积hull.area，不是无量纲紧致度比", ("nm²" if fish else "作者网格面积单位u²")
    if name.startswith("PC"):
        suffix = name[4:]
        desc = {"Length": "沿该主轴的投影跨度", "inertia moment": "到该轴的平方距离之和；每个体素中心单位权重，非质量惯性矩",
                "symmetry": "关于该主轴法向平面的镜像重合指标；越接近1越对称",
                "CA": "该主轴法向、过质心截面的面积", "Circum": "该主轴法向截面的周长估计"}[suffix]
        unit = {"Length": "nm" if fish else "u（体素坐标长度）",
                "inertia moment": "nm²（单位点权重平方距离和）" if fish else "u²（单位点权重平方距离和）",
                "symmetry": "无量纲", "CA": "nm²" if fish else "u²",
                "Circum": "nm" if fish else "u"}[suffix]
        return desc, unit
    if name == "mito diameter":
        return "采样体素中心之间的最大点对距离；不是等效球直径或局部宽度", ("nm" if fish else "u（体素坐标长度）")
    if name in ["length_along_skeleton_nm", "length along skeleton"]:
        return "线粒体覆盖的神经元骨架路径长度；不是线粒体自身骨架", ("nm" if fish else "未知：作者原表未明示单位，不直接标作nm")
    return "log10(同体积球的表面积 / 线粒体表面积)，按最终CSV核验", "无量纲"


def array_info(em_path, mask_path, meta, spacing):
    em, mask_raw = np.load(em_path, allow_pickle=False), np.load(mask_path, allow_pickle=False)
    assert em.ndim == mask_raw.ndim == 3 and em.shape == mask_raw.shape
    assert meta["array_axis_order"] == "ZYX"
    mask = mask_raw > 0
    count = int(mask.sum())
    assert count == int(meta["voxel_count"])
    bbox = np.array(meta["bbox_xyz"], dtype=int)
    delta = bbox[3:] - bbox[:3]
    assert np.array_equal(delta, np.array(mask.shape[::-1])), "Expected half-open XYZ bbox"
    indices = np.nonzero(mask)
    local = np.array([a.mean() for a in indices])
    centroid = bbox[:3] + local[::-1]
    info = dict(em_path=str(em_path), mask_path=str(mask_path), format="NumPy .npy", shape_zyx=list(em.shape),
                em_dtype=str(em.dtype), mask_dtype=str(mask_raw.dtype), em_range=[int(em.min()), int(em.max())],
                mask_values=np.unique(mask_raw).tolist(), foreground_voxels=count, total_voxels=int(mask.size),
                foreground_fraction=count/mask.size, spacing_xyz_nm=list(map(float, spacing)),
                bbox_xyz=meta["bbox_xyz"], bbox_convention="half-open, verified against array shape",
                mask_centroid_local_zyx=local.tolist(), mask_centroid_global_xyz=centroid.tolist(),
                derived_volume_nm3=float(count*np.prod(spacing)), derived_volume_um3=float(count*np.prod(spacing)/1e9),
                em_sha256=sha(em_path), mask_sha256=sha(mask_path), center_z_index=em.shape[0]//2)
    return em, mask, info


def picture(out, dataset, identity, em, mask, info):
    z = info["center_z_index"]
    spacing = np.array(info["spacing_xyz_nm"], dtype=float)
    extent = [0, em.shape[2]*spacing[0]/1000, em.shape[1]*spacing[1]/1000, 0]
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for slot, kind in [(1, "EM"), (2, "overlay"), (3, "mask")]:
        ax = fig.add_subplot(2, 2, slot)
        ax.imshow(mask[z] if kind == "mask" else em[z], cmap="gray", vmin=0, vmax=1 if kind == "mask" else 255,
                  extent=extent, interpolation="nearest")
        if kind == "overlay":
            ax.imshow(np.ma.masked_where(~mask[z], np.ones(mask[z].shape)), cmap=ListedColormap(["#ff8c00"]),
                      vmin=0, vmax=1, alpha=0.4, extent=extent, interpolation="nearest")
        ax.set_title({"EM": "EM center XY slice", "overlay": "Same slice: EM + mask", "mask": "Same slice: binary mask"}[kind] + f" / z={z}")
        ax.set_xlabel("Local X (um)")
        ax.set_ylabel("Local Y (um)")
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    sz = spacing[::-1]/1000
    verts, faces, _, _ = marching_cubes(np.pad(mask.astype(np.uint8), 1), level=0.5, spacing=sz)
    verts -= sz
    xyz = verts[:, ::-1]
    ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles=faces, color="#ee8a22",
                    linewidth=0, antialiased=False, shade=True)
    ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 1e-6))
    ax.set_xlabel("Local X (um)", fontsize=9, labelpad=6)
    ax.set_ylabel("Local Y (um)", fontsize=9, labelpad=6)
    ax.set_zlabel("Local Z (um)", fontsize=9, labelpad=6)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=8, pad=1)
    ax.view_init(elev=25, azim=-60)
    ax.set_title("3D surface of stored mask")
    fig.suptitle(f"{dataset.upper()} / {identity['cell_type']} / {identity['compartment']}\nbody {identity['body_id']} / instance {identity['global_instance']}", fontsize=15)
    path = out / f"{dataset}_instance.png"
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset-root", type=Path, required=True)
    ap.add_argument("--binary-results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    root = args.asset_root
    bundles = {}
    for dataset in ["fish2", "flyem"]:
        fish = dataset == "fish2"
        base = root / "neuron_mito_dataset/embeddings/fish2_reviewed_81" if fish else root / "flyem_lc21_source_instances_20261110"
        hp = base / "fish2_handcraft_81neurons_22_with_compartment.csv" if fish else base / "paper_features_lc21_v1/author_dryad/flyem_mito_author_features_exact.csv"
        candidates = [r for r in rows(hp) if r["compartment"] in ["axon", "dendrite"] and (not fish or r["compartment_qc"] == "ok")]
        median = float(np.median([float(r["size"]) for r in candidates]))
        h = min(candidates, key=lambda r: (abs(math.log(float(r["size"])/median)), r["record_id"]))
        rid, body = h["record_id"], h["body_id"]
        identity = {k: v for k, v in h.items() if k not in FEATURES and k != "length along skeleton"}
        sources = dict(handcrafted=str(hp), selection_pool=len(candidates), median_native_size=median, selected_native_size=h["size"])
        blocks = []
        if fish:
            _, paths = one(base / "instances/instance_paths.csv", "record_id", rid)
            directory = base / "instances"
            em_path, mask_path, meta_path = [directory / paths[k] for k in ["em_path", "mask_path", "metadata_path"]]
            raw = json.loads(meta_path.read_text())
            detail_file = root / "neuron_mito_dataset/handcraft_features/fish2_nov_reviewed_81_v1/handcrafted_features_with_skeleton.csv"
            _, detail = one(detail_file, "record_id", rid)
            spacing = [float(x) for x in detail["voxel_size_xyz_nm"].split(",")]
            sources["spacing_source"] = str(detail_file)
            ep = base / "fish2_model_embedding_81neurons_384.csv"
            csv_index, erow = one(ep, "record_id", rid)
            vector_csv = np.array([erow[f"embedding_{k:03d}"] for k in range(384)], dtype=np.float32)
            for field in erow:
                if not field.startswith("embedding_"):
                    assert erow[field] == h[field]
            matched = []
            for name in ["fusion50_all16428_20261112", "fusion50_fish2_14_20261112"]:
                directory = root / "neuron_mito_dataset/embeddings" / name
                found = [(i, r) for i, r in enumerate(rows(directory/"instance_map.csv")) if r["record_id"] == rid]
                if found:
                    assert len(found) == 1
                    matched.append((directory, found[0][1]))
            assert len(matched) == 1
            directory, mapping = matched[0]
            vector_path = directory / "fused_12.npy"
            vector_index = int(mapping["embedding_row"])
            vector = np.array(np.load(vector_path, mmap_mode="r", allow_pickle=False)[vector_index], dtype=np.float32)
            np.testing.assert_array_equal(vector_csv, vector)
            model = json.loads((directory / "complete.json").read_text())
            input_file = root / "neuron_mito_dataset/handcraft_features/fish2_nov_reviewed_81_v1/model_input_manifest.csv"
            _, original = one(input_file, "record_id", rid)
            for k, path in [("em_path", em_path), ("mask_path", mask_path), ("metadata_path", meta_path)]:
                assert sha(original[k]) == sha(path)
            blocks = [("原始实例metadata.json：全部字段", raw), ("实例路径表：全部字段", paths),
                      ("Embedding映射表：全部字段", mapping), ("原始模型输入记录", original),
                      ("关联特征计算记录中新增的实际字段", {k:v for k,v in detail.items() if k not in h})]
            supplemental = dict(feature_calculation_record=detail)
            vector_record = dict(path=str(vector_path), row_index_zero_based=vector_index, source_csv=str(ep),
                                 csv_data_row_zero_based=csv_index, csv_text_line_one_based=csv_index+2,
                                 checkpoint=model["checkpoint"], csv_vs_npy_max_abs_delta=0.0)
        else:
            instance = rid.rsplit("_", 1)[1]
            directory = base / "results" / body / "instances" / ("instance_" + instance)
            em_path, mask_path, meta_path = directory/"em.npy", directory/"mask.npy", directory/"metadata.json"
            raw = json.loads(meta_path.read_text())
            identity["global_instance"] = raw["global_instance"]
            spacing = raw["voxel_size_nm_xyz"]
            _, label = one(base / "paper_features_lc21_v1/author_dryad/paper22_author_compartments.csv", "record_id", rid)
            assert label["compartment_source"] == "published_mito_exact_center"
            assert str(raw["source_mito_label"]) == h["source_mito_label"] == label["source_mito_label"]
            published_path = base / "paper_features_lc21_v1/author_dryad/saved_mito_df" / f"{h['cell_type']}_{body}_mito_df.csv"
            published = [r for r in rows(published_path) if [float(r[k]) for k in ["x","y","z"]] == raw["center_xyz"]]
            assert len(published) == 1
            published = published[0]
            assert int(float(published["rowId"])) == int(float(h["author_row_id"]))
            assert int(published["class"]) == int(float(label["compartment_id"])) == 3
            for col in h:
                if col in published and col not in identity:
                    assert np.isclose(float(h[col]), float(published[col]), rtol=1e-12)
            mp = args.binary_results / "flyem_embeddings/instance_map.csv"
            _, mapping = one(mp, "record_id", rid)
            vector_path = mp.parent / "fused_12.npy"
            vector_index = int(mapping["embedding_row"])
            vector = np.array(np.load(vector_path, mmap_mode="r", allow_pickle=False)[vector_index], dtype=np.float32)
            export = json.loads((mp.parent / "provenance.json").read_text())
            cohort = json.loads((Path(export["config"]["point_run"]).parent / "refine_both_cls_ln/cohort.json").read_text())[int(mapping["cohort_index"])]
            assert cohort["key"][2] == rid and cohort["mask_path"] == str(mask_path) and cohort["em_path"] == str(em_path)
            neuron = json.loads((directory.parent.parent / "neuron_metadata.json").read_text())
            sources["published_author_row"] = str(published_path)
            sources["exact_center_label_table"] = str(base / "paper_features_lc21_v1/author_dryad/paper22_author_compartments.csv")
            extra = {k: v for k, v in published.items() if k not in h}
            blocks = [("原始实例metadata.json：全部字段", raw), ("作者标签记录：全部字段", label),
                      ("神经元metadata：全部字段", neuron), ("原始作者表中不在22维输入表里的其他字段", extra),
                      ("Embedding映射表：全部字段", mapping), ("模型cohort记录：全部字段", cohort)]
            supplemental = dict(published_row=published, published_path=str(published_path),
                                dataset_manifest=json.loads((base / "manifest.json").read_text()))
            vector_record = dict(path=str(vector_path), row_index_zero_based=vector_index, instance_map=str(mp),
                                 checkpoint=export["checkpoint"], embedding_provenance=str(mp.parent/"provenance.json"))
        assert str(raw["body_id"]) == body and int(raw["global_instance"]) == int(identity["global_instance"])
        em, mask, info = array_info(em_path, mask_path, raw, spacing)
        assert float(h["size"]) == (info["derived_volume_nm3"] if fish else info["foreground_voxels"])
        ordered = [col if fish or col != "length_along_skeleton_nm" else "length along skeleton" for col in FEATURES]
        values = np.array([h[c] for c in ordered], dtype=np.float64)
        assert len(values) == 22 and np.isfinite(values).all() and vector.shape == (384,) and np.isfinite(vector).all()
        radius = (3*float(h["size"])/(4*np.pi))**(1/3)
        np.testing.assert_allclose(float(h["SBI"]), np.log10(4*np.pi*radius*radius/float(h["mito SA"])), atol=1e-12)
        cache = args.binary_results / dataset
        cache_index, cache_meta = one(cache / "instances.csv", "record_id", rid)
        assert cache_meta["body_id"] == body and cache_meta["compartment"] == h["compartment"]
        cached_vector = np.load(cache/"embedding.npy", mmap_mode="r")[cache_index]
        # CSV-derived caches retain decimal float64 values; compare at the source float32 precision.
        np.testing.assert_array_equal(cached_vector.astype(np.float32), vector)
        vector_record["cache_vs_source_max_abs_delta"] = float(np.max(np.abs(cached_vector.astype(np.float64)-vector.astype(np.float64))))
        np.testing.assert_allclose(np.load(cache/"handcrafted.npy", mmap_mode="r")[cache_index], values, rtol=1e-12)
        vector_record["binary_cache_row_zero_based"] = cache_index
        vector_record["binary_cache_directory"] = str(cache)
        sample = out / f"{dataset}_embedding_sample.npy"
        np.save(sample, vector)
        v = vector.astype(np.float64)
        stats = dict(dimensions=384, dtype=str(vector.dtype), l2_norm=float(np.linalg.norm(v)), mean=float(v.mean()),
                     standard_deviation_ddof0=float(v.std()), minimum=float(v.min()), maximum=float(v.max()), first10=v[:10].tolist())
        figure = picture(out, dataset, identity, em, mask, info)
        bundles[dataset] = dict(identity=identity, sources=sources, handcrafted=h, raw_metadata=raw,
                               metadata_path=str(meta_path), metadata_blocks=blocks, supplemental=supplemental,
                               arrays=info, embedding=vector_record, embedding_statistics=stats,
                               extracted_embedding=str(sample), figure=str(figure),
                               alignment="PASS: raw IDs/bbox/count, handcrafted, original vectors, labels and classification caches agree")
    lines = ["# 两个真实线粒体实例：数据样本展示", "",
             "每个实例的原始包是EM三维裁块、二值mask和metadata.json；22维手工特征、384维向量及区室标签主要保存在另外的表/数组中，经实例ID连接，并不全部存在于实例文件夹里。",
             "选择规则：各自在原二分类的有效候选（Fish2额外要求QC=ok）中选size最接近中位数的实例；未读取预测成绩。size在两套表中的单位不同，中位数仅在各自数据集内部计算。",
             "下文全部路径均是113服务器路径。只读了选中实例的EM/mask与直接关联表、已有向量；没有训练、推理或重新提取特征。", ""]
    for dataset, b in bundles.items():
        fish = dataset == "fish2"
        h, info, emb = b["handcrafted"], b["arrays"], b["embedding"]
        lines += [f"## {'Fish2' if fish else 'FlyEM'}", "",
                  f"**实例：`{b['identity']['record_id']}`；神经元类型 `{b['identity']['cell_type']}`；区室 `{b['identity']['compartment']}`。**", "",
                  f"候选{b['sources']['selection_pool']}条，size中位数={b['sources']['median_native_size']}，所选size={b['sources']['selected_native_size']}。", "",
                  f"![{dataset} EM and mask]({dataset}_instance.png)", "",
                  "图中左上是原始EM的ROI中心XY切片，右上是同层mask叠加，左下是同层二值mask，右下是mask三维表面。灰度固定显示0–255；橙色表示该线粒体。3D按实际体素间距显示，坐标为局部µm。为绘制闭合边界仅在可视化副本外补一层0，原文件未修改。",
                  "", "### 身份、标签与QC", "", metadata_table(b["identity"]), "",
                  ("Fish2标签来自带修订号和距离QC的CSV，不在原始实例JSON中；审核源文件的完整追溯仍未解决。"
                   if fish else "FlyEM本例使用作者published_mito_exact_center标签；原始作者class=3，最近节点class=3仅作对照。原始实例JSON没有compartment字段，也没有Fish2式compartment_qc。"),
                  "", "### 原始EM与mask", "",
                  mdtable(["项目", "实际内容"], [[k, v] for k, v in {
                      "EM路径": info["em_path"], "mask路径": info["mask_path"], "metadata路径": b["metadata_path"],
                      "格式/轴顺序": ".npy；ZYX", "EM与mask形状": info["shape_zyx"], "EM dtype": info["em_dtype"],
                      "mask dtype": info["mask_dtype"], "EM灰度范围": info["em_range"], "mask实际值": info["mask_values"],
                      "体素尺寸XYZ（nm）": info["spacing_xyz_nm"], "mask前景体素数": info["foreground_voxels"],
                      "裁块总体素数": info["total_voxels"], "前景比例": info["foreground_fraction"],
                      "由mask计算体积（µm³）": info["derived_volume_um3"],
                      "原始bbox XYZ": info["bbox_xyz"], "bbox约定": "终点不包含；已与数组形状核对",
                      "由mask计算的局部质心ZYX（体素）": info["mask_centroid_local_zyx"],
                      "由bbox+mask计算的全局质心XYZ（体素）": info["mask_centroid_global_xyz"],
                      "中心切片Z索引（0基）": info["center_z_index"]}.items()]), "",
                  f"体素尺寸来源：`{b['sources']['spacing_source'] if fish else b['metadata_path']}`。", "",
                  "bbox/中心的全局体素坐标与图中的局部µm坐标是不同表达。记录里的center_xyz（若存在）保留原值，不自动当作mask的真实质心；世界坐标原点/变换未在这些记录中完整声明。",
                  "", "### 完整22维Handcrafted", "", f"来源：`{b['sources']['handcrafted']}`。数值按最终CSV原字符串列出。"]
        frows = []
        for col in FEATURES:
            raw_col = col if fish or col != "length_along_skeleton_nm" else "length along skeleton"
            meaning, unit = feature_info(raw_col, dataset)
            frows.append([raw_col, meaning, h[raw_col], unit])
        lines += ["", mdtable(["原始列名", "中文含义/计算口径", "实际数值", "单位"], frows), "",
                  "PC1/PC2/PC3是PCA主轴，按变化量排序，不是固定的X/Y/Z，也不是区室标签。inertia moment是几何平方距离和，不含实测质量。symmetry是镜像重合指标。mito diameter是采样最大点对距离，可能小于PC1 Length，不宜当作精确最大尺寸。",
                  ("Fish2长度/面积来自nm坐标。关联计算记录确认XYZ间距为8,8,30nm；size=33074×8×8×30=63502080nm³。该CSV的SBI采用球面积/实际面积的log10，已核对数值；不直接沿用底层编码器中相反符号的中间SBI。"
                   if fish else "FlyEM作者原始代码直接使用体素坐标；表中u表示原作者网格长度单位（该EM网格一体素为8nm），面积标为u²而非直接nm²。不能把本表原数值与Fish2同名列直接拼接。length along skeleton的单位没有在该作者CSV明确标注，保留未知；其代码包含路径求和及PC1长度回退，不能只凭字段名认定精确路径。"),
                  "", "### 384维Embedding摘要", "",
                  mdtable(["统计项", "实际值"], [[k, v] for k, v in {
                      "维数": 384, "数据类型": b["embedding_statistics"]["dtype"],
                      "L2范数": format(b["embedding_statistics"]["l2_norm"], ".10g"),
                      "均值": format(b["embedding_statistics"]["mean"], ".10g"),
                      "标准差（总体，ddof=0）": format(b["embedding_statistics"]["standard_deviation_ddof0"], ".10g"),
                      "最小值": format(b["embedding_statistics"]["minimum"], ".10g"),
                      "最大值": format(b["embedding_statistics"]["maximum"], ".10g")}.items()]), "",
                  mdtable(["维索引（0基）", "前10维实际值"], [[i, format(v, ".10g")] for i,v in enumerate(b["embedding_statistics"]["first10"])]), "",
                  mdtable(["定位项", "实际位置/索引"], [[k, v] for k, v in emb.items()]), "",
                  f"方便复核的单行完整副本：`{b['extracted_embedding']}`，shape=(384,)，由原行直接保存，未重新编码。",
                  "全部row_index和cache_row均为0基索引；CSV另列了含表头的1基文本行号（若适用）。L2范数/均值等只是学习向量统计，维度没有已知独立物理单位或逐维生物学命名；它不是384个体素，也不是22维特征的简单重复。",
                  "", "### 原始记录中的其他实际字段", ""]
        if not fish:
            lines += [f"作者原始行文件：`{b['sources']['published_author_row']}`。",
                      f"exact-center标签表：`{b['sources']['exact_center_label_table']}`。", ""]
        for title, values in b["metadata_blocks"]:
            lines += [f"**{title}**", "", metadata_table(values), ""]
        lines += ["### 同一实例对应关系核对", "", b["alignment"], "",
                  "已核对record_id/body_id、局部实例编号、bbox与数组形状、实际mask计数、标签、向量行、22维值，以及既有分类缓存的对应行。"]
        if fish:
            lines += ["Fish2 reviewed实例包的EM、mask、metadata还与model_input_manifest指向的原始来源包逐文件SHA256一致；CSV中的384维向量与来源NPY对应行逐位相等。"]
        else:
            lines += ["FlyEM的source_mito_label与实例JSON一致；作者表中的bodyId、rowId和记录XYZ精确匹配，原始21项作者特征一致，SBI公式已核对；向量cohort的EM/mask路径与此实例完全一致。"]
        lines += [""]
    f, y = bundles["fish2"], bundles["flyem"]
    lines += ["## 两个实例的数据结构对比", "",
        mdtable(["方面", "Fish2本例", "FlyEM本例"], [
            ["实例ID", "source_run_id__body_id__局部编号", "body_id_局部编号；另有source_mito_label"],
            ["原始实例包", "em.npy + mask.npy + metadata.json", "相同；body目录另有neuron_metadata.json"],
            ["EM / mask", f"{f['arrays']['shape_zyx']}，uint8 / uint8", f"{y['arrays']['shape_zyx']}，uint8 / uint8"],
            ["体素尺寸XYZ", "[8,8,30]nm，Z向各向异性", "[8,8,8]nm，各向同性"],
            ["mask前景体素", f["arrays"]["foreground_voxels"], y["arrays"]["foreground_voxels"]],
            ["22维手工特征", "CSV表，物理nm标尺；含神经元骨架长度", "作者原表，主要为网格标尺；骨架长度单位未确认"],
            ["同名列注意", "mito CA本例等于PC1 CA；compactness实际为凸包面积", "mito CA用神经元局部截面；compactness同样存凸包面积"],
            ["384维表示", "Fish2 Fusion50 epoch050，独立向量行", "FlyEM Fusion50固定epoch033，独立向量行"],
            ["标签/QC", "CSV中的axon；compartment_qc=ok；审核来源仍有限制", "CSV中的dendrite；作者exact-center；无同名compartment_qc"],
            ["不存储在单实例包里的内容", "手工特征、embedding、最终区室标签需连接外部表", "同样通过ID/作者精确匹配连接外部表"]]), "",
        "### 字段定义依据与未知项", "",
        "- compactness、PC几何量、表面积/截面/直径定义依据两套关联计算源代码；原始字段名不代表完全相同算法或相同单位。",
        f"- Fish2关联代码：`{root}/neuron_mito_dataset/handcraft_features/fish2_nov_reviewed_81_v1/code/mitochondria_feature_encoder.py`；最终SBI核对`{root.parent}/FlyEM_mito_Exp_worktrees/lc21_paper_features/HPC/mito_node_dists/compute_selected_morphology.py`的覆盖公式。",
        f"- FlyEM作者源码：`{root.parent}/FlyEM_mito_Exp_worktrees/lc21_paper_features/HPC/save_all_mitos/mito_main.py`、`{root.parent}/FlyEM_mito_Exp_worktrees/lc21_paper_features/HPC/save_mito_synapse/mito_synapse_main.py`及`{root.parent}/FlyEM_mito_Exp_worktrees/lc21_paper_features/util_files/voxel_utils.py`；本次只读，没有运行这些提取流程。",
        "- 未确认的缩写、索引系统、r0/r1/r2等字段以及length along skeleton单位均明确保留未知；没有用名称猜测物理量。附带sample_records.json保留完整原值及补充计算记录。",
        ""]
    (out / "report_zh.md").write_text("\n".join(lines))
    (out / "sample_records.json").write_text(json.dumps(bundles, ensure_ascii=False, indent=2) + "\n")
    deliverables = [p for p in sorted(out.iterdir()) if p.is_file() and p.suffix not in [".zip", ".log"] and p.name != "artifact_hashes.json"]
    manifest = {p.name: sha(p) for p in deliverables}
    (out / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(out / "showcase_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in deliverables + [out / "artifact_hashes.json"]:
            archive.write(p, p.name)
    print(json.dumps({d: dict(record_id=b["identity"]["record_id"], shape=b["arrays"]["shape_zyx"],
                     mask_voxels=b["arrays"]["foreground_voxels"], embedding=b["embedding"],
                     embedding_statistics=b["embedding_statistics"]) for d, b in bundles.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
