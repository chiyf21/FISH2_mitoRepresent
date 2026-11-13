"""Build the standalone FlyEM morphology dashboard from matched CSVs."""
import argparse
import csv
import json
import math
from pathlib import Path


FEATURES = {
    "size": ("体积", "作者发布的线粒体体积，单位为体素。"),
    "mito SA": ("表面积", "作者的线粒体表面积估计，单位为体素平方。"),
    "mito CA": ("线粒体截面积", "按局部神经元形态确定切面的线粒体截面积，单位为体素平方。"),
    "PC1 Length": ("第一主轴长度", "沿线粒体第一主轴的投影跨度，单位为体素。"),
    "PC1 CA": ("第一主轴截面积", "垂直于线粒体第一主轴的截面积，单位为体素平方。"),
    "mito diameter": ("直径", "作者计算的线粒体直径估计，单位为体素。"),
    "length along skeleton": ("沿骨架长度", "作者计算的沿神经元骨架长度，单位为体素，包含原方法的 PC1 回退值。"),
    "SBI": ("SBI 球形度指标", "等体积球表面积与线粒体表面积之比的 log₁₀，无量纲。"),
}
EM_FEATURES = {
    "em_mean": ("EM 内部平均灰度", "线粒体 mask 内的原始 8 位 EM 平均灰度。"),
    "em_std": ("EM 内部灰度标准差", "线粒体 mask 内灰度的标准差，单位为灰度级。"),
    "em_shell_mean": ("EM 周围平均灰度", "mask 外两步六邻接壳层的平均灰度，受原始裁块边界限制。"),
    "em_shell_minus_inside": ("EM 内外灰度对比", "外部壳层均值减去内部均值。正值表示线粒体内部更暗。"),
    "em_neighbor_abs_diff": ("EM 内部相邻灰度差", "mask 内相邻体素绝对灰度差的均值，反映局部纹理起伏。"),
}
KEYS = ("body_id", "record_id")
LABELS = ("compartment", "cell_type")
COORDINATES = ("author22_x", "author22_y", "fused_x", "fused_y")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in KEYS)
            if key in rows or not all(key):
                raise ValueError(f"Duplicate or empty instance key in {path}: {key}")
            rows[key] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("coordinates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--em-features", type=Path)
    args = parser.parse_args()
    features = read_csv(args.features)
    coordinates = read_csv(args.coordinates)
    if not features or features.keys() != coordinates.keys():
        raise ValueError("Expected identical nonempty instance keys")
    em = read_csv(args.em_features) if args.em_features else None
    if em is not None and em.keys() != features.keys():
        raise ValueError("Expected identical EM and morphology instance keys")
    metrics = FEATURES if em is None else {**FEATURES, **EM_FEATURES}
    rows = []
    for key, source in features.items():
        target = coordinates[key]
        labels = [source[label] for label in LABELS]
        if any(source[label] != target[label] for label in LABELS):
            raise ValueError(f"Annotation mismatch for {key}")
        values = [float(target[column]) for column in COORDINATES]
        values.extend(float(source[column]) for column in FEATURES)
        if em is not None:
            values.extend(float(em[key][column]) for column in EM_FEATURES)
        if not all(map(math.isfinite, values)):
            raise ValueError(f"Nonfinite value for {key}")
        rows.append([*key, *labels, *values])
    payload = json.dumps({"features": list(metrics.values()), "rows": rows},
                         ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    template = Path(__file__).with_name("umap_dashboard.html").read_text(encoding="utf-8")
    args.output.write_text(template.replace("__FLYEM_DATA__", payload.replace("<", "\\u003c")),
                           encoding="utf-8")
    print(f"Wrote {len(rows):,} instances and {len(metrics)} metrics to {args.output}")


if __name__ == "__main__":
    main()
