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
    args = parser.parse_args()
    features = read_csv(args.features)
    coordinates = read_csv(args.coordinates)
    if not features or features.keys() != coordinates.keys():
        raise ValueError("Expected identical nonempty instance keys")
    rows = []
    for key, source in features.items():
        target = coordinates[key]
        labels = [source[label] for label in LABELS]
        if any(source[label] != target[label] for label in LABELS):
            raise ValueError(f"Annotation mismatch for {key}")
        values = [float(target[column]) for column in COORDINATES]
        values.extend(float(source[column]) for column in FEATURES)
        if not all(map(math.isfinite, values)):
            raise ValueError(f"Nonfinite value for {key}")
        rows.append([*key, *labels, *values])
    payload = json.dumps({"features": list(FEATURES.values()), "rows": rows},
                         ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    template = Path(__file__).with_name("umap_dashboard.html").read_text(encoding="utf-8")
    args.output.write_text(template.replace("__FLYEM_DATA__", payload.replace("<", "\\u003c")),
                           encoding="utf-8")
    print(f"Wrote {len(rows):,} instances and {len(FEATURES)} metrics to {args.output}")


if __name__ == "__main__":
    main()
