"""Read completed LR results and make a table/figure; no fitting or inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

BASE = Path("/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream")
SOURCES = [
    BASE / "runs/20261115T015720Z_chiyf21_nohead50_maincov0004_L12_balanced1152_c9cd0415",
    BASE / "runs/20261115T015158Z_chiyf21_fusion50_L12_balanced1152_nohead50_comparison_4edf26ae",
    BASE / "runs/20261114T124741Z_codex_benchmark_celltype_epoch050_fused_lr_e3485d15",
]
NAMES = ["No Projection Head (Balanced1152D)",
         "With Projection Head — Balanced1152D",
         "With Projection Head — Fused384D"]
COLUMNS = [
    "Cell Type — Per Mito BA", "Cell Type — Per Mito Macro-F1",
    "Cell Type — Per Neuron BA", "Cell Type — Per Neuron Macro-F1",
    "Compartment — Per Mito BA", "Compartment — Per Mito Macro-F1",
    "Compartment — Per Compartment BA", "Compartment — Per Compartment Macro-F1",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(values, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.subplots_adjust(left=.065, right=.99, bottom=.23, top=.79)
    fig.suptitle("Fish2 fixed-LR comparison", fontsize=21, x=.065, ha="left", y=.98)
    fig.text(.065, .92, "Reviewed81 · five body-disjoint folds", color="#555555", fontsize=12)
    x = np.arange(8)
    width = .245
    colors = ["#8393A7", "#2676B8", "#E69F3A"]
    labels = ["No head: Balanced1152D*", "Head: Balanced1152D", "Head: Fused384D"]
    best = np.nanmax(values, axis=0)
    for i in range(3):
        positions = x + (i-1)*width
        heights = values[i]*100
        bars = ax.bar(positions, heights, width*.92, color=colors[i],
                      label=labels[i], zorder=3)
        for j, (rect, value) in enumerate(zip(bars, heights)):
            if np.isfinite(value):
                close = any(np.isfinite(values[k,j]) and abs(values[k,j]*100-value)<3
                            for k in range(3) if k != i)
                label_y = value+1.3+(3.0 if i == 1 and close else 0)
                ax.text(rect.get_x()+rect.get_width()/2, label_y, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=10,
                        weight="bold" if values[i,j] == best[j] else "normal")
            else:
                ax.text(positions[j], 3, "N/A", color=colors[i], ha="center",
                        va="bottom", fontsize=10, weight="bold")
    ax.set_ylim(0, 104)
    ax.set_xlim(-.6, 7.6)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Score (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E7EC", linewidth=.8)
    ax.axvline(3.5, color="#CBD2DA", linestyle="--", linewidth=1)
    ax.set_xticks(x, ["Mito\nBA", "Mito\nMacro-F1", "Neuron\nBA", "Neuron\nMacro-F1",
                      "Mito\nBA", "Mito\nMacro-F1", "Compartment\nBA", "Compartment\nMacro-F1"])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.text(1.5, 1.035, "Cell Type", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=14, weight="bold")
    ax.text(5.5, 1.035, "Compartment", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=14, weight="bold")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(.99,.9),
               ncol=3, frameon=False, fontsize=12)
    fig.text(.065, .09, "* No head also includes main covariance calibration (0.2 to 0.0004).",
             fontsize=11, color="#444444")
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(output/f"fish2_lr_threeway.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fused-compartment-run", type=Path, required=True)
    args = parser.parse_args()
    runs = [read(p/"run.json") for p in SOURCES]
    results = [read(p/"metrics.json") for p in SOURCES]
    values = np.full((3,8), np.nan)
    identities = []
    for i, (path, run, scores) in enumerate(zip(SOURCES, runs, results)):
        assert (path/"_SUCCESS.json").is_file() and run["status"] == "complete"
        assert run["classifier"] == "lr"
        for field in ["asset_manifest_sha256", "classifier_parameters", "aggregation"]:
            assert run[field] == runs[0][field], field
        assert run["feature_dimensions"]["embedding"] == [1152,1152,384][i]
        for task, offset in [("cell_type",0), ("compartment",4)]:
            if task not in scores:
                assert i == 2 and task == "compartment"
                continue
            assert sha(path/task/"instances.csv") == sha(Path(run["assets"])/f"{task}_instances.csv")
            for level, k in [("per_mito",0), ("per_group",2)]:
                value = scores[task]["embedding"][level]
                values[i,offset+k] = value["balanced_accuracy"]
                values[i,offset+k+1] = value["macro_f1"]
        identities.append(dict(representation=NAMES[i],run=str(path),metrics=str(path/"metrics.json"),
            metrics_sha256=sha(path/"metrics.json"),metadata_sha256=sha(path/"run.json"),
            tasks=list(scores),checkpoint=run["embedding"]["checkpoint"],
            embedding=run["embedding"]["representation"]))
    assert np.isfinite(values[:2]).all() and np.isfinite(values[2,:4]).all()
    path = args.fused_compartment_run.resolve()
    run, scores = read(path/"run.json"), read(path/"metrics.json")
    assert (path/"_SUCCESS.json").is_file() and run["status"] == "complete"
    assert run["tasks"] == ["compartment"] and run["classifier"] == "lr"
    assert run["feature_dimensions"]["embedding"] == 384
    assert run["embedding"]["sha256"] == runs[2]["embedding"]["sha256"]
    assert run["embedding"]["checkpoint"] == runs[2]["embedding"]["checkpoint"]
    for field in ["asset_manifest_sha256", "classifier_parameters", "aggregation"]:
        assert run[field] == runs[0][field], field
    assert sha(path/"compartment/instances.csv") == sha(
        Path(run["assets"])/"compartment_instances.csv")
    for level, offset, count in [("per_mito",4,5059), ("per_group",6,161)]:
        score = scores["compartment"]["embedding"][level]
        assert sum(sum(row) for row in score["confusion"]) == count
        values[2,offset] = score["balanced_accuracy"]
        values[2,offset+1] = score["macro_f1"]
    identities.append(dict(representation=NAMES[2]+" (Compartment)",run=str(path),
        metrics=str(path/"metrics.json"),metrics_sha256=sha(path/"metrics.json"),
        metadata_sha256=sha(path/"run.json"),tasks=["compartment"],
        checkpoint=run["embedding"]["checkpoint"],embedding=run["embedding"]["representation"]))
    previous = read(BASE/"reports/nohead_head_threeway_lr_20261115_01/sources.json")
    old_values = np.array([[np.nan if v is None else v for v in row]
                           for row in previous["raw_scores"]])
    known = np.isfinite(old_values)
    assert known.sum() == 20
    np.testing.assert_array_equal(values[known], old_values[known])
    assert np.isfinite(values).all()
    args.output.mkdir(parents=True, exist_ok=False)
    formatted = [[f"{v*100:.2f}%" if np.isfinite(v) else "N/A" for v in row] for row in values]
    with (args.output/"fish2_lr_threeway.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f);writer.writerow(["Representation"]+COLUMNS)
        writer.writerows([[name]+row for name,row in zip(NAMES,formatted)])
    maxima=np.nanmax(values,axis=0)
    lines=["| Representation | "+" | ".join(COLUMNS)+" |",
           "|---|"+"|".join(["---:"]*8)+"|"]
    for i,(name,row) in enumerate(zip(NAMES,formatted)):
        cells=[f"**{v}**" if np.isfinite(values[i,j]) and values[i,j]==maxima[j] else v
               for j,v in enumerate(row)]
        lines.append("| "+name+" | "+" | ".join(cells)+" |")
    lines += ["", "粗体为三种表征正式固定 LR 结果中的最高值。Fused384D 的四项 Compartment 指标已按相同协议补齐，其他 20 项原始指标保持不变。", "",
              "No Projection Head 同时包含主 Covariance 系数校准（0.2 → 0.0004），CLS Covariance 保持 0.05。", "",
              "来源（113，各指标直接读取对应正式 metrics.json；本次仅补跑 Fused384D Compartment 固定 LR）：", ""]
    lines += [f"- {identity['representation']}: "+chr(96)+identity["metrics"]+chr(96) for identity in identities]
    (args.output/"fish2_lr_threeway.md").write_text("\n".join(lines)+"\n")
    payload=dict(schema="fish2.threeway-summary.v1", columns=COLUMNS, representations=NAMES,
        raw_scores=[[float(v) if np.isfinite(v) else None for v in row] for row in values],
        units="fraction in raw_scores; percent with two decimals in CSV/Markdown",
        sources=identities, protocol_asset_sha256=runs[0]["asset_manifest_sha256"],
        missing=None, filled_metrics=4, unchanged_existing_metrics=20,
        no_fitting=True,no_encoder_inference=True,
        code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    (args.output/"sources.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    plot(values,args.output)
    with (args.output/"fish2_lr_threeway.csv").open(encoding="utf-8-sig") as f:
        csv_rows=list(csv.reader(f))
    assert len(csv_rows)==4 and csv_rows[0]==["Representation"]+COLUMNS
    assert all(len(r)==9 for r in csv_rows)
    assert all(value.endswith("%") for row in csv_rows[1:] for value in row[1:])
    for i,row in enumerate(csv_rows[1:]):
        for j,s in enumerate(row[1:]):
            if s!="N/A":
                assert abs(float(s[:-1])/100-values[i,j]) <= .000050001
    print("\n".join(lines))
    print("OUTPUT",args.output)


if __name__=="__main__":
    main()
