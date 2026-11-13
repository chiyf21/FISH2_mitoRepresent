"""Package reports and reproducibility metadata, excluding arrays and fitted weights."""
import argparse
import datetime
import json
import subprocess
import zipfile
from pathlib import Path

from experiment import dump, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["fish2", "flyem"])
    args = ap.parse_args()
    root = args.output.resolve()
    report = root / "report"
    validation = json.loads((report / "validation.json").read_text())
    assert validation["status"] == "PASS" and validation["fitted_models_verified"] == 30 * len(args.datasets)
    for dataset in args.datasets:
        for feature in ["handcrafted", "embedding", "dummy"]:
            assert (root / dataset / feature / "complete.json").exists()
    work = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.check_output(["git", *a], cwd=work).decode().strip()
    records = {p.name: json.loads(p.read_text()) for p in sorted(root.glob("*process*.json"))}
    manifest = dict(
        experiment="Compartment classification / handcrafted22 x learned384",
        classes={d: json.loads((root / d / "data_provenance.json").read_text())["classes"] for d in args.datasets},
        output=str(root), worktree=str(work), branch=git("branch", "--show-current"),
        code_commit=git("rev-parse", "HEAD"), origin=git("remote", "get-url", "origin"),
        finished_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        protocol="retained body-disjoint independent holdout; development nested 5x3 group CV",
        checkpoints={d: json.loads((root / d / "data_provenance.json").read_text())["checkpoint"] for d in args.datasets},
        source_hashes={str(p.relative_to(work)): sha(p) for p in sorted((work / "compartment_classification").glob("*.py"))},
        process_records=records, validation=validation,
        embedding_sources={d: json.loads((root / d / "data_provenance.json").read_text()).get("embedding_directory", str(root / "flyem_embeddings")) for d in args.datasets},
    )
    dump(root / "run_manifest.json", manifest)
    files = list(report.rglob("*"))
    files += [root / "run_manifest.json"]
    files += list(root.glob("*.log")) + list(root.glob("*process*.json"))
    files += list(root.glob("*resume*.json")) + list(root.glob("gpu_exclusive_guard*.json"))
    for dataset in args.datasets:
        files += [root / dataset / n for n in ["instances.csv", "splits.json", "data_provenance.json"]]
        if (root / dataset / "class_coverage.csv").exists():
            files.append(root / dataset / "class_coverage.csv")
        files += list((root / dataset).glob("*/fit_provenance.json"))
        files += list((root / dataset).glob("*/*/predictions.csv"))
        files += list((root / dataset).glob("*/complete.json"))
    for sub in ["flyem_smoke", "flyem_embeddings"]:
        files += [p for p in (root / sub).glob("*.json")]
        files += [root / sub / "instance_map.csv"]
    files = sorted({p for p in files if p.is_file()})
    dump(report / "artifact_hashes.json", {str(p.relative_to(root)): sha(p) for p in files if p.name != "artifact_hashes.json"})
    archive = root / "results_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(root))
    print(json.dumps(dict(bundle=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
                          files=len(files), code_commit=manifest["code_commit"]), indent=2))


if __name__ == "__main__":
    main()
