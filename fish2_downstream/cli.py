import argparse
import contextlib
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import sklearn

from . import assets, inputs
from .checkpoint import checkpoint_job
from .common import HERE, ROOT, TASKS, code_identity, dump, new_directory, read, sha, table
from .evaluate import evaluate, plot


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def selected_tasks(args):
    if args.classifier == "lr":
        if args.task != "compartment":
            raise ValueError("LR requires explicit --task compartment; cell-type remains RF")
        return ["compartment"]
    return list(TASKS) if args.task == "all" else [args.task]


def inspect_inputs(args):
    assets.validate(args.assets)
    descriptor, matrix, mapping = inputs.load(args.embedding)
    result = dict(representation=descriptor["representation"], source_shape=descriptor["shape"], tasks={})
    for task in selected_tasks(args):
        frame = table(Path(args.assets) / f"{task}_instances.csv")
        aligned = inputs.align(matrix, mapping, frame)
        result["tasks"][task] = dict(instances=len(frame), bodies=frame.body_id.nunique(),
            dimension=aligned.shape[1], protocol=TASKS[task])
    print(__import__("json").dumps(result, indent=2, ensure_ascii=False))
    return result


def run(args):
    config = assets.validate(args.assets)
    descriptor, matrix, mapping = inputs.load(args.embedding)
    tasks = selected_tasks(args)
    manifests = {task: table(Path(args.assets) / f"{task}_instances.csv") for task in tasks}
    aligned = {task: inputs.align(matrix, mapping, frame) for task, frame in manifests.items()}
    hand = np.load(Path(args.assets) / "handcrafted_22.npy", mmap_mode="r")
    out = new_directory(args.runs_root, args.owner, args.name)
    meta = dict(owner=args.owner, name=args.name, tasks=tasks, classifier=args.classifier,
        embedding_descriptor=str(Path(args.embedding).resolve()), embedding=descriptor,
        assets=str(Path(args.assets).resolve()), asset_manifest_sha256=sha(Path(args.assets) / "assets.json"),
        code=code_identity(), python=platform.python_version(), numpy=np.__version__,
        sklearn=sklearn.__version__, argv=sys.argv, status="running", workers=args.workers,
        protocol=config["tasks"], feature_dimensions=dict(embedding=matrix.shape[1], handcrafted=22),
        classifier_parameters=(dict(n_estimators=100, min_weight_fraction_leaf=.01, max_features="sqrt",
            sample_weight="inverse training class count", scaling="none", random_state="20261113 + fold")
            if args.classifier == "rf" else dict(regularization="L2", C=1., solver="lbfgs",
            class_weight="balanced", max_iter=3000, scaling="train-only StandardScaler per type-fold")),
        handcrafted_columns=config["handcrafted_columns"], aggregation="mean log10(clip(p,1e-15,1))")
    dump(out / "run.json", meta)
    with (out / "run.log").open("w") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        try:
            all_results = {}
            for task, frame in manifests.items():
                directory = out / task
                directory.mkdir()
                frame.to_csv(directory / "instances.csv", index=False)
                features = {}
                if args.with_handcrafted:
                    features["handcrafted"] = np.asarray(hand[frame.embedding_row.to_numpy(dtype=int)], dtype=np.float64)
                features["embedding"] = aligned[task]
                results = {}
                for feature, x in features.items():
                    print(task, feature, x.shape, flush=True)
                    scores, instance, grouped, records = evaluate(x, frame, TASKS[task], args.classifier, args.workers)
                    results[feature] = scores
                    instance.to_csv(directory / f"{feature}_instance_oof.csv", index=False)
                    grouped.to_csv(directory / f"{feature}_group_oof.csv", index=False)
                    dump(directory / f"{feature}_training.json", records)
                dump(directory / "metrics.json", results)
                plot(results, task, TASKS[task], directory / "confusion.png", args.classifier, matrix.shape[1])
                all_results[task] = results
            meta["status"] = "complete"
            dump(out / "run.json", meta)
            dump(out / "metrics.json", all_results)
            dump(out / "_SUCCESS.json", dict(status="complete", owner=args.owner))
            print("COMPLETE", out, flush=True)
        except Exception:
            meta["status"] = "failed"
            dump(out / "run.json", meta)
            dump(out / "_FAILED.json", dict(traceback=traceback.format_exc()))
            traceback.print_exc(file=log)
            raise
    return out


def main():
    parser = argparse.ArgumentParser(description="Fish2 fixed downstream tasks: inspect first, run into unique outputs.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks")
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, default=ROOT / "assets/reviewed81_v1")
    p = sub.add_parser("register")
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--instance-map", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--representation", required=True)
    p.add_argument("--feature-prefix", default="embedding_")
    p.add_argument("--columns", nargs="+")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--model-config", type=Path)
    p.add_argument("--notes", default="")
    for command in ["inspect", "run"]:
        p = sub.add_parser(command)
        p.add_argument("--assets", type=Path, default=ROOT / "assets/reviewed81_v1")
        p.add_argument("--embedding", type=Path, required=True)
        p.add_argument("--task", choices=["all", "compartment", "cell_type"], default="all")
        p.add_argument("--classifier", choices=["rf", "lr"], default="rf")
        if command == "run":
            p.add_argument("--owner", required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--with-handcrafted", action="store_true")
    p = sub.add_parser("checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--export-spec", type=Path, required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--jobs-root", type=Path, default=ROOT / "exports")
    p.add_argument("--gpu")
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("runs")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if args.command == "tasks":
        print(__import__("json").dumps(dict(protocols=TASKS, history=read(HERE / "history.json")), indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        print(assets.prepare(args.output))
    elif args.command == "register":
        print(inputs.register(args.matrix, args.output, args.representation, args.instance_map,
            args.feature_prefix, args.columns, args.checkpoint, args.model_config, args.notes))
    elif args.command == "inspect":
        inspect_inputs(args)
    elif args.command == "run":
        run(args)
    elif args.command == "checkpoint":
        checkpoint_job(args)
    else:
        for path in sorted(args.runs_root.glob("*/run.json")):
            record = read(path)
            print(record["status"], record["owner"], record["name"], path.parent)


if __name__ == "__main__":
    main()
