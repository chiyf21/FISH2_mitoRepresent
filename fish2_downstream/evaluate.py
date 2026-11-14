"""Same by-type compartment / pooled cell-type classifiers, with explicit grouping."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import SEED


def metrics(y, prediction, classes):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y, prediction, labels=labels)
    precision, recall, _, _ = precision_recall_fscore_support(y, prediction, labels=labels, zero_division=0)
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        macro_f1=float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        precision=dict(zip(classes, precision.tolist())), recall=dict(zip(classes, recall.tolist())),
        confusion=cm.tolist(), confusion_normalized=(cm / cm.sum(1, keepdims=True)).tolist())


def evaluate(x, frame, config, classifier, workers):
    y = frame.label.to_numpy(dtype=int)
    classes = config["classes"]
    probability = np.full((len(frame), len(classes)), np.nan)
    seen = np.zeros(len(frame), dtype=int)
    records = []
    if classifier == "lr" and config["label"] != "compartment":
        raise ValueError("Only the historical cell-type RF protocol is registered")
    for fold in range(5):
        scopes = sorted(frame.cell_type.unique()) if config["scope"] == "by_type" else [None]
        for scope in scopes:
            domain = frame.cell_type.eq(scope).to_numpy() if scope else np.ones(len(frame), dtype=bool)
            train = np.flatnonzero(domain & frame.fold.ne(fold).to_numpy())
            test = np.flatnonzero(domain & frame.fold.eq(fold).to_numpy())
            if not len(train) or not len(test):
                raise ValueError("Empty type-fold")
            if set(frame.body_id.iloc[train]) & set(frame.body_id.iloc[test]):
                raise ValueError("Body leakage")
            counts = np.bincount(y[train], minlength=len(classes))
            if np.any(counts == 0):
                raise ValueError("Registered task unexpectedly lacks a training class")
            if classifier == "rf":
                model = RandomForestClassifier(n_estimators=100, min_weight_fraction_leaf=.01,
                    max_features="sqrt", class_weight=None, random_state=SEED + fold, n_jobs=workers)
                model.fit(x[train], y[train], sample_weight=1.0 / counts[y[train]])
                learned_classes = model.classes_
                n_iter = None
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1., solver="lbfgs", class_weight="balanced", max_iter=3000,
                    random_state=SEED + fold))
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(x[train], y[train])
                learned_classes = model.named_steps["logisticregression"].classes_
                n_iter = model.named_steps["logisticregression"].n_iter_.tolist()
            probability[np.ix_(test, learned_classes.astype(int))] = model.predict_proba(x[test])
            seen[test] += 1
            records.append(dict(fold=fold, scope=scope or "all_types", train_instances=len(train),
                test_instances=len(test), train_bodies=frame.body_id.iloc[train].nunique(),
                test_bodies=frame.body_id.iloc[test].nunique(), class_counts=counts.tolist(), n_iter=n_iter))
        print(f"{config['label']} {classifier}: fold {fold + 1}/5", flush=True)
    if not np.all(seen == 1) or not np.isfinite(probability).all():
        raise ValueError("OOF coverage failure")
    np.testing.assert_allclose(probability.sum(1), 1)
    instance = frame.copy()
    for c, name in enumerate(classes):
        instance["p_" + name] = probability[:, c]
    instance["prediction"] = probability.argmax(1)
    group_rows = []
    for _, idx in frame.groupby(config["group"], sort=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        g = frame.iloc[idx]
        if g.label.nunique() != 1 or g.fold.nunique() != 1:
            raise ValueError("Aggregation group spans labels or models")
        mean = np.log10(np.clip(probability[idx], 1e-15, 1)).mean(0)
        group_rows.append(dict(body_id=g.body_id.iloc[0], cell_type=g.cell_type.iloc[0],
            true_class=classes[int(g.label.iloc[0])], label=int(g.label.iloc[0]),
            fold=int(g.fold.iloc[0]), n_mito=len(idx), prediction=int(mean.argmax())))
    grouped = pd.DataFrame(group_rows)
    if len(grouped) != config["groups"]:
        raise ValueError("Unexpected number of scored groups")
    scores = dict(per_mito=metrics(y, probability.argmax(1), classes),
                  per_group=metrics(grouped.label, grouped.prediction, classes))
    return scores, instance, grouped, records


def plot(results, task, config, path, classifier, dimension):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    features = list(results)
    fig, axes = plt.subplots(2, len(features), figsize=(6 * len(features), 12), squeeze=False)
    fig.subplots_adjust(left=.12, right=.97, bottom=.11, top=.84, wspace=.4, hspace=.55)
    for row, level in enumerate(["per_mito", "per_group"]):
        for col, feature in enumerate(features):
            ax = axes[row, col]
            r = results[feature][level]
            cm, norm = np.array(r["confusion"]), np.array(r["confusion_normalized"])
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, f"{cm[i,j]}\n{norm[i,j]:.1%}", ha="center", va="center",
                            fontsize=10 if len(cm) > 2 else 13,
                            color="white" if norm[i,j] > .55 else "black")
            ticks = list(range(len(config["classes"])))
            ax.set_xticks(ticks, config["classes"])
            ax.set_yticks(ticks, config["classes"])
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(f"{level} | {feature} ({22 if feature == 'handcrafted' else dimension}D)\nBA={r['balanced_accuracy']:.3f} | Macro-F1={r['macro_f1']:.3f}")
    aggregation = "body + true compartment (oracle grouping)" if task == "compartment" else "one vote per body"
    fig.suptitle(f"Fish2 | {task} | {classifier.upper()} | body-disjoint 5-fold OOF\n{aggregation}", fontsize=15)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
