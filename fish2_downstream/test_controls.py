"""Identity and grouping controls that must fail before any training starts."""
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd

from fish2_downstream.inputs import register, load, align
from fish2_downstream.common import new_directory, dump, read
from fish2_downstream.checkpoint import checkpoint_job
from fish2_downstream.assets import folds
from fish2_downstream.evaluate import evaluate
from fish2_downstream.cli import selected_tasks
from sklearn.preprocessing import StandardScaler


class InputsTest(unittest.TestCase):
    def test_id_join_survives_row_shuffle(self):
        x = np.array([[2., 20.], [1., 10.]])
        mapping = pd.DataFrame(dict(record_id=["b", "a"], body_id=["20", "10"], embedding_row=[0, 1]))
        manifest = pd.DataFrame(dict(record_id=["a", "b"], body_id=["10", "20"]))
        np.testing.assert_array_equal(align(x, mapping, manifest), [[1., 10.], [2., 20.]])
        with self.assertRaises(ValueError):
            align(x, mapping, pd.DataFrame(dict(record_id=["missing"], body_id=["10"])))
        with self.assertRaises(ValueError):
            align(x, mapping, pd.DataFrame(dict(record_id=["a"], body_id=["wrong"])))

    def test_metadata_columns_and_duplicate_instances_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            p = d / "bad.csv"
            pd.DataFrame(dict(record_id=["x", "x"], embedding_0=[1, 2])).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                register(p, d / "embedding.json", "test")
            pd.DataFrame(dict(record_id=["x"], body_id=["10"], embedding_0=[1])).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                register(p, d / "embedding.json", "test", columns=["body_id"])
            register(p, d / "embedding.json", "test")
            _, x, m = load(d / "embedding.json")
            self.assertEqual(x.shape, (1, 1))

    def test_body_cannot_have_multiple_type_labels(self):
        frame = pd.DataFrame(dict(body_id=["1", "1"], cell_type=["CON", "GC"]))
        with self.assertRaises(ValueError):
            folds(frame)

    def test_run_paths_unique_and_do_not_reuse_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            a = new_directory(directory, "person_a", "same")
            b = new_directory(directory, "person_b", "same")
            self.assertNotEqual(a, b)
            with self.assertRaises(ValueError):
                new_directory(directory, "../other", "bad")

    def test_checkpoint_plan_never_executes_export_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            checkpoint = d / "dummy.pt"
            checkpoint.write_bytes(b"not a loaded model")
            spec = d / "exporter.json"
            dump(spec, dict(cwd=str(d), argv=["python", "export.py", "{checkpoint}", "{export_dir}"],
                matrix="embedding.npy", instance_map="instance_map.csv", representation="test", dimension=2))
            args = Namespace(export_spec=spec, checkpoint=checkpoint, jobs_root=d / "jobs",
                             owner="test", name="plan", gpu=None, execute=False)
            with patch("fish2_downstream.checkpoint.code_identity", return_value={"commit": "test"}), patch("fish2_downstream.checkpoint.free_gpu", side_effect=AssertionError("GPU queried")), \
                 patch("fish2_downstream.checkpoint.subprocess.run", side_effect=AssertionError("Exporter executed")):
                checkpoint_job(args)
            plan = read(next((d / "jobs").glob("*/plan.json")))
            self.assertFalse(plan["executed"])
            self.assertEqual(plan["argv"][2], str(checkpoint))

    def test_cell_type_lr_routes_without_type_conditioning(self):
        self.assertEqual(selected_tasks(Namespace(task="cell_type", classifier="lr")), ["cell_type"])
        self.assertEqual(selected_tasks(Namespace(task="all", classifier="lr")), ["compartment", "cell_type"])

    def test_cell_type_lr_pools_classes_and_fits_scaler_on_training_only(self):
        classes = ["CON", "GC", "MON1", "MON2", "PC", "VMN"]
        rows = [dict(record_id=f"{k}-{fold}-{j}", body_id=f"{k}-{fold}", cell_type=label,
                     label=k, fold=fold) for fold in range(5)
                for k, label in enumerate(classes) for j in range(2)]
        frame = pd.DataFrame(rows)
        x = np.random.default_rng(5).normal(size=(len(frame), 4))
        config = dict(label="cell_type", classes=classes, scope="pooled", group=["body_id"], groups=30)
        seen = []
        original_fit = StandardScaler.fit
        def capture(scaler, values, y=None, **kwargs):
            seen.append(values.copy())
            return original_fit(scaler, values, y=y, **kwargs)
        with patch.object(StandardScaler, "fit", capture):
            scores, instance, groups, records = evaluate(x, frame, config, "lr", 1)
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(records), 5)
        self.assertTrue(all(row["scope"] == "all_types" for row in records))
        for fold in range(5):
            np.testing.assert_array_equal(seen[fold], x[frame.fold.ne(fold)])
        self.assertEqual(len(instance), 60)
        self.assertEqual(len(groups), 30)
        with self.assertRaises(ValueError):
            evaluate(x, frame, dict(config, scope="by_type"), "lr", 1)


if __name__ == "__main__":
    unittest.main()
