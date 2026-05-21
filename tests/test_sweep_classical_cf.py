import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = REPO_ROOT / "scripts" / "sweep_classical_cf.py"

spec = importlib.util.spec_from_file_location("sweep_classical_cf", SWEEP_PATH)
sweep_classical_cf = importlib.util.module_from_spec(spec)
sys.modules.setdefault("sweep_classical_cf", sweep_classical_cf)
spec.loader.exec_module(sweep_classical_cf)


class TestSlugBuilder(unittest.TestCase):
    def test_slug_orders_lightfm_keys_with_prefixes(self):
        slug = sweep_classical_cf.build_slug({"no_components": 128, "loss": "warp", "epochs": 20})
        self.assertEqual(slug, "n128_lwarp_e20")

    def test_slug_orders_als_keys_with_prefixes(self):
        slug = sweep_classical_cf.build_slug({"factors": 64, "regularization": 0.01, "alpha": 40.0, "iterations": 20})
        self.assertEqual(slug, "f64_r0.01_a40.0_i20")


class TestGridSpecParser(unittest.TestCase):
    def test_grid_and_fixed_combine_into_single_mapping(self):
        grid = sweep_classical_cf.parse_grid_spec(
            grid_args=["no-components=32,64", "loss=warp,bpr"],
            fixed_args=["epochs=20"],
        )
        self.assertEqual(grid["no_components"], ["32", "64"])
        self.assertEqual(grid["loss"], ["warp", "bpr"])
        self.assertEqual(grid["epochs"], ["20"])

    def test_duplicate_axis_in_grid_and_fixed_raises(self):
        with self.assertRaises(ValueError):
            sweep_classical_cf.parse_grid_spec(
                grid_args=["loss=warp,bpr"],
                fixed_args=["loss=warp"],
            )


class TestManifestSkipsExistingDir(unittest.TestCase):
    def test_skip_writes_row_with_skipped_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output_root = tmp / "sweeps"
            artifact_dir = output_root / "lightfm_n32_lwarp_e20"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "metadata.json").write_text(json.dumps({
                "row_count": 12345,
                "excluded_pair_count": 1555,
                "train_seconds": 10.0,
            }))

            manifest_path = tmp / "manifest.csv"
            sweep_classical_cf.append_manifest_row(manifest_path, {
                "slug": "n32_lwarp_e20",
                "model": "lightfm",
                "hyperparams_json": json.dumps({"no_components": "32", "loss": "warp", "epochs": "20"}, sort_keys=True),
                "exclude_pairs_path": "/dev/null",
                "artifact_dir": str(artifact_dir),
                "train_seconds": "10.0",
                "excluded_pair_count": "1555",
                "row_count": "12345",
                "skipped": "true",
            })

            with manifest_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["skipped"], "true")
            self.assertEqual(rows[0]["row_count"], "12345")
            self.assertEqual(rows[0]["excluded_pair_count"], "1555")


if __name__ == "__main__":
    unittest.main()
