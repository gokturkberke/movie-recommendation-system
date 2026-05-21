"""Hyperparameter sweep driver for LightFM and ALS classical-CF baselines.

For each combination produced by the cartesian product of `--grid` axes
(merged with `--fixed` keys), this script shells out to the existing
`scripts/train_lightfm_model.py` or `scripts/train_als_model.py` with
`--exclude-holdout-pairs` and the per-combo hyperparameters, saves each
artifact to `{output-root}/{model}_{slug}/`, and appends a manifest row.

Re-running with the same arguments is idempotent: combinations whose
output directory already contains `metadata.json` are skipped and the
manifest still records them with `skipped=true`.

Audit trail: docs/experiments/2026-05-25_hyperparam-sweep-loo.md item 1
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SLUG_PREFIX = {
    "no_components": "n",
    "loss": "l",
    "epochs": "e",
    "num_threads": "t",
    "factors": "f",
    "regularization": "r",
    "alpha": "a",
    "iterations": "i",
}

# Order params in slug for stable, readable directory names.
SLUG_ORDER = ("no_components", "loss", "epochs", "num_threads", "factors", "regularization", "alpha", "iterations")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["lightfm", "als"], help="Which train script to drive.")
    parser.add_argument("--grid", action="append", default=[], help="One occurrence per swept dimension, e.g. no-components=32,64,128 (repeatable).")
    parser.add_argument("--fixed", action="append", default=[], help="One occurrence per fixed dimension, e.g. epochs=20 (repeatable).")
    parser.add_argument("--exclude-holdout-pairs", required=True, help="Path to the LOO exclusion CSV; forwarded to every train command.")
    parser.add_argument("--output-root", required=True, help="Root directory for per-combo artifact subdirs.")
    parser.add_argument("--manifest-path", required=True, help="CSV path that lists every (combo, artifact_dir) row.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for the train subprocess.")
    return parser.parse_args()


def parse_grid_spec(grid_args, fixed_args):
    """Convert --grid / --fixed strings into the dimension -> values mapping."""
    grid = {}
    for spec in grid_args:
        key, _, values = spec.partition("=")
        key = key.strip().replace("-", "_")
        values = [v.strip() for v in values.split(",") if v.strip()]
        if not values:
            raise ValueError(f"grid axis {key!r} has no values")
        grid[key] = values
    for spec in fixed_args:
        key, _, value = spec.partition("=")
        key = key.strip().replace("-", "_")
        value = value.strip()
        if not value:
            raise ValueError(f"fixed axis {key!r} has no value")
        if key in grid:
            raise ValueError(f"axis {key!r} appears in both --grid and --fixed")
        grid[key] = [value]
    return grid


def build_slug(params):
    """Deterministic, filesystem-safe encoding of a params dict."""
    parts = []
    for key in SLUG_ORDER:
        if key not in params:
            continue
        prefix = SLUG_PREFIX[key]
        value = str(params[key])
        parts.append(f"{prefix}{value}")
    leftovers = sorted(set(params) - set(SLUG_ORDER))
    for key in leftovers:
        parts.append(f"{key}{params[key]}")
    return "_".join(parts)


def build_train_command(python_exe, model, params, exclude_holdout_pairs, output_dir):
    script = "scripts/train_lightfm_model.py" if model == "lightfm" else "scripts/train_als_model.py"
    command = [python_exe, str(REPO_ROOT / script), "--output-dir", str(output_dir), "--exclude-holdout-pairs", str(exclude_holdout_pairs)]
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        command.extend([flag, str(value)])
    return command


def iter_combinations(grid_dict):
    keys = list(grid_dict.keys())
    for values in product(*(grid_dict[key] for key in keys)):
        yield dict(zip(keys, values))


def run_combo(python_exe, model, params, exclude_holdout_pairs, output_dir, dry_run=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_train_command(python_exe, model, params, exclude_holdout_pairs, output_dir)
    if dry_run:
        return {"command": command, "stdout": ""}
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"train subprocess failed for {output_dir}: exit={result.returncode}")
    return {"command": command, "stdout": result.stdout}


def read_metadata(output_dir):
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


MANIFEST_COLUMNS = ["slug", "model", "hyperparams_json", "exclude_pairs_path", "artifact_dir", "train_seconds", "excluded_pair_count", "row_count", "skipped"]


def append_manifest_row(manifest_path, row):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})


def main():
    args = parse_args()
    grid = parse_grid_spec(args.grid, args.fixed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Sweep model={args.model}, grid={ {k: v for k, v in grid.items()} }", file=sys.stderr)

    for params in iter_combinations(grid):
        slug = build_slug(params)
        artifact_dir = output_root / f"{args.model}_{slug}"
        existing = read_metadata(artifact_dir)
        if existing is not None:
            print(f"  SKIP {artifact_dir} (metadata.json exists)", file=sys.stderr)
            append_manifest_row(
                args.manifest_path,
                {
                    "slug": slug,
                    "model": args.model,
                    "hyperparams_json": json.dumps(params, sort_keys=True),
                    "exclude_pairs_path": str(args.exclude_holdout_pairs),
                    "artifact_dir": str(artifact_dir),
                    "train_seconds": existing.get("train_seconds", ""),
                    "excluded_pair_count": existing.get("excluded_pair_count", ""),
                    "row_count": existing.get("row_count", ""),
                    "skipped": "true",
                },
            )
            continue

        print(f"  TRAIN {artifact_dir}", file=sys.stderr)
        run_combo(args.python, args.model, params, args.exclude_holdout_pairs, artifact_dir, dry_run=False)
        metadata = read_metadata(artifact_dir)
        if metadata is None:
            raise RuntimeError(f"train subprocess for {artifact_dir} did not produce metadata.json")
        append_manifest_row(
            args.manifest_path,
            {
                "slug": slug,
                "model": args.model,
                "hyperparams_json": json.dumps(params, sort_keys=True),
                "exclude_pairs_path": str(args.exclude_holdout_pairs),
                "artifact_dir": str(artifact_dir),
                "train_seconds": metadata.get("train_seconds", ""),
                "excluded_pair_count": metadata.get("excluded_pair_count", ""),
                "row_count": metadata.get("row_count", ""),
                "skipped": "false",
            },
        )


if __name__ == "__main__":
    main()
