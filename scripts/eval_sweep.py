"""Run minimal segmented eval per sweep artifact and aggregate results.

Reads the sweep manifest produced by `scripts/sweep_classical_cf.py`. For
each row, calls `scripts/evaluate_baselines.py` with only the matching
classical-CF model flag (`--include-lightfm` or `--include-als`) at
`--max-users 300 --holdout-count 3 --user-sample-seed 42
--segment-by-history`. Parses each resulting JSON for aggregate NDCG@10 /
HitRate@10 plus per-segment NDCG@10 (cold, warm, regular, heavy) and
appends a row to the results CSV.

Audit trail: docs/experiments/2026-05-25_hyperparam-sweep-loo.md item 3
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

EVAL_DEFAULT_FLAGS = [
    "--max-users", "300",
    "--k", "5,10,20",
    "--holdout-count", "3",
    "--user-sample-seed", "42",
    "--segment-by-history",
]

SEGMENTS = ["cold_0_10", "warm_10_50", "regular_50_200", "heavy_200_plus"]

RESULTS_COLUMNS = [
    "slug",
    "model",
    "hyperparams_json",
    "artifact_dir",
    "aggregate_ndcg10",
    "aggregate_hit10",
    "cold_ndcg10",
    "warm_ndcg10",
    "regular_ndcg10",
    "heavy_ndcg10",
    "cold_n",
    "warm_n",
    "regular_n",
    "heavy_n",
    "run_json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Sweep manifest CSV from scripts/sweep_classical_cf.py.")
    parser.add_argument("--output", required=True, help="Results CSV path; appended to.")
    parser.add_argument("--eval-output-root", default="/private/tmp/sweep_eval", help="Per-artifact eval output dir.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sbert-faiss-index-dir", default="artifacts/indexes/sbert_faiss")
    parser.add_argument("--skip-existing", action="store_true", help="Skip manifest rows already present in the output CSV.")
    return parser.parse_args()


def existing_slugs(output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        return set()
    with output_path.open() as handle:
        reader = csv.DictReader(handle)
        return {row["slug"] for row in reader if row.get("slug")}


def append_row(output_path, row):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULTS_COLUMNS})


def run_one(python_exe, model, artifact_dir, eval_output_dir, sbert_index_dir):
    if model == "lightfm":
        include_flags = ["--include-lightfm", "--lightfm-artifacts-dir", str(artifact_dir)]
    elif model == "als":
        include_flags = ["--include-als", "--als-artifacts-dir", str(artifact_dir)]
    else:
        raise ValueError(f"unsupported model {model!r}")

    command = [
        python_exe,
        str(REPO_ROOT / "scripts" / "evaluate_baselines.py"),
        *EVAL_DEFAULT_FLAGS,
        *include_flags,
        "--output-dir", str(eval_output_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"eval subprocess failed for {artifact_dir}")
    summary_path = Path(eval_output_dir) / "metrics_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"eval subprocess for {artifact_dir} did not write {summary_path}")
    return summary_path


def parse_eval(model, summary_path):
    data = json.loads(Path(summary_path).read_text())
    model_key = "lightfm_warp" if model == "lightfm" else "als_implicit"
    block = data["top_n"][model_key]["10"]
    segments = block.get("segments", {})
    return {
        "aggregate_ndcg10": block.get("ndcg_at_k"),
        "aggregate_hit10": block.get("hit_rate_at_k"),
        "cold_ndcg10": segments.get("cold_0_10", {}).get("ndcg_at_k", 0.0),
        "warm_ndcg10": segments.get("warm_10_50", {}).get("ndcg_at_k", 0.0),
        "regular_ndcg10": segments.get("regular_50_200", {}).get("ndcg_at_k", 0.0),
        "heavy_ndcg10": segments.get("heavy_200_plus", {}).get("ndcg_at_k", 0.0),
        "cold_n": segments.get("cold_0_10", {}).get("evaluated_user_count", 0),
        "warm_n": segments.get("warm_10_50", {}).get("evaluated_user_count", 0),
        "regular_n": segments.get("regular_50_200", {}).get("evaluated_user_count", 0),
        "heavy_n": segments.get("heavy_200_plus", {}).get("evaluated_user_count", 0),
    }


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    eval_output_root = Path(args.eval_output_root)
    eval_output_root.mkdir(parents=True, exist_ok=True)

    already_done = existing_slugs(output_path) if args.skip_existing else set()

    with manifest_path.open() as handle:
        rows = list(csv.DictReader(handle))

    for index, row in enumerate(rows, start=1):
        slug = row["slug"]
        model = row["model"]
        artifact_dir = row["artifact_dir"]
        if slug in already_done:
            print(f"[{index}/{len(rows)}] SKIP {slug} (already in results)", file=sys.stderr)
            continue
        eval_output_dir = eval_output_root / f"{model}_{slug}"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(rows)}] EVAL {model} {slug}", file=sys.stderr)
        summary_path = run_one(args.python, model, artifact_dir, eval_output_dir, args.sbert_faiss_index_dir)
        parsed = parse_eval(model, summary_path)
        append_row(output_path, {
            "slug": slug,
            "model": model,
            "hyperparams_json": row.get("hyperparams_json", ""),
            "artifact_dir": artifact_dir,
            **parsed,
            "run_json": str(summary_path),
        })


if __name__ == "__main__":
    main()
