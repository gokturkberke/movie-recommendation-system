"""Extract the union of evaluation holdout (userId, movieId) pairs across seeds.

Mirrors `run_evaluation`'s user selection and temporal split so the produced
CSV matches exactly what the canonical eval would hold out. The CSV is the
input to `train_lightfm_model.py --exclude-holdout-pairs` and
`train_als_model.py --exclude-holdout-pairs` for leave-one-out retraining.

Audit trail: docs/experiments/2026-05-24_leave-one-out-leakage-fix.md item 2
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

from config import EVALUATION_DEFAULTS
from data_access import load_ratings
from evaluation import temporal_train_test_split
from evaluation_runner import filter_to_users, select_evaluation_user_ids


def parse_args():
    defaults = EVALUATION_DEFAULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-sample-seeds", default="42,7,1337", help="Comma-separated list of --user-sample-seed values to union.")
    parser.add_argument("--max-users", type=int, default=300)
    parser.add_argument("--holdout-count", type=int, default=3)
    parser.add_argument("--min-interactions", type=int, default=int(defaults.get("min_interactions", 5)))
    parser.add_argument("--positive-threshold", type=float, default=float(defaults.get("positive_threshold", 4.0)))
    parser.add_argument("--output-path", required=True, help="Destination CSV path; written with userId,movieId columns.")
    return parser.parse_args()


def collect_holdout_pairs(ratings, seeds, max_users, holdout_count, min_interactions, positive_threshold):
    pairs = set()
    per_seed_counts = []
    for seed in seeds:
        selected_user_ids = select_evaluation_user_ids(
            ratings,
            max_users=max_users,
            min_interactions=min_interactions,
            holdout_count=holdout_count,
            random_seed=seed,
        )
        sampled = filter_to_users(ratings, selected_user_ids)
        _, holdout = temporal_train_test_split(
            sampled,
            holdout_count=holdout_count,
            min_interactions=min_interactions,
        )
        positive = holdout[pd.to_numeric(holdout["rating"], errors="coerce") >= positive_threshold]
        seed_pairs = {
            (int(row.userId), int(row.movieId))
            for row in positive[["userId", "movieId"]].dropna().itertuples(index=False)
        }
        pairs.update(seed_pairs)
        per_seed_counts.append((seed, len(seed_pairs)))
    return pairs, per_seed_counts


def main():
    args = parse_args()
    seeds = [int(value.strip()) for value in args.user_sample_seeds.split(",") if value.strip()]

    print(f"Loading ratings...", file=sys.stderr)
    ratings = load_ratings()

    print(f"Extracting holdouts for seeds {seeds} at max_users={args.max_users}, holdout_count={args.holdout_count}, positive_threshold={args.positive_threshold}", file=sys.stderr)
    pairs, per_seed_counts = collect_holdout_pairs(
        ratings,
        seeds,
        max_users=args.max_users,
        holdout_count=args.holdout_count,
        min_interactions=args.min_interactions,
        positive_threshold=args.positive_threshold,
    )

    for seed, count in per_seed_counts:
        print(f"  seed {seed}: {count} positive-holdout pairs", file=sys.stderr)
    print(f"Union size after dedup: {len(pairs)}", file=sys.stderr)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(sorted(pairs), columns=["userId", "movieId"])
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
