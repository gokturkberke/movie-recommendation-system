import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

from config import EVALUATION_DEFAULTS, project_path
from data_access import load_ratings
from experimental.als_recommender import (
    build_confidence_matrix,
    require_implicit_dependency,
    save_als_artifacts,
    train_als_model,
)


_ALS_DEFAULTS = EVALUATION_DEFAULTS.get("als") or {}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train and save an Implicit ALS model for MovieLens ratings.")
    parser.add_argument("--output-dir", default=str(project_path(_ALS_DEFAULTS.get("artifacts_dir", "artifacts/models/als"))), help="Directory for ALS artifacts.")
    parser.add_argument("--factors", type=int, default=int(_ALS_DEFAULTS.get("factors", 64)), help="ALS latent factor count.")
    parser.add_argument("--regularization", type=float, default=float(_ALS_DEFAULTS.get("regularization", 0.01)), help="ALS regularization value.")
    parser.add_argument("--iterations", type=int, default=int(_ALS_DEFAULTS.get("iterations", 20)), help="ALS training iterations.")
    parser.add_argument("--alpha", type=float, default=float(_ALS_DEFAULTS.get("alpha", 40.0)), help="Implicit confidence scaling factor.")
    parser.add_argument("--positive-threshold", type=float, default=float(_ALS_DEFAULTS.get("positive_threshold", 4.0)), help="Minimum rating treated as implicit positive feedback.")
    parser.add_argument("--use-gpu", action="store_true", default=bool(_ALS_DEFAULTS.get("use_gpu", False)), help="Use implicit GPU backend when available.")
    parser.add_argument("--exclude-holdout-pairs", default=None, help="Optional CSV path with userId,movieId columns; rows are removed from the training matrix before model fit.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_implicit_dependency()
    ratings = load_ratings()
    exclude_pairs = None
    if args.exclude_holdout_pairs:
        excl_df = pd.read_csv(args.exclude_holdout_pairs)
        exclude_pairs = set(
            (int(row.userId), int(row.movieId))
            for row in excl_df[["userId", "movieId"]].dropna().itertuples(index=False)
        )
    user_items, user_index, item_index = build_confidence_matrix(
        ratings,
        positive_threshold=args.positive_threshold,
        alpha=args.alpha,
        exclude_pairs=exclude_pairs,
    )
    started_at = time.perf_counter()
    model = train_als_model(
        user_items,
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        use_gpu=args.use_gpu,
    )
    train_seconds = time.perf_counter() - started_at
    metadata = {
        "factors": int(args.factors),
        "regularization": float(args.regularization),
        "iterations": int(args.iterations),
        "alpha": float(args.alpha),
        "positive_threshold": float(args.positive_threshold),
        "use_gpu": bool(args.use_gpu),
        "row_count": int(user_items.nnz),
        "train_seconds": float(train_seconds),
    }
    if exclude_pairs is not None:
        metadata["excluded_pair_count"] = int(len(exclude_pairs))
        metadata["exclude_pairs_path"] = str(args.exclude_holdout_pairs)
    artifacts = save_als_artifacts(
        model,
        user_index,
        item_index,
        user_items,
        output_dir=args.output_dir,
        metadata=metadata,
    )
    print(json.dumps(artifacts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
