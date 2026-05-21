import argparse
import json
import time

import pandas as pd

from config import EVALUATION_DEFAULTS, project_path
from data_access import load_ratings
from experimental.lightfm_recommender import (
    build_interaction_matrix,
    require_lightfm_dependency,
    save_lightfm_artifacts,
    train_lightfm_model,
)


_LIGHTFM_DEFAULTS = EVALUATION_DEFAULTS.get("lightfm") or {}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train and save a LightFM WARP model for MovieLens ratings.")
    parser.add_argument("--output-dir", default=str(project_path(_LIGHTFM_DEFAULTS.get("artifacts_dir", "artifacts/models/lightfm"))), help="Directory for LightFM artifacts.")
    parser.add_argument("--no-components", type=int, default=int(_LIGHTFM_DEFAULTS.get("no_components", 64)), help="LightFM latent component count.")
    parser.add_argument("--loss", default=_LIGHTFM_DEFAULTS.get("loss", "warp"), help="LightFM loss, for example warp or bpr.")
    parser.add_argument("--epochs", type=int, default=int(_LIGHTFM_DEFAULTS.get("epochs", 20)), help="Training epochs.")
    parser.add_argument("--positive-threshold", type=float, default=float(_LIGHTFM_DEFAULTS.get("positive_threshold", 4.0)), help="Minimum rating treated as implicit positive feedback.")
    parser.add_argument("--num-threads", type=int, default=int(_LIGHTFM_DEFAULTS.get("num_threads", 4)), help="Training thread count.")
    parser.add_argument("--exclude-holdout-pairs", default=None, help="Optional CSV path with userId,movieId columns; rows are removed from the training matrix before model fit.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_lightfm_dependency()
    ratings = load_ratings()
    exclude_pairs = None
    if args.exclude_holdout_pairs:
        excl_df = pd.read_csv(args.exclude_holdout_pairs)
        exclude_pairs = set(
            (int(row.userId), int(row.movieId))
            for row in excl_df[["userId", "movieId"]].dropna().itertuples(index=False)
        )
    interactions, user_index, item_index = build_interaction_matrix(
        ratings,
        positive_threshold=args.positive_threshold,
        exclude_pairs=exclude_pairs,
    )
    started_at = time.perf_counter()
    model = train_lightfm_model(
        interactions,
        no_components=args.no_components,
        loss=args.loss,
        epochs=args.epochs,
        num_threads=args.num_threads,
    )
    train_seconds = time.perf_counter() - started_at
    metadata = {
        "no_components": int(args.no_components),
        "loss": str(args.loss),
        "epochs": int(args.epochs),
        "num_threads": int(args.num_threads),
        "positive_threshold": float(args.positive_threshold),
        "row_count": int(interactions.nnz),
        "train_seconds": float(train_seconds),
    }
    if exclude_pairs is not None:
        metadata["excluded_pair_count"] = int(len(exclude_pairs))
        metadata["exclude_pairs_path"] = str(args.exclude_holdout_pairs)
    artifacts = save_lightfm_artifacts(
        model,
        user_index,
        item_index,
        output_dir=args.output_dir,
        metadata=metadata,
    )
    print(json.dumps(artifacts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
