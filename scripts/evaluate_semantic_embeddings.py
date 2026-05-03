import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_access import load_movies, load_ratings, load_tags
from evaluation import popularity_recommendations, temporal_train_test_split
from evaluate_baselines import (
    build_content_recommendations,
    build_metric_report,
    filter_to_users,
    json_ready,
    parse_k_values,
    recommendation_examples,
    select_evaluation_user_ids,
)
from experimental.semantic_embeddings import build_semantic_watch_history_recommendations


def run_semantic_evaluation(
    max_users=100,
    k_values=None,
    holdout_count=1,
    min_interactions=5,
    positive_threshold=4.0,
    n_components=64,
    random_state=42,
    example_count=0,
):
    k_values = k_values or [10]
    max_k = max(k_values)

    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()
    selected_user_ids = select_evaluation_user_ids(
        ratings,
        max_users=max_users,
        min_interactions=min_interactions,
        holdout_count=holdout_count,
    )
    sampled_ratings = filter_to_users(ratings, selected_user_ids)
    train, holdout = temporal_train_test_split(
        sampled_ratings,
        holdout_count=holdout_count,
        min_interactions=min_interactions,
    )

    eval_user_ids = holdout["userId"].dropna().drop_duplicates().tolist() if "userId" in holdout.columns else []
    candidate_items = movies[["movieId"]] if "movieId" in movies.columns else None
    popularity = popularity_recommendations(
        train,
        candidate_items=candidate_items,
        user_ids=eval_user_ids,
        k=max_k,
        positive_threshold=positive_threshold,
    )
    content_recommendations = build_content_recommendations(
        train,
        movies,
        tags,
        eval_user_ids,
        max_k,
        positive_threshold=positive_threshold,
    )
    semantic_recommendations = build_semantic_watch_history_recommendations(
        train,
        movies,
        tags,
        eval_user_ids,
        top_n=max_k,
        n_components=n_components,
        random_state=random_state,
        positive_threshold=positive_threshold,
    )

    report = {
        "config": {
            "max_users": int(max_users),
            "k_values": k_values,
            "holdout_count": int(holdout_count),
            "min_interactions": int(min_interactions),
            "positive_threshold": float(positive_threshold),
            "components": int(n_components),
            "random_state": int(random_state),
            "example_count": int(example_count),
        },
        "data": {
            "ratings_rows": int(len(ratings)),
            "movies_rows": int(len(movies)),
            "tags_rows": int(len(tags)),
            "selected_user_count": int(len(selected_user_ids)),
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "evaluated_user_count": int(len(eval_user_ids)),
        },
        "top_n": {
            "popularity": build_metric_report(
                popularity,
                holdout,
                train,
                movies,
                k_values,
                score_col="score",
                positive_threshold=positive_threshold,
            ),
            "content_watch_history": build_metric_report(
                content_recommendations,
                holdout,
                train,
                movies,
                k_values,
                baseline_recommendations=popularity,
                positive_threshold=positive_threshold,
            ),
            "semantic_embeddings": build_metric_report(
                semantic_recommendations,
                holdout,
                train,
                movies,
                k_values,
                baseline_recommendations=popularity,
                score_col="similarity_score",
                positive_threshold=positive_threshold,
            ),
        },
    }

    if example_count > 0:
        report["examples"] = {
            "popularity": recommendation_examples(
                popularity,
                movies=movies,
                limit=example_count,
            ),
            "content_watch_history": recommendation_examples(
                content_recommendations,
                movies=movies,
                limit=example_count,
                include_reasons=True,
            ),
            "semantic_embeddings": recommendation_examples(
                semantic_recommendations,
                movies=movies,
                limit=example_count,
            ),
        }

    return json_ready(report)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate offline semantic embedding recommendations.")
    parser.add_argument("--max-users", type=int, default=100, help="Maximum eligible users to evaluate. Use 0 for all.")
    parser.add_argument("--k", default="10", help="Comma-separated top-N cutoffs, for example 5,10,20.")
    parser.add_argument("--holdout-count", type=int, default=1, help="Latest interactions held out per user.")
    parser.add_argument("--min-interactions", type=int, default=5, help="Minimum interactions required per user.")
    parser.add_argument("--positive-threshold", type=float, default=4.0, help="Rating threshold treated as positive.")
    parser.add_argument("--components", type=int, default=64, help="Number of latent semantic dimensions.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state used by TruncatedSVD.")
    parser.add_argument("--example-count", type=int, default=0, help="Include this many recommendation examples.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    report = run_semantic_evaluation(
        max_users=args.max_users,
        k_values=parse_k_values(args.k),
        holdout_count=args.holdout_count,
        min_interactions=args.min_interactions,
        positive_threshold=args.positive_threshold,
        n_components=args.components,
        random_state=args.random_state,
        example_count=args.example_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

