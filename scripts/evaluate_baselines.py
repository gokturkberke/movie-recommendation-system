import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_access import load_movies, load_ratings, load_surprise_model, load_tags
from evaluation import (
    popularity_recommendations,
    rating_prediction_metrics,
    temporal_train_test_split,
    top_n_metrics,
)
from recommenders import (
    build_movie_stats,
    build_tfidf_matrix,
    explain_hybrid_recommendation,
    hybrid_signal_contributions,
    recommend_based_on_watch_history_content,
)


def parse_k_values(raw_value):
    values = []
    for value in str(raw_value).split(","):
        value = value.strip()
        if not value:
            continue
        k = int(value)
        if k < 1:
            raise ValueError("k values must be positive integers")
        values.append(k)
    if not values:
        raise ValueError("at least one k value is required")
    return sorted(set(values))


def select_evaluation_user_ids(
    ratings,
    max_users=100,
    min_interactions=2,
    holdout_count=1,
    user_col="userId",
):
    if ratings.empty or user_col not in ratings.columns:
        return []

    min_required = max(min_interactions, holdout_count + 1)
    interaction_counts = ratings.groupby(user_col).size()
    user_ids = interaction_counts[interaction_counts >= min_required].sort_index().index.tolist()
    if max_users and max_users > 0:
        user_ids = user_ids[:max_users]
    return user_ids


def filter_to_users(frame, user_ids, user_col="userId"):
    if frame.empty or user_col not in frame.columns:
        return frame.copy()
    return frame[frame[user_col].isin(user_ids)].copy()


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return json_ready(value.item())
    return value


def build_metric_report(
    recommendations,
    holdout,
    train,
    movies,
    k_values,
    baseline_recommendations=None,
    score_col=None,
    positive_threshold=4.0,
):
    return {
        str(k): top_n_metrics(
            recommendations,
            holdout,
            train=train,
            movies=movies,
            baseline_recommendations=baseline_recommendations,
            k=k,
            score_col=score_col,
            positive_threshold=positive_threshold,
        )
        for k in k_values
    }


def recommendation_examples(
    recommendations,
    movies=None,
    limit=0,
    include_reasons=False,
):
    if limit <= 0 or recommendations.empty:
        return []

    examples = recommendations.head(limit).copy()
    if movies is not None and not movies.empty and "movieId" in examples.columns and "movieId" in movies.columns:
        detail_columns = ["movieId"] + [
            column
            for column in ["title", "genres"]
            if column not in examples.columns and column in movies.columns
        ]
        if len(detail_columns) > 1:
            examples = examples.merge(movies[detail_columns].drop_duplicates(subset=["movieId"]), on="movieId", how="left")

    display_columns = [
        "userId",
        "movieId",
        "title",
        "genres",
        "score",
        "final_score",
        "similarity_score",
        "bayesian_rating",
        "rating_count",
        "popularity_score",
        "diversity_bonus",
        "matched_seed_count",
    ]
    hybrid_columns = {
        "similarity_score",
        "final_score",
        "bayesian_rating",
        "popularity_score",
        "diversity_bonus",
        "matched_seed_count",
    }

    rows = []
    for _, row in examples.iterrows():
        item = {}
        for column in display_columns:
            if column in row and pd.notna(row[column]):
                item[column] = json_ready(row[column])
        if include_reasons and hybrid_columns.intersection(examples.columns):
            item["reason"] = explain_hybrid_recommendation(row)
            item["score_contributions"] = json_ready(hybrid_signal_contributions(row))
        rows.append(item)
    return rows


def build_content_recommendations(
    train,
    movies,
    tags,
    user_ids,
    top_n,
    positive_threshold=4.0,
):
    if train.empty or movies.empty or not user_ids:
        return pd.DataFrame(columns=["userId", "movieId"])

    tfidf_matrix, _, movies_with_content = build_tfidf_matrix(movies.copy(), tags.copy())
    if tfidf_matrix is None or movies_with_content.empty:
        return pd.DataFrame(columns=["userId", "movieId"])

    movie_stats = build_movie_stats(train[["movieId", "rating"]])
    recommendations = []
    ratings = train.copy()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    for user_id in user_ids:
        user_history = ratings[
            (ratings["userId"] == user_id)
            & (ratings["rating"] >= positive_threshold)
        ]
        seed_ids = user_history["movieId"].dropna().drop_duplicates().tolist()
        if not seed_ids:
            continue

        user_recommendations = recommend_based_on_watch_history_content(
            seed_ids,
            movies_with_content,
            tfidf_matrix,
            movies,
            movie_stats=movie_stats,
            top_n=top_n,
        )
        if user_recommendations.empty:
            continue
        user_recommendations = user_recommendations.copy()
        user_recommendations["userId"] = user_id
        recommendations.append(user_recommendations)

    if not recommendations:
        return pd.DataFrame(columns=["userId", "movieId"])
    return pd.concat(recommendations, ignore_index=True)


def build_svd_holdout_predictions(model, holdout):
    if model is None or holdout.empty:
        return pd.DataFrame(columns=["userId", "movieId", "actual_rating", "predicted_rating"])

    rows = []
    for row in holdout.itertuples(index=False):
        rows.append(
            {
                "userId": getattr(row, "userId"),
                "movieId": getattr(row, "movieId"),
                "actual_rating": getattr(row, "rating"),
                "predicted_rating": model.predict(
                    uid=getattr(row, "userId"),
                    iid=getattr(row, "movieId"),
                ).est,
            }
        )
    return pd.DataFrame(rows)


def run_evaluation(
    max_users=100,
    k_values=None,
    holdout_count=1,
    min_interactions=5,
    positive_threshold=4.0,
    include_content=False,
    include_svd=False,
    example_count=0,
    include_reasons=False,
):
    k_values = k_values or [10]
    max_k = max(k_values)

    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags() if include_content else pd.DataFrame()
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

    report = {
        "config": {
            "max_users": int(max_users),
            "k_values": k_values,
            "holdout_count": int(holdout_count),
            "min_interactions": int(min_interactions),
            "positive_threshold": float(positive_threshold),
            "include_content": bool(include_content),
            "include_svd": bool(include_svd),
            "example_count": int(example_count),
            "include_reasons": bool(include_reasons),
        },
        "data": {
            "ratings_rows": int(len(ratings)),
            "movies_rows": int(len(movies)),
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
            )
        },
    }
    if example_count > 0:
        report["examples"] = {
            "popularity": recommendation_examples(
                popularity,
                movies=movies,
                limit=example_count,
                include_reasons=False,
            )
        }

    if include_content:
        content_recommendations = build_content_recommendations(
            train,
            movies,
            tags,
            eval_user_ids,
            max_k,
            positive_threshold=positive_threshold,
        )
        report["top_n"]["content_watch_history"] = build_metric_report(
            content_recommendations,
            holdout,
            train,
            movies,
            k_values,
            baseline_recommendations=popularity,
            positive_threshold=positive_threshold,
        )
        if example_count > 0:
            report["examples"]["content_watch_history"] = recommendation_examples(
                content_recommendations,
                movies=movies,
                limit=example_count,
                include_reasons=include_reasons,
            )

    if include_svd:
        model, model_error = load_surprise_model()
        report["svd_rating_prediction"] = {"error": model_error}
        if model is not None:
            svd_predictions = build_svd_holdout_predictions(model, holdout)
            report["svd_rating_prediction"] = rating_prediction_metrics(svd_predictions)

    return json_ready(report)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate offline recommendation baselines.")
    parser.add_argument("--max-users", type=int, default=100, help="Maximum eligible users to evaluate. Use 0 for all.")
    parser.add_argument("--k", default="10", help="Comma-separated top-N cutoffs, for example 5,10,20.")
    parser.add_argument("--holdout-count", type=int, default=1, help="Latest interactions held out per user.")
    parser.add_argument("--min-interactions", type=int, default=5, help="Minimum interactions required per user.")
    parser.add_argument("--positive-threshold", type=float, default=4.0, help="Rating threshold treated as positive.")
    parser.add_argument("--include-content", action="store_true", help="Evaluate watch-history content recommendations.")
    parser.add_argument("--include-svd", action="store_true", help="Evaluate SVD holdout rating prediction.")
    parser.add_argument("--example-count", type=int, default=0, help="Include this many recommendation examples.")
    parser.add_argument("--include-reasons", action="store_true", help="Include hybrid explanation text in examples.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    report = run_evaluation(
        max_users=args.max_users,
        k_values=parse_k_values(args.k),
        holdout_count=args.holdout_count,
        min_interactions=args.min_interactions,
        positive_threshold=args.positive_threshold,
        include_content=args.include_content,
        include_svd=args.include_svd,
        example_count=args.example_count,
        include_reasons=args.include_reasons,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
