import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import EVALUATION_DEFAULTS, EVALUATION_OUTPUT_DIR
from data_access import load_movies, load_ratings, load_surprise_model, load_tags
from evaluation import (
    measure_per_user_latency,
    popularity_recommendations,
    random_recommendations,
    rating_prediction_metrics,
    svd_topk_recommendations,
    temporal_train_test_split,
    tfidf_content_recommendations,
    top_n_metrics,
)
from experimental.semantic_embeddings import (
    fit_semantic_embeddings,
    semantic_recommendations_for_seed_ids,
)
from recommenders import (
    build_movie_stats,
    build_tfidf_matrix,
    explain_hybrid_recommendation,
    hybrid_signal_contributions,
    recommend_based_on_watch_history_content,
)


METRIC_CSV_COLUMNS = [
    "model",
    "k",
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "ndcg_at_k",
    "catalog_coverage",
    "user_coverage",
    "diversity",
    "novelty",
    "evaluated_user_count",
    "recommended_item_count",
    "latency_mean_ms",
    "latency_p95_ms",
]

_SEMANTIC_DEFAULTS = EVALUATION_DEFAULTS.get("semantic") or {}
_DEFAULT_K_VALUES = EVALUATION_DEFAULTS.get("k_values") or [10]
_DEFAULT_K_STR = ",".join(str(int(value)) for value in _DEFAULT_K_VALUES)


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


def make_hybrid_per_user(
    train,
    movies,
    movies_with_content,
    tfidf_matrix,
    movie_stats,
    top_n,
    positive_threshold=4.0,
):
    ratings = train.copy() if train is not None else pd.DataFrame()
    if not ratings.empty and "rating" in ratings.columns:
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    stats = movie_stats if movie_stats is not None and not movie_stats.empty else None

    def recommend(user_id):
        if (
            ratings.empty
            or movies is None
            or movies.empty
            or tfidf_matrix is None
            or movies_with_content is None
            or movies_with_content.empty
        ):
            return pd.DataFrame()
        user_history = ratings[
            (ratings["userId"] == user_id)
            & (ratings["rating"] >= positive_threshold)
        ]
        seed_ids = user_history["movieId"].dropna().drop_duplicates().tolist()
        if not seed_ids:
            return pd.DataFrame()
        recommendations = recommend_based_on_watch_history_content(
            seed_ids,
            movies_with_content,
            tfidf_matrix,
            movies,
            movie_stats=stats,
            top_n=top_n,
        )
        if recommendations.empty:
            return recommendations
        recommendations = recommendations.copy()
        recommendations["userId"] = user_id
        return recommendations

    return recommend


def make_semantic_per_user(
    train,
    movies,
    embedding_index,
    top_n,
    positive_threshold=4.0,
):
    ratings = train.copy() if train is not None else pd.DataFrame()
    if not ratings.empty and "rating" in ratings.columns:
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

    def recommend(user_id):
        if (
            ratings.empty
            or movies is None
            or movies.empty
            or embedding_index is None
            or embedding_index.embeddings.size == 0
        ):
            return pd.DataFrame()
        user_history = ratings[
            (ratings["userId"] == user_id)
            & (ratings["rating"] >= positive_threshold)
        ]
        seed_ids = user_history["movieId"].dropna().drop_duplicates().tolist()
        if not seed_ids:
            return pd.DataFrame()
        recommendations = semantic_recommendations_for_seed_ids(
            seed_ids,
            embedding_index,
            movies,
            watched_movie_ids=seed_ids,
            top_n=top_n,
        )
        if recommendations.empty:
            return recommendations
        recommendations = recommendations.copy()
        recommendations["userId"] = user_id
        return recommendations

    return recommend


def run_per_user(recommend_for_user, user_ids, measure_latency):
    if measure_latency:
        return measure_per_user_latency(recommend_for_user, user_ids)
    frames = []
    for user_id in user_ids:
        frame = recommend_for_user(user_id)
        if frame is not None and not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, None


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


def evaluate_baseline(
    name,
    recommend_for_user,
    eval_user_ids,
    holdout,
    train,
    movies,
    k_values,
    measure_latency,
    score_col,
    baseline_recommendations=None,
    positive_threshold=4.0,
):
    recommendations, latency = run_per_user(recommend_for_user, eval_user_ids, measure_latency)
    metrics = build_metric_report(
        recommendations,
        holdout,
        train,
        movies,
        k_values,
        baseline_recommendations=baseline_recommendations,
        score_col=score_col,
        positive_threshold=positive_threshold,
    )
    return {
        "name": name,
        "recommendations": recommendations,
        "metrics": metrics,
        "latency": latency,
    }


def build_summary_rows(report):
    top_n = report.get("top_n") or {}
    latency = report.get("latency") or {}
    rows = []
    for model_name in sorted(top_n.keys()):
        per_k = top_n[model_name] or {}
        latency_summary = latency.get(model_name) or {}
        for k_str in sorted(per_k.keys(), key=lambda value: int(value)):
            metrics = per_k[k_str] or {}
            rows.append({
                "model": model_name,
                "k": int(k_str),
                "precision_at_k": float(metrics.get("precision_at_k", 0.0)),
                "recall_at_k": float(metrics.get("recall_at_k", 0.0)),
                "hit_rate_at_k": float(metrics.get("hit_rate_at_k", 0.0)),
                "ndcg_at_k": float(metrics.get("ndcg_at_k", 0.0)),
                "catalog_coverage": float(metrics.get("catalog_coverage", 0.0)),
                "user_coverage": float(metrics.get("user_coverage", 0.0)),
                "diversity": float(metrics.get("diversity", 0.0)),
                "novelty": float(metrics.get("novelty", 0.0)),
                "evaluated_user_count": int(metrics.get("evaluated_user_count", 0)),
                "recommended_item_count": int(metrics.get("recommended_item_count", 0)),
                "latency_mean_ms": float(latency_summary.get("mean_ms", 0.0)) if latency_summary else 0.0,
                "latency_p95_ms": float(latency_summary.get("p95_ms", 0.0)) if latency_summary else 0.0,
            })
    return rows


def write_artifacts(report, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    payload = {key: value for key, value in report.items() if key != "artifacts"}
    summary_rows = build_summary_rows(payload)
    summary_df = pd.DataFrame(summary_rows, columns=METRIC_CSV_COLUMNS)

    json_text = json.dumps(json_ready(payload), indent=2, sort_keys=True, default=str)

    paths = {
        "metrics_json": output_path / "metrics_summary.json",
        "metrics_json_versioned": output_path / f"metrics_summary_{timestamp}.json",
        "metrics_csv": output_path / "metrics_summary.csv",
        "metrics_csv_versioned": output_path / f"metrics_summary_{timestamp}.csv",
        "run_config": output_path / "run_config.json",
    }
    paths["metrics_json"].write_text(json_text)
    paths["metrics_json_versioned"].write_text(json_text)
    summary_df.to_csv(paths["metrics_csv"], index=False)
    summary_df.to_csv(paths["metrics_csv_versioned"], index=False)
    paths["run_config"].write_text(
        json.dumps(json_ready(payload.get("config", {})), indent=2, sort_keys=True, default=str)
    )
    return {key: str(value) for key, value in paths.items()}


def run_evaluation(
    max_users=100,
    k_values=None,
    holdout_count=1,
    min_interactions=5,
    positive_threshold=4.0,
    include_random=False,
    include_tfidf=False,
    include_content=False,
    include_semantic=False,
    include_svd_topk=False,
    include_svd=False,
    measure_latency=True,
    output_dir=None,
    random_seed=42,
    semantic_components=64,
    semantic_random_state=42,
    example_count=0,
    include_reasons=False,
):
    k_values = k_values or [10]
    max_k = max(k_values)

    ratings = load_ratings()
    movies = load_movies()
    needs_content_resources = include_tfidf or include_content
    needs_tags = needs_content_resources or include_semantic
    tags = load_tags() if needs_tags else pd.DataFrame()
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

    eval_user_ids = (
        holdout["userId"].dropna().drop_duplicates().tolist()
        if "userId" in holdout.columns
        else []
    )
    candidate_items = movies[["movieId"]] if "movieId" in movies.columns else None

    tfidf_matrix = None
    movies_with_content = pd.DataFrame()
    movie_stats = pd.DataFrame()
    if needs_content_resources and not movies.empty:
        tfidf_matrix, _, movies_with_content = build_tfidf_matrix(movies.copy(), tags.copy())
        if include_content and not train.empty and "rating" in train.columns:
            movie_stats = build_movie_stats(train[["movieId", "rating"]])

    svd_model = None
    svd_model_error = None
    if include_svd or include_svd_topk:
        svd_model, svd_model_error = load_surprise_model()

    embedding_index = None
    if include_semantic and not movies.empty:
        embedding_index = fit_semantic_embeddings(
            movies,
            tags,
            n_components=semantic_components,
            random_state=semantic_random_state,
        )

    report = {
        "config": {
            "max_users": int(max_users),
            "k_values": k_values,
            "holdout_count": int(holdout_count),
            "min_interactions": int(min_interactions),
            "positive_threshold": float(positive_threshold),
            "include_random": bool(include_random),
            "include_tfidf": bool(include_tfidf),
            "include_content": bool(include_content),
            "include_semantic": bool(include_semantic),
            "include_svd_topk": bool(include_svd_topk),
            "include_svd": bool(include_svd),
            "measure_latency": bool(measure_latency),
            "random_seed": int(random_seed),
            "semantic_components": int(semantic_components),
            "semantic_random_state": int(semantic_random_state),
            "semantic_method": "tfidf+truncated_svd" if include_semantic else None,
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
        "top_n": {},
        "latency": {},
    }
    if example_count > 0:
        report["examples"] = {}

    def record(result):
        report["top_n"][result["name"]] = result["metrics"]
        if result["latency"] is not None:
            report["latency"][result["name"]] = result["latency"]
        if example_count > 0:
            report["examples"][result["name"]] = recommendation_examples(
                result["recommendations"],
                movies=movies,
                limit=example_count,
                include_reasons=include_reasons and result["name"] == "hybrid_content",
            )

    popularity_closure = lambda user_id: popularity_recommendations(
        train,
        candidate_items=candidate_items,
        user_ids=[user_id],
        k=max_k,
        positive_threshold=positive_threshold,
    )
    popularity_result = evaluate_baseline(
        "popularity",
        popularity_closure,
        eval_user_ids,
        holdout,
        train,
        movies,
        k_values,
        measure_latency,
        score_col="score",
        positive_threshold=positive_threshold,
    )
    record(popularity_result)
    popularity_recommendations_df = popularity_result["recommendations"]

    if include_random:
        def random_closure(user_id):
            per_user_seed = (random_seed * 1000003 + int(user_id)) & 0x7FFFFFFF
            return random_recommendations(
                train,
                candidate_items,
                [user_id],
                k=max_k,
                seed=per_user_seed,
            )
        record(evaluate_baseline(
            "random",
            random_closure,
            eval_user_ids,
            holdout,
            train,
            movies,
            k_values,
            measure_latency,
            score_col="score",
            baseline_recommendations=popularity_recommendations_df,
            positive_threshold=positive_threshold,
        ))

    if include_tfidf and tfidf_matrix is not None and not movies_with_content.empty:
        def tfidf_closure(user_id):
            return tfidf_content_recommendations(
                train,
                [user_id],
                movies_with_content,
                tfidf_matrix,
                k=max_k,
                positive_threshold=positive_threshold,
            )
        record(evaluate_baseline(
            "tfidf_content",
            tfidf_closure,
            eval_user_ids,
            holdout,
            train,
            movies,
            k_values,
            measure_latency,
            score_col="score",
            baseline_recommendations=popularity_recommendations_df,
            positive_threshold=positive_threshold,
        ))

    if include_content and tfidf_matrix is not None and not movies_with_content.empty:
        hybrid_closure = make_hybrid_per_user(
            train,
            movies,
            movies_with_content,
            tfidf_matrix,
            movie_stats,
            max_k,
            positive_threshold=positive_threshold,
        )
        record(evaluate_baseline(
            "hybrid_content",
            hybrid_closure,
            eval_user_ids,
            holdout,
            train,
            movies,
            k_values,
            measure_latency,
            score_col=None,
            baseline_recommendations=popularity_recommendations_df,
            positive_threshold=positive_threshold,
        ))

    if include_semantic and embedding_index is not None and embedding_index.embeddings.size:
        semantic_closure = make_semantic_per_user(
            train,
            movies,
            embedding_index,
            max_k,
            positive_threshold=positive_threshold,
        )
        record(evaluate_baseline(
            "semantic_content",
            semantic_closure,
            eval_user_ids,
            holdout,
            train,
            movies,
            k_values,
            measure_latency,
            score_col="similarity_score",
            baseline_recommendations=popularity_recommendations_df,
            positive_threshold=positive_threshold,
        ))

    if include_svd_topk and svd_model is not None:
        def svd_topk_closure(user_id):
            return svd_topk_recommendations(
                svd_model,
                train,
                candidate_items,
                [user_id],
                k=max_k,
            )
        record(evaluate_baseline(
            "svd_topk",
            svd_topk_closure,
            eval_user_ids,
            holdout,
            train,
            movies,
            k_values,
            measure_latency,
            score_col="score",
            baseline_recommendations=popularity_recommendations_df,
            positive_threshold=positive_threshold,
        ))

    if include_svd:
        report["svd_rating_prediction"] = {"error": svd_model_error}
        if svd_model is not None:
            svd_predictions = build_svd_holdout_predictions(svd_model, holdout)
            report["svd_rating_prediction"] = rating_prediction_metrics(svd_predictions)

    if output_dir is not None:
        report["artifacts"] = write_artifacts(report, output_dir)

    return json_ready(report)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate offline recommendation baselines.")
    parser.add_argument("--max-users", type=int, default=int(EVALUATION_DEFAULTS.get("max_users", 100)), help="Maximum eligible users to evaluate. Use 0 for all.")
    parser.add_argument("--k", default=_DEFAULT_K_STR, help="Comma-separated top-N cutoffs, for example 5,10,20.")
    parser.add_argument("--holdout-count", type=int, default=int(EVALUATION_DEFAULTS.get("holdout_count", 1)), help="Latest interactions held out per user.")
    parser.add_argument("--min-interactions", type=int, default=int(EVALUATION_DEFAULTS.get("min_interactions", 5)), help="Minimum interactions required per user.")
    parser.add_argument("--positive-threshold", type=float, default=float(EVALUATION_DEFAULTS.get("positive_threshold", 4.0)), help="Rating threshold treated as positive.")
    parser.add_argument("--include-random", action="store_true", help="Evaluate the random baseline.")
    parser.add_argument("--include-tfidf", action="store_true", help="Evaluate the pure TF-IDF content baseline (no hybrid rerank).")
    parser.add_argument("--include-content", action="store_true", help="Evaluate the watch-history hybrid (TF-IDF + Bayesian + popularity + diversity).")
    parser.add_argument("--include-semantic", action="store_true", help="Evaluate the semantic content baseline (TF-IDF + TruncatedSVD LSA, watch-history seeds, max-similarity aggregation).")
    parser.add_argument("--semantic-components", type=int, default=int(_SEMANTIC_DEFAULTS.get("components", 64)), help="Latent dimensions for the semantic embedding index (TruncatedSVD).")
    parser.add_argument("--semantic-random-state", type=int, default=int(_SEMANTIC_DEFAULTS.get("random_state", 42)), help="Random state used by the semantic TruncatedSVD fit.")
    parser.add_argument("--include-svd-topk", action="store_true", help="Evaluate SVD top-K recommendations from the trained Surprise model.")
    parser.add_argument("--include-svd", action="store_true", help="Evaluate SVD holdout rating prediction (RMSE/MAE).")
    parser.add_argument("--no-measure-latency", action="store_true", help="Disable per-user latency measurement.")
    parser.add_argument("--random-seed", type=int, default=int(EVALUATION_DEFAULTS.get("random_seed", 42)), help="Seed used by the random baseline.")
    parser.add_argument("--output-dir", default=str(EVALUATION_OUTPUT_DIR), help="Directory for metrics_summary.json/csv. Use empty string to disable saving.")
    parser.add_argument("--example-count", type=int, default=0, help="Include this many recommendation examples.")
    parser.add_argument("--include-reasons", action="store_true", help="Include hybrid explanation text in examples.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir else None
    report = run_evaluation(
        max_users=args.max_users,
        k_values=parse_k_values(args.k),
        holdout_count=args.holdout_count,
        min_interactions=args.min_interactions,
        positive_threshold=args.positive_threshold,
        include_random=args.include_random,
        include_tfidf=args.include_tfidf,
        include_content=args.include_content,
        include_semantic=args.include_semantic,
        include_svd_topk=args.include_svd_topk,
        include_svd=args.include_svd,
        measure_latency=not args.no_measure_latency,
        output_dir=output_dir,
        random_seed=args.random_seed,
        semantic_components=args.semantic_components,
        semantic_random_state=args.semantic_random_state,
        example_count=args.example_count,
        include_reasons=args.include_reasons,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
