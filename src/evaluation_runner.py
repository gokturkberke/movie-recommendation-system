import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import EVALUATION_DEFAULTS, project_path
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
from experimental.sbert_faiss import (
    load_sbert_faiss_index,
    sbert_faiss_recommendations_for_seed_ids,
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
    "map_at_k",
    "mrr_at_k",
    "catalog_coverage",
    "user_coverage",
    "diversity",
    "novelty",
    "evaluated_user_count",
    "recommended_item_count",
    "rmse",
    "mae",
    "rating_prediction_count",
    "latency_mean_ms",
    "latency_p95_ms",
]


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


def make_sbert_faiss_per_user(
    train,
    movies,
    sbert_index,
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
            or sbert_index is None
            or sbert_index.embeddings.size == 0
        ):
            return pd.DataFrame()
        user_history = ratings[
            (ratings["userId"] == user_id)
            & (ratings["rating"] >= positive_threshold)
        ]
        seed_ids = user_history["movieId"].dropna().drop_duplicates().tolist()
        if not seed_ids:
            return pd.DataFrame()
        recommendations = sbert_faiss_recommendations_for_seed_ids(
            seed_ids,
            sbert_index,
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
                "map_at_k": float(metrics.get("map_at_k", 0.0)),
                "mrr_at_k": float(metrics.get("mrr_at_k", 0.0)),
                "catalog_coverage": float(metrics.get("catalog_coverage", 0.0)),
                "user_coverage": float(metrics.get("user_coverage", 0.0)),
                "diversity": float(metrics.get("diversity", 0.0)),
                "novelty": float(metrics.get("novelty", 0.0)),
                "evaluated_user_count": int(metrics.get("evaluated_user_count", 0)),
                "recommended_item_count": int(metrics.get("recommended_item_count", 0)),
                "rmse": math.nan,
                "mae": math.nan,
                "rating_prediction_count": 0,
                "latency_mean_ms": float(latency_summary.get("mean_ms", 0.0)) if latency_summary else 0.0,
                "latency_p95_ms": float(latency_summary.get("p95_ms", 0.0)) if latency_summary else 0.0,
            })

    svd_rating_prediction = report.get("svd_rating_prediction") or {}
    if "count" in svd_rating_prediction:
        rows.append({
            "model": "svd_rating_prediction",
            "k": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
            "mrr_at_k": 0.0,
            "catalog_coverage": 0.0,
            "user_coverage": 0.0,
            "diversity": 0.0,
            "novelty": 0.0,
            "evaluated_user_count": 0,
            "recommended_item_count": 0,
            "rmse": float(svd_rating_prediction.get("rmse", math.nan)),
            "mae": float(svd_rating_prediction.get("mae", math.nan)),
            "rating_prediction_count": int(svd_rating_prediction.get("count", 0)),
            "latency_mean_ms": 0.0,
            "latency_p95_ms": 0.0,
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
    include_sbert_faiss=False,
    include_svd_topk=False,
    include_svd=False,
    measure_latency=True,
    output_dir=None,
    random_seed=42,
    semantic_components=64,
    semantic_random_state=42,
    sbert_faiss_index_dir=None,
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

    sbert_index = None
    sbert_faiss_error = None
    if include_sbert_faiss:
        sbert_defaults = EVALUATION_DEFAULTS.get("sbert_faiss") or {}
        resolved_index_dir = project_path(sbert_faiss_index_dir or sbert_defaults.get("index_dir", "artifacts/indexes/sbert_faiss"))
        try:
            sbert_index = load_sbert_faiss_index(resolved_index_dir)
        except (FileNotFoundError, ImportError, ValueError) as exc:
            sbert_faiss_error = str(exc)

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
            "include_sbert_faiss": bool(include_sbert_faiss),
            "include_svd_topk": bool(include_svd_topk),
            "include_svd": bool(include_svd),
            "measure_latency": bool(measure_latency),
            "random_seed": int(random_seed),
            "semantic_components": int(semantic_components),
            "semantic_random_state": int(semantic_random_state),
            "semantic_method": "tfidf+truncated_svd" if include_semantic else None,
            "sbert_faiss_index_dir": str(resolved_index_dir) if include_sbert_faiss else None,
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
    if sbert_faiss_error:
        report["sbert_faiss_error"] = sbert_faiss_error

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

    if include_sbert_faiss and sbert_index is not None and sbert_index.embeddings.size:
        sbert_faiss_closure = make_sbert_faiss_per_user(
            train,
            movies,
            sbert_index,
            max_k,
            positive_threshold=positive_threshold,
        )
        record(evaluate_baseline(
            "sbert_faiss_content",
            sbert_faiss_closure,
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
