import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import CONTENT_CANDIDATE_POOL_SIZE
from data_access import load_movies, load_ratings, load_tags
from evaluation import temporal_train_test_split
from evaluation_runner import filter_to_users, json_ready, select_evaluation_user_ids
from recommenders import (
    HYBRID_SCORE_COLUMNS,
    aggregate_watch_history_candidates,
    apply_hybrid_base_score,
    apply_similarity_only_scores,
    build_movie_stats,
    build_tfidf_matrix,
    ensure_output_columns,
    filter_watched_movies,
    find_movie_match_by_id,
    merge_hybrid_movie_stats,
    output_columns,
    prepare_hybrid_candidates,
    select_diverse_hybrid_candidates,
)


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000.0


def add_stage(stages, name, duration_ms):
    stages[name] = stages.get(name, 0.0) + float(duration_ms)


def timed(stages, name, func):
    start = time.perf_counter()
    result = func()
    add_stage(stages, name, elapsed_ms(start))
    return result


def profile_hybrid_rerank(candidates, movie_stats, top_n, stages, prefix):
    if candidates.empty:
        return candidates

    reranked = timed(stages, f"{prefix}_prepare_ms", lambda: prepare_hybrid_candidates(candidates))
    if movie_stats is None or movie_stats.empty:
        return timed(
            stages,
            f"{prefix}_similarity_only_rerank_ms",
            lambda: apply_similarity_only_scores(reranked, top_n=top_n),
        )

    reranked = timed(stages, f"{prefix}_merge_stats_ms", lambda: merge_hybrid_movie_stats(reranked, movie_stats))
    reranked = timed(stages, f"{prefix}_base_scoring_ms", lambda: apply_hybrid_base_score(reranked))
    return timed(stages, f"{prefix}_diversity_selection_ms", lambda: select_diverse_hybrid_candidates(reranked, top_n=top_n))


def seed_recommendations_for_profile(
    seed_movie_id,
    movies_with_content,
    tfidf_matrix,
    movies,
    watched_ids,
    movie_stats,
    stages,
):
    match_index = timed(
        stages,
        "seed_matching_ms",
        lambda: find_movie_match_by_id(seed_movie_id, movies_with_content),
    )
    if match_index is None:
        return pd.DataFrame(columns=output_columns(movies) + HYBRID_SCORE_COLUMNS)

    def compute_similarity():
        match_position = movies_with_content.index.get_loc(match_index)
        return match_position, cosine_similarity(tfidf_matrix[match_position], tfidf_matrix).flatten()

    match_position, cosine_sim_vector = timed(stages, "similarity_computation_ms", compute_similarity)

    def select_candidates():
        similar_indices = cosine_sim_vector.argsort()[-(CONTENT_CANDIDATE_POOL_SIZE + 1) :][::-1]
        return [index for index in similar_indices if index != match_position][:CONTENT_CANDIDATE_POOL_SIZE]

    similar_indices = timed(stages, "candidate_selection_ms", select_candidates)
    if not similar_indices:
        return pd.DataFrame(columns=output_columns(movies) + HYBRID_SCORE_COLUMNS)

    def merge_and_filter():
        scores = movies_with_content.iloc[similar_indices][["movieId"]].copy()
        scores["similarity_score"] = cosine_sim_vector[similar_indices]
        recommendations = movies[movies["movieId"].isin(scores["movieId"])].copy()
        recommendations = recommendations.merge(scores, on="movieId", how="left")
        return filter_watched_movies(recommendations, watched_ids)

    recommendations = timed(stages, "candidate_filtering_ms", merge_and_filter)
    recommendations = profile_hybrid_rerank(
        recommendations,
        movie_stats,
        top_n=CONTENT_CANDIDATE_POOL_SIZE,
        stages=stages,
        prefix="per_seed_rerank",
    )
    return timed(
        stages,
        "per_seed_output_shaping_ms",
        lambda: ensure_output_columns(recommendations, movies, HYBRID_SCORE_COLUMNS).reset_index(drop=True),
    )


def profile_user(
    user_id,
    train,
    movies,
    movies_with_content,
    tfidf_matrix,
    movie_stats,
    top_n,
    positive_threshold,
):
    stages = {}
    total_start = time.perf_counter()

    def select_seeds():
        user_history = train[
            (train["userId"] == user_id)
            & (pd.to_numeric(train["rating"], errors="coerce") >= positive_threshold)
        ]
        return user_history["movieId"].dropna().drop_duplicates().tolist()

    seed_ids = timed(stages, "seed_selection_ms", select_seeds)
    watched_ids = set(int(movie_id) for movie_id in seed_ids)
    candidate_frames = []

    for seed_movie_id in seed_ids:
        seed_frame = seed_recommendations_for_profile(
            seed_movie_id,
            movies_with_content,
            tfidf_matrix,
            movies,
            watched_ids,
            movie_stats,
            stages,
        )
        if not seed_frame.empty:
            seed_frame = seed_frame.copy()
            seed_frame["seed_movie_id"] = seed_movie_id
            candidate_frames.append(seed_frame)

    if candidate_frames:
        combined = timed(stages, "score_aggregation_ms", lambda: aggregate_watch_history_candidates(candidate_frames))
        combined = timed(stages, "watched_filtering_ms", lambda: filter_watched_movies(combined, watched_ids))
        recommendations = profile_hybrid_rerank(
            combined,
            movie_stats,
            top_n=top_n,
            stages=stages,
            prefix="final_rerank",
        )
        recommendations = timed(
            stages,
            "final_output_shaping_ms",
            lambda: ensure_output_columns(recommendations, movies, HYBRID_SCORE_COLUMNS).head(top_n).reset_index(drop=True),
        )
    else:
        recommendations = pd.DataFrame(columns=output_columns(movies) + HYBRID_SCORE_COLUMNS)

    total_ms = elapsed_ms(total_start)
    stages["total_ms"] = total_ms
    return {
        "userId": user_id,
        "seed_count": len(seed_ids),
        "recommendation_count": int(len(recommendations)),
        "stages_ms": stages,
    }


def summarize_stage_timings(user_profiles):
    stage_names = sorted({
        stage
        for profile in user_profiles
        for stage in profile["stages_ms"]
    })
    summary = {}
    for stage_name in stage_names:
        values = pd.Series([
            profile["stages_ms"].get(stage_name, 0.0)
            for profile in user_profiles
        ], dtype="float64")
        summary[stage_name] = {
            "mean_ms": float(values.mean()) if not values.empty else 0.0,
            "p95_ms": float(values.quantile(0.95)) if not values.empty else 0.0,
            "total_ms": float(values.sum()) if not values.empty else 0.0,
        }
    return summary


def flatten_user_profiles(user_profiles):
    rows = []
    stage_names = sorted({
        stage
        for profile in user_profiles
        for stage in profile["stages_ms"]
    })
    for profile in user_profiles:
        row = {
            "userId": profile["userId"],
            "seed_count": profile["seed_count"],
            "recommendation_count": profile["recommendation_count"],
        }
        for stage_name in stage_names:
            row[stage_name] = profile["stages_ms"].get(stage_name, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_profile(
    max_users=5,
    top_n=10,
    holdout_count=1,
    min_interactions=5,
    positive_threshold=4.0,
    output_dir=None,
):
    setup_stages = {}
    ratings = timed(setup_stages, "data_access_ratings_ms", load_ratings)
    movies = timed(setup_stages, "data_access_movies_ms", load_movies)
    tags = timed(setup_stages, "data_access_tags_ms", load_tags)

    selected_user_ids = timed(
        setup_stages,
        "user_selection_ms",
        lambda: select_evaluation_user_ids(
            ratings,
            max_users=max_users,
            min_interactions=min_interactions,
            holdout_count=holdout_count,
        ),
    )
    sampled_ratings = timed(setup_stages, "filter_to_users_ms", lambda: filter_to_users(ratings, selected_user_ids))
    train, holdout = timed(
        setup_stages,
        "temporal_split_ms",
        lambda: temporal_train_test_split(
            sampled_ratings,
            holdout_count=holdout_count,
            min_interactions=min_interactions,
        ),
    )
    tfidf_matrix, _, movies_with_content = timed(
        setup_stages,
        "tfidf_resource_setup_ms",
        lambda: build_tfidf_matrix(movies.copy(), tags.copy()),
    )
    movie_stats = timed(setup_stages, "movie_stats_setup_ms", lambda: build_movie_stats(train[["movieId", "rating"]]))

    eval_user_ids = holdout["userId"].dropna().drop_duplicates().tolist() if "userId" in holdout.columns else []
    user_profiles = [
        profile_user(
            user_id,
            train,
            movies,
            movies_with_content,
            tfidf_matrix,
            movie_stats,
            top_n=top_n,
            positive_threshold=positive_threshold,
        )
        for user_id in eval_user_ids
    ]

    report = {
        "config": {
            "max_users": int(max_users),
            "top_n": int(top_n),
            "holdout_count": int(holdout_count),
            "min_interactions": int(min_interactions),
            "positive_threshold": float(positive_threshold),
        },
        "data": {
            "ratings_rows": int(len(ratings)),
            "movies_rows": int(len(movies)),
            "tags_rows": int(len(tags)),
            "selected_user_count": int(len(selected_user_ids)),
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "profiled_user_count": int(len(user_profiles)),
        },
        "setup_stages_ms": setup_stages,
        "per_user_stage_summary_ms": summarize_stage_timings(user_profiles),
        "users": user_profiles,
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "hybrid_latency_profile.json"
        csv_path = output_path / "hybrid_latency_profile.csv"
        json_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True, default=str))
        flatten_user_profiles(user_profiles).to_csv(csv_path, index=False)
        report["artifacts"] = {
            "profile_json": str(json_path),
            "profile_csv": str(csv_path),
        }

    return json_ready(report)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Profile the current watch-history hybrid recommendation path.")
    parser.add_argument("--max-users", type=int, default=5, help="Maximum eligible users to profile.")
    parser.add_argument("--k", type=int, default=10, help="Recommendation list size.")
    parser.add_argument("--holdout-count", type=int, default=1, help="Latest interactions held out per user.")
    parser.add_argument("--min-interactions", type=int, default=5, help="Minimum interactions required per user.")
    parser.add_argument("--positive-threshold", type=float, default=4.0, help="Rating threshold used as positive seed history.")
    parser.add_argument("--output-dir", default="/private/tmp/hybrid_profile", help="Directory for profile JSON/CSV artifacts.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    report = run_profile(
        max_users=args.max_users,
        top_n=args.k,
        holdout_count=args.holdout_count,
        min_interactions=args.min_interactions,
        positive_threshold=args.positive_threshold,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
