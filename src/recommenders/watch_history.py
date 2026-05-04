"""Watch-history recommendations: aggregate per-seed candidates with hybrid rerank."""

import pandas as pd

from config import CONTENT_CANDIDATE_POOL_SIZE, WATCH_HISTORY_WEIGHTS

from .common import (
    HYBRID_SCORE_COLUMNS,
    ensure_output_columns,
    filter_watched_movies,
    normalize_movie_ids,
    numeric_series,
    output_columns,
)
from .content import recommend_similar_movies_by_id
from .hybrid import rerank_hybrid_candidates


def extract_watched_movies_and_genres(watched_movie_ids, movies):
    watched_ids = normalize_movie_ids(watched_movie_ids)
    if not watched_ids or movies.empty or "movieId" not in movies.columns:
        return pd.DataFrame(), set()

    movies_copy = movies.copy()
    watched_df = movies_copy[movies_copy["movieId"].isin(watched_ids)].drop_duplicates(subset=["movieId"])
    if watched_df.empty:
        return pd.DataFrame(), set()

    watched_df = watched_df.reset_index(drop=True)
    genres = set()
    if "genres" in watched_df.columns:
        for genres_str in watched_df["genres"].dropna().values:
            genres.update(str(genres_str).split("|"))
    return watched_df, genres


def genre_based_recommendations(movies, genres, watched_movie_ids, top_n):
    columns = output_columns(movies)
    if not genres:
        return pd.DataFrame(columns=columns)

    matches = movies[movies["genres"].apply(lambda value: isinstance(value, str) and any(genre in value.split("|") for genre in genres))]
    recommendations = matches.copy()
    recommendations = filter_watched_movies(recommendations, watched_movie_ids)
    return ensure_output_columns(recommendations, movies).head(top_n).reset_index(drop=True)


def fallback_recommendations(movies, watched_movie_ids, top_n):
    recommendations = movies.copy()
    recommendations = filter_watched_movies(recommendations, watched_movie_ids)
    if recommendations.empty:
        return pd.DataFrame(columns=output_columns(movies))
    sample_size = min(top_n, len(recommendations))
    return ensure_output_columns(recommendations.sample(n=sample_size, random_state=42), movies).reset_index(drop=True)


def recommend_by_watched_genres(watched_movie_ids, movies, top_n=10):
    columns = output_columns(movies)
    if not normalize_movie_ids(watched_movie_ids):
        return pd.DataFrame(columns=columns)

    watched_movies, genres = extract_watched_movies_and_genres(watched_movie_ids, movies.copy())
    watched_ids = watched_movies["movieId"] if not watched_movies.empty and "movieId" in watched_movies.columns else pd.Series(dtype="int64")
    recommendations = genre_based_recommendations(movies, genres, watched_ids, top_n)
    if recommendations.empty:
        recommendations = fallback_recommendations(movies, watched_ids, top_n)
    return ensure_output_columns(recommendations, movies).head(top_n).reset_index(drop=True)


def aggregate_watch_history_candidates(candidate_frames):
    if not candidate_frames:
        return pd.DataFrame()

    combined = pd.concat(candidate_frames, ignore_index=True)
    if combined.empty or "movieId" not in combined.columns:
        return combined

    combined["similarity_score"] = numeric_series(combined, "similarity_score", 0.0)
    if "seed_movie_id" not in combined.columns:
        combined["seed_movie_id"] = combined.index

    sort_columns = [column for column in ["final_score", "similarity_score"] if column in combined.columns]
    if sort_columns:
        representatives = combined.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    else:
        representatives = combined.copy()
    representatives = representatives.drop_duplicates(subset=["movieId"], keep="first")

    aggregated_scores = (
        combined.groupby("movieId")
        .agg(
            max_similarity_score=("similarity_score", "max"),
            mean_similarity_score=("similarity_score", "mean"),
            matched_seed_count=("seed_movie_id", "nunique"),
        )
        .reset_index()
    )
    aggregated_scores["watch_history_score"] = (
        WATCH_HISTORY_WEIGHTS["max_similarity"] * aggregated_scores["max_similarity_score"]
        + WATCH_HISTORY_WEIGHTS["mean_similarity"] * aggregated_scores["mean_similarity_score"]
        + WATCH_HISTORY_WEIGHTS["matched_seed_count_bonus"] * aggregated_scores["matched_seed_count"]
    )

    score_columns = [
        "watch_history_score",
        "max_similarity_score",
        "mean_similarity_score",
        "matched_seed_count",
    ]
    representatives = representatives.drop(columns=score_columns, errors="ignore")
    aggregated = representatives.merge(aggregated_scores, on="movieId", how="left")
    aggregated["similarity_score"] = aggregated["watch_history_score"]
    return aggregated.drop(columns=["seed_movie_id"], errors="ignore")


def recommend_based_on_watch_history_content(
    watched_movie_ids,
    movies_with_content,
    tfidf_matrix,
    movies,
    movie_stats=None,
    top_n=10,
):
    watched_ids = normalize_movie_ids(watched_movie_ids)
    columns = output_columns(movies)
    if not watched_ids:
        return pd.DataFrame(columns=columns + HYBRID_SCORE_COLUMNS)

    recommendation_frames = []
    for seed_movie_id in watched_ids:
        seed_recommendations, matched_title = recommend_similar_movies_by_id(
            seed_movie_id,
            movies_with_content,
            tfidf_matrix,
            movies,
            watched_movie_ids=watched_ids,
            movie_stats=movie_stats,
            top_n=CONTENT_CANDIDATE_POOL_SIZE,
            internal_candidate_count=CONTENT_CANDIDATE_POOL_SIZE,
        )
        if matched_title and not seed_recommendations.empty:
            seed_recommendations = seed_recommendations.copy()
            seed_recommendations["seed_movie_id"] = seed_movie_id
            recommendation_frames.append(seed_recommendations)

    if not recommendation_frames:
        return pd.DataFrame(columns=columns + HYBRID_SCORE_COLUMNS)

    combined = aggregate_watch_history_candidates(recommendation_frames)
    combined = filter_watched_movies(combined, watched_ids)
    combined = rerank_hybrid_candidates(combined, movie_stats=movie_stats, top_n=top_n)
    return ensure_output_columns(combined, movies, HYBRID_SCORE_COLUMNS).head(top_n).reset_index(drop=True)
