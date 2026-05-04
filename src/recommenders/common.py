"""Shared helpers used across every recommendation family."""

import math
import re

import pandas as pd

from config import BAYESIAN_MIN_RATINGS


BASE_OUTPUT_COLUMNS = ["movieId", "title", "genres"]
HYBRID_SCORE_COLUMNS = [
    "similarity_score",
    "final_score",
    "bayesian_rating",
    "rating_count",
    "popularity_score",
    "diversity_bonus",
    "watch_history_score",
    "max_similarity_score",
    "mean_similarity_score",
    "matched_seed_count",
]


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s*\(\d{4}\)", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def output_columns(movies):
    columns = BASE_OUTPUT_COLUMNS.copy()
    if "tmdbId" in movies.columns:
        columns.append("tmdbId")
    return columns


def ensure_output_columns(df, movies=None, include_score=None):
    columns = output_columns(movies if movies is not None else df)
    if include_score:
        if isinstance(include_score, (list, tuple)):
            columns.extend(include_score)
        else:
            columns.append(include_score)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns]


def normalize_movie_ids(movie_ids):
    if movie_ids is None:
        return set()

    if isinstance(movie_ids, (str, bytes)) or not hasattr(movie_ids, "__iter__"):
        movie_ids = [movie_ids]

    normalized = set()
    for movie_id in movie_ids:
        if pd.isna(movie_id):
            continue
        try:
            normalized.add(int(movie_id))
        except (TypeError, ValueError):
            normalized.add(movie_id)
    return normalized


def filter_watched_movies(df, watched_movie_ids):
    watched_ids = normalize_movie_ids(watched_movie_ids)
    if not watched_ids or df.empty or "movieId" not in df.columns:
        return df
    return df[~df["movieId"].isin(watched_ids)]


def numeric_series(df, column, default=0.0):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def clamp_score(value, default=0.0):
    if pd.isna(value):
        return default
    return min(max(float(value), 0.0), 1.0)


def numeric_value(value, default=0.0):
    if pd.isna(value):
        return default
    return float(value)


def movie_ids_from_titles(titles, movies):
    if titles is None or movies.empty or "title" not in movies.columns:
        return set()
    if isinstance(titles, (str, bytes)) or not hasattr(titles, "__iter__"):
        titles = [titles]
    titles = list(titles)
    if not titles:
        return set()
    matched = movies[movies["title"].isin(set(titles))]
    if matched.empty or "movieId" not in matched.columns:
        return set()
    return normalize_movie_ids(matched["movieId"])


def build_movie_stats(ratings, min_rating_count=BAYESIAN_MIN_RATINGS):
    columns = [
        "movieId",
        "avg_rating",
        "rating_count",
        "bayesian_rating",
        "bayesian_rating_normalized",
        "popularity_score",
    ]
    if ratings is None or ratings.empty or not {"movieId", "rating"}.issubset(ratings.columns):
        return pd.DataFrame(columns=columns)

    ratings_copy = ratings[["movieId", "rating"]].copy()
    ratings_copy["rating"] = pd.to_numeric(ratings_copy["rating"], errors="coerce")
    ratings_copy = ratings_copy.dropna(subset=["movieId", "rating"])
    if ratings_copy.empty:
        return pd.DataFrame(columns=columns)

    stats = (
        ratings_copy.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )
    global_mean = ratings_copy["rating"].mean()
    v = stats["rating_count"].astype(float)
    r = stats["avg_rating"].astype(float)
    m = float(min_rating_count)
    stats["bayesian_rating"] = (v / (v + m)) * r + (m / (v + m)) * global_mean
    stats["bayesian_rating_normalized"] = (stats["bayesian_rating"] / 5.0).clip(0, 1)

    max_count = stats["rating_count"].max()
    if max_count and max_count > 0:
        max_popularity = math.log(float(max_count) + 1.0)
        stats["popularity_score"] = stats["rating_count"].astype(float).add(1.0).apply(math.log) / max_popularity
    else:
        stats["popularity_score"] = 0.0

    return stats[columns]


def split_genres(genres):
    if pd.isna(genres):
        return set()
    return {
        genre.strip()
        for genre in str(genres).split("|")
        if genre.strip() and genre.strip() != "(no genres listed)"
    }


def genre_overlap_ratio(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
