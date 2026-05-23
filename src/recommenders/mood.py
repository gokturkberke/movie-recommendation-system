"""Mood-based recommendations: filter by configured mood-to-genre map."""

import pandas as pd

from config import MOOD_GENRE_MAP

from .common import (
    ensure_output_columns,
    filter_watched_movies,
    movie_ids_from_titles,
    normalize_movie_ids,
    output_columns,
)


def recommend_by_mood(
    mood,
    movies,
    watched_movie_ids=None,
    watched_titles=None,
    top_n=10,
    movie_stats=None,
    min_bayesian_rating=None,
):
    columns = output_columns(movies)
    genres_for_mood = MOOD_GENRE_MAP.get(str(mood).lower())
    if not genres_for_mood or movies.empty:
        return pd.DataFrame(columns=columns)

    movies_copy = movies.copy()
    movies_copy["genres"] = movies_copy["genres"].astype(str)
    mask = movies_copy["genres"].apply(lambda genres: any(genre in genres for genre in genres_for_mood))
    filtered = movies_copy[mask]
    if filtered.empty:
        return pd.DataFrame(columns=columns)

    if (
        movie_stats is not None
        and not movie_stats.empty
        and "movieId" in movie_stats.columns
        and "bayesian_rating" in movie_stats.columns
        and min_bayesian_rating is not None
    ):
        eligible_ids = movie_stats.loc[
            movie_stats["bayesian_rating"] >= float(min_bayesian_rating),
            "movieId",
        ]
        filtered_quality = filtered[filtered["movieId"].isin(set(eligible_ids))]
        if not filtered_quality.empty:
            filtered = filtered_quality

    watched_ids = normalize_movie_ids(watched_movie_ids)
    watched_ids.update(movie_ids_from_titles(watched_titles, movies))
    sample_size = min(top_n + len(watched_ids) + 5, len(filtered))
    recommendations = filtered.sample(n=sample_size, random_state=42)
    recommendations = filter_watched_movies(recommendations, watched_ids)
    return ensure_output_columns(recommendations, movies).head(top_n).reset_index(drop=True)
