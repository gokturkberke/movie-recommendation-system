"""Collaborative filtering via Surprise SVD (raw predictions + persona filter)."""

import pandas as pd

from config import INITIAL_CANDIDATE_POOL_SIZE

from .common import (
    ensure_output_columns,
    filter_watched_movies,
    movie_ids_from_titles,
    normalize_movie_ids,
    output_columns,
)


def raw_svd_predictions(user_id, model, movies, ratings, candidate_pool_size=None):
    if model is None or movies.empty:
        return pd.DataFrame(columns=["movieId", "predicted_score"])

    all_movie_ids = movies["movieId"].unique()
    if ratings is not None and not ratings.empty:
        rated_movie_ids = ratings[ratings["userId"] == user_id]["movieId"].unique()
    else:
        rated_movie_ids = []
    movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]

    predictions = [
        {"movieId": movie_id, "predicted_score": model.predict(uid=user_id, iid=movie_id).est}
        for movie_id in movies_to_predict
    ]
    predictions_df = pd.DataFrame(predictions)
    if predictions_df.empty:
        return pd.DataFrame(columns=["movieId", "predicted_score"])

    predictions_df = predictions_df.sort_values("predicted_score", ascending=False)
    if candidate_pool_size:
        return predictions_df.head(candidate_pool_size)
    return predictions_df


def recommend_for_user(user_id, model, movies, ratings, watched_movie_ids=None, watched_titles=None, top_n=10):
    watched_ids = normalize_movie_ids(watched_movie_ids)
    watched_ids.update(movie_ids_from_titles(watched_titles, movies))
    pool_size = top_n + len(watched_ids) + 20
    predictions = raw_svd_predictions(user_id, model, movies, ratings, candidate_pool_size=pool_size)
    if predictions.empty:
        return pd.DataFrame(columns=output_columns(movies))

    recommendations = predictions[["movieId"]].merge(movies[output_columns(movies)], on="movieId", how="left")
    recommendations = filter_watched_movies(recommendations, watched_ids)
    return ensure_output_columns(recommendations, movies).head(top_n).reset_index(drop=True)


def recommend_for_persona(user_id, target_genre_columns, model, movies, ratings, watched_movie_ids=None, watched_titles=None, top_n=10):
    watched_ids = normalize_movie_ids(watched_movie_ids)
    watched_ids.update(movie_ids_from_titles(watched_titles, movies))
    predictions = raw_svd_predictions(
        user_id,
        model,
        movies,
        ratings,
        candidate_pool_size=INITIAL_CANDIDATE_POOL_SIZE,
    )
    if predictions.empty:
        return pd.DataFrame(columns=output_columns(movies) + ["predicted_score"])

    detail_columns = output_columns(movies) + [column for column in target_genre_columns if column in movies.columns]
    candidates = predictions[["movieId", "predicted_score"]].merge(movies[detail_columns], on="movieId", how="left")

    valid_target_columns = [column for column in target_genre_columns if column in candidates.columns]
    if valid_target_columns:
        for column in valid_target_columns:
            candidates[column] = candidates[column].fillna(0).astype(int)
        candidates = candidates[candidates[valid_target_columns].sum(axis=1) > 0]

    candidates = filter_watched_movies(candidates, watched_ids)

    return ensure_output_columns(candidates, movies, "predicted_score").head(top_n).reset_index(drop=True)
