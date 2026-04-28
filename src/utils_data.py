from pathlib import Path

from config import CLEANED_DATA_DIR
from data_access import load_movies, load_ratings, load_tags, load_surprise_model
from recommenders import (
    clean_text,
    extract_watched_movies_and_genres,
    fallback_recommendations,
    genre_based_recommendations,
    pick_random_movie,
    raw_svd_predictions,
)
from tmdb_client import get_movie_details


def load_trained_surprise_model(model_filename="svd_trained_model.pkl"):
    model_path = Path(model_filename)
    if not model_path.is_absolute():
        model_path = CLEANED_DATA_DIR / model_path
    model, _ = load_surprise_model(model_path)
    return model


def get_movie_details_from_tmdb(tmdb_id, api_key):
    return get_movie_details(tmdb_id, api_key)


def _get_raw_svd_predictions(user_id, surprise_model, movies_df, ratings_df, candidate_pool_size=None):
    return raw_svd_predictions(user_id, surprise_model, movies_df, ratings_df, candidate_pool_size)


def _extract_watched_movies_and_genres(watched_movie_ids, movies_input_df, similarity_threshold=85):
    return extract_watched_movies_and_genres(watched_movie_ids, movies_input_df)


def _get_genre_based_recommendations(movies_df, all_genres_set, watched_movie_ids, top_n):
    return genre_based_recommendations(movies_df, all_genres_set, watched_movie_ids, top_n)


def _get_fallback_recommendations(movies_df, watched_movie_ids, top_n):
    return fallback_recommendations(movies_df, watched_movie_ids, top_n)
