from pathlib import Path

import pandas as pd

from config import CLEANED_DATA_DIR, DATA_DIR, SVD_MODEL_PATH


EXPECTED_MOVIE_COLUMNS = [
    "movieId",
    "title",
    "genres",
    "title_display",
    "title_clean",
    "title_for_matching",
    "genres_for_matching",
]


def empty_links_frame():
    return pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])


def load_csv(path, required=False):
    path = Path(path)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        if required:
            raise
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_movies(data_dir=CLEANED_DATA_DIR):
    movies = load_csv(Path(data_dir) / "movies_clean.csv")
    for column in EXPECTED_MOVIE_COLUMNS:
        if column not in movies.columns:
            movies[column] = ""
    if "title_original" in movies.columns:
        title_display = movies["title_original"].fillna("").astype(str).str.strip()
        movies["title_display"] = title_display
        movies["title_clean"] = title_display
        movies["title"] = title_display
    return movies


def load_tags(data_dir=CLEANED_DATA_DIR):
    tags = load_csv(Path(data_dir) / "tags_clean.csv")
    if tags.empty:
        return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    return tags


def load_ratings(data_dir=CLEANED_DATA_DIR):
    ratings = load_csv(Path(data_dir) / "ratings_clean.csv")
    if ratings.empty:
        return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp", "rating_z"])
    return ratings


def load_links(data_dir=DATA_DIR):
    links = load_csv(Path(data_dir) / "links.csv")
    if links.empty:
        return empty_links_frame()

    if "tmdbId" not in links.columns:
        links["tmdbId"] = pd.NA
    links = links[pd.notna(links["tmdbId"])].copy()
    if not links.empty:
        links["tmdbId"] = links["tmdbId"].astype(int)
    return links


def merge_tmdb_ids(movies, links):
    if movies.empty or links.empty or "tmdbId" not in links.columns:
        return movies.copy()
    if "tmdbId" in movies.columns:
        return movies.copy()
    return movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")


def load_surprise_model(model_path=SVD_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        return None, f"Saved SVD model not found at {model_path}."

    try:
        from surprise import dump
    except Exception as exc:
        return None, f"Surprise could not be imported: {exc}"

    try:
        loaded_object = dump.load(str(model_path))
        return loaded_object[1], None
    except Exception as exc:
        return None, f"SVD model could not be loaded: {exc}"
