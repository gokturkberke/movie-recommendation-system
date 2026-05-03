"""MovieLens cleaning pipeline.

Reads raw CSVs from ``raw_data_dir`` and writes cleaned CSVs to ``cleaned_data_dir``.
Each stage is exposed as a standalone function so callers (tests, notebooks, the
CLI in ``scripts/preprocess_dataset.py``) can run a single stage if they want.
"""

import os
import re
from pathlib import Path

import pandas as pd

from config import CLEANED_DATA_DIR, DATA_DIR


DEFAULT_MIN_RATINGS_PER_USER = 5
DEFAULT_MIN_RATINGS_PER_MOVIE = 5


def clean_title_display(title):
    if pd.isnull(title):
        return ""
    text = str(title)
    text = re.sub(r"[^a-zA-Z0-9\s()]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_matching(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s*\(\d{4}\)", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tag(tag):
    text = str(tag).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def clean_movies(raw_data_dir, cleaned_data_dir):
    raw_data_dir = Path(raw_data_dir)
    cleaned_data_dir = Path(cleaned_data_dir)

    movies = pd.read_csv(raw_data_dir / "movies.csv")
    movies = movies.drop_duplicates(subset=["movieId"])
    movies = movies.dropna(subset=["title"])
    movies = movies[~movies["genres"].str.contains(r"\(no genres listed\)", na=False)]
    movies["title"] = movies["title"].fillna("").astype(str).str.strip()
    movies["genres"] = movies["genres"].fillna("").astype(str).str.strip()

    movies["title_original"] = movies["title"]
    movies["genres_original"] = movies["genres"]
    movies["title_display"] = movies["title_original"].apply(clean_title_display)
    movies["title_clean"] = movies["title_display"]
    movies["title"] = movies["title_display"]
    movies["title_for_matching"] = movies["title_original"].apply(clean_text_for_matching)
    movies["genres_for_matching"] = movies["genres_original"].apply(
        lambda value: clean_text_for_matching(value.replace("|", " "))
    )

    movies["genres_original_list"] = (
        movies["genres_original"].fillna("").astype(str).str.split("|")
    )
    all_genres = sorted({
        genre
        for sublist in movies["genres_original_list"]
        for genre in sublist
        if genre
    })
    for genre in all_genres:
        column = f"genre_{clean_text_for_matching(genre)}"
        movies[column] = movies["genres_original_list"].apply(
            lambda values, target=genre: 1 if target in values else 0
        )
    movies = movies.drop(columns=["genres_original_list"])

    output_path = cleaned_data_dir / "movies_clean.csv"
    movies.to_csv(output_path, index=False)
    print(f"movies_clean.csv saved -> {output_path}")
    return movies


def clean_ratings(
    raw_data_dir,
    cleaned_data_dir,
    min_ratings_per_user=DEFAULT_MIN_RATINGS_PER_USER,
    min_ratings_per_movie=DEFAULT_MIN_RATINGS_PER_MOVIE,
):
    raw_data_dir = Path(raw_data_dir)
    cleaned_data_dir = Path(cleaned_data_dir)

    ratings = pd.read_csv(raw_data_dir / "ratings.csv")
    ratings = ratings.drop_duplicates(subset=["userId", "movieId", "timestamp"])
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

    user_counts = ratings["userId"].value_counts()
    ratings = ratings[ratings["userId"].isin(user_counts[user_counts >= min_ratings_per_user].index)]

    movie_counts = ratings["movieId"].value_counts()
    ratings = ratings[ratings["movieId"].isin(movie_counts[movie_counts >= min_ratings_per_movie].index)]

    ratings["rating_z"] = (ratings["rating"] - ratings["rating"].mean()) / ratings["rating"].std()

    output_path = cleaned_data_dir / "ratings_clean.csv"
    ratings.to_csv(output_path, index=False)
    print(f"ratings_clean.csv saved -> {output_path}")
    return ratings


def clean_tags(raw_data_dir, cleaned_data_dir):
    raw_data_dir = Path(raw_data_dir)
    cleaned_data_dir = Path(cleaned_data_dir)

    tags = pd.read_csv(raw_data_dir / "tags.csv")
    tags = tags.drop_duplicates(subset=["userId", "movieId", "tag", "timestamp"])
    tags = tags.dropna(subset=["userId", "movieId", "tag"])
    tags["tag"] = tags["tag"].apply(clean_tag)

    output_path = cleaned_data_dir / "tags_clean.csv"
    tags.to_csv(output_path, index=False)
    print(f"tags_clean.csv saved -> {output_path}")
    return tags


def run_preprocessing(
    raw_data_dir=DATA_DIR,
    cleaned_data_dir=CLEANED_DATA_DIR,
    min_ratings_per_user=DEFAULT_MIN_RATINGS_PER_USER,
    min_ratings_per_movie=DEFAULT_MIN_RATINGS_PER_MOVIE,
):
    raw_data_dir = Path(raw_data_dir)
    cleaned_data_dir = Path(cleaned_data_dir)
    os.makedirs(cleaned_data_dir, exist_ok=True)

    clean_movies(raw_data_dir, cleaned_data_dir)
    clean_ratings(
        raw_data_dir,
        cleaned_data_dir,
        min_ratings_per_user=min_ratings_per_user,
        min_ratings_per_movie=min_ratings_per_movie,
    )
    clean_tags(raw_data_dir, cleaned_data_dir)
    print(f"Preprocessing complete. Cleaned files are in {cleaned_data_dir}")
