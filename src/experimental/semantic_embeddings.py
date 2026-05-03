from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from recommenders import (
    build_tfidf_matrix,
    ensure_output_columns,
    filter_watched_movies,
    normalize_movie_ids,
    output_columns,
)


SEMANTIC_SCORE_COLUMNS = ["similarity_score", "matched_seed_count"]


@dataclass
class SemanticEmbeddingIndex:
    embeddings: np.ndarray
    movies_with_content: pd.DataFrame
    vectorizer: object
    svd_model: object
    component_count: int


def empty_semantic_recommendations(movies):
    return pd.DataFrame(columns=output_columns(movies) + SEMANTIC_SCORE_COLUMNS)


def fit_semantic_embeddings(movies, tags, n_components=64, random_state=42):
    if n_components < 1:
        raise ValueError("n_components must be at least 1")

    tfidf_matrix, vectorizer, movies_with_content = build_tfidf_matrix(movies.copy(), tags.copy())
    if tfidf_matrix is None or movies_with_content.empty:
        return SemanticEmbeddingIndex(
            embeddings=np.zeros((0, 0), dtype=np.float32),
            movies_with_content=movies_with_content,
            vectorizer=vectorizer,
            svd_model=None,
            component_count=0,
        )

    if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] < 2:
        embeddings = normalize(tfidf_matrix, norm="l2", copy=True).toarray().astype(np.float32)
        return SemanticEmbeddingIndex(
            embeddings=embeddings,
            movies_with_content=movies_with_content,
            vectorizer=vectorizer,
            svd_model=None,
            component_count=int(embeddings.shape[1]),
        )

    component_count = min(int(n_components), int(tfidf_matrix.shape[1] - 1))
    svd_model = TruncatedSVD(n_components=component_count, random_state=random_state)
    embeddings = svd_model.fit_transform(tfidf_matrix)
    embeddings = normalize(embeddings, norm="l2", copy=False).astype(np.float32)
    return SemanticEmbeddingIndex(
        embeddings=embeddings,
        movies_with_content=movies_with_content,
        vectorizer=vectorizer,
        svd_model=svd_model,
        component_count=component_count,
    )


def movie_position_lookup(movies_with_content):
    if movies_with_content.empty or "movieId" not in movies_with_content.columns:
        return {}
    return {
        movie_id: position
        for position, movie_id in enumerate(movies_with_content["movieId"].tolist())
    }


def semantic_recommendations_for_seed_ids(
    seed_movie_ids,
    embedding_index,
    movies_for_output,
    watched_movie_ids=None,
    top_n=10,
):
    seed_ids = normalize_movie_ids(seed_movie_ids)
    watched_ids = normalize_movie_ids(watched_movie_ids)
    watched_ids.update(seed_ids)
    if not seed_ids or embedding_index.embeddings.size == 0:
        return empty_semantic_recommendations(movies_for_output)

    positions_by_movie_id = movie_position_lookup(embedding_index.movies_with_content)
    seed_positions = [
        positions_by_movie_id[movie_id]
        for movie_id in seed_ids
        if movie_id in positions_by_movie_id
    ]
    if not seed_positions:
        return empty_semantic_recommendations(movies_for_output)

    seed_vectors = embedding_index.embeddings[seed_positions]
    similarity_matrix = embedding_index.embeddings @ seed_vectors.T
    similarity_scores = similarity_matrix.max(axis=1)
    matched_seed_counts = (similarity_matrix > 0).sum(axis=1)

    scores = embedding_index.movies_with_content[["movieId"]].copy()
    scores["similarity_score"] = similarity_scores
    scores["matched_seed_count"] = matched_seed_counts
    scores = scores[~scores["movieId"].isin(watched_ids)]

    recommendations = movies_for_output[movies_for_output["movieId"].isin(scores["movieId"])].copy()
    recommendations = recommendations.merge(scores, on="movieId", how="left")
    recommendations = filter_watched_movies(recommendations, watched_ids)
    if recommendations.empty:
        return empty_semantic_recommendations(movies_for_output)

    recommendations = recommendations.sort_values(
        ["similarity_score", "matched_seed_count", "movieId"],
        ascending=[False, False, True],
    )
    recommendations = recommendations.head(top_n).reset_index(drop=True)
    return ensure_output_columns(
        recommendations,
        movies_for_output,
        SEMANTIC_SCORE_COLUMNS,
    )


def build_semantic_watch_history_recommendations(
    train,
    movies,
    tags,
    user_ids,
    top_n=10,
    n_components=64,
    random_state=42,
    positive_threshold=4.0,
):
    output = []
    if train.empty or movies.empty or not user_ids:
        columns = ["userId"] + output_columns(movies) + SEMANTIC_SCORE_COLUMNS
        return pd.DataFrame(columns=columns)

    embedding_index = fit_semantic_embeddings(
        movies,
        tags,
        n_components=n_components,
        random_state=random_state,
    )
    ratings = train.copy()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    for user_id in pd.Series(user_ids).dropna().drop_duplicates().tolist():
        user_history = ratings[
            (ratings["userId"] == user_id)
            & (ratings["rating"] >= positive_threshold)
        ]
        seed_ids = user_history["movieId"].dropna().drop_duplicates().tolist()
        if not seed_ids:
            continue
        recommendations = semantic_recommendations_for_seed_ids(
            seed_ids,
            embedding_index,
            movies,
            watched_movie_ids=seed_ids,
            top_n=top_n,
        )
        if recommendations.empty:
            continue
        recommendations = recommendations.copy()
        recommendations["userId"] = user_id
        output.append(recommendations)

    columns = ["userId"] + output_columns(movies) + SEMANTIC_SCORE_COLUMNS
    if not output:
        return pd.DataFrame(columns=columns)
    return pd.concat(output, ignore_index=True)[columns]

