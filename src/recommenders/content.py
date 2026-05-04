"""Content-based (TF-IDF) recommendations and title matching."""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz

from config import CONTENT_CANDIDATE_POOL_SIZE

from .common import (
    HYBRID_SCORE_COLUMNS,
    clean_text,
    ensure_output_columns,
    filter_watched_movies,
    movie_ids_from_titles,
    normalize_movie_ids,
    output_columns,
)
from .hybrid import rerank_hybrid_candidates


def build_tfidf_matrix(movies, tags):
    if movies.empty:
        return None, None, movies.copy()

    movies_with_content = movies.copy()
    tags_copy = tags.copy()
    if tags_copy.empty or "tag" not in tags_copy.columns:
        tags_grouped = pd.DataFrame(columns=["movieId", "tag"])
    else:
        tags_copy["tag"] = tags_copy["tag"].fillna("").apply(clean_text)
        tags_copy = tags_copy.drop_duplicates(subset=["movieId", "tag"])
        tags_grouped = tags_copy.groupby("movieId")["tag"].apply(lambda values: " ".join(values)).reset_index()

    movies_with_content = movies_with_content.merge(tags_grouped, on="movieId", how="left")
    for column in ["title_for_matching", "genres_for_matching", "tag"]:
        if column not in movies_with_content.columns:
            movies_with_content[column] = ""
        movies_with_content[column] = movies_with_content[column].fillna("").astype(str)

    movies_with_content["content"] = (
        movies_with_content["title_for_matching"]
        + " "
        + movies_with_content["genres_for_matching"]
        + " "
        + movies_with_content["tag"]
    )

    tfidf = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = tfidf.fit_transform(movies_with_content["content"].fillna(""))
    except ValueError:
        return None, None, movies_with_content
    return tfidf_matrix, tfidf, movies_with_content


def find_movie_match(movie_title, movies_with_content):
    cleaned_movie_title = clean_text(movie_title)
    if not cleaned_movie_title or "title_for_matching" not in movies_with_content.columns:
        return None

    titles = movies_with_content["title_for_matching"].fillna("").astype(str)
    exact_matches = movies_with_content[titles == cleaned_movie_title]
    if not exact_matches.empty:
        return exact_matches.index[0]

    contains_matches = movies_with_content[titles.str.contains(cleaned_movie_title, na=False, regex=False)]
    if not contains_matches.empty:
        return contains_matches.index[0]

    best_score = 0
    best_index = None
    for index, title_for_matching in titles.items():
        score = fuzz.ratio(cleaned_movie_title, title_for_matching)
        if score > best_score:
            best_score = score
            best_index = index

    if best_score > 80:
        return best_index
    return None


def find_movie_match_by_id(movie_id, movies_with_content):
    movie_ids = normalize_movie_ids([movie_id])
    if not movie_ids or movies_with_content.empty or "movieId" not in movies_with_content.columns:
        return None
    matches = movies_with_content[movies_with_content["movieId"].isin(movie_ids)]
    if matches.empty:
        return None
    return matches.index[0]


def title_match_score(query, candidate):
    query = clean_text(query)
    candidate = clean_text(candidate)
    if not query or not candidate:
        return 0

    if query == candidate:
        return 120
    if candidate.startswith(query):
        return 115
    if query in candidate:
        return 105

    token_scores = []
    for index, token in enumerate(candidate.split()):
        if len(token) <= 1:
            continue
        token_score = fuzz.ratio(query, token)
        if index > 0:
            token_score = 105 if query == token else token_score - 15
        token_scores.append(token_score)
    token_score = max(token_scores) if token_scores else 0
    full_score = fuzz.ratio(query, candidate)
    token_set_score = fuzz.token_set_ratio(query, candidate) if " " in query else 0
    return max(token_score, full_score, token_set_score)


def suggest_movie_titles(query, movies, limit=8, min_score=78):
    cleaned_query = clean_text(query)
    if len(cleaned_query) < 2 or movies.empty:
        return pd.DataFrame(columns=["movieId", "title", "genres", "match_score"])

    suggestions = []
    seen_movie_ids = set()
    for _, row in movies.iterrows():
        movie_id = row.get("movieId")
        if movie_id in seen_movie_ids:
            continue
        candidate = row.get("title_for_matching") or row.get("title")
        score = title_match_score(cleaned_query, candidate)
        if score >= min_score:
            title = row.get("title", "")
            years = re.findall(r"\((\d{4})\)", str(title))
            suggestions.append(
                {
                    "movieId": movie_id,
                    "title": title,
                    "genres": row.get("genres", ""),
                    "match_score": score,
                    "release_year": int(years[-1]) if years else 0,
                }
            )
            seen_movie_ids.add(movie_id)

    if not suggestions:
        return pd.DataFrame(columns=["movieId", "title", "genres", "match_score"])

    suggestions_df = pd.DataFrame(suggestions)
    suggestions_df = suggestions_df.sort_values(
        ["match_score", "release_year", "title"],
        ascending=[False, False, True],
    )
    return suggestions_df.head(limit).reset_index(drop=True)


def recommend_similar_movies(
    movie_title,
    movies_with_content,
    tfidf_matrix,
    movies_for_output,
    watched_movie_ids=None,
    watched_titles=None,
    movie_stats=None,
    top_n=10,
    internal_candidate_count=CONTENT_CANDIDATE_POOL_SIZE,
):
    columns = output_columns(movies_for_output)
    empty = pd.DataFrame(columns=columns + HYBRID_SCORE_COLUMNS)
    if tfidf_matrix is None or movies_with_content.empty:
        return empty, None

    match_index = find_movie_match(movie_title, movies_with_content)
    if match_index is None:
        return empty, None

    watched_ids = normalize_movie_ids(watched_movie_ids)
    watched_ids.update(movie_ids_from_titles(watched_titles, movies_for_output))
    return build_recommendations_from_match_index(
        match_index,
        movies_with_content,
        tfidf_matrix,
        movies_for_output,
        watched_movie_ids=watched_ids,
        movie_stats=movie_stats,
        top_n=top_n,
        internal_candidate_count=internal_candidate_count,
    )


def recommend_similar_movies_by_id(
    movie_id,
    movies_with_content,
    tfidf_matrix,
    movies_for_output,
    watched_movie_ids=None,
    movie_stats=None,
    top_n=10,
    internal_candidate_count=CONTENT_CANDIDATE_POOL_SIZE,
):
    columns = output_columns(movies_for_output)
    empty = pd.DataFrame(columns=columns + HYBRID_SCORE_COLUMNS)
    if tfidf_matrix is None or movies_with_content.empty:
        return empty, None

    match_index = find_movie_match_by_id(movie_id, movies_with_content)
    if match_index is None:
        return empty, None

    return build_recommendations_from_match_index(
        match_index,
        movies_with_content,
        tfidf_matrix,
        movies_for_output,
        watched_movie_ids=watched_movie_ids,
        movie_stats=movie_stats,
        top_n=top_n,
        internal_candidate_count=internal_candidate_count,
    )


def build_recommendations_from_match_index(
    match_index,
    movies_with_content,
    tfidf_matrix,
    movies_for_output,
    watched_movie_ids=None,
    movie_stats=None,
    top_n=10,
    internal_candidate_count=CONTENT_CANDIDATE_POOL_SIZE,
):
    columns = output_columns(movies_for_output)
    empty = pd.DataFrame(columns=columns + HYBRID_SCORE_COLUMNS)
    matched_movie_id = movies_with_content.loc[match_index, "movieId"]
    matched_row = movies_for_output[movies_for_output["movieId"] == matched_movie_id]
    if matched_row.empty:
        matched_title = movies_with_content.loc[match_index].get("title", "Title Unavailable")
    else:
        matched_title = matched_row["title"].iloc[0]

    match_position = movies_with_content.index.get_loc(match_index)
    cosine_sim_vector = cosine_similarity(tfidf_matrix[match_position], tfidf_matrix).flatten()
    similar_indices = cosine_sim_vector.argsort()[-(internal_candidate_count + 1) :][::-1]
    similar_indices = [idx for idx in similar_indices if idx != match_position][:internal_candidate_count]
    if not similar_indices:
        return empty, matched_title

    scores = movies_with_content.iloc[similar_indices][["movieId"]].copy()
    scores["similarity_score"] = cosine_sim_vector[similar_indices]
    recommendations = movies_for_output[movies_for_output["movieId"].isin(scores["movieId"])].copy()
    recommendations = recommendations.merge(scores, on="movieId", how="left")
    recommendations = filter_watched_movies(recommendations, watched_movie_ids)
    recommendations = rerank_hybrid_candidates(recommendations, movie_stats=movie_stats, top_n=top_n)
    return ensure_output_columns(recommendations, movies_for_output, HYBRID_SCORE_COLUMNS).head(top_n).reset_index(drop=True), matched_title
