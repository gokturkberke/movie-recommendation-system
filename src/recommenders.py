import math
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz

from config import INITIAL_CANDIDATE_POOL_SIZE, MOOD_GENRE_MAP


BASE_OUTPUT_COLUMNS = ["movieId", "title", "genres"]
CONTENT_CANDIDATE_POOL_SIZE = 100
BAYESIAN_MIN_RATINGS = 100
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
HYBRID_WEIGHTS = {
    "content_similarity": 0.60,
    "bayesian_rating": 0.25,
    "popularity": 0.10,
    "diversity": 0.05,
}
WATCH_HISTORY_WEIGHTS = {
    "max_similarity": 0.70,
    "mean_similarity": 0.20,
    "matched_seed_count_bonus": 0.10,
}


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


def prepare_hybrid_candidates(candidates):
    reranked = candidates.copy()
    reranked = reranked.drop(
        columns=[column for column in ["final_score", "base_score", "diversity_bonus"] if column in reranked.columns],
        errors="ignore",
    )
    reranked["similarity_score"] = numeric_series(reranked, "similarity_score", 0.0)
    return reranked


def apply_similarity_only_scores(candidates, top_n=10):
    reranked = candidates.copy()
    reranked["final_score"] = reranked["similarity_score"]
    reranked["bayesian_rating"] = pd.NA
    reranked["rating_count"] = pd.NA
    reranked["popularity_score"] = 0.0
    reranked["diversity_bonus"] = 0.0
    return reranked.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)


def merge_hybrid_movie_stats(candidates, movie_stats):
    stat_columns = [
        "movieId",
        "bayesian_rating",
        "bayesian_rating_normalized",
        "rating_count",
        "popularity_score",
    ]
    available_stat_columns = [column for column in stat_columns if column in movie_stats.columns]
    reranked = candidates.drop(
        columns=[column for column in available_stat_columns if column != "movieId" and column in candidates.columns],
        errors="ignore",
    )
    reranked = reranked.merge(movie_stats[available_stat_columns], on="movieId", how="left")
    for column in ["bayesian_rating_normalized", "popularity_score"]:
        reranked[column] = numeric_series(reranked, column, 0.0)
    return reranked


def weighted_hybrid_base_score(similarity_score, bayesian_score, popularity_score):
    return (
        HYBRID_WEIGHTS["content_similarity"] * similarity_score
        + HYBRID_WEIGHTS["bayesian_rating"] * bayesian_score
        + HYBRID_WEIGHTS["popularity"] * popularity_score
    )


def weighted_hybrid_final_score(base_score, diversity_bonus):
    return base_score + HYBRID_WEIGHTS["diversity"] * diversity_bonus


def apply_hybrid_base_score(candidates):
    reranked = candidates.copy()
    reranked["base_score"] = weighted_hybrid_base_score(
        reranked["similarity_score"],
        reranked["bayesian_rating_normalized"],
        reranked["popularity_score"],
    )
    return reranked


def hybrid_signal_contributions(row):
    similarity = numeric_value(row.get("similarity_score", 0.0))
    bayesian = numeric_value(row.get("bayesian_rating_normalized", 0.0))
    popularity = numeric_value(row.get("popularity_score", 0.0))
    diversity = numeric_value(row.get("diversity_bonus", 0.0))
    return {
        "content_similarity": HYBRID_WEIGHTS["content_similarity"] * similarity,
        "bayesian_rating": HYBRID_WEIGHTS["bayesian_rating"] * bayesian,
        "popularity": HYBRID_WEIGHTS["popularity"] * popularity,
        "diversity": HYBRID_WEIGHTS["diversity"] * diversity,
    }


def explain_hybrid_recommendation(row):
    reasons = []
    similarity = clamp_score(row.get("similarity_score", 0.0))
    bayesian_rating = pd.to_numeric(pd.Series([row.get("bayesian_rating", pd.NA)]), errors="coerce").iloc[0]
    rating_count = pd.to_numeric(pd.Series([row.get("rating_count", pd.NA)]), errors="coerce").iloc[0]
    popularity = clamp_score(row.get("popularity_score", 0.0))
    diversity = clamp_score(row.get("diversity_bonus", 0.0))
    matched_seed_count = pd.to_numeric(pd.Series([row.get("matched_seed_count", pd.NA)]), errors="coerce").iloc[0]

    if similarity >= 0.65:
        reasons.append("strong content similarity")
    elif similarity > 0:
        reasons.append("content similarity")
    if pd.notna(bayesian_rating) and bayesian_rating >= 4.0:
        reasons.append("high Bayesian rating")
    if pd.notna(rating_count) and rating_count >= BAYESIAN_MIN_RATINGS:
        reasons.append("well-supported rating signal")
    elif popularity >= 0.60:
        reasons.append("popular with viewers")
    if diversity >= 0.50:
        reasons.append("adds genre variety")
    if pd.notna(matched_seed_count) and matched_seed_count > 1:
        reasons.append("matches multiple watched movies")

    if not reasons:
        return "Ranked by the available hybrid signals."
    return "Ranked for " + ", ".join(reasons[:3]) + "."


def diversity_bonus_for_candidate(candidate_genres, selected_genres):
    if not selected_genres:
        return 1.0
    max_overlap = max(genre_overlap_ratio(candidate_genres, genres) for genres in selected_genres)
    return 1.0 - max_overlap


def select_diverse_hybrid_candidates(candidates, top_n=10):
    reranked = candidates.copy()
    remaining = reranked.copy()
    selected_rows = []
    selected_genres = []
    while not remaining.empty and len(selected_rows) < top_n:
        scored = []
        for index, row in remaining.iterrows():
            candidate_genres = split_genres(row.get("genres", ""))
            diversity_bonus = diversity_bonus_for_candidate(candidate_genres, selected_genres)
            final_score = weighted_hybrid_final_score(row["base_score"], diversity_bonus)
            scored.append((final_score, row["base_score"], row["similarity_score"], index, diversity_bonus))

        _, _, _, best_index, best_diversity = max(scored, key=lambda item: (item[0], item[1], item[2]))
        best_row = remaining.loc[best_index].copy()
        best_row["diversity_bonus"] = best_diversity
        best_row["final_score"] = weighted_hybrid_final_score(best_row["base_score"], best_diversity)
        selected_rows.append(best_row)
        selected_genres.append(split_genres(best_row.get("genres", "")))
        remaining = remaining.drop(index=best_index)

    if not selected_rows:
        return pd.DataFrame(columns=reranked.columns)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def rerank_hybrid_candidates(candidates, movie_stats=None, top_n=10):
    if candidates.empty:
        return candidates

    reranked = prepare_hybrid_candidates(candidates)

    if movie_stats is None or movie_stats.empty:
        return apply_similarity_only_scores(reranked, top_n)

    reranked = merge_hybrid_movie_stats(reranked, movie_stats)
    reranked = apply_hybrid_base_score(reranked)
    return select_diverse_hybrid_candidates(reranked, top_n)


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


def recommend_by_mood(mood, movies, watched_movie_ids=None, watched_titles=None, top_n=10):
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

    watched_ids = normalize_movie_ids(watched_movie_ids)
    watched_ids.update(movie_ids_from_titles(watched_titles, movies))
    sample_size = min(top_n + len(watched_ids) + 5, len(filtered))
    recommendations = filtered.sample(n=sample_size, random_state=42)
    recommendations = filter_watched_movies(recommendations, watched_ids)
    return ensure_output_columns(recommendations, movies).head(top_n).reset_index(drop=True)


def pick_random_movie(movies, selected_genres=None):
    if movies.empty:
        return None

    filtered = movies.copy()
    if selected_genres:
        genre_mask = pd.Series(False, index=filtered.index)
        for genre in selected_genres:
            genre_mask |= filtered["genres"].astype(str).str.contains(genre, case=False, na=False, regex=False)
        filtered = filtered[genre_mask]

    if filtered.empty:
        return None
    return filtered.sample(n=1, random_state=None).iloc[0]


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
