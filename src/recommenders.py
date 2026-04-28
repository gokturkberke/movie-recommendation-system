import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz

from config import INITIAL_CANDIDATE_POOL_SIZE, MOOD_GENRE_MAP


BASE_OUTPUT_COLUMNS = ["movieId", "title", "genres"]


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
        columns.append(include_score)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns]


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
    watched_titles=None,
    top_n=10,
    internal_candidate_count=20,
):
    columns = output_columns(movies_for_output)
    empty = pd.DataFrame(columns=columns + ["similarity_score"])
    if tfidf_matrix is None or movies_with_content.empty:
        return empty, None

    match_index = find_movie_match(movie_title, movies_with_content)
    if match_index is None:
        return empty, None

    matched_movie_id = movies_with_content.loc[match_index, "movieId"]
    matched_row = movies_for_output[movies_for_output["movieId"] == matched_movie_id]
    if matched_row.empty:
        matched_title = movies_with_content.loc[match_index].get("title", "Title Unavailable")
    else:
        matched_title = matched_row["title"].iloc[0]

    cosine_sim_vector = cosine_similarity(tfidf_matrix[match_index], tfidf_matrix).flatten()
    similar_indices = cosine_sim_vector.argsort()[-(internal_candidate_count + 1) :][::-1]
    similar_indices = [idx for idx in similar_indices if idx != match_index][:internal_candidate_count]
    if not similar_indices:
        return empty, matched_title

    scores = movies_with_content.iloc[similar_indices][["movieId"]].copy()
    scores["similarity_score"] = cosine_sim_vector[similar_indices]
    recommendations = movies_for_output[movies_for_output["movieId"].isin(scores["movieId"])].copy()
    recommendations = recommendations.merge(scores, on="movieId", how="left")

    if watched_titles and "title" in recommendations.columns:
        recommendations = recommendations[~recommendations["title"].isin(set(watched_titles))]

    recommendations = recommendations.sort_values("similarity_score", ascending=False)
    return ensure_output_columns(recommendations, movies_for_output, "similarity_score").head(top_n).reset_index(drop=True), matched_title


def extract_watched_movies_and_genres(watched_titles, movies, similarity_threshold=85):
    if not watched_titles or movies.empty:
        return pd.DataFrame(), set()

    movies_copy = movies.copy()
    movies_copy["title"] = movies_copy["title"].astype(str)
    watched_frames = []
    remaining_titles = list(watched_titles)

    for title_query in watched_titles:
        exact_matches = movies_copy[movies_copy["title"] == str(title_query)]
        if not exact_matches.empty:
            watched_frames.append(exact_matches)
            if title_query in remaining_titles:
                remaining_titles.remove(title_query)

    already_added_movie_ids = set()
    if watched_frames:
        exact_df = pd.concat(watched_frames)
        already_added_movie_ids.update(exact_df["movieId"].unique())

    if remaining_titles and "title_for_matching" in movies_copy.columns:
        movies_copy["title_for_matching_fuzzy"] = movies_copy["title_for_matching"].fillna("").astype(str).apply(clean_text)
        for title_query in remaining_titles:
            cleaned_title = clean_text(title_query)
            if not cleaned_title:
                continue

            best_score = 0
            best_index = None
            for index, row in movies_copy.iterrows():
                if row.get("movieId") in already_added_movie_ids:
                    continue
                score = fuzz.partial_ratio(cleaned_title, row["title_for_matching_fuzzy"])
                if score > best_score:
                    best_score = score
                    best_index = index

            if best_score >= similarity_threshold and best_index is not None:
                movie_id = movies_copy.loc[best_index, "movieId"]
                watched_frames.append(movies_copy.loc[[best_index]])
                already_added_movie_ids.add(movie_id)

    if not watched_frames:
        return pd.DataFrame(), set()

    watched_df = pd.concat(watched_frames).drop_duplicates(subset=["movieId"]).reset_index(drop=True)
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
    if watched_movie_ids is not None and not watched_movie_ids.empty:
        recommendations = recommendations[~recommendations["movieId"].isin(watched_movie_ids)]
    return ensure_output_columns(recommendations, movies).head(top_n).reset_index(drop=True)


def fallback_recommendations(movies, watched_movie_ids, top_n):
    recommendations = movies.copy()
    if watched_movie_ids is not None and not watched_movie_ids.empty:
        recommendations = recommendations[~recommendations["movieId"].isin(watched_movie_ids)]
    if recommendations.empty:
        return pd.DataFrame(columns=output_columns(movies))
    sample_size = min(top_n, len(recommendations))
    return ensure_output_columns(recommendations.sample(n=sample_size, random_state=42), movies).reset_index(drop=True)


def recommend_by_watched_genres(watched_titles, movies, top_n=10):
    columns = output_columns(movies)
    if not watched_titles:
        return pd.DataFrame(columns=columns)

    watched_movies, genres = extract_watched_movies_and_genres(watched_titles, movies.copy())
    watched_ids = watched_movies["movieId"] if not watched_movies.empty and "movieId" in watched_movies.columns else pd.Series(dtype="int64")
    recommendations = genre_based_recommendations(movies, genres, watched_ids, top_n)
    if recommendations.empty:
        recommendations = fallback_recommendations(movies, watched_ids, top_n)
    return ensure_output_columns(recommendations, movies).head(top_n).reset_index(drop=True)


def recommend_based_on_watch_history_content(watched_titles, movies_with_content, tfidf_matrix, movies, top_n=10):
    if not watched_titles:
        return pd.DataFrame(columns=output_columns(movies))

    watched_movies, _ = extract_watched_movies_and_genres(watched_titles, movies.copy())
    if not watched_movies.empty and "title" in watched_movies.columns:
        watched_titles_to_exclude = set(watched_movies["title"].unique())
    else:
        watched_titles_to_exclude = set(watched_titles)

    recommendation_frames = []
    for seed_title in watched_titles:
        seed_recommendations, matched_title = recommend_similar_movies(
            seed_title,
            movies_with_content,
            tfidf_matrix,
            movies,
            watched_titles=watched_titles_to_exclude,
            top_n=top_n + 5,
            internal_candidate_count=top_n + 15,
        )
        if matched_title and not seed_recommendations.empty:
            recommendation_frames.append(seed_recommendations)

    if not recommendation_frames:
        return pd.DataFrame(columns=output_columns(movies))

    combined = pd.concat(recommendation_frames)
    combined = combined.sort_values("similarity_score", ascending=False)
    combined = combined.drop_duplicates(subset=["movieId"], keep="first")
    combined = combined[~combined["title"].isin(watched_titles_to_exclude)]
    return ensure_output_columns(combined, movies).head(top_n).reset_index(drop=True)


def recommend_by_mood(mood, movies, watched_titles=None, top_n=10):
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

    sample_size = min(top_n + (len(watched_titles) if watched_titles else 0) + 5, len(filtered))
    recommendations = filtered.sample(n=sample_size, random_state=42)
    if watched_titles:
        recommendations = recommendations[~recommendations["title"].isin(set(watched_titles))]
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


def recommend_for_user(user_id, model, movies, ratings, watched_titles=None, top_n=10):
    pool_size = top_n + (len(watched_titles) if watched_titles else 0) + 20
    predictions = raw_svd_predictions(user_id, model, movies, ratings, candidate_pool_size=pool_size)
    if predictions.empty:
        return pd.DataFrame(columns=output_columns(movies))

    recommendations = predictions[["movieId"]].merge(movies[output_columns(movies)], on="movieId", how="left")
    if watched_titles and "title" in recommendations.columns:
        recommendations = recommendations[~recommendations["title"].isin(set(watched_titles))]
    return ensure_output_columns(recommendations, movies).head(top_n).reset_index(drop=True)


def recommend_for_persona(user_id, target_genre_columns, model, movies, ratings, watched_titles=None, top_n=10):
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

    if watched_titles and "title" in candidates.columns:
        candidates = candidates[~candidates["title"].isin(set(watched_titles))]

    return ensure_output_columns(candidates, movies, "predicted_score").head(top_n).reset_index(drop=True)
