"""Hybrid scoring: similarity + Bayesian rating + popularity + diversity."""

import pandas as pd

from config import BAYESIAN_MIN_RATINGS, HYBRID_WEIGHTS

from .common import (
    clamp_score,
    genre_overlap_ratio,
    numeric_series,
    numeric_value,
    split_genres,
)


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
