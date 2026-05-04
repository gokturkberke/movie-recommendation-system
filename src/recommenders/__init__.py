"""Recommendation engine, organized by family.

Public symbols are re-exported here so existing callers can keep using
``from recommenders import recommend_similar_movies`` without changes.
"""

from .common import (
    BASE_OUTPUT_COLUMNS,
    HYBRID_SCORE_COLUMNS,
    build_movie_stats,
    clamp_score,
    clean_text,
    ensure_output_columns,
    filter_watched_movies,
    genre_overlap_ratio,
    movie_ids_from_titles,
    normalize_movie_ids,
    numeric_series,
    numeric_value,
    output_columns,
    split_genres,
)
from .content import (
    build_recommendations_from_match_index,
    build_tfidf_matrix,
    find_movie_match,
    find_movie_match_by_id,
    recommend_similar_movies,
    recommend_similar_movies_by_id,
    suggest_movie_titles,
    title_match_score,
)
from .hybrid import (
    apply_hybrid_base_score,
    apply_similarity_only_scores,
    diversity_bonus_for_candidate,
    explain_hybrid_recommendation,
    hybrid_signal_contributions,
    merge_hybrid_movie_stats,
    prepare_hybrid_candidates,
    rerank_hybrid_candidates,
    select_diverse_hybrid_candidates,
    weighted_hybrid_base_score,
    weighted_hybrid_final_score,
)
from .mood import recommend_by_mood
from .picker import pick_random_movie
from .svd import (
    raw_svd_predictions,
    recommend_for_persona,
    recommend_for_user,
)
from .watch_history import (
    aggregate_watch_history_candidates,
    extract_watched_movies_and_genres,
    fallback_recommendations,
    genre_based_recommendations,
    recommend_based_on_watch_history_content,
    recommend_by_watched_genres,
)


__all__ = [
    # constants
    "BASE_OUTPUT_COLUMNS",
    "HYBRID_SCORE_COLUMNS",
    # common helpers
    "build_movie_stats",
    "clamp_score",
    "clean_text",
    "ensure_output_columns",
    "filter_watched_movies",
    "genre_overlap_ratio",
    "movie_ids_from_titles",
    "normalize_movie_ids",
    "numeric_series",
    "numeric_value",
    "output_columns",
    "split_genres",
    # content
    "build_recommendations_from_match_index",
    "build_tfidf_matrix",
    "find_movie_match",
    "find_movie_match_by_id",
    "recommend_similar_movies",
    "recommend_similar_movies_by_id",
    "suggest_movie_titles",
    "title_match_score",
    # hybrid
    "apply_hybrid_base_score",
    "apply_similarity_only_scores",
    "diversity_bonus_for_candidate",
    "explain_hybrid_recommendation",
    "hybrid_signal_contributions",
    "merge_hybrid_movie_stats",
    "prepare_hybrid_candidates",
    "rerank_hybrid_candidates",
    "select_diverse_hybrid_candidates",
    "weighted_hybrid_base_score",
    "weighted_hybrid_final_score",
    # mood / picker / svd
    "recommend_by_mood",
    "pick_random_movie",
    "raw_svd_predictions",
    "recommend_for_persona",
    "recommend_for_user",
    # watch history
    "aggregate_watch_history_candidates",
    "extract_watched_movies_and_genres",
    "fallback_recommendations",
    "genre_based_recommendations",
    "recommend_based_on_watch_history_content",
    "recommend_by_watched_genres",
]
