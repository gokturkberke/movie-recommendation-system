import math

import pandas as pd


DEFAULT_USER_COL = "userId"
DEFAULT_ITEM_COL = "movieId"
DEFAULT_RATING_COL = "rating"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_SCORE_COL = "score"


def require_columns(df, columns, frame_name="DataFrame"):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def temporal_train_test_split(
    ratings,
    holdout_count=1,
    min_interactions=2,
    user_col=DEFAULT_USER_COL,
    timestamp_col=DEFAULT_TIMESTAMP_COL,
):
    """Split each eligible user's latest interactions into a temporal holdout."""
    require_columns(ratings, [user_col, timestamp_col], "ratings")
    if holdout_count < 1:
        raise ValueError("holdout_count must be at least 1")
    if ratings.empty:
        return ratings.copy(), ratings.copy()

    ordered = ratings.copy()
    ordered["_input_order"] = range(len(ordered))
    ordered = ordered.sort_values([user_col, timestamp_col, "_input_order"])

    train_parts = []
    holdout_parts = []
    for _, user_rows in ordered.groupby(user_col, sort=False):
        if len(user_rows) >= min_interactions and len(user_rows) > holdout_count:
            train_parts.append(user_rows.iloc[:-holdout_count])
            holdout_parts.append(user_rows.iloc[-holdout_count:])
        else:
            train_parts.append(user_rows)

    columns = [column for column in ordered.columns if column != "_input_order"]
    train = pd.concat(train_parts, ignore_index=True)[columns] if train_parts else ordered.iloc[0:0][columns]
    holdout = pd.concat(holdout_parts, ignore_index=True)[columns] if holdout_parts else ordered.iloc[0:0][columns]
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def rating_prediction_metrics(
    predictions,
    actual_col="actual_rating",
    predicted_col="predicted_rating",
):
    """Compute RMSE and MAE for rating predictions."""
    require_columns(predictions, [actual_col, predicted_col], "predictions")
    valid = predictions[[actual_col, predicted_col]].dropna()
    if valid.empty:
        return {"rmse": math.nan, "mae": math.nan, "count": 0}

    errors = pd.to_numeric(valid[predicted_col], errors="coerce") - pd.to_numeric(valid[actual_col], errors="coerce")
    errors = errors.dropna()
    if errors.empty:
        return {"rmse": math.nan, "mae": math.nan, "count": 0}

    return {
        "rmse": math.sqrt(float((errors ** 2).mean())),
        "mae": float(errors.abs().mean()),
        "count": int(len(errors)),
    }


def candidate_item_ids(candidate_items, item_col=DEFAULT_ITEM_COL):
    if candidate_items is None:
        return None
    if isinstance(candidate_items, pd.DataFrame):
        require_columns(candidate_items, [item_col], "candidate_items")
        values = candidate_items[item_col]
    else:
        values = pd.Series(candidate_items)
    return values.dropna().drop_duplicates().tolist()


def popularity_recommendations(
    train_ratings,
    candidate_items=None,
    user_ids=None,
    k=10,
    positive_threshold=4.0,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    rating_col=DEFAULT_RATING_COL,
    score_col=DEFAULT_SCORE_COL,
):
    """Build a simple popularity baseline while excluding each user's train items."""
    require_columns(train_ratings, [user_col, item_col, rating_col], "train_ratings")
    candidate_ids = candidate_item_ids(candidate_items, item_col=item_col)
    output_columns = [user_col, item_col, score_col, "positive_count", "rating_count", "avg_rating"]

    if user_ids is None:
        user_ids = train_ratings[user_col].dropna().drop_duplicates().tolist()
    else:
        user_ids = pd.Series(user_ids).dropna().drop_duplicates().tolist()

    if not user_ids:
        return pd.DataFrame(columns=output_columns)

    ratings = train_ratings[[user_col, item_col, rating_col]].dropna(subset=[user_col, item_col]).copy()
    ratings[rating_col] = pd.to_numeric(ratings[rating_col], errors="coerce")
    ratings["is_positive"] = ratings[rating_col] >= positive_threshold

    if ratings.empty and not candidate_ids:
        return pd.DataFrame(columns=output_columns)

    if ratings.empty:
        stats = pd.DataFrame({item_col: candidate_ids})
        stats["positive_count"] = 0
        stats["rating_count"] = 0
        stats["avg_rating"] = 0.0
    else:
        stats = (
            ratings.groupby(item_col)
            .agg(
                positive_count=("is_positive", "sum"),
                rating_count=(rating_col, "count"),
                avg_rating=(rating_col, "mean"),
            )
            .reset_index()
        )
        if candidate_ids is not None:
            missing_ids = sorted(set(candidate_ids) - set(stats[item_col]))
            if missing_ids:
                missing = pd.DataFrame({item_col: missing_ids})
                missing["positive_count"] = 0
                missing["rating_count"] = 0
                missing["avg_rating"] = 0.0
                stats = pd.concat([stats, missing], ignore_index=True)
            stats = stats[stats[item_col].isin(candidate_ids)]

    stats[score_col] = stats["positive_count"].astype(float)
    stats = stats.sort_values(
        [score_col, "rating_count", "avg_rating", item_col],
        ascending=[False, False, False, True],
    )

    seen_by_user = ratings.groupby(user_col)[item_col].apply(set).to_dict() if not ratings.empty else {}
    recommendations = []
    for user_id in user_ids:
        seen_items = seen_by_user.get(user_id, set())
        user_candidates = stats[~stats[item_col].isin(seen_items)].head(k).copy()
        user_candidates[user_col] = user_id
        recommendations.append(user_candidates[[user_col, item_col, score_col, "positive_count", "rating_count", "avg_rating"]])

    if not recommendations:
        return pd.DataFrame(columns=output_columns)
    return pd.concat(recommendations, ignore_index=True)


def top_n_metrics(
    recommendations,
    holdout,
    train=None,
    movies=None,
    baseline_recommendations=None,
    k=10,
    positive_threshold=4.0,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    rating_col=DEFAULT_RATING_COL,
    genres_col="genres",
    score_col=None,
):
    """Compute top-N relevance and catalog diagnostics for ranked recommendations."""
    require_columns(recommendations, [user_col, item_col], "recommendations")
    require_columns(holdout, [user_col, item_col, rating_col], "holdout")

    ranked = top_k_by_user(recommendations, k, user_col, item_col, score_col)
    positive_holdout = holdout[pd.to_numeric(holdout[rating_col], errors="coerce") >= positive_threshold]
    relevant_by_user = positive_holdout.groupby(user_col)[item_col].apply(set).to_dict()
    evaluated_users = list(relevant_by_user.keys())

    if not evaluated_users:
        return empty_top_n_metrics(k)

    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []
    for user_id in evaluated_users:
        relevant_items = relevant_by_user[user_id]
        recommended_items = ranked.get(user_id, [])
        hits = [item_id for item_id in recommended_items if item_id in relevant_items]
        precisions.append(len(hits) / float(k))
        recalls.append(len(hits) / float(len(relevant_items)))
        hit_rates.append(1.0 if hits else 0.0)
        ndcgs.append(ndcg_at_k(recommended_items, relevant_items, k))

    catalog_size = infer_catalog_size(recommendations, holdout, train, movies, item_col)
    recommended_items = {
        item_id
        for user_id in evaluated_users
        for item_id in ranked.get(user_id, [])
    }
    user_count_with_recommendations = sum(1 for user_id in evaluated_users if ranked.get(user_id))

    baseline_ranked = {}
    if baseline_recommendations is not None:
        require_columns(baseline_recommendations, [user_col, item_col], "baseline_recommendations")
        baseline_ranked = top_k_by_user(baseline_recommendations, k, user_col, item_col, score_col)

    return {
        "k": int(k),
        "precision_at_k": mean(precisions),
        "recall_at_k": mean(recalls),
        "ndcg_at_k": mean(ndcgs),
        "hit_rate_at_k": mean(hit_rates),
        "catalog_coverage": len(recommended_items) / float(catalog_size) if catalog_size else 0.0,
        "user_coverage": user_count_with_recommendations / float(len(evaluated_users)),
        "diversity": mean(intra_list_diversities(ranked, evaluated_users, movies, item_col, genres_col)),
        "novelty": novelty_score(ranked, evaluated_users, train, item_col),
        "serendipity": serendipity_score(ranked, baseline_ranked, relevant_by_user, evaluated_users),
        "evaluated_user_count": int(len(evaluated_users)),
        "recommended_item_count": int(len(recommended_items)),
    }


def top_k_by_user(recommendations, k, user_col, item_col, score_col=None):
    if recommendations.empty:
        return {}

    ranked = recommendations.copy()
    ranked["_input_order"] = range(len(ranked))
    if score_col and score_col in ranked.columns:
        ranked = ranked.sort_values([user_col, score_col, "_input_order"], ascending=[True, False, True])
    else:
        ranked = ranked.sort_values([user_col, "_input_order"], ascending=[True, True])
    ranked = ranked.drop_duplicates(subset=[user_col, item_col], keep="first")
    top_k = ranked.groupby(user_col, sort=False).head(k)
    return top_k.groupby(user_col)[item_col].apply(list).to_dict()


def ndcg_at_k(recommended_items, relevant_items, k):
    dcg = 0.0
    for rank, item_id in enumerate(recommended_items[:k], start=1):
        if item_id in relevant_items:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


def infer_catalog_size(recommendations, holdout, train, movies, item_col):
    item_sets = []
    for frame in [recommendations, holdout, train, movies]:
        if frame is not None and not frame.empty and item_col in frame.columns:
            item_sets.append(set(frame[item_col].dropna()))
    return len(set().union(*item_sets)) if item_sets else 0


def intra_list_diversities(ranked, evaluated_users, movies, item_col, genres_col):
    if movies is None or movies.empty or item_col not in movies.columns or genres_col not in movies.columns:
        return [0.0 for _ in evaluated_users]

    genre_lookup = movies.drop_duplicates(subset=[item_col]).set_index(item_col)[genres_col].apply(split_genres).to_dict()
    diversities = []
    for user_id in evaluated_users:
        recommended_items = ranked.get(user_id, [])
        genre_sets = [genre_lookup.get(item_id, set()) for item_id in recommended_items]
        pair_scores = []
        for left_index in range(len(genre_sets)):
            for right_index in range(left_index + 1, len(genre_sets)):
                pair_scores.append(1.0 - jaccard_similarity(genre_sets[left_index], genre_sets[right_index]))
        diversities.append(mean(pair_scores) if pair_scores else 0.0)
    return diversities


def novelty_score(ranked, evaluated_users, train, item_col):
    if train is None or train.empty or item_col not in train.columns:
        return 0.0

    item_counts = train[item_col].value_counts().to_dict()
    total_interactions = int(sum(item_counts.values()))
    catalog_size = max(len(item_counts), 1)
    scores = []
    for user_id in evaluated_users:
        for item_id in ranked.get(user_id, []):
            smoothed_probability = (item_counts.get(item_id, 0) + 1.0) / (total_interactions + catalog_size)
            scores.append(-math.log2(smoothed_probability))
    return mean(scores)


def serendipity_score(ranked, baseline_ranked, relevant_by_user, evaluated_users):
    if not baseline_ranked:
        return 0.0

    scores = []
    for user_id in evaluated_users:
        relevant_items = relevant_by_user[user_id]
        recommended_hits = set(ranked.get(user_id, [])) & relevant_items
        if not relevant_items:
            scores.append(0.0)
            continue
        baseline_items = set(baseline_ranked.get(user_id, []))
        unexpected_hits = recommended_hits - baseline_items
        scores.append(len(unexpected_hits) / float(len(relevant_items)))
    return mean(scores)


def split_genres(genres):
    if pd.isna(genres):
        return set()
    return {
        genre.strip()
        for genre in str(genres).split("|")
        if genre.strip() and genre.strip() != "(no genres listed)"
    }


def jaccard_similarity(left, right):
    if not left and not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def empty_top_n_metrics(k):
    return {
        "k": int(k),
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "ndcg_at_k": 0.0,
        "hit_rate_at_k": 0.0,
        "catalog_coverage": 0.0,
        "user_coverage": 0.0,
        "diversity": 0.0,
        "novelty": 0.0,
        "serendipity": 0.0,
        "evaluated_user_count": 0,
        "recommended_item_count": 0,
    }
