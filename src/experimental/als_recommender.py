from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import sparse

from recommenders import ensure_output_columns, filter_watched_movies, normalize_movie_ids, output_columns


ALS_SCORE_COLUMNS = ["similarity_score"]
DEFAULT_MODEL_NAME = "als_model.npz"
DEFAULT_USER_ITEMS_NAME = "user_items.npz"
DEFAULT_USER_INDEX_NAME = "user_index.csv"
DEFAULT_ITEM_INDEX_NAME = "item_index.csv"
DEFAULT_METADATA_NAME = "metadata.json"


@dataclass
class AlsArtifacts:
    model: object
    user_index: dict
    item_index: dict
    metadata: dict
    user_items: sparse.csr_matrix


def require_implicit_dependency():
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError as exc:
        raise ImportError("implicit is required for Implicit ALS artifacts.") from exc
    return AlternatingLeastSquares


def build_confidence_matrix(ratings, positive_threshold=4.0, alpha=40.0):
    if ratings is None or ratings.empty:
        return sparse.csr_matrix((0, 0), dtype=np.float32), {}, {}
    required_columns = {"userId", "movieId", "rating"}
    if not required_columns.issubset(ratings.columns):
        return sparse.csr_matrix((0, 0), dtype=np.float32), {}, {}

    filtered = ratings[["userId", "movieId", "rating"]].copy()
    filtered["rating"] = pd.to_numeric(filtered["rating"], errors="coerce")
    filtered = filtered.dropna(subset=["userId", "movieId", "rating"])
    filtered = filtered[filtered["rating"] >= float(positive_threshold)].copy()
    if filtered.empty:
        return sparse.csr_matrix((0, 0), dtype=np.float32), {}, {}

    filtered["userId"] = filtered["userId"].astype(int)
    filtered["movieId"] = filtered["movieId"].astype(int)
    user_ids = sorted(filtered["userId"].unique().tolist())
    movie_ids = sorted(filtered["movieId"].unique().tolist())
    user_index = {int(user_id): position for position, user_id in enumerate(user_ids)}
    item_index = {int(movie_id): position for position, movie_id in enumerate(movie_ids)}

    rows = filtered["userId"].map(user_index).to_numpy(dtype=np.int32)
    cols = filtered["movieId"].map(item_index).to_numpy(dtype=np.int32)
    confidence = 1.0 + float(alpha) * (filtered["rating"].astype(float) - float(positive_threshold)).clip(lower=0.0)
    user_items = sparse.coo_matrix(
        (confidence.to_numpy(dtype=np.float32), (rows, cols)),
        shape=(len(user_index), len(item_index)),
        dtype=np.float32,
    ).tocsr()
    user_items.sum_duplicates()
    return user_items, user_index, item_index


def train_als_model(confidence_matrix, factors=64, regularization=0.01, iterations=20, use_gpu=False):
    AlternatingLeastSquares = require_implicit_dependency()
    model = AlternatingLeastSquares(
        factors=int(factors),
        regularization=float(regularization),
        iterations=int(iterations),
        use_gpu=bool(use_gpu),
    )
    model.fit(confidence_matrix, show_progress=False)
    return model


def _index_frame(index, id_column):
    return pd.DataFrame(
        [{id_column: int(identifier), "position": int(position)} for identifier, position in index.items()]
    ).sort_values("position")


def save_als_artifacts(model, user_index, item_index, user_items, output_dir, metadata=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / DEFAULT_MODEL_NAME
    user_items_path = output_path / DEFAULT_USER_ITEMS_NAME
    user_index_path = output_path / DEFAULT_USER_INDEX_NAME
    item_index_path = output_path / DEFAULT_ITEM_INDEX_NAME
    metadata_path = output_path / DEFAULT_METADATA_NAME

    model.save(model_path)
    sparse.save_npz(user_items_path, user_items)
    _index_frame(user_index, "userId").to_csv(user_index_path, index=False)
    _index_frame(item_index, "movieId").to_csv(item_index_path, index=False)

    payload = dict(metadata or {})
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_count": int(len(user_index)),
            "item_count": int(len(item_index)),
            "model_path": str(model_path),
            "user_items_path": str(user_items_path),
            "user_index_path": str(user_index_path),
            "item_index_path": str(item_index_path),
        }
    )
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {
        "model_path": str(model_path),
        "user_items_path": str(user_items_path),
        "user_index_path": str(user_index_path),
        "item_index_path": str(item_index_path),
        "metadata_path": str(metadata_path),
        "metadata": payload,
    }


def _load_index(path, id_column):
    frame = pd.read_csv(path)
    if id_column not in frame.columns or "position" not in frame.columns:
        raise ValueError(f"{path} must contain {id_column} and position columns.")
    return {
        int(row[id_column]): int(row["position"])
        for _, row in frame[[id_column, "position"]].dropna().iterrows()
    }


def load_als_artifacts(input_dir):
    input_path = Path(input_dir)
    model_path = input_path / DEFAULT_MODEL_NAME
    user_items_path = input_path / DEFAULT_USER_ITEMS_NAME
    user_index_path = input_path / DEFAULT_USER_INDEX_NAME
    item_index_path = input_path / DEFAULT_ITEM_INDEX_NAME
    metadata_path = input_path / DEFAULT_METADATA_NAME

    missing = [
        str(path)
        for path in [model_path, user_items_path, user_index_path, item_index_path, metadata_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing ALS artifact files: {missing}")

    AlternatingLeastSquares = require_implicit_dependency()
    model = AlternatingLeastSquares(factors=1).load(model_path)
    user_items = sparse.load_npz(user_items_path).tocsr()
    user_index = _load_index(user_index_path, "userId")
    item_index = _load_index(item_index_path, "movieId")
    metadata = json.loads(metadata_path.read_text())
    return AlsArtifacts(
        model=model,
        user_index=user_index,
        item_index=item_index,
        metadata=metadata,
        user_items=user_items,
    )


def empty_als_recommendations(movies):
    return pd.DataFrame(columns=output_columns(movies) + ALS_SCORE_COLUMNS)


def als_recommendations_for_user(
    user_id,
    artifacts,
    movies_for_output,
    watched_movie_ids=None,
    top_n=10,
):
    if artifacts is None or artifacts.model is None or not artifacts.item_index:
        return empty_als_recommendations(movies_for_output)
    try:
        user_position = artifacts.user_index[int(user_id)]
    except (KeyError, TypeError, ValueError):
        return empty_als_recommendations(movies_for_output)

    watched_ids = normalize_movie_ids(watched_movie_ids)
    requested_n = min(len(artifacts.item_index), int(top_n) + len(watched_ids))
    if requested_n <= 0:
        return empty_als_recommendations(movies_for_output)

    # user_items was built from the full pre-split rating matrix, so it
    # contains evaluation-time holdout interactions; rely on the post-hoc
    # filter_watched_movies (train-only) for exclusion instead.
    item_positions, scores = artifacts.model.recommend(
        int(user_position),
        artifacts.user_items[int(user_position)],
        N=requested_n,
        filter_already_liked_items=False,
    )
    item_positions = np.asarray(item_positions).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if item_positions.size == 0:
        return empty_als_recommendations(movies_for_output)

    movie_id_by_position = {
        int(position): int(movie_id)
        for movie_id, position in artifacts.item_index.items()
    }
    score_rows = [
        {
            "movieId": movie_id_by_position[int(position)],
            "similarity_score": float(score),
        }
        for position, score in zip(item_positions, scores)
        if int(position) in movie_id_by_position
    ]
    if not score_rows:
        return empty_als_recommendations(movies_for_output)

    scores_df = pd.DataFrame(score_rows)
    recommendations = movies_for_output[movies_for_output["movieId"].isin(scores_df["movieId"])].copy()
    recommendations = recommendations.merge(scores_df, on="movieId", how="left")
    recommendations = filter_watched_movies(recommendations, watched_ids)
    if recommendations.empty:
        return empty_als_recommendations(movies_for_output)

    recommendations = recommendations.sort_values(
        ["similarity_score", "movieId"],
        ascending=[False, True],
    ).head(int(top_n))
    return ensure_output_columns(
        recommendations.reset_index(drop=True),
        movies_for_output,
        ALS_SCORE_COLUMNS,
    )
