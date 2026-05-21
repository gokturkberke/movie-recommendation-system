from dataclasses import dataclass
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recommenders import ensure_output_columns, filter_watched_movies, output_columns


LIGHTFM_SCORE_COLUMNS = ["similarity_score"]
DEFAULT_MODEL_NAME = "lightfm_model.pkl"
DEFAULT_USER_INDEX_NAME = "user_index.csv"
DEFAULT_ITEM_INDEX_NAME = "item_index.csv"
DEFAULT_METADATA_NAME = "metadata.json"


@dataclass
class LightfmArtifacts:
    model: object
    user_index: dict
    item_index: dict
    metadata: dict


def require_lightfm_dependency():
    try:
        from lightfm import LightFM
    except ImportError as exc:
        raise ImportError("lightfm is required for LightFM WARP artifacts.") from exc
    return LightFM


def build_interaction_matrix(ratings, positive_threshold=4.0, exclude_pairs=None):
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

    if exclude_pairs:
        excl_list = [(int(u), int(m)) for u, m in exclude_pairs]
        if excl_list:
            excl_index = pd.MultiIndex.from_tuples(excl_list)
            filtered_index = pd.MultiIndex.from_arrays([filtered["userId"], filtered["movieId"]])
            filtered = filtered[~filtered_index.isin(excl_index)].copy()
            if filtered.empty:
                return sparse.csr_matrix((0, 0), dtype=np.float32), {}, {}
    user_ids = sorted(filtered["userId"].unique().tolist())
    movie_ids = sorted(filtered["movieId"].unique().tolist())
    user_index = {int(user_id): position for position, user_id in enumerate(user_ids)}
    item_index = {int(movie_id): position for position, movie_id in enumerate(movie_ids)}

    rows = filtered["userId"].map(user_index).to_numpy(dtype=np.int32)
    cols = filtered["movieId"].map(item_index).to_numpy(dtype=np.int32)
    data = np.ones(len(filtered), dtype=np.float32)
    interactions = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(user_index), len(item_index)),
        dtype=np.float32,
    ).tocsr()
    interactions.sum_duplicates()
    return interactions, user_index, item_index


def train_lightfm_model(interactions, no_components=64, loss="warp", epochs=20, num_threads=4):
    LightFM = require_lightfm_dependency()
    model = LightFM(no_components=int(no_components), loss=str(loss))
    model.fit(interactions, epochs=int(epochs), num_threads=int(num_threads))
    return model


def _index_frame(index, id_column):
    return pd.DataFrame(
        [{id_column: int(identifier), "position": int(position)} for identifier, position in index.items()]
    ).sort_values("position")


def save_lightfm_artifacts(model, user_index, item_index, output_dir, metadata=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / DEFAULT_MODEL_NAME
    user_index_path = output_path / DEFAULT_USER_INDEX_NAME
    item_index_path = output_path / DEFAULT_ITEM_INDEX_NAME
    metadata_path = output_path / DEFAULT_METADATA_NAME

    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    _index_frame(user_index, "userId").to_csv(user_index_path, index=False)
    _index_frame(item_index, "movieId").to_csv(item_index_path, index=False)

    payload = dict(metadata or {})
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_count": int(len(user_index)),
            "item_count": int(len(item_index)),
            "model_path": str(model_path),
            "user_index_path": str(user_index_path),
            "item_index_path": str(item_index_path),
        }
    )
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {
        "model_path": str(model_path),
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


def load_lightfm_artifacts(input_dir):
    input_path = Path(input_dir)
    model_path = input_path / DEFAULT_MODEL_NAME
    user_index_path = input_path / DEFAULT_USER_INDEX_NAME
    item_index_path = input_path / DEFAULT_ITEM_INDEX_NAME
    metadata_path = input_path / DEFAULT_METADATA_NAME

    missing = [
        str(path)
        for path in [model_path, user_index_path, item_index_path, metadata_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing LightFM artifact files: {missing}")

    require_lightfm_dependency()
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    user_index = _load_index(user_index_path, "userId")
    item_index = _load_index(item_index_path, "movieId")
    metadata = json.loads(metadata_path.read_text())
    return LightfmArtifacts(
        model=model,
        user_index=user_index,
        item_index=item_index,
        metadata=metadata,
    )


def empty_lightfm_recommendations(movies):
    return pd.DataFrame(columns=output_columns(movies) + LIGHTFM_SCORE_COLUMNS)


def lightfm_recommendations_for_user(
    user_id,
    artifacts,
    movies_for_output,
    watched_movie_ids=None,
    top_n=10,
):
    if artifacts is None or artifacts.model is None or not artifacts.item_index:
        return empty_lightfm_recommendations(movies_for_output)
    try:
        user_position = artifacts.user_index[int(user_id)]
    except (KeyError, TypeError, ValueError):
        return empty_lightfm_recommendations(movies_for_output)

    movie_positions = np.asarray(list(artifacts.item_index.values()), dtype=np.int32)
    movie_ids = np.asarray(list(artifacts.item_index.keys()), dtype=np.int64)
    user_positions = np.full(len(movie_positions), int(user_position), dtype=np.int32)
    scores = artifacts.model.predict(user_positions, movie_positions)
    scores_df = pd.DataFrame(
        {
            "movieId": movie_ids.astype(int),
            "similarity_score": np.asarray(scores, dtype=np.float64),
        }
    )

    recommendations = movies_for_output[movies_for_output["movieId"].isin(scores_df["movieId"])].copy()
    recommendations = recommendations.merge(scores_df, on="movieId", how="left")
    recommendations = filter_watched_movies(recommendations, watched_movie_ids)
    if recommendations.empty:
        return empty_lightfm_recommendations(movies_for_output)

    recommendations = recommendations.sort_values(
        ["similarity_score", "movieId"],
        ascending=[False, True],
    ).head(int(top_n))
    return ensure_output_columns(
        recommendations.reset_index(drop=True),
        movies_for_output,
        LIGHTFM_SCORE_COLUMNS,
    )
