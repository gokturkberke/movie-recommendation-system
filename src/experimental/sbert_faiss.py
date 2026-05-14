from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recommenders import ensure_output_columns, filter_watched_movies, normalize_movie_ids, output_columns


SBERT_FAISS_SCORE_COLUMNS = ["similarity_score", "matched_seed_count"]
DEFAULT_INDEX_NAME = "sbert_faiss.index"
DEFAULT_EMBEDDINGS_NAME = "embeddings.npy"
DEFAULT_MOVIE_IDS_NAME = "movie_ids.csv"
DEFAULT_METADATA_NAME = "metadata.json"


@dataclass
class SbertFaissIndex:
    index: object
    embeddings: np.ndarray
    movie_ids: list
    metadata: dict


def require_sbert_faiss_dependencies():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required for SBERT+FAISS artifacts.") from exc

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required for SBERT+FAISS artifacts.") from exc

    return SentenceTransformer, faiss


def build_movie_text_corpus(movies, tags):
    if movies.empty or "movieId" not in movies.columns:
        return pd.DataFrame(columns=["movieId", "content"])

    movies_copy = movies.copy()
    if tags is None or tags.empty or "tag" not in tags.columns:
        tag_text = pd.DataFrame(columns=["movieId", "tag"])
    else:
        tags_copy = tags[["movieId", "tag"]].copy()
        tags_copy["tag"] = tags_copy["tag"].fillna("").astype(str)
        tag_text = tags_copy.groupby("movieId")["tag"].apply(lambda values: " ".join(values)).reset_index()

    corpus = movies_copy.merge(tag_text, on="movieId", how="left")
    for column in ["title", "genres", "tag"]:
        if column not in corpus.columns:
            corpus[column] = ""
        corpus[column] = corpus[column].fillna("").astype(str)

    corpus["content"] = (
        corpus["title"]
        + " "
        + corpus["genres"].str.replace("|", " ", regex=False)
        + " "
        + corpus["tag"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    return corpus[["movieId", "content"]].drop_duplicates(subset=["movieId"]).reset_index(drop=True)


def build_sbert_faiss_artifacts(
    movies,
    tags,
    output_dir,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=64,
    sample_size=None,
):
    SentenceTransformer, faiss = require_sbert_faiss_dependencies()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus = build_movie_text_corpus(movies, tags)
    if sample_size:
        corpus = corpus.head(int(sample_size)).copy()
    if corpus.empty:
        raise ValueError("No movie rows are available for SBERT+FAISS indexing.")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        corpus["content"].tolist(),
        batch_size=int(batch_size),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)

    index_path = output_path / DEFAULT_INDEX_NAME
    embeddings_path = output_path / DEFAULT_EMBEDDINGS_NAME
    movie_ids_path = output_path / DEFAULT_MOVIE_IDS_NAME
    metadata_path = output_path / DEFAULT_METADATA_NAME

    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings)
    corpus[["movieId"]].to_csv(movie_ids_path, index=False)

    metadata = {
        "model_name": model_name,
        "batch_size": int(batch_size),
        "row_count": int(len(corpus)),
        "embedding_dim": int(embeddings.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path),
        "embeddings_path": str(embeddings_path),
        "movie_ids_path": str(movie_ids_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return {
        "index_path": str(index_path),
        "embeddings_path": str(embeddings_path),
        "movie_ids_path": str(movie_ids_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def load_sbert_faiss_index(index_dir):
    _, faiss = require_sbert_faiss_dependencies()
    index_path = Path(index_dir)
    faiss_path = index_path / DEFAULT_INDEX_NAME
    embeddings_path = index_path / DEFAULT_EMBEDDINGS_NAME
    movie_ids_path = index_path / DEFAULT_MOVIE_IDS_NAME
    metadata_path = index_path / DEFAULT_METADATA_NAME

    missing = [
        str(path)
        for path in [faiss_path, embeddings_path, movie_ids_path, metadata_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing SBERT+FAISS artifact files: {missing}")

    index = faiss.read_index(str(faiss_path))
    embeddings = np.load(embeddings_path).astype("float32")
    movie_ids = pd.read_csv(movie_ids_path)["movieId"].dropna().astype(int).tolist()
    metadata = json.loads(metadata_path.read_text())
    return SbertFaissIndex(index=index, embeddings=embeddings, movie_ids=movie_ids, metadata=metadata)


def empty_sbert_faiss_recommendations(movies):
    return pd.DataFrame(columns=output_columns(movies) + SBERT_FAISS_SCORE_COLUMNS)


def sbert_faiss_recommendations_for_seed_ids(
    seed_movie_ids,
    sbert_index,
    movies_for_output,
    watched_movie_ids=None,
    top_n=10,
    search_k=None,
):
    seed_ids = normalize_movie_ids(seed_movie_ids)
    watched_ids = normalize_movie_ids(watched_movie_ids)
    watched_ids.update(seed_ids)
    if not seed_ids or sbert_index is None or sbert_index.embeddings.size == 0:
        return empty_sbert_faiss_recommendations(movies_for_output)

    positions_by_movie_id = {
        int(movie_id): position
        for position, movie_id in enumerate(sbert_index.movie_ids)
    }
    seed_positions = [
        positions_by_movie_id[movie_id]
        for movie_id in seed_ids
        if movie_id in positions_by_movie_id
    ]
    if not seed_positions:
        return empty_sbert_faiss_recommendations(movies_for_output)

    search_k = int(search_k or min(len(sbert_index.movie_ids), top_n + len(watched_ids) + 50))
    seed_vectors = sbert_index.embeddings[seed_positions].astype("float32")
    scores, indices = sbert_index.index.search(seed_vectors, search_k)

    rows = []
    for seed_row, seed_scores in zip(indices, scores):
        for index_position, score in zip(seed_row, seed_scores):
            if index_position < 0 or index_position >= len(sbert_index.movie_ids):
                continue
            movie_id = int(sbert_index.movie_ids[index_position])
            if movie_id in watched_ids:
                continue
            rows.append({"movieId": movie_id, "similarity_score": float(score)})

    if not rows:
        return empty_sbert_faiss_recommendations(movies_for_output)

    scores_df = pd.DataFrame(rows)
    aggregated_scores = (
        scores_df.groupby("movieId")
        .agg(
            similarity_score=("similarity_score", "max"),
            matched_seed_count=("similarity_score", "count"),
        )
        .reset_index()
    )
    recommendations = movies_for_output[movies_for_output["movieId"].isin(aggregated_scores["movieId"])].copy()
    recommendations = recommendations.merge(aggregated_scores, on="movieId", how="left")
    recommendations = filter_watched_movies(recommendations, watched_ids)
    if recommendations.empty:
        return empty_sbert_faiss_recommendations(movies_for_output)

    recommendations = recommendations.sort_values(
        ["similarity_score", "matched_seed_count", "movieId"],
        ascending=[False, False, True],
    ).head(top_n)
    return ensure_output_columns(recommendations.reset_index(drop=True), movies_for_output, SBERT_FAISS_SCORE_COLUMNS)


def build_sbert_faiss_watch_history_recommendations(
    train,
    movies,
    sbert_index,
    user_ids,
    top_n=10,
    positive_threshold=4.0,
):
    columns = ["userId"] + output_columns(movies) + SBERT_FAISS_SCORE_COLUMNS
    if train.empty or movies.empty or sbert_index is None or not user_ids:
        return pd.DataFrame(columns=columns)

    ratings = train.copy()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    output = []
    for user_id in pd.Series(user_ids).dropna().drop_duplicates().tolist():
        user_history = ratings[
            (ratings["userId"] == user_id)
            & (ratings["rating"] >= positive_threshold)
        ]
        seed_ids = user_history["movieId"].dropna().drop_duplicates().tolist()
        if not seed_ids:
            continue
        recommendations = sbert_faiss_recommendations_for_seed_ids(
            seed_ids,
            sbert_index,
            movies,
            watched_movie_ids=seed_ids,
            top_n=top_n,
        )
        if recommendations.empty:
            continue
        recommendations = recommendations.copy()
        recommendations["userId"] = user_id
        output.append(recommendations)

    if not output:
        return pd.DataFrame(columns=columns)
    return pd.concat(output, ignore_index=True)[columns]
