import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from experimental.sbert_faiss import (
    DEFAULT_EMBEDDINGS_NAME,
    DEFAULT_INDEX_NAME,
    DEFAULT_METADATA_NAME,
    DEFAULT_MOVIE_IDS_NAME,
    SbertFaissIndex,
    build_movie_text_corpus,
    load_sbert_faiss_index,
    sbert_faiss_recommendations_for_seed_ids,
)


class FakeFaissIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def search(self, queries, k):
        scores = queries @ self.embeddings.T
        order = np.argsort(-scores, axis=1)[:, :k]
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        return sorted_scores.astype("float32"), order.astype("int64")


def fixture_movies():
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "Space Adventure", "genres": "Adventure|Sci-Fi"},
            {"movieId": 2, "title": "Deep Space", "genres": "Sci-Fi|Drama"},
            {"movieId": 3, "title": "Romantic Comedy", "genres": "Comedy|Romance"},
        ]
    )


def fixture_tags():
    return pd.DataFrame(
        [
            {"movieId": 1, "tag": "space travel"},
            {"movieId": 2, "tag": "spaceship"},
            {"movieId": 3, "tag": "date night"},
        ]
    )


class TestSbertFaiss(unittest.TestCase):
    def test_build_movie_text_corpus_merges_tags_and_metadata(self):
        corpus = build_movie_text_corpus(fixture_movies(), fixture_tags())

        self.assertEqual(corpus["movieId"].tolist(), [1, 2, 3])
        self.assertIn("Space Adventure", corpus.iloc[0]["content"])
        self.assertIn("Adventure Sci-Fi", corpus.iloc[0]["content"])
        self.assertIn("space travel", corpus.iloc[0]["content"])

    def test_sbert_faiss_recommendations_exclude_watched_seed(self):
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype="float32",
        )
        index = SbertFaissIndex(
            index=FakeFaissIndex(embeddings),
            embeddings=embeddings,
            movie_ids=[1, 2, 3],
            metadata={"model_name": "fake"},
        )

        recommendations = sbert_faiss_recommendations_for_seed_ids(
            [1],
            index,
            fixture_movies(),
            watched_movie_ids=[1],
            top_n=2,
        )

        self.assertEqual(recommendations.iloc[0]["movieId"], 2)
        self.assertNotIn(1, recommendations["movieId"].tolist())
        self.assertIn("similarity_score", recommendations.columns)
        self.assertIn("matched_seed_count", recommendations.columns)

    def test_load_existing_index_does_not_require_sentence_transformers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            embeddings = np.asarray([[1.0, 0.0]], dtype="float32")
            (index_dir / DEFAULT_INDEX_NAME).write_bytes(b"fake-index")
            np.save(index_dir / DEFAULT_EMBEDDINGS_NAME, embeddings)
            pd.DataFrame({"movieId": [1]}).to_csv(index_dir / DEFAULT_MOVIE_IDS_NAME, index=False)
            (index_dir / DEFAULT_METADATA_NAME).write_text('{"model_name": "fake"}')

            fake_index = object()
            fake_faiss = Mock()
            fake_faiss.read_index.return_value = fake_index

            with (
                patch.dict(sys.modules, {"sentence_transformers": None}),
                patch("experimental.sbert_faiss.require_faiss_dependency", return_value=fake_faiss),
            ):
                loaded = load_sbert_faiss_index(index_dir)

            self.assertIs(loaded.index, fake_index)
            self.assertEqual(loaded.movie_ids, [1])
            np.testing.assert_array_equal(loaded.embeddings, embeddings)
            fake_faiss.read_index.assert_called_once_with(str(index_dir / DEFAULT_INDEX_NAME))


if __name__ == "__main__":
    unittest.main()
