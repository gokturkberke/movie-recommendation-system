import unittest

import numpy as np
import pandas as pd

from experimental.sbert_faiss import (
    SbertFaissIndex,
    build_movie_text_corpus,
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


if __name__ == "__main__":
    unittest.main()
