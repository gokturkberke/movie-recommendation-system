import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experimental.semantic_embeddings import (
    SemanticEmbeddingIndex,
    build_semantic_watch_history_recommendations,
    fit_semantic_embeddings,
    semantic_recommendations_for_seed_ids,
)


def fixture_movies():
    return pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Toy Story",
                "genres": "Adventure|Animation|Comedy",
                "title_for_matching": "toy story",
                "genres_for_matching": "adventure animation comedy",
            },
            {
                "movieId": 2,
                "title": "Toy Story 2",
                "genres": "Adventure|Animation|Comedy",
                "title_for_matching": "toy story 2",
                "genres_for_matching": "adventure animation comedy",
            },
            {
                "movieId": 3,
                "title": "Heat",
                "genres": "Action|Crime|Thriller",
                "title_for_matching": "heat",
                "genres_for_matching": "action crime thriller",
            },
        ]
    )


def fixture_tags():
    return pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "tag": "pixar toys friendship", "timestamp": 1},
            {"userId": 1, "movieId": 2, "tag": "pixar toys sequel", "timestamp": 2},
            {"userId": 2, "movieId": 3, "tag": "crime heist intense", "timestamp": 3},
        ]
    )


class TestSemanticEmbeddings(unittest.TestCase):
    def test_fit_semantic_embeddings_builds_dense_index(self):
        movies = fixture_movies()
        tags = fixture_tags()

        embedding_index = fit_semantic_embeddings(movies, tags, n_components=2, random_state=7)

        self.assertEqual(embedding_index.embeddings.shape[0], 3)
        self.assertEqual(embedding_index.component_count, 2)
        self.assertEqual(embedding_index.movies_with_content["movieId"].tolist(), [1, 2, 3])

    def test_seed_recommendations_exclude_watched_movie_ids(self):
        movies = fixture_movies()
        embeddings = normalize(
            np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            norm="l2",
        )
        embedding_index = SemanticEmbeddingIndex(
            embeddings=embeddings,
            movies_with_content=movies.copy(),
            vectorizer=None,
            svd_model=None,
            component_count=2,
        )

        recommendations = semantic_recommendations_for_seed_ids(
            [1],
            embedding_index,
            movies,
            watched_movie_ids=[1],
            top_n=2,
        )

        self.assertEqual(recommendations["movieId"].tolist(), [2, 3])
        self.assertNotIn(1, recommendations["movieId"].tolist())
        self.assertIn("similarity_score", recommendations.columns)

    def test_watch_history_recommendations_return_user_rows(self):
        train = pd.DataFrame(
            [
                {"userId": 10, "movieId": 1, "rating": 5.0, "timestamp": 1},
                {"userId": 10, "movieId": 3, "rating": 2.0, "timestamp": 2},
            ]
        )

        recommendations = build_semantic_watch_history_recommendations(
            train,
            fixture_movies(),
            fixture_tags(),
            user_ids=[10],
            top_n=2,
            n_components=2,
            random_state=7,
        )

        self.assertFalse(recommendations.empty)
        self.assertEqual(recommendations["userId"].drop_duplicates().tolist(), [10])
        self.assertNotIn(1, recommendations["movieId"].tolist())


if __name__ == "__main__":
    unittest.main()

