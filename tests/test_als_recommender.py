import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from experimental.als_recommender import (
    AlsArtifacts,
    als_recommendations_for_user,
    build_confidence_matrix,
)
from recommenders import BASE_OUTPUT_COLUMNS


class FakeAlsModel:
    def recommend(self, user_pos, user_items, N, filter_already_liked_items=True):
        self.last_call = {
            "user_pos": user_pos,
            "user_items_shape": user_items.shape,
            "N": N,
            "filter_already_liked_items": filter_already_liked_items,
        }
        return np.asarray([1, 2, 0], dtype=np.int64)[:N], np.asarray([0.9, 0.6, 0.2], dtype=np.float64)[:N]


def fixture_movies():
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "Known Seen", "genres": "Drama"},
            {"movieId": 2, "title": "Best Candidate", "genres": "Action"},
            {"movieId": 3, "title": "Second Candidate", "genres": "Comedy"},
        ]
    )


def fixture_artifacts():
    return AlsArtifacts(
        model=FakeAlsModel(),
        user_index={10: 0},
        item_index={1: 0, 2: 1, 3: 2},
        metadata={"factors": 64},
        user_items=sparse.csr_matrix([[1.0, 0.0, 0.0]], dtype=np.float32),
    )


class TestAlsRecommender(unittest.TestCase):
    def test_build_confidence_matrix_filters_and_scales_positive_ratings(self):
        ratings = pd.DataFrame(
            [
                {"userId": 10, "movieId": 1, "rating": 5.0},
                {"userId": 10, "movieId": 2, "rating": 3.5},
                {"userId": 11, "movieId": 3, "rating": 4.0},
            ]
        )

        user_items, user_index, item_index = build_confidence_matrix(
            ratings,
            positive_threshold=4.0,
            alpha=40.0,
        )

        self.assertEqual(user_items.shape, (2, 2))
        self.assertEqual(user_items.nnz, 2)
        self.assertEqual(user_index, {10: 0, 11: 1})
        self.assertEqual(item_index, {1: 0, 3: 1})
        values = sorted(user_items.data.tolist())
        self.assertEqual(values, [1.0, 41.0])

    def test_als_recommendations_exclude_watched_movies(self):
        artifacts = fixture_artifacts()
        recommendations = als_recommendations_for_user(
            10,
            artifacts,
            fixture_movies(),
            watched_movie_ids=[2],
            top_n=2,
        )

        self.assertEqual(recommendations["movieId"].tolist(), [3, 1])
        self.assertNotIn(2, recommendations["movieId"].tolist())
        self.assertEqual(recommendations.columns.tolist(), BASE_OUTPUT_COLUMNS + ["similarity_score"])
        self.assertEqual(artifacts.model.last_call["N"], 3)
        self.assertFalse(artifacts.model.last_call["filter_already_liked_items"])

    def test_als_recommendations_excludes_watched_via_post_filter(self):
        artifacts = fixture_artifacts()
        recommendations = als_recommendations_for_user(
            10,
            artifacts,
            fixture_movies(),
            watched_movie_ids=[3],
            top_n=2,
        )

        self.assertEqual(recommendations["movieId"].tolist(), [2, 1])
        self.assertNotIn(3, recommendations["movieId"].tolist())
        self.assertFalse(artifacts.model.last_call["filter_already_liked_items"])

    def test_als_recommendations_unknown_user_returns_empty_frame(self):
        recommendations = als_recommendations_for_user(
            999,
            fixture_artifacts(),
            fixture_movies(),
            top_n=2,
        )

        self.assertTrue(recommendations.empty)
        self.assertEqual(recommendations.columns.tolist(), BASE_OUTPUT_COLUMNS + ["similarity_score"])


if __name__ == "__main__":
    unittest.main()
