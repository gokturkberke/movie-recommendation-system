import unittest

import numpy as np
import pandas as pd

from experimental.lightfm_recommender import (
    build_interaction_matrix,
    lightfm_recommendations_for_user,
)
from recommenders import BASE_OUTPUT_COLUMNS


class FakeLightfmModel:
    def __init__(self, scores_by_position):
        self.scores_by_position = scores_by_position

    def predict(self, user_ids, item_ids):
        return np.asarray(
            [self.scores_by_position[int(position)] for position in item_ids],
            dtype=np.float64,
        )


class FakeLightfmArtifacts:
    def __init__(self):
        self.model = FakeLightfmModel({0: 0.2, 1: 0.9, 2: 0.6})
        self.user_index = {10: 0}
        self.item_index = {1: 0, 2: 1, 3: 2}
        self.metadata = {"loss": "warp"}


def fixture_movies():
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "Known Seen", "genres": "Drama"},
            {"movieId": 2, "title": "Best Candidate", "genres": "Action"},
            {"movieId": 3, "title": "Second Candidate", "genres": "Comedy"},
        ]
    )


class TestLightfmRecommender(unittest.TestCase):
    def test_build_interaction_matrix_filters_below_threshold(self):
        ratings = pd.DataFrame(
            [
                {"userId": 10, "movieId": 1, "rating": 5.0},
                {"userId": 10, "movieId": 2, "rating": 3.5},
                {"userId": 11, "movieId": 3, "rating": 4.0},
            ]
        )

        interactions, user_index, item_index = build_interaction_matrix(
            ratings,
            positive_threshold=4.0,
        )

        self.assertEqual(interactions.shape, (2, 2))
        self.assertEqual(interactions.nnz, 2)
        self.assertEqual(user_index, {10: 0, 11: 1})
        self.assertEqual(item_index, {1: 0, 3: 1})

    def test_lightfm_recommendations_exclude_watched_movies(self):
        recommendations = lightfm_recommendations_for_user(
            10,
            FakeLightfmArtifacts(),
            fixture_movies(),
            watched_movie_ids=[2],
            top_n=2,
        )

        self.assertEqual(recommendations["movieId"].tolist(), [3, 1])
        self.assertNotIn(2, recommendations["movieId"].tolist())
        self.assertEqual(recommendations.columns.tolist(), BASE_OUTPUT_COLUMNS + ["similarity_score"])

    def test_lightfm_recommendations_unknown_user_returns_empty_frame(self):
        recommendations = lightfm_recommendations_for_user(
            999,
            FakeLightfmArtifacts(),
            fixture_movies(),
            top_n=2,
        )

        self.assertTrue(recommendations.empty)
        self.assertEqual(recommendations.columns.tolist(), BASE_OUTPUT_COLUMNS + ["similarity_score"])

    def test_build_interaction_matrix_excludes_specified_pairs(self):
        ratings = pd.DataFrame(
            [
                {"userId": 10, "movieId": 1, "rating": 5.0},
                {"userId": 10, "movieId": 2, "rating": 4.5},
                {"userId": 11, "movieId": 1, "rating": 5.0},
                {"userId": 11, "movieId": 3, "rating": 4.0},
            ]
        )

        interactions, user_index, item_index = build_interaction_matrix(
            ratings,
            positive_threshold=4.0,
            exclude_pairs={(10, 1), (11, 3)},
        )

        self.assertEqual(interactions.nnz, 2)
        self.assertEqual(set(user_index.keys()), {10, 11})
        self.assertEqual(set(item_index.keys()), {1, 2})

    def test_build_interaction_matrix_none_exclude_is_no_op(self):
        ratings = pd.DataFrame(
            [
                {"userId": 10, "movieId": 1, "rating": 5.0},
                {"userId": 11, "movieId": 3, "rating": 4.0},
            ]
        )

        baseline = build_interaction_matrix(ratings, positive_threshold=4.0)
        none_result = build_interaction_matrix(ratings, positive_threshold=4.0, exclude_pairs=None)
        empty_result = build_interaction_matrix(ratings, positive_threshold=4.0, exclude_pairs=set())

        self.assertEqual(baseline[0].nnz, none_result[0].nnz)
        self.assertEqual(baseline[0].nnz, empty_result[0].nnz)
        self.assertEqual(baseline[1], none_result[1])
        self.assertEqual(baseline[1], empty_result[1])


if __name__ == "__main__":
    unittest.main()
