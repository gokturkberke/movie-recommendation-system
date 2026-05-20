import unittest

import pandas as pd

from evaluation_runner import select_evaluation_user_ids


def fixture_ratings(eligible_user_ids):
    rows = []
    for user_id in eligible_user_ids:
        for movie_id in range(1, 6):
            rows.append({"userId": user_id, "movieId": movie_id, "rating": 5.0, "timestamp": movie_id})
    return pd.DataFrame(rows)


class TestSelectEvaluationUserIds(unittest.TestCase):
    eligible = [10, 20, 30, 40, 50, 60, 70]

    def test_default_is_deterministic_first_n(self):
        ratings = fixture_ratings(self.eligible)
        result = select_evaluation_user_ids(
            ratings,
            max_users=3,
            min_interactions=2,
            holdout_count=1,
        )
        self.assertEqual(result, [10, 20, 30])

    def test_seed_returns_stable_random_sample(self):
        ratings = fixture_ratings(self.eligible)
        result_a = select_evaluation_user_ids(
            ratings,
            max_users=3,
            min_interactions=2,
            holdout_count=1,
            random_seed=42,
        )
        result_b = select_evaluation_user_ids(
            ratings,
            max_users=3,
            min_interactions=2,
            holdout_count=1,
            random_seed=42,
        )
        self.assertEqual(result_a, result_b)
        self.assertEqual(len(result_a), 3)
        self.assertEqual(len(set(result_a)), 3)
        self.assertTrue(set(result_a).issubset(set(self.eligible)))
        self.assertNotEqual(result_a, [10, 20, 30])

    def test_different_seeds_produce_different_samples(self):
        ratings = fixture_ratings(self.eligible)
        result_42 = select_evaluation_user_ids(
            ratings,
            max_users=3,
            min_interactions=2,
            holdout_count=1,
            random_seed=42,
        )
        result_7 = select_evaluation_user_ids(
            ratings,
            max_users=3,
            min_interactions=2,
            holdout_count=1,
            random_seed=7,
        )
        self.assertNotEqual(result_42, result_7)
        self.assertTrue(set(result_42).issubset(set(self.eligible)))
        self.assertTrue(set(result_7).issubset(set(self.eligible)))


if __name__ == "__main__":
    unittest.main()
