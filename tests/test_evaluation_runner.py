import unittest

import pandas as pd

from evaluation import segment_users_by_history
from evaluation_runner import build_metric_report, select_evaluation_user_ids


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


class TestSegmentUsersByHistory(unittest.TestCase):
    def _train_with_counts(self, count_by_user):
        rows = []
        for user_id, count in count_by_user.items():
            for movie_id in range(1, count + 1):
                rows.append({"userId": user_id, "movieId": movie_id, "rating": 5.0, "timestamp": movie_id})
        return pd.DataFrame(rows)

    def test_default_buckets_partition_users_by_count(self):
        train = self._train_with_counts({1: 3, 2: 12, 3: 80, 4: 250, 5: 500})
        segments = segment_users_by_history(train)
        self.assertEqual(segments["cold_0_10"], {1})
        self.assertEqual(segments["warm_10_50"], {2})
        self.assertEqual(segments["regular_50_200"], {3})
        self.assertEqual(segments["heavy_200_plus"], {4, 5})

    def test_open_upper_includes_boundary_value(self):
        train = self._train_with_counts({10: 200, 11: 199})
        segments = segment_users_by_history(train)
        self.assertIn(10, segments["heavy_200_plus"])
        self.assertIn(11, segments["regular_50_200"])
        self.assertNotIn(10, segments["regular_50_200"])


class TestBuildMetricReportSegmented(unittest.TestCase):
    def test_single_covering_segment_matches_aggregate(self):
        recommendations = pd.DataFrame(
            [
                {"userId": 10, "movieId": 1, "score": 0.9},
                {"userId": 10, "movieId": 2, "score": 0.8},
                {"userId": 20, "movieId": 1, "score": 0.7},
                {"userId": 20, "movieId": 3, "score": 0.6},
            ]
        )
        holdout = pd.DataFrame(
            [
                {"userId": 10, "movieId": 1, "rating": 5.0, "timestamp": 1},
                {"userId": 20, "movieId": 1, "rating": 5.0, "timestamp": 1},
            ]
        )
        train = pd.DataFrame(
            [
                {"userId": 10, "movieId": 99, "rating": 5.0, "timestamp": 0},
                {"userId": 20, "movieId": 99, "rating": 5.0, "timestamp": 0},
            ]
        )
        movies = pd.DataFrame([{"movieId": m, "genres": "Drama"} for m in [1, 2, 3, 99]])

        segment_user_ids = {"covers_all": {10, 20}}
        report = build_metric_report(
            recommendations,
            holdout,
            train,
            movies,
            k_values=[2],
            score_col="score",
            segment_user_ids=segment_user_ids,
        )

        self.assertIn("segments", report["2"])
        aggregate = {k: v for k, v in report["2"].items() if k != "segments"}
        segmented = report["2"]["segments"]["covers_all"]
        for metric in ("precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"):
            self.assertAlmostEqual(aggregate[metric], segmented[metric], places=10)

    def test_segment_with_no_positive_holdout_is_skipped(self):
        recommendations = pd.DataFrame(
            [{"userId": 10, "movieId": 1, "score": 0.9}]
        )
        holdout = pd.DataFrame(
            [{"userId": 10, "movieId": 1, "rating": 2.0, "timestamp": 1}]
        )
        train = pd.DataFrame([{"userId": 10, "movieId": 99, "rating": 5.0, "timestamp": 0}])
        movies = pd.DataFrame([{"movieId": m, "genres": "Drama"} for m in [1, 99]])
        report = build_metric_report(
            recommendations,
            holdout,
            train,
            movies,
            k_values=[2],
            score_col="score",
            positive_threshold=4.0,
            segment_user_ids={"only_low_rating": {10}},
        )
        self.assertNotIn("segments", report["2"])


if __name__ == "__main__":
    unittest.main()
