import tempfile
import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_access import latest_release_info, load_movies, load_surprise_model
from recommenders import (
    build_tfidf_matrix,
    pick_random_movie,
    recommend_based_on_watch_history_content,
    recommend_by_mood,
    recommend_for_user,
    recommend_similar_movies,
)
from tmdb_client import get_tmdb_id


class FakePrediction:
    def __init__(self, est):
        self.est = est


class FakeSvdModel:
    def predict(self, uid, iid):
        scores = {
            2: 4.9,
            3: 3.8,
            4: 4.5,
        }
        return FakePrediction(scores.get(iid, 2.5))


def fixture_movies():
    return pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Toy Story",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
                "title_for_matching": "toy story",
                "genres_for_matching": "adventure animation children comedy fantasy",
                "genre_comedy": 1,
                "genre_action": 0,
                "tmdbId": 862,
            },
            {
                "movieId": 2,
                "title": "Toy Story 2",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
                "title_for_matching": "toy story 2",
                "genres_for_matching": "adventure animation children comedy fantasy",
                "genre_comedy": 1,
                "genre_action": 0,
                "tmdbId": 863,
            },
            {
                "movieId": 3,
                "title": "Heat",
                "genres": "Action|Crime|Thriller",
                "title_for_matching": "heat",
                "genres_for_matching": "action crime thriller",
                "genre_comedy": 0,
                "genre_action": 1,
                "tmdbId": pd.NA,
            },
            {
                "movieId": 4,
                "title": "Casino",
                "genres": "Crime|Drama",
                "title_for_matching": "casino",
                "genres_for_matching": "crime drama",
                "genre_comedy": 0,
                "genre_action": 0,
                "tmdbId": pd.NA,
            },
        ]
    )


def fixture_tags():
    return pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "tag": "toys friendship pixar", "timestamp": 1},
            {"userId": 1, "movieId": 2, "tag": "toys sequel pixar", "timestamp": 1},
            {"userId": 2, "movieId": 3, "tag": "crime heist intense", "timestamp": 1},
            {"userId": 2, "movieId": 4, "tag": "crime mafia drama", "timestamp": 1},
        ]
    )


class TestMovieRecommendations(unittest.TestCase):
    def setUp(self):
        self.movies = fixture_movies()
        self.tags = fixture_tags()
        self.tfidf_matrix, self.vectorizer, self.movies_with_content = build_tfidf_matrix(
            self.movies.copy(),
            self.tags.copy(),
        )

    def test_content_based_fuzzy_match_and_watched_exclusion(self):
        recommendations, matched_title = recommend_similar_movies(
            "toy storie",
            self.movies_with_content,
            self.tfidf_matrix,
            self.movies,
            watched_titles={"Toy Story"},
            top_n=3,
        )

        self.assertEqual(matched_title, "Toy Story")
        self.assertNotIn("Toy Story", recommendations["title"].tolist())
        self.assertIn("Toy Story 2", recommendations["title"].tolist())

    def test_movie_title_suggestions_handle_common_typo(self):
        movies = pd.concat(
            [
                self.movies,
                pd.DataFrame(
                    [
                        {
                            "movieId": 5,
                            "title": "Joker (2019)",
                            "genres": "Crime|Drama|Thriller",
                            "title_for_matching": "joker",
                            "genres_for_matching": "crime drama thriller",
                        },
                        {
                            "movieId": 6,
                            "title": "OKA! (2011)",
                            "genres": "Drama",
                            "title_for_matching": "oka",
                            "genres_for_matching": "drama",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

        from recommenders import suggest_movie_titles

        suggestions = suggest_movie_titles("jokar", movies, limit=5)

        self.assertFalse(suggestions.empty)
        self.assertEqual(suggestions.iloc[0]["title"], "Joker (2019)")
        self.assertNotIn("OKA! (2011)", suggestions["title"].tolist())

    def test_mood_recommendations_filter_to_mapped_genres(self):
        recommendations = recommend_by_mood("happy", self.movies, watched_titles=set(), top_n=2)

        self.assertFalse(recommendations.empty)
        for genres in recommendations["genres"]:
            self.assertTrue(any(genre in genres for genre in ["Comedy", "Family", "Animation", "Romance"]))

    def test_watch_history_recommendations_are_unique_and_unwatched(self):
        recommendations = recommend_based_on_watch_history_content(
            ["Toy Story"],
            self.movies_with_content,
            self.tfidf_matrix,
            self.movies,
            top_n=3,
        )

        self.assertNotIn("Toy Story", recommendations["title"].tolist())
        self.assertEqual(len(recommendations["movieId"]), recommendations["movieId"].nunique())

    def test_svd_recommendations_use_fake_model_without_loading_real_model(self):
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 1, "rating": 5.0},
                {"userId": 1, "movieId": 3, "rating": 4.0},
            ]
        )

        recommendations = recommend_for_user(
            1,
            FakeSvdModel(),
            self.movies,
            ratings,
            watched_titles=set(),
            top_n=2,
        )

        self.assertEqual(recommendations["movieId"].tolist(), [2, 4])
        self.assertNotIn(1, recommendations["movieId"].tolist())

    def test_missing_model_returns_error_without_import_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model, error = load_surprise_model(Path(tmp_dir) / "missing.pkl")

        self.assertIsNone(model)
        self.assertIn("not found", error)

    def test_load_movies_restores_year_in_display_title_from_existing_clean_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            movies_path = Path(tmp_dir) / "movies_clean.csv"
            pd.DataFrame(
                [
                    {
                        "movieId": 1,
                        "title": "Sabrina",
                        "genres": "Comedy|Romance",
                        "title_original": "Sabrina (1954)",
                        "title_for_matching": "sabrina",
                        "genres_for_matching": "comedy romance",
                    },
                    {
                        "movieId": 2,
                        "title": "Sabrina",
                        "genres": "Comedy|Romance",
                        "title_original": "Sabrina (1995)",
                        "title_for_matching": "sabrina",
                        "genres_for_matching": "comedy romance",
                    },
                ]
            ).to_csv(movies_path, index=False)

            movies = load_movies(tmp_dir)

        self.assertEqual(movies["title"].tolist(), ["Sabrina (1954)", "Sabrina (1995)"])
        self.assertEqual(movies["title_display"].tolist(), ["Sabrina (1954)", "Sabrina (1995)"])
        self.assertEqual(movies["title_for_matching"].tolist(), ["sabrina", "sabrina"])

    def test_latest_release_info_uses_release_year_from_titles(self):
        movies = pd.DataFrame(
            [
                {"movieId": 1, "title": "Older Movie (1995)", "genres": "Drama"},
                {"movieId": 2, "title": "New Movie (2019)", "genres": "Drama"},
                {"movieId": 3, "title": "Another New Movie (2019)", "genres": "Comedy"},
            ]
        )

        latest_year, latest_count, latest_movies = latest_release_info(movies)

        self.assertEqual(latest_year, 2019)
        self.assertEqual(latest_count, 2)
        self.assertEqual(set(latest_movies["title"]), {"New Movie (2019)", "Another New Movie (2019)"})

    def test_tmdb_id_falls_back_to_links(self):
        row = pd.Series({"movieId": 3, "title": "Heat"})
        links = pd.DataFrame([{"movieId": 3, "tmdbId": 949}])

        self.assertEqual(get_tmdb_id(row, links), 949)

    def test_random_movie_returns_none_for_empty_filter(self):
        movie = pick_random_movie(self.movies, selected_genres=["Documentary"])

        self.assertIsNone(movie)

    def test_app_import_smoke(self):
        import app  # noqa: F401


if __name__ == "__main__":
    unittest.main()
