"""AppTest-driven integration coverage for the Streamlit random page.

Slower than unit tests because each AppTest.from_file boots the full
data pipeline. Kept in a separate module so unit-test iteration speed
is not affected.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import config as cfg


APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app.py")
RANDOM_MENU = cfg.MENU_ITEMS[3]


def _go_random(at):
    for radio in at.sidebar.radio:
        if radio.key == "main_menu_choice":
            radio.set_value(RANDOM_MENU)
            break
    at.run()


def _click(at, key):
    for button in at.button:
        if button.key == key:
            button.click()
            return True
    return False


def _session_value(at, key):
    if key in at.session_state:
        return at.session_state[key]
    return None


class TestRandomPageIntegration(unittest.TestCase):
    def setUp(self):
        self.at = AppTest.from_file(APP_PATH, default_timeout=120)
        self.at.run()
        _go_random(self.at)

    def test_random_invalid_pick_cleared_when_watched_added(self):
        self.assertTrue(_click(self.at, "random_pick"))
        self.at.run()
        pick = _session_value(self.at, "random_pick_movie")
        self.assertIsNotNone(pick)
        picked_id = int(pick["movieId"])

        self.at.session_state["watched_movie_ids"] = {picked_id}
        self.at.run()

        self.assertIsNone(_session_value(self.at, "random_pick_movie"))
        movie_subheaders = [s.value for s in self.at.subheader if s.value.startswith("1. ")]
        self.assertEqual(movie_subheaders, [])
        captions = [c.value for c in self.at.caption if c.value]
        self.assertTrue(
            any("filtered out by your watch history or genre selection" in c for c in captions),
            f"Expected stale-pick caption; saw: {captions}",
        )

    def test_random_invalid_pick_cleared_when_genre_changes(self):
        self.assertTrue(_click(self.at, "random_pick"))
        self.at.run()
        pick = _session_value(self.at, "random_pick_movie")
        self.assertIsNotNone(pick)
        picked_genres = set(str(pick["genres"]).split("|"))

        candidate_genres = ["Documentary", "War", "Western", "Film-Noir", "IMAX", "Musical"]
        incompatible = next((g for g in candidate_genres if g not in picked_genres), None)
        self.assertIsNotNone(
            incompatible,
            f"Test fixture too broad: pick already has every candidate genre ({picked_genres}).",
        )

        for multiselect in self.at.multiselect:
            if multiselect.key == "random_genres":
                multiselect.set_value([incompatible])
                break
        self.at.run()

        self.assertIsNone(_session_value(self.at, "random_pick_movie"))
        movie_subheaders = [s.value for s in self.at.subheader if s.value.startswith("1. ")]
        self.assertEqual(movie_subheaders, [])

    def test_random_pick_another_excludes_current_id(self):
        row_a = pd.Series({"movieId": 9001, "title": "Mock A (1999)", "genres": "Drama", "tmdbId": pd.NA})
        row_b = pd.Series({"movieId": 9002, "title": "Mock B (2001)", "genres": "Drama", "tmdbId": pd.NA})
        fake_pick = MagicMock(side_effect=[row_a, row_b, row_b])

        with patch("recommenders.pick_random_movie", fake_pick):
            at = AppTest.from_file(APP_PATH, default_timeout=120)
            at.run()
            _go_random(at)

            self.assertTrue(_click(at, "random_pick"))
            at.run()
            self.assertEqual(_session_value(at, "random_pick_movie")["movieId"], 9001)

            self.assertTrue(_click(at, "random_pick_another"))
            at.run()
            self.assertEqual(_session_value(at, "random_pick_movie")["movieId"], 9002)

        self.assertGreaterEqual(fake_pick.call_count, 2)
        second_call_kwargs = fake_pick.call_args_list[1].kwargs
        self.assertEqual(second_call_kwargs.get("excluded_movie_ids"), {9001})

    def test_random_pick_another_warns_when_no_alternative(self):
        row_a = pd.Series({"movieId": 9001, "title": "Mock A (1999)", "genres": "Drama", "tmdbId": pd.NA})
        fake_pick = MagicMock(side_effect=[row_a, None])

        with patch("recommenders.pick_random_movie", fake_pick):
            at = AppTest.from_file(APP_PATH, default_timeout=120)
            at.run()
            _go_random(at)

            self.assertTrue(_click(at, "random_pick"))
            at.run()
            self.assertEqual(_session_value(at, "random_pick_movie")["movieId"], 9001)

            self.assertTrue(_click(at, "random_pick_another"))
            at.run()

            preserved = _session_value(at, "random_pick_movie")
            self.assertIsNotNone(preserved, "Existing pick must remain in session state when no alternative exists.")
            self.assertEqual(preserved["movieId"], 9001)

            warnings = [w.value for w in at.warning]
            self.assertTrue(
                any("No other unseen movie matched" in w for w in warnings),
                f"Expected no-alternative warning; saw warnings: {warnings}",
            )

        self.assertEqual(fake_pick.call_count, 2)
        second_call_kwargs = fake_pick.call_args_list[1].kwargs
        self.assertEqual(second_call_kwargs.get("excluded_movie_ids"), {9001})


if __name__ == "__main__":
    unittest.main()
