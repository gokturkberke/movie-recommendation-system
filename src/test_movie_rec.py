import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"

sys.path.insert(0, str(TESTS_DIR))

from test_movie_rec import TestMovieRecommendations  # noqa: E402,F401


if __name__ == "__main__":
    unittest.main(module="test_movie_rec")
