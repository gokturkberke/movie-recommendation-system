"""CLI for the MovieLens preprocessing pipeline.

The pipeline body lives in ``src/preprocessing.py``. This script is a thin
argparse wrapper.
"""

import argparse
from pathlib import Path

from config import CLEANED_DATA_DIR, DATA_DIR
from preprocessing import (
    DEFAULT_MIN_RATINGS_PER_MOVIE,
    DEFAULT_MIN_RATINGS_PER_USER,
    run_preprocessing,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Clean MovieLens raw CSVs into cleaned_data/.")
    parser.add_argument("--raw-data-dir", default=str(DATA_DIR), help="Directory containing raw MovieLens CSVs.")
    parser.add_argument("--cleaned-data-dir", default=str(CLEANED_DATA_DIR), help="Directory for cleaned outputs.")
    parser.add_argument("--min-ratings-per-user", type=int, default=DEFAULT_MIN_RATINGS_PER_USER, help="Drop users with fewer than this many ratings.")
    parser.add_argument("--min-ratings-per-movie", type=int, default=DEFAULT_MIN_RATINGS_PER_MOVIE, help="Drop movies with fewer than this many ratings.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_preprocessing(
        raw_data_dir=Path(args.raw_data_dir),
        cleaned_data_dir=Path(args.cleaned_data_dir),
        min_ratings_per_user=args.min_ratings_per_user,
        min_ratings_per_movie=args.min_ratings_per_movie,
    )


if __name__ == "__main__":
    main()
