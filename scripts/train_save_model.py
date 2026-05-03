"""CLI for training and persisting the Surprise SVD model.

The training body lives in ``src/training.py``. This script is a thin
argparse wrapper.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLEANED_DATA_DIR, SVD_MODEL_PATH  # noqa: E402
from training import DEFAULT_RATING_SCALE, run_training  # noqa: E402


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the Surprise SVD model on cleaned ratings.")
    parser.add_argument("--cleaned-data-dir", default=str(CLEANED_DATA_DIR), help="Directory containing ratings_clean.csv.")
    parser.add_argument("--model-path", default=str(SVD_MODEL_PATH), help="Output path for the trained model artifact.")
    parser.add_argument("--rating-min", type=float, default=DEFAULT_RATING_SCALE[0], help="Lower bound of the rating scale.")
    parser.add_argument("--rating-max", type=float, default=DEFAULT_RATING_SCALE[1], help="Upper bound of the rating scale.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_training(
        cleaned_data_dir=Path(args.cleaned_data_dir),
        model_path=Path(args.model_path),
        rating_scale=(args.rating_min, args.rating_max),
    )


if __name__ == "__main__":
    main()
