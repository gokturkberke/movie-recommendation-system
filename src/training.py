"""Train a Surprise SVD model on cleaned ratings and persist it to disk.

The pipeline body lives here so callers can invoke it programmatically. The
CLI shell at ``scripts/train_save_model.py`` is just argparse wiring.
"""

from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD, dump

from config import CLEANED_DATA_DIR, SVD_MODEL_PATH


DEFAULT_RATING_SCALE = (0.5, 5.0)


def load_ratings_data_for_training(cleaned_data_dir):
    cleaned_data_dir = Path(cleaned_data_dir)
    ratings_file = cleaned_data_dir / "ratings_clean.csv"
    print(f"Loading cleaned ratings from {ratings_file}")
    if not ratings_file.exists():
        print(f"ERROR: {ratings_file} not found. Run scripts/preprocess_dataset.py first.")
        return None
    return pd.read_csv(ratings_file)


def train_and_save_surprise_model(
    ratings_df,
    model_output_path,
    rating_scale=DEFAULT_RATING_SCALE,
):
    if ratings_df is None or ratings_df.empty:
        print("Ratings dataframe is empty. Aborting model training.")
        return None

    print("Building Surprise dataset...")
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

    print("Building full trainset...")
    full_trainset = data.build_full_trainset()

    print("Training SVD model (this can take a while on large datasets)...")
    algo = SVD()
    algo.fit(full_trainset)

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving trained model to {model_output_path}")
    dump.dump(str(model_output_path), algo=algo)
    print("Model saved.")
    return algo


def run_training(
    cleaned_data_dir=CLEANED_DATA_DIR,
    model_path=SVD_MODEL_PATH,
    rating_scale=DEFAULT_RATING_SCALE,
):
    print("Training pipeline starting.")
    ratings_df = load_ratings_data_for_training(cleaned_data_dir)
    train_and_save_surprise_model(ratings_df, model_path, rating_scale=rating_scale)
    print("Training pipeline finished.")
