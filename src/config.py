import os
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

DEFAULT_CONFIG = {
    "paths": {
        "data_dir": "data",
        "cleaned_data_dir": "cleaned_data",
        "svd_model_path": "cleaned_data/svd_trained_model.pkl",
    },
    "recommendations": {
        "initial_candidate_pool_size": 300,
        "content_candidate_pool_size": 100,
        "bayesian_min_ratings": 100,
        "hybrid_weights": {
            "content_similarity": 0.60,
            "bayesian_rating": 0.25,
            "popularity": 0.10,
            "diversity": 0.05,
        },
        "watch_history_weights": {
            "max_similarity": 0.70,
            "mean_similarity": 0.20,
            "matched_seed_count_bonus": 0.10,
        },
    },
    "mood_genre_map": {
        "happy": ["Comedy", "Family", "Animation", "Romance"],
        "sad": ["Drama", "Romance"],
        "adventurous": ["Action", "Adventure", "Thriller"],
        "scared": ["Horror", "Thriller", "Mystery"],
        "excited": ["Action", "Adventure", "Sci-Fi"],
        "nostalgic": ["Animation", "Family", "Fantasy"],
        "thoughtful": ["Documentary", "Drama"],
        "surprised": ["Mystery", "Thriller"],
    },
    "evaluation": {
        "max_users": 100,
        "k_values": [5, 10, 20],
        "holdout_count": 1,
        "min_interactions": 5,
        "positive_threshold": 4.0,
        "random_seed": 42,
        "output_dir": "artifacts/evaluation",
        "semantic": {
            "components": 64,
            "random_state": 42,
        },
    },
    "tmdb": {
        "timeout_seconds": 8,
    },
    "ui": {
        "menu_items": [
            "\U0001F3AF Content-Based Recommendation",
            "\U0001F465 Collaborative Filtering",
            "\U0001F60A Mood-Based Recommendation",
            "\U0001F3B2 Random Movie",
            "\U0001F39E\U0000FE0F Watch History & Recommendations",
            "\U00002139\U0000FE0F About & Help",
        ],
        "demo_profiles": {
            "Select a Demo Profile...": {"id": None, "target_genre_cols": []},
            "\U0001F3AC Comedy Fan": {"id": 88539, "target_genre_cols": ["genre_comedy"]},
            "\U0001F4A5 Action & Thriller Seeker": {"id": 129440, "target_genre_cols": ["genre_action", "genre_thriller"]},
            "\U0001F3AD Drama Enthusiast": {"id": 110971, "target_genre_cols": ["genre_drama"]},
            "\U0001F52E Sci-Fi & Fantasy Voyager": {"id": 78616, "target_genre_cols": ["genre_scifi", "genre_fantasy"]},
            "\U0001F9F8 Animation & Family Watcher": {"id": 93359, "target_genre_cols": ["genre_animation", "genre_children"]},
        },
    },
}


def deep_merge(base, override):
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(config_path=CONFIG_PATH):
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


APP_CONFIG = deep_merge(DEFAULT_CONFIG, load_yaml_config())


def config_value(path, default=None):
    value = APP_CONFIG
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def project_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


DATA_DIR = project_path(config_value(["paths", "data_dir"], "data"))
CLEANED_DATA_DIR = project_path(config_value(["paths", "cleaned_data_dir"], "cleaned_data"))
SVD_MODEL_PATH = project_path(config_value(["paths", "svd_model_path"], "cleaned_data/svd_trained_model.pkl"))


def get_tmdb_api_key():
    """Read the TMDB API key without hardcoding secrets in source control."""
    env_key = os.getenv("TMDB_API_KEY")
    if env_key:
        return env_key

    try:
        import streamlit as st

        return st.secrets.get("TMDB_API_KEY")
    except Exception:
        pass

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as secrets_file:
            return tomllib.load(secrets_file).get("TMDB_API_KEY")

    return None

MOOD_GENRE_MAP = config_value(["mood_genre_map"], {})
INITIAL_CANDIDATE_POOL_SIZE = int(config_value(["recommendations", "initial_candidate_pool_size"], 300))
CONTENT_CANDIDATE_POOL_SIZE = int(config_value(["recommendations", "content_candidate_pool_size"], 100))
BAYESIAN_MIN_RATINGS = int(config_value(["recommendations", "bayesian_min_ratings"], 100))
HYBRID_WEIGHTS = config_value(["recommendations", "hybrid_weights"], {})
WATCH_HISTORY_WEIGHTS = config_value(["recommendations", "watch_history_weights"], {})
TMDB_TIMEOUT = int(config_value(["tmdb", "timeout_seconds"], 8))
MENU_ITEMS = config_value(["ui", "menu_items"], [])
DEMO_PROFILES_WITH_GENRES = config_value(["ui", "demo_profiles"], {})
EVALUATION_DEFAULTS = config_value(["evaluation"], {})
EVALUATION_OUTPUT_DIR = project_path(config_value(["evaluation", "output_dir"], "artifacts/evaluation"))
