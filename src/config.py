import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data"
SVD_MODEL_PATH = CLEANED_DATA_DIR / "svd_trained_model.pkl"


def get_tmdb_api_key():
    """Read the TMDB API key without hardcoding secrets in source control."""
    env_key = os.getenv("TMDB_API_KEY")
    if env_key:
        return env_key

    try:
        import streamlit as st

        return st.secrets.get("TMDB_API_KEY")
    except Exception:
        return None

MOOD_GENRE_MAP = {
    "happy": ["Comedy", "Family", "Animation", "Romance"],
    "sad": ["Drama", "Romance"],
    "adventurous": ["Action", "Adventure", "Thriller"],
    "scared": ["Horror", "Thriller", "Mystery"],
    "excited": ["Action", "Adventure", "Sci-Fi"],
    "nostalgic": ["Animation", "Family", "Fantasy"],
    "thoughtful": ["Documentary", "Drama"],
    "surprised": ["Mystery", "Thriller"],
}

# Recommendation System Parameters
INITIAL_CANDIDATE_POOL_SIZE = 300

# UI Configuration
MENU_ITEMS = [
    "🎯 Content-Based Recommendation",
    "👥 Collaborative Filtering",
    "😊 Mood-Based Recommendation",
    "🎲 Random Movie",
    "📽️ Watch History & Recommendations",
    "ℹ️ About & Help",
]

DEMO_PROFILES_WITH_GENRES = {
    "Select a Demo Profile...": {"id": None, "target_genre_cols": []},
    "🎬 Comedy Fan": {"id": 88539, "target_genre_cols": ['genre_comedy']},
    "💥 Action & Thriller Seeker": {"id": 129440, "target_genre_cols": ['genre_action', 'genre_thriller']},
    "🎭 Drama Enthusiast": {"id": 110971, "target_genre_cols": ['genre_drama']},
    "🔮 Sci-Fi & Fantasy Voyager": {"id": 78616, "target_genre_cols": ['genre_scifi', 'genre_fantasy']},
    "🧸 Animation & Family Watcher": {"id": 93359, "target_genre_cols": ['genre_animation', 'genre_children']}
}
