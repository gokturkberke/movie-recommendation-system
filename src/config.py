# src/config.py

# TMDB API Key
TMDB_API_KEY = "d4dd76aa404d680766dbacc0e83552bd"

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
    "🕵️ Unwatched Movies"
]

DEMO_PROFILES_WITH_GENRES = {
    "Select a Demo Profile...": {"id": None, "target_genre_cols": []},
    "🎬 Comedy Fan": {"id": 88539, "target_genre_cols": ['genre_comedy']},
    "💥 Action & Thriller Seeker": {"id": 129440, "target_genre_cols": ['genre_action', 'genre_thriller']},
    "🎭 Drama Enthusiast": {"id": 110971, "target_genre_cols": ['genre_drama']},
    "🔮 Sci-Fi & Fantasy Voyager": {"id": 78616, "target_genre_cols": ['genre_scifi', 'genre_fantasy']},
    "🧸 Animation & Family Watcher": {"id": 93359, "target_genre_cols": ['genre_animation', 'genre_children']}
}
