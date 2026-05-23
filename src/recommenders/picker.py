"""Random movie picker (lightweight discovery feature)."""

import pandas as pd

from .common import filter_watched_movies


def pick_random_movie(movies, selected_genres=None, watched_movie_ids=None):
    if movies.empty:
        return None

    filtered = movies.copy()
    if selected_genres:
        genre_mask = pd.Series(False, index=filtered.index)
        for genre in selected_genres:
            genre_mask |= filtered["genres"].astype(str).str.contains(genre, case=False, na=False, regex=False)
        filtered = filtered[genre_mask]

    filtered = filter_watched_movies(filtered, watched_movie_ids)

    if filtered.empty:
        return None
    return filtered.sample(n=1, random_state=None).iloc[0]
