"""Random movie picker (lightweight discovery feature)."""

import pandas as pd


def pick_random_movie(movies, selected_genres=None):
    if movies.empty:
        return None

    filtered = movies.copy()
    if selected_genres:
        genre_mask = pd.Series(False, index=filtered.index)
        for genre in selected_genres:
            genre_mask |= filtered["genres"].astype(str).str.contains(genre, case=False, na=False, regex=False)
        filtered = filtered[genre_mask]

    if filtered.empty:
        return None
    return filtered.sample(n=1, random_state=None).iloc[0]
