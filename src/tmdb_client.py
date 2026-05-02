import requests
import pandas as pd

from config import TMDB_TIMEOUT


def get_tmdb_id(row, links_df=None):
    if "tmdbId" in row and pd.notna(row["tmdbId"]):
        return int(row["tmdbId"])

    if links_df is None or links_df.empty or "movieId" not in row or pd.isna(row["movieId"]):
        return None

    link_info = links_df[links_df["movieId"] == row["movieId"]]
    if link_info.empty or "tmdbId" not in link_info.columns or pd.isna(link_info.iloc[0]["tmdbId"]):
        return None
    return int(link_info.iloc[0]["tmdbId"])


def get_movie_details(tmdb_id, api_key, timeout=TMDB_TIMEOUT):
    if not api_key or pd.isna(tmdb_id):
        return None

    api_url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    try:
        response = requests.get(
            api_url,
            params={"api_key": api_key, "language": "en-US"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    data = response.json()
    poster_path = data.get("poster_path")
    return {
        "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
        "overview": data.get("overview", ""),
        "tmdb_title": data.get("title", ""),
    }
