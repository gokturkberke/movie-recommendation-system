import pandas as pd
import streamlit as st

from config import DEMO_PROFILES_WITH_GENRES, MENU_ITEMS, MOOD_GENRE_MAP, get_tmdb_api_key
from data_access import (
    latest_release_info,
    load_links,
    load_movies,
    load_ratings,
    load_ratings_for_stats,
    load_surprise_model,
    load_tags,
    merge_tmdb_ids,
)
from recommenders import (
    build_tfidf_matrix,
    build_movie_stats,
    normalize_movie_ids,
    pick_random_movie,
    recommend_based_on_watch_history_content,
    recommend_by_mood,
    recommend_for_persona,
    recommend_for_user,
    recommend_similar_movies,
    recommend_similar_movies_by_id,
    suggest_movie_titles,
)
from tmdb_client import get_movie_details, get_tmdb_id


APP_VERSION = "Movie Recommendation System v1.4"


@st.cache_data(show_spinner=False)
def cached_movies():
    return load_movies()


@st.cache_data(show_spinner=False)
def cached_tags():
    return load_tags()


@st.cache_data(show_spinner=False)
def cached_links():
    return load_links()


@st.cache_data(show_spinner=False)
def cached_ratings():
    return load_ratings()


@st.cache_data(show_spinner=False)
def cached_tfidf(movies, tags):
    return build_tfidf_matrix(movies, tags)


@st.cache_data(show_spinner="Building movie rating statistics...")
def cached_movie_stats():
    return build_movie_stats(load_ratings_for_stats())


@st.cache_resource(show_spinner=False)
def cached_svd_model():
    return load_surprise_model()


@st.cache_data(show_spinner=False)
def cached_tmdb_details(tmdb_id, api_key):
    return get_movie_details(tmdb_id, api_key)


def initialize_session_state():
    if "watched_movie_ids" not in st.session_state:
        st.session_state.watched_movie_ids = set()
    st.session_state.watched_movie_ids = normalize_movie_ids(st.session_state.watched_movie_ids)


def explain(text):
    st.caption(text)


def render_movie(row, index, links_df, tmdb_api_key, show_score=False):
    title = row.get("title", "Title not available")
    genres = row.get("genres", "Genres not available")

    st.subheader(f"{index}. {title}")
    st.write(f"**Genres:** {genres}")
    if show_score and pd.notna(row.get("predicted_score", pd.NA)):
        st.caption(f"Predicted rating: {row.get('predicted_score'):.2f}")

    tmdb_id = get_tmdb_id(row, links_df)
    if not tmdb_id:
        st.caption("TMDB ID is not available for this movie.")
        st.markdown("---")
        return

    if not tmdb_api_key:
        st.caption("Poster and overview are disabled because TMDB_API_KEY is not configured.")
        st.markdown("---")
        return

    details = cached_tmdb_details(tmdb_id, tmdb_api_key)
    if not details:
        st.caption("TMDB details could not be retrieved.")
        st.markdown("---")
        return

    poster_url = details.get("poster_url")
    overview = details.get("overview")
    if poster_url:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(poster_url, width=150)
        with col2:
            st.caption(f"Overview: {overview}" if overview else "Overview not available.")
    else:
        st.caption("Poster not found on TMDB.")
        if overview:
            st.caption(f"Overview: {overview}")
    st.markdown("---")


def render_movie_list(recommendations, links_df, tmdb_api_key, show_score=False):
    if recommendations.empty:
        st.warning("No recommendations found.")
        return

    with st.expander("See Recommendations", expanded=True):
        for index, (_, row) in enumerate(recommendations.reset_index(drop=True).iterrows(), start=1):
            render_movie(row, index, links_df, tmdb_api_key, show_score=show_score)


def render_content_based_page(context):
    st.success("**Content-Based Recommendation**")
    explain("Finds similar movies with TF-IDF, then re-ranks candidates with content similarity, Bayesian rating, popularity, and light diversity.")

    if not context["content_enabled"]:
        st.error("Content-based recommendations are unavailable because the TF-IDF matrix could not be built.")
        return

    movie_title = st.text_input("Enter a movie title you like:", key="content_movie_title")
    selected_movie_id = None
    selected_title = None
    suggestions = suggest_movie_titles(movie_title, context["movies"], limit=8)
    if not suggestions.empty:
        option_lookup = {
            index: f"{row.title} — {row.genres}"
            for index, row in suggestions.iterrows()
        }
        selected_index = st.selectbox(
            "Closest matches in your dataset:",
            options=[None] + list(option_lookup.keys()),
            format_func=lambda value: "Select a match..." if value is None else option_lookup[value],
            key="content_movie_suggestion",
        )
        if selected_index is not None:
            selected_movie_id = suggestions.loc[selected_index, "movieId"]
            selected_title = suggestions.loc[selected_index, "title"]
        else:
            st.caption("Select one of these if your typed title is misspelled or ambiguous.")

    if not st.button("Get Recommendations", key="content_get_recommendations"):
        return

    title_to_recommend = selected_title or movie_title
    if not str(title_to_recommend).strip():
        st.warning("Please enter a movie title.")
        return

    movie_stats = cached_movie_stats()
    if selected_movie_id is not None:
        recommendations, matched_title = recommend_similar_movies_by_id(
            selected_movie_id,
            context["movies_with_content"],
            context["tfidf_matrix"],
            context["movies"],
            watched_movie_ids=st.session_state.watched_movie_ids,
            movie_stats=movie_stats,
            top_n=10,
        )
    else:
        recommendations, matched_title = recommend_similar_movies(
            title_to_recommend,
            context["movies_with_content"],
            context["tfidf_matrix"],
            context["movies"],
            watched_movie_ids=st.session_state.watched_movie_ids,
            movie_stats=movie_stats,
            top_n=10,
        )
    if matched_title:
        st.info(f"Recommendations are based on: **{matched_title}**")
    render_movie_list(recommendations, context["links"], context["tmdb_api_key"])


def render_collaborative_page(context):
    st.success("**Collaborative Filtering Recommendation**")
    explain("Uses the trained Surprise SVD model to predict ratings for movies the selected user has not rated.")

    model, model_error = cached_svd_model()
    ratings = cached_ratings()
    if model_error:
        st.error(model_error)
        return
    if ratings.empty:
        st.error("ratings_clean.csv is missing or empty. Collaborative filtering is disabled.")
        return

    chosen_profile_name = st.selectbox(
        "Explore recommendations for a demo profile:",
        options=list(DEMO_PROFILES_WITH_GENRES.keys()),
        key="collab_demo_profile",
        index=0,
    )

    user_id = None
    target_genres = []
    profile = DEMO_PROFILES_WITH_GENRES.get(chosen_profile_name, {})
    if profile.get("id") is not None:
        user_id = profile["id"]
        target_genres = profile["target_genre_cols"]

    if user_id is None:
        manual_user_id = st.text_input(
            "Or enter a MovieLens User ID:",
            key="collab_manual_user_id",
            help="Use a numeric userId from ratings_clean.csv.",
        ).strip()
        if manual_user_id:
            if manual_user_id.isdigit():
                user_id = int(manual_user_id)
            else:
                st.warning("Please enter a numeric User ID.")

    if not st.button("Get Collaborative Recommendations", key="collab_get_recommendations"):
        return

    if user_id is None:
        st.warning("Please select a demo profile or enter a valid User ID.")
        return

    if target_genres:
        st.markdown(f"### SVD recommendations for {chosen_profile_name}")
        recommendations = recommend_for_persona(
            user_id,
            target_genres,
            model,
            context["movies"],
            ratings,
            watched_movie_ids=st.session_state.watched_movie_ids,
            top_n=10,
        )
        render_movie_list(recommendations, context["links"], context["tmdb_api_key"], show_score=True)
    else:
        st.markdown(f"### SVD recommendations for User ID {user_id}")
        recommendations = recommend_for_user(
            user_id,
            model,
            context["movies"],
            ratings,
            watched_movie_ids=st.session_state.watched_movie_ids,
            top_n=10,
        )
        render_movie_list(recommendations, context["links"], context["tmdb_api_key"])


def render_mood_page(context):
    st.success("**Mood-Based Recommendation**")
    explain("Maps your selected mood to genres, filters the movie pool, and samples unseen titles.")

    mood = st.selectbox("Select your mood:", list(MOOD_GENRE_MAP.keys()), key="mood_select")
    if not st.button("Get Mood-Based Recommendations", key="mood_get_recommendations"):
        return

    recommendations = recommend_by_mood(
        mood,
        context["movies"],
        watched_movie_ids=st.session_state.watched_movie_ids,
        top_n=10,
    )
    render_movie_list(recommendations, context["links"], context["tmdb_api_key"])


def available_genres(movies):
    if movies.empty or "genres" not in movies.columns:
        return []
    genres = movies["genres"].str.split("|").explode().str.strip().dropna().unique()
    return sorted([genre for genre in genres if genre and genre != "(no genres listed)"])


def render_random_page(context):
    st.success("**Random Movie Recommendation**")
    explain("Samples one movie from the dataset, optionally after applying genre filters.")

    selected_genres = st.multiselect(
        "Filter by genre:",
        options=available_genres(context["movies"]),
        key="random_genres",
    )

    if not st.button("Pick a Random Movie", key="random_pick"):
        return

    movie = pick_random_movie(context["movies"], selected_genres=selected_genres)
    if movie is None:
        st.warning("No movie matched the current filter.")
        return
    render_movie(movie, 1, context["links"], context["tmdb_api_key"])


def render_watch_history_page(context):
    st.success("**Watch History & Personalized Recommendations**")
    explain("Treats your watched movie IDs as content seeds, combines similar results, and removes already watched movies.")

    watched_ids = normalize_movie_ids(st.session_state.watched_movie_ids)
    st.session_state.watched_movie_ids = watched_ids

    movies_by_id = context["movies"].drop_duplicates(subset=["movieId"]).copy()
    movies_by_id = movies_by_id.dropna(subset=["movieId"])
    title_lookup = dict(zip(movies_by_id["movieId"].astype(int), movies_by_id["title"]))

    selectable_movies = movies_by_id[~movies_by_id["movieId"].isin(watched_ids)].sort_values("title")
    selectable_movie_ids = selectable_movies["movieId"].astype(int).tolist()

    selected_to_add = st.multiselect(
        "Select movies to add to your watch history:",
        options=selectable_movie_ids,
        format_func=lambda movie_id: title_lookup.get(movie_id, f"Movie ID {movie_id}"),
        key="watch_history_add",
    )
    if st.button("Add Selected to Watch History", key="watch_history_add_button"):
        if not selected_to_add:
            st.warning("Please select at least one movie.")
        else:
            st.session_state.watched_movie_ids.update(normalize_movie_ids(selected_to_add))
            st.success(f"{len(selected_to_add)} movie(s) added.")
            st.rerun()

    watched_ids = normalize_movie_ids(st.session_state.watched_movie_ids)
    watched_movies = movies_by_id[movies_by_id["movieId"].isin(watched_ids)].sort_values("title")
    watched_movie_ids = watched_movies["movieId"].astype(int).tolist()
    if watched_movie_ids:
        watched_df = watched_movies[["title", "genres"]].rename(columns={"title": "Title", "genres": "Genres"})
        watched_df.index = range(1, len(watched_df) + 1)
        st.dataframe(watched_df, use_container_width=True, height=min(300, len(watched_df) * 40 + 40))

        selected_to_remove = st.multiselect(
            "Select movies to remove:",
            options=watched_movie_ids,
            format_func=lambda movie_id: title_lookup.get(movie_id, f"Movie ID {movie_id}"),
            key="watch_history_remove",
        )
        if st.button("Remove Selected", key="watch_history_remove_button"):
            if not selected_to_remove:
                st.warning("Please select at least one movie.")
            else:
                st.session_state.watched_movie_ids.difference_update(normalize_movie_ids(selected_to_remove))
                st.success(f"{len(selected_to_remove)} movie(s) removed.")
                st.rerun()
    else:
        st.info("Your watch history is empty.")

    if not st.button("Get Recommendations Based on Watch History", key="watch_history_get_recommendations"):
        return

    if not st.session_state.watched_movie_ids:
        st.warning("Add movies to your watch history first.")
        return
    if not context["content_enabled"]:
        st.error("Content-based components are unavailable.")
        return

    movie_stats = cached_movie_stats()
    recommendations = recommend_based_on_watch_history_content(
        list(st.session_state.watched_movie_ids),
        context["movies_with_content"],
        context["tfidf_matrix"],
        context["movies"],
        movie_stats=movie_stats,
        top_n=10,
    )
    render_movie_list(recommendations, context["links"], context["tmdb_api_key"])


def render_about_page(context):
    st.success("**About & Help**")
    latest_year, latest_count, latest_movies = latest_release_info(context["movies"])
    st.markdown(
        """
        ### Recommendation methods

        **Content-Based:** retrieves similar movies with TF-IDF over title, genre, and tag text, then re-ranks them with similarity, Bayesian rating, popularity, and light diversity.

        **Collaborative Filtering:** uses a trained Surprise SVD model to estimate ratings for unseen movies.

        **Mood-Based:** maps moods to genre groups and samples unwatched movies from matching genres.

        **Watch History:** stores watched movies by `movieId`, uses them as content seeds, and merges the strongest hybrid-ranked results.

        ### Local data

        The app expects MovieLens files under `data/` and cleaned/model artifacts under `cleaned_data/`.
        Poster and overview rendering requires `TMDB_API_KEY` from the environment or Streamlit secrets.
        """
    )
    if latest_year:
        st.info(f"Loaded dataset latest release year: **{latest_year}** ({latest_count} movies).")
        st.dataframe(
            latest_movies[["title", "genres"]].head(10).reset_index(drop=True),
            use_container_width=True,
        )


def load_context():
    movies = cached_movies()
    tags = cached_tags()
    links = cached_links()

    if movies.empty:
        st.error("Movie data could not be loaded. The application cannot continue.")
        st.stop()

    tfidf_matrix, tfidf_vectorizer, movies_with_content = cached_tfidf(movies.copy(), tags.copy())
    movies = merge_tmdb_ids(movies, links)
    movies_with_content = merge_tmdb_ids(movies_with_content, links)
    tmdb_api_key = get_tmdb_api_key()

    return {
        "movies": movies,
        "tags": tags,
        "links": links,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer,
        "movies_with_content": movies_with_content,
        "content_enabled": tfidf_matrix is not None and tfidf_vectorizer is not None and not movies_with_content.empty,
        "tmdb_api_key": tmdb_api_key,
    }


def main():
    st.markdown("<h1 style='color:#1976d2;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("## 📋 Menu")
    initialize_session_state()
    context = load_context()

    if not context["tmdb_api_key"]:
        st.sidebar.warning("TMDB_API_KEY is not configured. Posters and overviews are disabled.")

    choice = st.sidebar.radio("Choose a recommendation method:", MENU_ITEMS, key="main_menu_choice")

    if choice == MENU_ITEMS[0]:
        render_content_based_page(context)
    elif choice == MENU_ITEMS[1]:
        render_collaborative_page(context)
    elif choice == MENU_ITEMS[2]:
        render_mood_page(context)
    elif choice == MENU_ITEMS[3]:
        render_random_page(context)
    elif choice == MENU_ITEMS[4]:
        render_watch_history_page(context)
    elif choice == MENU_ITEMS[5]:
        render_about_page(context)

    st.sidebar.markdown("---")
    st.sidebar.info(APP_VERSION)


if __name__ == "__main__":
    main()
