import streamlit as st
import pandas as pd
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from surprise import dump # Ensure this import is present
from thefuzz import fuzz
import requests # Added requests import
from config import TMDB_API_KEY, MOOD_GENRE_MAP, INITIAL_CANDIDATE_POOL_SIZE, MENU_ITEMS, DEMO_PROFILES_WITH_GENRES # TMDB_API_KEY'in buradan geldiğini varsayıyorum
from utils_data import (
    load_movies,
    load_ratings,
    load_tags,
    load_trained_surprise_model,
    clean_text,
    get_movie_details_from_tmdb,
    _get_raw_svd_predictions,
    pick_random_movie,
    _extract_watched_movies_and_genres,
    _get_genre_based_recommendations,
    _get_fallback_recommendations
)

def get_tfidf_matrix(movies, tags):
    if movies.empty:
        st.warning("Movies DataFrame is empty. Cannot generate TF-IDF matrix for content-based recommendations.")
        return None, None, movies

    tags['tag'] = tags['tag'].fillna('').apply(clean_text)
    tags = tags.drop_duplicates(subset=['movieId', 'tag'])
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

    movies = movies.merge(tags_grouped, on='movieId', how='left')

    movies['title_for_matching'] = movies['title_for_matching'].fillna('').astype(str)
    movies['genres_for_matching'] = movies['genres_for_matching'].fillna('').astype(str)
    movies['tag'] = movies['tag'].fillna('').astype(str)

    movies['content'] = movies['title_for_matching'] + ' ' + movies['genres_for_matching'] + ' ' + movies['tag']
    movies['content'] = movies['content'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])

    if tfidf_matrix.shape[1] == 0:
        st.warning("No features were learned from movie content for TF-IDF. Content-based recommendations might be ineffective.")

    return tfidf_matrix, tfidf, movies

def get_user_recommendations(user_id, surprise_model, movies_df, ratings_df, watched_titles, top_n=10):
    # Determine the number of candidates to fetch
    num_candidates_to_fetch = top_n + (len(watched_titles) if watched_titles else 0) + 20

    raw_predictions_df = _get_raw_svd_predictions(user_id, surprise_model, movies_df, ratings_df, candidate_pool_size=num_candidates_to_fetch)

    if raw_predictions_df.empty:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'] + (['tmdbId'] if 'tmdbId' in movies_df.columns else []))

    # Merge with movie details
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_df.columns:
        cols_to_return.append('tmdbId')

    valid_cols_for_merge = [col for col in cols_to_return if col in movies_df.columns]
    if 'movieId' not in valid_cols_for_merge: # movieId is essential for merge
        valid_cols_for_merge.insert(0, 'movieId')
        valid_cols_for_merge = list(set(valid_cols_for_merge))


    recommended_movies_df = pd.merge(
        raw_predictions_df[['movieId']],
        movies_df[valid_cols_for_merge],
        on='movieId',
        how='left'
    )

    # Filter out watched titles
    if watched_titles and not recommended_movies_df.empty and 'title' in recommended_movies_df.columns:
        recommended_movies_df = recommended_movies_df[~recommended_movies_df['title'].isin(watched_titles)]

    # Re-order based on original prediction scores
    if not recommended_movies_df.empty and 'movieId' in recommended_movies_df.columns:
        order_map = {movie_id: i for i, movie_id in enumerate(raw_predictions_df['movieId'])}
        recommended_movies_df = recommended_movies_df[recommended_movies_df['movieId'].isin(order_map)]

        if not recommended_movies_df.empty:
            recommended_movies_df['sort_order'] = recommended_movies_df['movieId'].map(order_map)
            recommended_movies_df.sort_values('sort_order', inplace=True)
            recommended_movies_df.drop(columns=['sort_order'], inplace=True)


    # Ensure all expected columns are present before returning
    final_cols_to_return = ['movieId', 'title', 'genres'] + (['tmdbId'] if 'tmdbId' in movies_df.columns else [])
    for col in final_cols_to_return:
        if col not in recommended_movies_df.columns:
            recommended_movies_df[col] = pd.NA

    return recommended_movies_df[final_cols_to_return].head(top_n)


def get_filtered_svd_recommendations_for_persona(
    user_id,
    persona_target_genre_cols, # e.g., ['genre_comedy'], ['genre_action', 'genre_adventure']
    model,                     # surprise_model
    movies_data,               # main movies DataFrame with one-hot encoded genres
    ratings_data,              # full ratings DataFrame
    watched_titles,            # user's general watch history (titles)
    top_n_final=10
):
    """
    Gets SVD recommendations, filters by specified persona target genres,
    and removes watched movies. Returns a DataFrame.
    """
    cols_to_bring_from_movies_data = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_data.columns:
        cols_to_bring_from_movies_data.append('tmdbId')

    for gc in persona_target_genre_cols:
        if gc in movies_data.columns and gc not in cols_to_bring_from_movies_data:
            cols_to_bring_from_movies_data.append(gc)
        elif gc not in movies_data.columns:
            st.error(f"Error: Persona target genre column '{gc}' not found in movies_data DataFrame. "
                     f"Please check your 'movies_clean.csv' and preprocessing script to ensure "
                     f"one-hot encoded genre columns (e.g., 'genre_comedy') exist.")
            return pd.DataFrame()


    # 1. Get raw SVD predictions
    raw_predictions_df = _get_raw_svd_predictions(user_id, model, movies_data, ratings_data, candidate_pool_size=INITIAL_CANDIDATE_POOL_SIZE)

    if raw_predictions_df.empty:
        return pd.DataFrame()

    # 2. Merge raw predictions with movie details
    candidate_movies_with_details = pd.merge(
        raw_predictions_df[['movieId', 'predicted_score']],
        movies_data[cols_to_bring_from_movies_data],
        on='movieId',
        how='left'
    )

    for genre_col in persona_target_genre_cols:
        if genre_col in candidate_movies_with_details.columns:
            candidate_movies_with_details[genre_col] = candidate_movies_with_details[genre_col].fillna(0).astype(int)

    # 3. Filter by persona target genres
    if persona_target_genre_cols:
        valid_persona_genre_cols_for_filtering = [col for col in persona_target_genre_cols if col in candidate_movies_with_details.columns]

        if not valid_persona_genre_cols_for_filtering:
             st.warning("No valid target persona genre columns found in candidate movies for filtering. Showing unfiltered SVD recommendations (but still excluding watched).")
             filtered_recommendations_df = candidate_movies_with_details.copy()
        else:
            filter_mask = candidate_movies_with_details[valid_persona_genre_cols_for_filtering].sum(axis=1) > 0
            filtered_recommendations_df = candidate_movies_with_details[filter_mask]
    else:
        filtered_recommendations_df = candidate_movies_with_details.copy()

    if filtered_recommendations_df.empty:
        print(f"No movies found for User ID {user_id} matching persona genres from the SVD pool (or pool was empty).")
        return pd.DataFrame()

    # 4. Filter out watched movies
    if watched_titles and not filtered_recommendations_df.empty:
        if 'title' in filtered_recommendations_df.columns:
            filtered_recommendations_df = filtered_recommendations_df[
                ~filtered_recommendations_df['title'].isin(watched_titles)
            ]

    # 5. Get the top N results
    final_df_to_show = filtered_recommendations_df.head(top_n_final)

    # 6. Define and ensure final output columns
    output_cols = ['movieId', 'title', 'genres', 'predicted_score']
    if 'tmdbId' in movies_data.columns and 'tmdbId' not in output_cols :
        output_cols.append('tmdbId')

    for col in output_cols:
        if col not in final_df_to_show.columns:
            final_df_to_show[col] = pd.NA

    return final_df_to_show[[col for col in output_cols if col in final_df_to_show.columns]].reset_index(drop=True)

def recommend_by_mood(mood, movies, watched_movies, top_n=10):
    genres_for_mood = MOOD_GENRE_MAP.get(mood.lower())

    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies.columns:
        cols_to_return.append('tmdbId')

    if not genres_for_mood:
        return pd.DataFrame(columns=cols_to_return)

    movies_copy = movies.copy()
    movies_copy['genres'] = movies_copy['genres'].astype(str)

    mask = movies_copy['genres'].apply(lambda g: any(genre_item in g for genre_item in genres_for_mood))
    filtered_movies = movies_copy[mask]

    if filtered_movies.empty:
        return pd.DataFrame(columns=cols_to_return)

    num_to_sample = min(top_n + (len(watched_movies) if watched_movies else 0) + 5, len(filtered_movies))

    if num_to_sample <= 0:
        return pd.DataFrame(columns=cols_to_return)

    recommendations = filtered_movies.sample(n=num_to_sample, random_state=42)[cols_to_return].copy()

    if watched_movies and not recommendations.empty:
        recommendations = recommendations[~recommendations['title'].isin(watched_movies)]

    return recommendations.head(top_n).reset_index(drop=True)

def recommend_by_watched_genres(watched_titles, movies, top_n=10):
    final_cols = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies.columns:
        final_cols.append('tmdbId')

    if not watched_titles:
        return pd.DataFrame(columns=final_cols)

    watched_movies_df, all_genres = _extract_watched_movies_and_genres(watched_titles, movies.copy())
    watched_movie_ids = pd.Series(dtype='int64')
    if not watched_movies_df.empty and 'movieId' in watched_movies_df.columns:
        watched_movie_ids = watched_movies_df['movieId']

    recommendations = pd.DataFrame(columns=final_cols)
    if all_genres:
        recommendations = _get_genre_based_recommendations(movies, all_genres, watched_movie_ids, top_n)

    if recommendations.empty:
        recommendations = _get_fallback_recommendations(movies, watched_movie_ids, top_n)

    if recommendations.empty:
        return pd.DataFrame(columns=final_cols)

    for col in final_cols:
        if col not in recommendations.columns:
            recommendations[col] = pd.NA

    return recommendations[final_cols].head(top_n).reset_index(drop=True)

def recommend_similar_movies_partial(
    movie_title,
    movies_with_content_for_tfidf,
    tfidf_matrix,
    movies_for_output_columns,
    watched_movie_titles_to_exclude,
    top_n=10,
    internal_candidate_count=20
):
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_for_output_columns.columns:
        cols_to_return.append('tmdbId')

    if not movie_title or not str(movie_title).strip():
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    cleaned_movie_title = clean_text(str(movie_title)).lower()
    if not cleaned_movie_title:
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    if 'title_for_matching' not in movies_with_content_for_tfidf.columns:
        st.error("Critical: 'title_for_matching' not in DataFrame for TF-IDF. Cannot find movie.")
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    movies_with_content_for_tfidf['title_for_matching'] = movies_with_content_for_tfidf['title_for_matching'].fillna('').astype(str)
    matches = movies_with_content_for_tfidf[movies_with_content_for_tfidf['title_for_matching'].str.contains(cleaned_movie_title, na=False)]

    matched_movie_original_title = None

    if matches.empty:
        best_fuzz_score = 0
        best_fuzz_idx = -1
        for idx_val, row_title_for_matching in movies_with_content_for_tfidf['title_for_matching'].items():
            score = fuzz.ratio(cleaned_movie_title, row_title_for_matching)
            if score > best_fuzz_score:
                best_fuzz_score = score
                best_fuzz_idx = idx_val

        if best_fuzz_score > 80 and best_fuzz_idx != -1:
            matches = movies_with_content_for_tfidf.loc[[best_fuzz_idx]]
        else:
            return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    if matches.empty:
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    idx = matches.index[0]

    matched_movie_id_from_tfidf_source = movies_with_content_for_tfidf.loc[idx, 'movieId']

    if 'movieId' not in movies_for_output_columns.columns or 'title' not in movies_for_output_columns.columns:
        st.error("Critical: 'movieId' or 'title' not in the DataFrame for output columns.")
        matched_movie_original_title = movies_with_content_for_tfidf.loc[idx, 'title'] if 'title' in movies_with_content_for_tfidf else "Title Unavailable"
    else:
        matched_movie_row_for_display = movies_for_output_columns[movies_for_output_columns['movieId'] == matched_movie_id_from_tfidf_source]
        if matched_movie_row_for_display.empty:
            matched_movie_original_title = movies_with_content_for_tfidf.loc[idx, 'title'] if 'title' in movies_with_content_for_tfidf else "Title Unavailable"
        else:
            matched_movie_original_title = matched_movie_row_for_display['title'].iloc[0]


    cosine_sim_vector = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similar_indices_with_self = cosine_sim_vector.argsort()[-(internal_candidate_count + 1):][::-1]
    similar_indices_for_tfidf_df = [sim_idx for sim_idx in similar_indices_with_self if sim_idx != idx][:internal_candidate_count]

    if not similar_indices_for_tfidf_df:
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), matched_movie_original_title

    if 'movieId' not in movies_with_content_for_tfidf.columns:
        st.error("Critical: 'movieId' not in DataFrame for TF-IDF. Cannot create recommendations.")
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), matched_movie_original_title

    temp_recs_df = movies_with_content_for_tfidf.iloc[similar_indices_for_tfidf_df][['movieId']].copy()
    temp_recs_df['similarity_score'] = cosine_sim_vector[similar_indices_for_tfidf_df]

    if 'movieId' not in movies_for_output_columns.columns:
        st.error("Critical: 'movieId' not in the DataFrame for output columns. Cannot merge recommendations.")
        recommendations = movies_with_content_for_tfidf.iloc[similar_indices_for_tfidf_df][cols_to_return].copy()
        recommendations['similarity_score'] = cosine_sim_vector[similar_indices_for_tfidf_df]
    else:
        recommendations = movies_for_output_columns[
            movies_for_output_columns['movieId'].isin(temp_recs_df['movieId'])
        ].copy()
        recommendations = recommendations.merge(
            temp_recs_df[['movieId', 'similarity_score']],
            on='movieId',
            how='left'
        )

    if watched_movie_titles_to_exclude and not recommendations.empty:
        if 'title' in recommendations.columns:
             recommendations = recommendations[~recommendations['title'].isin(watched_movie_titles_to_exclude)]
        else:
            pass

    final_recommendations = recommendations.sort_values(by='similarity_score', ascending=False)

    output_columns_with_score = cols_to_return + ['similarity_score']
    for col in output_columns_with_score:
        if col not in final_recommendations.columns:
            final_recommendations[col] = pd.NA

    return final_recommendations[output_columns_with_score].head(top_n).reset_index(drop=True), matched_movie_original_title

def recommend_based_on_watch_history_content(
    watched_titles_list,
    movies_with_tags_for_tfidf,
    tfidf_matrix,
    main_movies_df,
    top_n=10
):
    if not watched_titles_list:
        return pd.DataFrame()

    all_recommendations_list = []
    actual_watched_movies_df, _ = _extract_watched_movies_and_genres(watched_titles_list, main_movies_df.copy())
    watched_movie_titles_to_exclude = set()
    if not actual_watched_movies_df.empty and 'title' in actual_watched_movies_df.columns:
        watched_movie_titles_to_exclude = set(actual_watched_movies_df['title'].unique())
    else:
        watched_movie_titles_to_exclude = set(watched_titles_list)

    for movie_title_seed in watched_titles_list:
        recs_for_seed_df, matched_title = recommend_similar_movies_partial(
            movie_title=movie_title_seed,
            movies_with_content_for_tfidf=movies_with_tags_for_tfidf,
            tfidf_matrix=tfidf_matrix,
            movies_for_output_columns=main_movies_df,
            watched_movie_titles_to_exclude=watched_movie_titles_to_exclude,
            top_n=top_n + 5,
            internal_candidate_count=top_n + 15
        )
        if matched_title and not recs_for_seed_df.empty:
            all_recommendations_list.append(recs_for_seed_df)

    if not all_recommendations_list:
        st.info("Could not generate seed recommendations from watch history.")
        return pd.DataFrame()

    if not all_recommendations_list:
        return pd.DataFrame()
    try:
        combined_recs_df = pd.concat(all_recommendations_list)
    except ValueError:
        st.info("No recommendations to combine from watch history.")
        return pd.DataFrame()

    if combined_recs_df.empty:
        st.info("Combined recommendations are empty before filtering duplicates.")
        return pd.DataFrame()

    combined_recs_df = combined_recs_df.sort_values(by='similarity_score', ascending=False)
    combined_recs_df = combined_recs_df.drop_duplicates(subset=['movieId'], keep='first')
    final_recommendations_df = combined_recs_df[~combined_recs_df['title'].isin(watched_movie_titles_to_exclude)]

    final_output_cols = ['movieId', 'title', 'genres']
    if 'tmdbId' in main_movies_df.columns:
        final_output_cols.append('tmdbId')

    for col in final_output_cols:
        if col not in final_recommendations_df.columns:
            final_recommendations_df[col] = pd.NA

    return final_recommendations_df[final_output_cols].head(top_n).reset_index(drop=True)

def show_table(df):
    if not df.empty:
        df_display = df.copy()
        df_display.index = range(1, len(df_display) + 1)
        st.dataframe(df_display)
    else:
        st.info("No data to display.")

def main():
    st.markdown("<h1 style='color:#1976d2;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("## 📋 Menu")

    base_dir_for_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    cleaned_data_path_in_app = os.path.join(base_dir_for_data, 'cleaned_data')

    movies = load_movies(data_path=cleaned_data_path_in_app)
    ratings = load_ratings(data_path=cleaned_data_path_in_app)
    tags = load_tags(data_path=cleaned_data_path_in_app)

    links_df = None
    links_file_path = os.path.join(base_dir_for_data, 'data', 'links.csv')
    try:
        links_df = pd.read_csv(links_file_path)
        if links_df.empty:
            st.warning("Warning: links.csv is empty. Poster functionality might be affected.")
        else:
            links_df = links_df[pd.notna(links_df['tmdbId'])].copy()
            if not links_df.empty:
                links_df['tmdbId'] = links_df['tmdbId'].astype(int)
    except FileNotFoundError:
        st.error(f"ERROR: links.csv not found at {links_file_path}. Poster functionality will be disabled.")
        links_df = pd.DataFrame(columns=['movieId', 'tmdbId'])
    except Exception as e:
        st.error(f"ERROR: An unexpected error occurred while loading links.csv: {e}")
        links_df = pd.DataFrame(columns=['movieId', 'tmdbId'])


    if movies.empty:
        st.error("Movie data could not be loaded. The application cannot continue.")
        st.stop()

    if 'movie_id_to_title' not in st.session_state:
        if not movies.empty and 'movieId' in movies.columns and 'title' in movies.columns:
            st.session_state.movie_id_to_title = pd.Series(movies['title'].values, index=movies['movieId']).to_dict()
        else:
            st.session_state.movie_id_to_title = {}

    if st.session_state.get('movies_added_to_watch_history_flag', False):
        st.session_state.add_selected_movies_multiselect = []
        st.session_state.movies_added_to_watch_history_flag = False

    tfidf_matrix, tfidf_vectorizer, movies_with_tags = get_tfidf_matrix(movies.copy(), tags.copy())

    if not movies.empty and (links_df is not None and not links_df.empty) and 'tmdbId' in links_df.columns:
        movies = movies.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
    if not movies_with_tags.empty and (links_df is not None and not links_df.empty) and 'tmdbId' in links_df.columns:
        movies_with_tags = movies_with_tags.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')

    content_based_enabled = tfidf_matrix is not None and tfidf_vectorizer is not None and not movies_with_tags.empty
    if not content_based_enabled:
        st.warning(
            "TF-IDF matrix and related components could not be generated. "
            "Content-based recommendations will be disabled."
        )

    surprise_model = load_trained_surprise_model()
    if surprise_model is None:
        st.warning("Collaborative filtering model could not be loaded. This feature may be unavailable.")

    if 'watched_movies' not in st.session_state:
        st.session_state['watched_movies'] = set()
    if 'add_selected_movies_multiselect' not in st.session_state:
        st.session_state.add_selected_movies_multiselect = []
    if 'movies_added_to_watch_history_flag' not in st.session_state:
        st.session_state.movies_added_to_watch_history_flag = False

    choice = st.sidebar.radio("Choose a recommendation method:", MENU_ITEMS, key="main_menu_choice")

    # =================== CONTENT-BASED ===================
    if choice == MENU_ITEMS[0]:
        st.success("**Content-Based Recommendation**")
        if not content_based_enabled:
            st.error("Content-based recommendation is currently unavailable.")
        else:
            movie_title_cb = st.text_input("🎬 Enter a movie title you like (no need for year):", key="cb_movie_title_input")
            if st.button("Get Recommendations", key="cb_get_recs_button"):
                if not movie_title_cb.strip():
                    st.warning("Please enter a movie title.")
                else:
                    recs_df, matched_title = recommend_similar_movies_partial(
                        movie_title=movie_title_cb,
                        movies_with_content_for_tfidf=movies_with_tags,
                        tfidf_matrix=tfidf_matrix,
                        movies_for_output_columns=movies,
                        watched_movie_titles_to_exclude=st.session_state.get('watched_movies', set()),
                        top_n=10
                    )
                    if matched_title:
                        st.info(f"Showing recommendations based on: **{matched_title}**")

                    if not recs_df.empty:
                        with st.expander("See Recommendations", expanded=True):
                            if 'movieId' not in recs_df.columns and 'tmdbId' not in recs_df.columns:
                                st.warning("Recommendation data is missing 'movieId' or 'tmdbId' for poster lookup.")
                                temp_display_df = recs_df.copy()
                                if 'title' not in temp_display_df.columns: temp_display_df['title'] = "N/A"
                                if 'genres' not in temp_display_df.columns: temp_display_df['genres'] = "N/A"
                                show_table(temp_display_df[['title', 'genres']])
                            else:
                                for index, row in recs_df.iterrows():
                                    title_display = row.get('title', "Title not available")
                                    genres_display = row.get('genres', "Genres not available")
                                    st.subheader(f"{index + 1}. {title_display}")
                                    st.write(f"**Genres:** {genres_display}")

                                    tmdb_id_to_fetch = None
                                    if 'tmdbId' in row and pd.notna(row['tmdbId']):
                                        tmdb_id_to_fetch = int(row['tmdbId'])
                                    elif 'movieId' in row and pd.notna(row['movieId']) and (links_df is not None and not links_df.empty):
                                        link_info = links_df[links_df['movieId'] == row['movieId']]
                                        if not link_info.empty and 'tmdbId' in link_info.columns and pd.notna(link_info.iloc[0]['tmdbId']):
                                            tmdb_id_to_fetch = int(link_info.iloc[0]['tmdbId'])

                                    if tmdb_id_to_fetch:
                                        movie_details = get_movie_details_from_tmdb(tmdb_id_to_fetch, TMDB_API_KEY)
                                        if movie_details and movie_details.get("poster_url"):
                                            col1, col2 = st.columns([1, 3])
                                            with col1:
                                                st.image(movie_details["poster_url"], width=150)
                                            with col2:
                                                if movie_details.get("overview"):
                                                    st.caption(f"Overview: {movie_details['overview']}")
                                                else:
                                                    st.caption("Overview not available.")
                                        elif movie_details:
                                            st.caption("Poster not found on TMDB.")
                                            if movie_details.get("overview"):
                                                    st.caption(f"Overview: {movie_details['overview']}")
                                        else:
                                            st.caption("Details (including poster) not found on TMDB.")
                                    else:
                                        st.caption("TMDB ID not found for this movie, so poster cannot be displayed.")
                                    st.markdown("---")
                    else:
                        st.warning("No recommendations found. Try a different title.")

    # =================== COLLABORATIVE FILTERING ===================
    elif choice == MENU_ITEMS[1]:
        st.success("**Collaborative Filtering Recommendation**")
        st.markdown(""" ### Personalized SVD Recommendations

        This section provides SVD-based movie suggestions.

        -   **Demo Profiles:** Recommendations for these profiles are SVD-generated and then post-filtered by the profile's target genre(s).
        -   **Manual User ID:** Get general, unfiltered SVD recommendations by entering a MovieLens User ID.
        """)

        DEMO_PROFILES_WITH_GENRES = {
            "Select a Demo Profile...": {"id": None, "target_genre_cols": []},
            "🎬 Comedy Fan": {"id": 88539, "target_genre_cols": ['genre_comedy']},
            "💥 Action & Thriller Seeker": {"id": 129440, "target_genre_cols": ['genre_action', 'genre_thriller']},
            "🎭 Drama Enthusiast": {"id": 110971, "target_genre_cols": ['genre_drama']},
            "🔮 Sci-Fi & Fantasy Voyager": {"id": 78616, "target_genre_cols": ['genre_scifi', 'genre_fantasy']},
            "🧸 Animation & Family Watcher": {"id": 93359, "target_genre_cols": ['genre_animation', 'genre_children']}
        }

        chosen_profile_name = st.selectbox(
            "Explore recommendations for a demo profile:",
            options=list(DEMO_PROFILES_WITH_GENRES.keys()),
            key="cf_demo_profile_selectbox_v4",
            index=0
        )

        user_id_to_process = None
        target_cols_for_filter = []

        persona_definition = DEMO_PROFILES_WITH_GENRES.get(chosen_profile_name)
        if persona_definition and persona_definition["id"] is not None:
            user_id_to_process = persona_definition["id"]
            target_cols_for_filter = persona_definition["target_genre_cols"]

        if user_id_to_process is None:
            manual_user_id_input_str = st.text_input(
                "Or, enter a specific MovieLens User ID (e.g., 1):",
                key="cf_manual_userid_input_v4",
                help="Enter a numeric User ID from the MovieLens dataset for general SVD recommendations."
            ).strip()

            if manual_user_id_input_str:
                if manual_user_id_input_str.isdigit():
                    user_id_to_process = int(manual_user_id_input_str)
                else:
                    if chosen_profile_name == "Select a Demo Profile...":
                        st.warning("Please enter a valid numeric User ID or select a demo profile.")

        if st.button("Get Collaborative Recommendations", key="cf_get_recs_button_v4"):
            if user_id_to_process is not None:
                if surprise_model is not None:
                    if ratings is not None and not ratings.empty:

                        recs_df = pd.DataFrame()

                        if target_cols_for_filter:
                            st.markdown(f"### Showing SVD Recommendations for '{chosen_profile_name}' profile (Filtered by Target Genre(s))")
                            recs_df = get_filtered_svd_recommendations_for_persona(
                                user_id=user_id_to_process,
                                persona_target_genre_cols=target_cols_for_filter,
                                model=surprise_model,
                                movies_data=movies,
                                ratings_data=ratings,
                                watched_titles=st.session_state.get('watched_movies', set()),
                                top_n_final=10
                            )
                        elif chosen_profile_name == "Select a Demo Profile..." and user_id_to_process is not None:
                             st.markdown(f"### Showing General SVD Recommendations for User ID: {user_id_to_process}")
                             recs_df = get_user_recommendations(
                                         user_id=user_id_to_process,
                                         surprise_model=surprise_model,
                                         movies_df=movies,
                                         ratings_df=ratings,
                                         watched_titles=st.session_state.get('watched_movies', set()),
                                         top_n=10
                                     )
                        else:
                            if chosen_profile_name != "Select a Demo Profile...":
                                st.error(f"Target genres are not properly defined for the selected profile: {chosen_profile_name}. "
                                         "Please check the 'target_genre_cols' in DEMO_PROFILES_WITH_GENRES.")

                        if not recs_df.empty:
                            with st.expander("See Recommendations", expanded=True):
                                for i, row in recs_df.reset_index(drop=True).iterrows():
                                    title_display = row.get('title', "Title not available")
                                    genres_display = row.get('genres', "Genres not available")

                                    st.subheader(f"{i + 1}. {title_display}")
                                    st.write(f"**Genres:** {genres_display}")

                                    tmdb_id_to_fetch = None
                                    if 'tmdbId' in row and pd.notna(row['tmdbId']):
                                        tmdb_id_to_fetch = int(row['tmdbId'])
                                    elif 'movieId' in row and pd.notna(row['movieId']) and (links_df is not None and not links_df.empty):
                                        link_info = links_df[links_df['movieId'] == row['movieId']]
                                        if not link_info.empty and 'tmdbId' in link_info.columns and pd.notna(link_info.iloc[0]['tmdbId']):
                                            tmdb_id_to_fetch = int(link_info.iloc[0]['tmdbId'])

                                    if tmdb_id_to_fetch:
                                        movie_details = get_movie_details_from_tmdb(tmdb_id_to_fetch, TMDB_API_KEY)
                                        if movie_details and movie_details.get("poster_url"):
                                            col1, col2 = st.columns([1,3])
                                            with col1:
                                                st.image(movie_details["poster_url"], width=150)
                                            with col2:
                                                overview_text = movie_details.get('overview', 'Overview not available.')
                                                st.caption(f"Overview: {overview_text}")
                                        elif movie_details:
                                            overview_text = movie_details.get('overview', 'Overview not available.')
                                            st.caption(f"Poster not found on TMDB. Overview: {overview_text}")
                                        else:
                                            st.caption("Details (including poster) could not be retrieved from TMDB.")
                                    else:
                                        st.caption("TMDB ID not found, so poster and overview cannot be displayed.")
                                    st.markdown("---")
                        else:
                            if user_id_to_process:
                                if chosen_profile_name != "Select a Demo Profile..." and target_cols_for_filter :
                                    st.info(f"No SVD recommendations found matching the genres for the '{chosen_profile_name}' profile after filtering. "
                                            "This might mean the SVD model didn't rank target genre movies high enough for this user profile, "
                                            "or all such movies were already in the global watch history. "
                                            "You could try a different profile or the Content-Based recommender.")
                                else:
                                    st.warning(f"No new recommendations found for User ID {user_id_to_process}. This could be due to various reasons "
                                               "(e.g., user has rated many movies, all potential recommendations are in the global watch history, "
                                               "or the User ID is not in the model's training data if entered manually).")
                    else:
                        st.error("Ratings data is not available or empty. Cannot generate collaborative recommendations.")
                else:
                    st.error("The collaborative filtering model (Surprise model) is currently unavailable. Please ensure it's trained and loaded correctly.")
            else:
                st.warning("Please select a demo profile or enter a valid User ID to get recommendations.")

    # =================== MOOD-BASED ===================
    elif choice == MENU_ITEMS[2]:
        st.success("**Mood-Based Recommendation**")
        mood_selected = st.selectbox("Select your mood:", list(MOOD_GENRE_MAP.keys()), key="mood_selectbox_input_v2")
        if st.button("Get Mood-Based Recommendations", key="mood_get_recs_button_v2"):
            recs_df = recommend_by_mood(
                mood_selected,
                movies,
                st.session_state.get('watched_movies', set()),
                top_n=10
            )
            if not recs_df.empty:
                with st.expander("See Recommendations", expanded=True):
                    for i, row in recs_df.reset_index(drop=True).iterrows():
                        title_display = row.get('title', "Title not available")
                        genres_display = row.get('genres', "Genres not available")
                        st.subheader(f"{i + 1}. {title_display}")
                        st.write(f"**Genres:** {genres_display}")

                        tmdb_id_to_fetch = None
                        if 'tmdbId' in row and pd.notna(row['tmdbId']):
                            tmdb_id_to_fetch = int(row['tmdbId'])
                        elif 'movieId' in row and pd.notna(row['movieId']) and (links_df is not None and not links_df.empty):
                            link_info = links_df[links_df['movieId'] == row['movieId']]
                            if not link_info.empty and 'tmdbId' in link_info.columns and pd.notna(link_info.iloc[0]['tmdbId']):
                                tmdb_id_to_fetch = int(link_info.iloc[0]['tmdbId'])

                        if tmdb_id_to_fetch:
                            movie_details = get_movie_details_from_tmdb(tmdb_id_to_fetch, TMDB_API_KEY)
                            if movie_details and movie_details.get("poster_url"):
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.image(movie_details["poster_url"], width=150)
                                with col2:
                                    if movie_details.get("overview"):
                                        st.caption(f"Overview: {movie_details['overview']}")
                                    else:
                                        st.caption("Overview not available.")
                            elif movie_details:
                                st.caption("Poster not found on TMDB.")
                                if movie_details.get("overview"):
                                    st.caption(f"Overview: {movie_details['overview']}")
                            else:
                                st.caption("Details (including poster) not found on TMDB.")
                        else:
                            st.caption("TMDB ID not found, poster cannot be displayed.")
                        st.markdown("---")
            else:
                st.warning("No movies found for this mood or all were in your watch history.")

    # =================== RANDOM MOVIE ===================
    elif choice == MENU_ITEMS[3]:
        st.success("**Random Movie Recommendation**")

        available_genres = []
        if not movies.empty and 'genres' in movies.columns:
            all_genres_list = movies['genres'].str.split('|').explode().str.strip().unique()
            available_genres = sorted([genre for genre in all_genres_list if genre and genre != '(no genres listed)'])

        selected_genres_for_random = []
        if available_genres:
            selected_genres_for_random = st.multiselect(
                "Filter by Genre(s) (optional):",
                options=available_genres,
                key="random_movie_genre_filter_multiselect_v1"
            )
        else:
            st.caption("No genres available for filtering or movies data is not loaded correctly.")

        if st.button("Pick a Random Movie", key="random_movie_button_v3"):
            filtered_movies_for_random = movies.copy()

            if selected_genres_for_random:
                genre_mask = pd.Series([False] * len(filtered_movies_for_random), index=filtered_movies_for_random.index)
                for genre_filter in selected_genres_for_random:
                    genre_mask |= filtered_movies_for_random['genres'].str.contains(genre_filter, case=False, na=False)

                filtered_movies_for_random = filtered_movies_for_random[genre_mask]

            if not filtered_movies_for_random.empty:
                movie_picked = pick_random_movie(filtered_movies_for_random)

                if movie_picked is not None:
                    st.info(f"**Title:** {movie_picked.get('title', 'N/A')}")
                    st.info(f"**Genres:** {movie_picked.get('genres', 'N/A')}")

                    tmdb_id_to_fetch = None
                    if 'tmdbId' in movie_picked and pd.notna(movie_picked['tmdbId']):
                        tmdb_id_to_fetch = int(movie_picked['tmdbId'])
                    elif 'movieId' in movie_picked and pd.notna(movie_picked['movieId']) and (links_df is not None and not links_df.empty):
                        link_info = links_df[links_df['movieId'] == movie_picked['movieId']]
                        if not link_info.empty and 'tmdbId' in link_info.columns and pd.notna(link_info.iloc[0]['tmdbId']):
                            tmdb_id_to_fetch = int(link_info.iloc[0]['tmdbId'])

                    if tmdb_id_to_fetch:
                        movie_details = get_movie_details_from_tmdb(tmdb_id_to_fetch, TMDB_API_KEY)
                        if movie_details and movie_details.get("poster_url"):
                            st.image(movie_details["poster_url"], width=200)
                        if movie_details and movie_details.get("overview"):
                            st.caption(f"Overview: {movie_details['overview']}")
                        elif movie_details:
                            st.caption("Poster or overview not available on TMDB.")
                        else:
                            st.caption("Details could not be retrieved from TMDB.")
                    else:
                        st.caption("TMDB ID not found for this movie, so poster and overview cannot be displayed.")
                else:
                    st.warning("Could not pick a random movie from the filtered selection (it might be empty after filtering).")
            else:
                if selected_genres_for_random:
                    st.warning(f"No movies found matching the selected genre(s): {', '.join(selected_genres_for_random)}. Try different genres or no filter.")
                else:
                    st.warning("No movies available in the database to pick from.")

    # =================== WATCH HISTORY & RECOMMENDATIONS ===================
    elif choice == MENU_ITEMS[4]:
        st.success("**Watch History & Personalized Recommendations**")

        if 'watched_movies' not in st.session_state:
            st.session_state.watched_movies = set()

        if not movies.empty and 'title' in movies.columns:
            all_movie_titles = movies['title'].dropna().sort_values().unique().tolist()
            selectable_movies = [
                title for title in all_movie_titles
                if title not in st.session_state.watched_movies
            ]
        else:
            selectable_movies = []
            all_movie_titles = []

        if selectable_movies:
            st.multiselect(
                "Select movies to add to your watch history:",
                options=selectable_movies,
                key="multiselect_add_watched_movies_key"
            )
            if st.button("Add Selected to Watch History", key="add_selected_to_watch_history_button_v3"):
                selected_movies_to_add = st.session_state.multiselect_add_watched_movies_key
                if selected_movies_to_add:
                    for movie_title_add in selected_movies_to_add:
                        st.session_state.watched_movies.add(movie_title_add)
                    st.success(f"{len(selected_movies_to_add)} movie(s) added to your watch history.")
                    st.rerun()
                else:
                    st.warning("Please select at least one movie to add.")
        elif not movies.empty and 'title' in movies.columns and not all_movie_titles:
             st.warning("Movie list is empty or contains no valid titles to select from.")
        elif movies.empty or 'title' not in movies.columns:
            st.warning("Movie list is not available to make selections.")
        else:
            st.info("All movies from the list are already in your watch history or the movie list is empty.")


        if st.session_state.watched_movies:
            st.write("Your current watch history:")
            watched_list_for_df = sorted(list(st.session_state.watched_movies))

            watched_df = pd.DataFrame(watched_list_for_df, columns=['Title'])
            watched_df.index = range(1, len(watched_df) + 1)
            st.dataframe(watched_df, height=min(300, len(watched_df) * 40 + 40), use_container_width=True)

            st.markdown("---")
            st.subheader("Manage Your Watch History")

            if watched_list_for_df:
                movies_to_remove_selection = st.multiselect(
                    "Select movies to remove from your watch history:",
                    options=watched_list_for_df,
                    key="multiselect_remove_watched_movies_key"
                )
                if st.button("Remove Selected from Watch History", key="remove_selected_from_watch_history_button_v3"):
                    if movies_to_remove_selection:
                        removed_count = 0
                        for movie_title_remove in movies_to_remove_selection:
                            if movie_title_remove in st.session_state.watched_movies:
                                st.session_state.watched_movies.remove(movie_title_remove)
                                removed_count += 1
                        if removed_count > 0:
                            st.success(f"{removed_count} movie(s) removed from your watch history.")
                            st.rerun()
                        else:
                            st.info("Selected movies were not found in the current watch history (perhaps selection was cleared or they were already removed).")
                    else:
                        st.warning("Please select at least one movie to remove.")
        else:
            st.info("Your watch history is currently empty. Add movies using the selection field above.")

        if st.button("Get Recommendations Based on Watch History", key="get_recs_watch_history_button_v3"):
            watched_titles_set = st.session_state.watched_movies
            if not watched_titles_set:
                st.warning("Your watch history is empty. Please add some movies to get personalized suggestions.")
            else:
                if not content_based_enabled:
                    st.error("Content-based components are not available for watch history recommendations.")
                else:
                    recs_based_on_watched = recommend_based_on_watch_history_content(
                        watched_titles_list=list(watched_titles_set),
                        movies_with_tags_for_tfidf=movies_with_tags,
                        tfidf_matrix=tfidf_matrix,
                        main_movies_df=movies,
                        top_n=10
                    )

                    if not recs_based_on_watched.empty:
                        st.subheader("Recommendations based on your watch history:")
                        with st.expander("See Recommendations", expanded=True):
                            for i, row in recs_based_on_watched.reset_index(drop=True).iterrows():
                                title_display = row.get('title', "Title not available")
                                genres_display = row.get('genres', "Genres not available")
                                st.subheader(f"{i + 1}. {title_display}")
                                st.write(f"**Genres:** {genres_display}")

                                tmdb_id_to_fetch = None
                                if 'tmdbId' in row and pd.notna(row['tmdbId']):
                                    tmdb_id_to_fetch = int(row['tmdbId'])
                                elif 'movieId' in row and pd.notna(row['movieId']) and (links_df is not None and not links_df.empty):
                                    link_info = links_df[links_df['movieId'] == row['movieId']]
                                    if not link_info.empty and 'tmdbId' in link_info.columns and pd.notna(link_info.iloc[0]['tmdbId']):
                                        tmdb_id_to_fetch = int(link_info.iloc[0]['tmdbId'])

                                if tmdb_id_to_fetch:
                                    movie_details = get_movie_details_from_tmdb(tmdb_id_to_fetch, TMDB_API_KEY)
                                    if movie_details and movie_details.get("poster_url"):
                                        col1, col2 = st.columns([1, 3])
                                        with col1:
                                            st.image(movie_details["poster_url"], width=150)
                                        with col2:
                                            if movie_details.get("overview"):
                                                st.caption(f"Overview: {movie_details['overview']}")
                                            else:
                                                st.caption("Overview not available.")
                                    elif movie_details:
                                        st.caption("Poster not found on TMDB.")
                                        if movie_details.get("overview"):
                                            st.caption(f"Overview: {movie_details['overview']}")
                                    else:
                                        st.caption("Details (including poster) not found on TMDB.")
                                else:
                                    st.caption("TMDB ID not found, poster cannot be displayed.")
                                st.markdown("---")
                    else:
                        st.warning("No new recommendations found based on your current watch history. Try adding more diverse movies.")

    # =================== ABOUT / HELP ===================
    elif choice == MENU_ITEMS[5]:
        st.success("**About & Help**")
        st.markdown(""" ### Welcome to the Movie Recommendation System!

        This application demonstrates various movie recommendation techniques:

        *   **Content-Based Filtering:** Recommends movies similar to one you like, based on movie titles, genres, and tags.
        *   **Collaborative Filtering (SVD):**
            *   **Demo Profiles:** Shows recommendations for pre-defined user profiles with specific genre preferences. These are SVD-based, then filtered by the profile's target genre(s).
            *   **Manual User ID:** Provides general SVD recommendations for any MovieLens User ID.
        *   **Mood-Based:** Suggests movies based on your selected mood, mapped to relevant genres.
        *   **Random Movie Picker:** Lets you discover a random movie, optionally filtered by genre.
        *   **Watch History & Personalized Recs:**
            *   Build and manage your own watch history.
            *   Get content-based recommendations derived from your entire watch history.

        **Data Sources:**
        *   MovieLens 25M Dataset (movies, ratings, tags).
        *   TMDB API for movie posters and overviews.

        **Key Technologies:**
        *   Streamlit for the web application interface.
        *   Pandas for data manipulation.
        *   Scikit-learn for TF-IDF vectorization and cosine similarity.
        *   Surprise library for the SVD collaborative filtering model.
        *   TheFuzz for fuzzy string matching of movie titles.

        **How to Use:**
        1.  Select a recommendation method from the sidebar menu.
        2.  Follow the on-screen prompts (e.g., enter a movie title, select a mood, or choose a demo profile).
        3.  View the generated recommendations.
        4.  Use the "Watch History" section to curate a list of movies you've seen and get recommendations based on it.

        **Notes on Demo Profiles (Collaborative Filtering):**
        The User IDs for demo profiles were chosen by analyzing the MovieLens dataset to find users with strong preferences for specific genres. This involved looking for users who:
        *   Rated a sufficient number of movies overall (e.g., >30 ratings).
        *   Had a significant portion of their ratings within the target genre(s) (e.g., >20% of total ratings).
        *   Gave high ratings (e.g., >= 4.0) to a good number of movies within those target genres.
        *   Specifically, at least 5 high scores (4.0+) in the target genre.
        This helps ensure that the SVD model, when applied to these user IDs, is likely to recommend movies relevant to the demo profile's theme, which are then further refined by the genre filter.

        **Troubleshooting:**
        *   If a model (e.g., Surprise SVD) is not loaded, related features will be disabled. Ensure `model.pkl` is present.
        *   If `links.csv` (for TMDB IDs) is missing, poster functionality will be affected.
        *   Content-based recommendations require `movies_clean.csv` and `tags_clean.csv`.
        *   Ensure your `TMDB_API_KEY` is correctly set in `config.py` for poster and overview fetching.

        Enjoy exploring movie recommendations!
        """)

    # =================== FOOTER ===================
    st.sidebar.markdown("---")
    st.sidebar.info("Movie Recommendation System v1.2")

if __name__ == '__main__':
    main()