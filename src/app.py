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
    # Determine the number of candidates to fetch, considering watched titles for potential exclusion later
    # This aims to ensure we have enough *net new* recommendations after filtering.
    # The candidate_pool_size for raw predictions can be larger than top_n + len(watched_titles)
    # to give more room for subsequent filtering and to ensure diversity.
    # Let's use a slightly larger pool than strictly necessary, e.g., top_n + len(watched_titles) + a buffer.
    # Or, we can fetch a fixed larger number like INITIAL_CANDIDATE_POOL_SIZE if that's deemed sufficient.
    # For this refactoring, let's use a dynamic size based on top_n and watched_titles, plus a buffer.
    num_candidates_to_fetch = top_n + (len(watched_titles) if watched_titles else 0) + 20 

    raw_predictions_df = _get_raw_svd_predictions(user_id, surprise_model, movies_df, ratings_df, candidate_pool_size=num_candidates_to_fetch)

    if raw_predictions_df.empty:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'] + (['tmdbId'] if 'tmdbId' in movies_df.columns else []))

    # Merge with movie details
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_df.columns:
        cols_to_return.append('tmdbId')
    
    # Ensure we only try to merge with columns that exist in movies_df
    valid_cols_for_merge = [col for col in cols_to_return if col in movies_df.columns]
    if 'movieId' not in valid_cols_for_merge: # movieId is essential for merge
        valid_cols_for_merge.insert(0, 'movieId')
        valid_cols_for_merge = list(set(valid_cols_for_merge))


    recommended_movies_df = pd.merge(
        raw_predictions_df[['movieId']], # Only need movieId for merging initially
        movies_df[valid_cols_for_merge],
        on='movieId',
        how='left'
    )
    
    # Filter out watched titles
    if watched_titles and not recommended_movies_df.empty and 'title' in recommended_movies_df.columns:
        recommended_movies_df = recommended_movies_df[~recommended_movies_df['title'].isin(watched_titles)]

    # Re-order based on original prediction scores if necessary, though merge usually preserves left df order
    # If raw_predictions_df was already sorted, and merge was 'left', order should be mostly fine.
    # However, to be absolutely sure, we can re-apply the order from raw_predictions_df
    # This requires 'movieId' to be in recommended_movies_df
    if not recommended_movies_df.empty and 'movieId' in recommended_movies_df.columns:
        # Create a mapping of movieId to its original sort order from raw_predictions_df
        order_map = {movie_id: i for i, movie_id in enumerate(raw_predictions_df['movieId'])}
        # Filter recommended_movies_df to only include movies that are in the order_map
        # (handles cases where some movies might have been dropped if not in movies_df)
        recommended_movies_df = recommended_movies_df[recommended_movies_df['movieId'].isin(order_map)]

        if not recommended_movies_df.empty: # Check again after filtering
            recommended_movies_df['sort_order'] = recommended_movies_df['movieId'].map(order_map)
            recommended_movies_df.sort_values('sort_order', inplace=True)
            recommended_movies_df.drop(columns=['sort_order'], inplace=True)


    # Ensure all expected columns are present before returning
    final_cols_to_return = ['movieId', 'title', 'genres'] + (['tmdbId'] if 'tmdbId' in movies_df.columns else [])
    for col in final_cols_to_return:
        if col not in recommended_movies_df.columns:
            recommended_movies_df[col] = pd.NA # Or some other appropriate default

    return recommended_movies_df[final_cols_to_return].head(top_n)


def get_filtered_svd_recommendations_for_persona(
    user_id,
    persona_target_genre_cols, # ['genre_comedy'], ['genre_action', 'genre_adventure'] gibi
    model,                     # surprise_model
    movies_data,               # one-hot encoded genre'ları içeren ana movies DataFrame'i
    ratings_data,              # tam ratings DataFrame'i (veya modelin eğitildiği veri)
    watched_titles,            # Kullanıcının genel izleme geçmişi (başlıklar)
    top_n_final=10
    # initial_candidate_pool_size is now imported from config
):
    """
    SVD önerilerini alır, belirtilen persona hedef türlerine göre filtreler
    ve izlenmiş filmleri çıkarır. Sonuç olarak bir DataFrame döndürür.
    """
    # Define columns to bring from movies_data at the beginning
    cols_to_bring_from_movies_data = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_data.columns:
        cols_to_bring_from_movies_data.append('tmdbId')
    
    # Add persona target genre columns to the list of columns to bring, if they exist in movies_data
    for gc in persona_target_genre_cols:
        if gc in movies_data.columns and gc not in cols_to_bring_from_movies_data:
            cols_to_bring_from_movies_data.append(gc)
        elif gc not in movies_data.columns:
            st.error(f"Error: Persona target genre column '{gc}' not found in movies_data DataFrame. "
                     f"Please check your 'movies_clean.csv' and preprocessing script to ensure "
                     f"one-hot encoded genre columns (e.g., 'genre_comedy') exist.")
            return pd.DataFrame()


    # 1. Get raw SVD predictions using the helper function and INITIAL_CANDIDATE_POOL_SIZE from config
    raw_predictions_df = _get_raw_svd_predictions(user_id, model, movies_data, ratings_data, candidate_pool_size=INITIAL_CANDIDATE_POOL_SIZE)

    if raw_predictions_df.empty:
        return pd.DataFrame() # Return empty df with appropriate columns later if needed

    # 2. Merge raw predictions with movie details (including genre columns for filtering)
    # We only need 'movieId' and 'predicted_score' from raw_predictions_df for the merge,
    # and the specified columns from movies_data.
    candidate_movies_with_details = pd.merge(
        raw_predictions_df[['movieId', 'predicted_score']], 
        movies_data[cols_to_bring_from_movies_data], # Use the predefined list
        on='movieId',
        how='left' # Keep all predictions, fill missing movie details with NaN if any (should not happen with clean data)
    )
    
    # Fill NaN in genre columns that were merged (important for the sum operation later)
    # This step ensures that if a movie somehow didn't have a value for a genre_col after merge, it's treated as 0.
    for genre_col in persona_target_genre_cols:
        if genre_col in candidate_movies_with_details.columns: 
            candidate_movies_with_details[genre_col] = candidate_movies_with_details[genre_col].fillna(0).astype(int)

    # 3. Filter by persona target genres
    if persona_target_genre_cols: 
        # Ensure we only use genre columns that are actually present in the merged DataFrame
        valid_persona_genre_cols_for_filtering = [col for col in persona_target_genre_cols if col in candidate_movies_with_details.columns]
        
        if not valid_persona_genre_cols_for_filtering:
             st.warning("No valid target persona genre columns found in candidate movies for filtering. Showing unfiltered SVD recommendations (but still excluding watched).")
             filtered_recommendations_df = candidate_movies_with_details.copy() 
        else:
            # Movies must have at least one of the target persona genres
            filter_mask = candidate_movies_with_details[valid_persona_genre_cols_for_filtering].sum(axis=1) > 0
            filtered_recommendations_df = candidate_movies_with_details[filter_mask]
    else: 
        # No persona genres specified, so no genre filtering
        filtered_recommendations_df = candidate_movies_with_details.copy()

    if filtered_recommendations_df.empty:
        print(f"No movies found for User ID {user_id} matching persona genres from the SVD pool (or pool was empty).")
        return pd.DataFrame() # Consider returning with specific columns
        
    # 4. Filter out watched movies (by title)
    if watched_titles and not filtered_recommendations_df.empty:
        if 'title' in filtered_recommendations_df.columns:
            filtered_recommendations_df = filtered_recommendations_df[
                ~filtered_recommendations_df['title'].isin(watched_titles)
            ]
        # else: 'title' column should exist due to cols_to_bring_from_movies_data

    # 5. Get the top N results, already sorted by 'predicted_score' from _get_raw_svd_predictions
    final_df_to_show = filtered_recommendations_df.head(top_n_final)
    
    # 6. Define and ensure final output columns
    output_cols = ['movieId', 'title', 'genres', 'predicted_score'] # 'predicted_score' is useful for context
    if 'tmdbId' in movies_data.columns and 'tmdbId' not in output_cols : 
        output_cols.append('tmdbId')

    # Ensure all desired output columns are present, adding them with NA if missing
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

    matched_movie_original_title = None # Initialize

    if matches.empty:
        best_fuzz_score = 0
        best_fuzz_idx = -1
        # Ensure 'title_for_matching' is used for fuzzy matching if it exists and is prepared
        # The fuzzy matching should iterate over the same source that tfidf_matrix is based on.
        for idx_val, row_title_for_matching in movies_with_content_for_tfidf['title_for_matching'].items():
            score = fuzz.ratio(cleaned_movie_title, row_title_for_matching) # Compare with cleaned title_for_matching
            if score > best_fuzz_score:
                best_fuzz_score = score
                best_fuzz_idx = idx_val # Store the index from movies_with_content_for_tfidf

        if best_fuzz_score > 80 and best_fuzz_idx != -1:
            matches = movies_with_content_for_tfidf.loc[[best_fuzz_idx]]
            # matched_movie_original_title would be set after idx is determined from 'matches'
        else:
            return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None # No good match found

    # This check should be after 'matches' is confirmed to be non-empty
    if matches.empty: # Should not be hit if logic above is correct, but as a safeguard
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    idx = matches.index[0] # Index from movies_with_content_for_tfidf
    
    # Determine matched_movie_original_title using the 'movieId' from the TF-IDF source DF
    # and looking it up in the 'movies_for_output_columns' DF.
    matched_movie_id_from_tfidf_source = movies_with_content_for_tfidf.loc[idx, 'movieId']

    if 'movieId' not in movies_for_output_columns.columns or 'title' not in movies_for_output_columns.columns:
        st.error("Critical: 'movieId' or 'title' not in the DataFrame for output columns.")
        # Fallback to title from the tfidf source if output df is problematic
        matched_movie_original_title = movies_with_content_for_tfidf.loc[idx, 'title'] if 'title' in movies_with_content_for_tfidf else "Title Unavailable"
    else:
        matched_movie_row_for_display = movies_for_output_columns[movies_for_output_columns['movieId'] == matched_movie_id_from_tfidf_source]
        if matched_movie_row_for_display.empty:
            # Fallback if movieId not found in output_df, though this indicates a data consistency issue
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

    # pd.concat([]) ValueError: No objects to concatenate
    if not all_recommendations_list: # Double check before concat
        return pd.DataFrame()
    try:
        combined_recs_df = pd.concat(all_recommendations_list)
    except ValueError: # If list is still empty for some reason
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
        df_display = df.copy() # Use a different variable name to avoid modifying input df
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
                                    st.subheader(f"{index + 1}. {title_display}") # Use df index + 1 for numbering
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

    # =================== COLLABORATIVE FILTERING (YENİDEN DÜZENLENMİŞ) ===================
    elif choice == MENU_ITEMS[1]:
        st.success("**Collaborative Filtering Recommendation**")
        st.markdown("""
        ### Personalized SVD Recommendations

        This section provides SVD-based movie suggestions.

        -   **Demo Profiles:** Recommendations for these profiles are SVD-generated and then post-filtered by the profile's target genre(s). The userId for each demo profile was selected based on specific criteria to ensure strong genre preference: at least 30 total ratings, ≥20% of ratings in the target genre, ≥50% of those in-genre ratings being 4.0+, and at least 5 high scores (4.0+) in the genre.
        -   **Manual User ID:** Get general, unfiltered SVD recommendations by entering a MovieLens User ID.
        """)

        # 1. Demo Profilleri Tanımlayın 
        # LÜTFEN BU userId'LERİ KENDİ VERİ SETİNİZDEN BULDUĞUNUZ
        # GERÇEK VE ANLAMLI ID'LERLE DEĞİŞTİRİN!
        # 'target_genre_cols' içindeki sütun adları, movies DataFrame'inizdeki
        # one-hot encoded tür sütun adlarıyla BİREBİR AYNI OLMALIDIR.
        # Örneğin, preprocess_dataset.py'de clean_text_for_matching("Comedy") -> "comedy" ise,
        # sütun adı 'genre_comedy' olmalıdır.
        DEMO_PROFILES_WITH_GENRES = {
            "Select a Demo Profile...": {"id": None, "target_genre_cols": []},
            "🎬 Comedy Fan": {"id": 88539, "target_genre_cols": ['genre_comedy']},
            "💥 Action & Thriller Seeker": {"id": 129440, "target_genre_cols": ['genre_action', 'genre_thriller']},
            "🎭 Drama Enthusiast": {"id": 110971, "target_genre_cols": ['genre_drama']},
            "🔮 Sci-Fi & Fantasy Voyager": {"id": 78616, "target_genre_cols": ['genre_scifi', 'genre_fantasy']},
            "🧸 Animation & Family Watcher": {"id": 93359, "target_genre_cols": ['genre_animation', 'genre_children']}
        }

        # 2. Kullanıcının Demo Profil Seçmesi İçin Selectbox
        chosen_profile_name = st.selectbox(
            "Explore recommendations for a demo profile:",
            options=list(DEMO_PROFILES_WITH_GENRES.keys()),
            key="cf_demo_profile_selectbox_v4", # Yeni ve benzersiz bir key
            index=0
        )

        # 3. Öneri İçin Kullanılacak Nihai User ID ve Filtre Türleri
        user_id_to_process = None
        target_cols_for_filter = []

        persona_definition = DEMO_PROFILES_WITH_GENRES.get(chosen_profile_name)
        if persona_definition and persona_definition["id"] is not None:
            user_id_to_process = persona_definition["id"]
            target_cols_for_filter = persona_definition["target_genre_cols"]

        # 4. Manuel User ID Giriş Alanı
        if user_id_to_process is None: # Demo profil seçilmemişse manuel girişe izin ver
            manual_user_id_input_str = st.text_input(
                "Or, enter a specific MovieLens User ID (e.g., 1):",
                key="cf_manual_userid_input_v4", # Yeni key
                help="Enter a numeric User ID from the MovieLens dataset for general SVD recommendations."
            ).strip()

            if manual_user_id_input_str:
                if manual_user_id_input_str.isdigit():
                    user_id_to_process = int(manual_user_id_input_str)
                    # Manuel ID için hedef tür filtresi yok (target_cols_for_filter boş kalacak)
                else:
                    if chosen_profile_name == "Select a Demo Profile...":
                        st.warning("Please enter a valid numeric User ID or select a demo profile.")

        # 5. Öneri Alma Butonu
        if st.button("Get Collaborative Recommendations", key="cf_get_recs_button_v4"): # Yeni key
            if user_id_to_process is not None:
                if surprise_model is not None:
                    if ratings is not None and not ratings.empty:

                        recs_df = pd.DataFrame() # Başlangıçta boş DataFrame

                        if target_cols_for_filter: # Demo profil seçilmiş VE hedef türleri tanımlanmışsa
                            st.markdown(f"### Showing SVD Recommendations for '{chosen_profile_name}' profile (Filtered by Target Genre(s))")
                            recs_df = get_filtered_svd_recommendations_for_persona(
                                user_id=user_id_to_process,
                                persona_target_genre_cols=target_cols_for_filter,
                                model=surprise_model,
                                movies_data=movies, # Ana movies DataFrame'i (one-hot genre'lı)
                                ratings_data=ratings, # Tam ratings DataFrame'i
                                watched_titles=st.session_state.get('watched_movies', set()),
                                top_n_final=10
                            )
                        elif chosen_profile_name == "Select a Demo Profile..." and user_id_to_process is not None: # Manuel ID girilmişse (filtresiz)
                             st.markdown(f"### Showing General SVD Recommendations for User ID: {user_id_to_process}")
                             recs_df = get_user_recommendations( # Sizin mevcut filtresiz fonksiyonunuz
                                         user_id=user_id_to_process,
                                         surprise_model=surprise_model,
                                         movies_df=movies,
                                         ratings_df=ratings,
                                         watched_titles=st.session_state.get('watched_movies', set()),
                                         top_n=10
                                     )
                        else: # Beklenmedik bir durum veya demo profil seçilmiş ama hedef türü yok (yapılandırma hatası)
                            if chosen_profile_name != "Select a Demo Profile...": # Sadece demo profil seçiliyken bu hatayı ver
                                st.error(f"Target genres are not properly defined for the selected profile: {chosen_profile_name}. "
                                         "Please check the 'target_genre_cols' in DEMO_PROFILES_WITH_GENRES.")
                        
                        # --- ÖNERİ GÖSTERME KISMI ---
                        if not recs_df.empty:
                            with st.expander("See Recommendations", expanded=True):
                                for i, row in recs_df.reset_index(drop=True).iterrows():
                                    title_display = row.get('title', "Title not available")
                                    genres_display = row.get('genres', "Genres not available")
                                    # predicted_score_display = row.get('predicted_score', None) # Eğer skor göstermek isterseniz

                                    st.subheader(f"{i + 1}. {title_display}")
                                    st.write(f"**Genres:** {genres_display}")
                                    # if predicted_score_display is not None:
                                    #     st.write(f"Predicted Score: {predicted_score_display:.2f}")


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
                        else: # recs_df boş ise
                            if user_id_to_process: # Eğer bir kullanıcı ID'si işlenmeye çalışıldıysa
                                if chosen_profile_name != "Select a Demo Profile..." and target_cols_for_filter :
                                    st.info(f"No SVD recommendations found matching the genres for the '{chosen_profile_name}' profile after filtering. "
                                            "This might mean the SVD model didn't rank target genre movies high enough for this user profile, "
                                            "or all such movies were already in the global watch history. "
                                            "You could try a different profile or the Content-Based recommender.")
                                else: # Manuel ID veya demo profil için filtrelenmemiş ama yine de boş
                                    st.warning(f"No new recommendations found for User ID {user_id_to_process}. This could be due to various reasons "
                                               "(e.g., user has rated many movies, all potential recommendations are in the global watch history, "
                                               "or the User ID is not in the model's training data if entered manually).")
                            # else: # user_id_to_process None ise zaten aşağıdaki "Please select..." uyarısı çıkacak
                    else:
                        st.error("Ratings data is not available or empty. Cannot generate collaborative recommendations.")
                else:
                    st.error("The collaborative filtering model (Surprise model) is currently unavailable. Please ensure it's trained and loaded correctly.")
            else:
                st.warning("Please select a demo profile or enter a valid User ID to get recommendations.")

    # =================== MOOD-BASED ===================
    elif choice == MENU_ITEMS[2]:
        st.success("**Mood-Based Recommendation**")
        mood_selected = st.selectbox("Select your mood:", list(MOOD_GENRE_MAP.keys()), key="mood_selectbox_input_v2") # Yeni key
        if st.button("Get Mood-Based Recommendations", key="mood_get_recs_button_v2"): # Yeni key
            recs_df = recommend_by_mood(
                mood_selected,
                movies,
                st.session_state.get('watched_movies', set()),
                top_n=10
            )
            if not recs_df.empty:
                with st.expander("See Recommendations", expanded=True):
                    for i, row in recs_df.reset_index(drop=True).iterrows(): # Numaralandırma için reset_index
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

        # Get available genres for the filter
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
                # Filter movies that contain AT LEAST ONE of the selected genres
                # This requires genres to be in string format and then checking for substring presence for each selected genre
                # A movie is kept if any of its genres match any of the selected genres.
                # We assume genres in the DataFrame are like "Action|Adventure|Sci-Fi"
                
                # Create a boolean mask, True for rows that match at least one selected genre
                genre_mask = pd.Series([False] * len(filtered_movies_for_random), index=filtered_movies_for_random.index)
                for genre_filter in selected_genres_for_random:
                    # Ensure case-insensitivity if needed, though genre list is usually consistent
                    genre_mask |= filtered_movies_for_random['genres'].str.contains(genre_filter, case=False, na=False)
                
                filtered_movies_for_random = filtered_movies_for_random[genre_mask]

            if not filtered_movies_for_random.empty:
                movie_picked = pick_random_movie(filtered_movies_for_random) # pick_random_movie now handles empty df
                
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
                        elif movie_details: # Details fetched but no poster/overview
                            st.caption("Poster or overview not available on TMDB.")
                        else: # No details fetched
                            st.caption("Details could not be retrieved from TMDB.")
                    else:
                        st.caption("TMDB ID not found for this movie, so poster and overview cannot be displayed.")
                else: # movie_picked was None
                    st.warning("Could not pick a random movie from the filtered selection (it might be empty after filtering).")
            else:
                if selected_genres_for_random:
                    st.warning(f"No movies found matching the selected genre(s): {', '.join(selected_genres_for_random)}. Try different genres or no filter.")
                else:
                    st.warning("No movies available in the database to pick from.")

    # =================== WATCH HISTORY & RECOMMENDATIONS ===================
    elif choice == MENU_ITEMS[4]:
        st.success("**Watch History & Personalized Recommendations**")

        # Initialize session state for watched_movies if it doesn't exist
        if 'watched_movies' not in st.session_state:
            st.session_state.watched_movies = set()

        if not movies.empty and 'title' in movies.columns:
            all_movie_titles = movies['title'].dropna().sort_values().unique().tolist()
            selectable_movies = [
                title for title in all_movie_titles
                if title not in st.session_state.watched_movies # Use .watched_movies directly
            ]
        else:
            selectable_movies = []
            all_movie_titles = [] # Ensure it's defined

        if selectable_movies:
            # Key for multiselect should be consistent if its value is read from st.session_state directly
            # Using a more descriptive key for the widget itself if needed, e.g., "add_to_watch_history_multiselect"
            # The state is accessed via st.session_state[key_name]
            st.multiselect(
                "Select movies to add to your watch history:",
                options=selectable_movies,
                key="multiselect_add_watched_movies_key" 
            )
            if st.button("Add Selected to Watch History", key="add_selected_to_watch_history_button_v3"): # New key
                selected_movies_to_add = st.session_state.multiselect_add_watched_movies_key
                if selected_movies_to_add:
                    for movie_title_add in selected_movies_to_add:
                        st.session_state.watched_movies.add(movie_title_add)
                    st.success(f"{len(selected_movies_to_add)} movie(s) added to your watch history.")
                    # st.session_state.movies_added_to_watch_history_flag = True # This flag might not be necessary if rerun is used
                    st.rerun()
                else:
                    st.warning("Please select at least one movie to add.")
        elif not movies.empty and 'title' in movies.columns and not all_movie_titles: # Corrected condition
             st.warning("Movie list is empty or contains no valid titles to select from.")
        elif movies.empty or 'title' not in movies.columns:
            st.warning("Movie list is not available to make selections.")
        else: # This means selectable_movies is empty but all_movie_titles is not, so all movies are watched
            st.info("All movies from the list are already in your watch history or the movie list is empty.")


        if st.session_state.watched_movies: # Check if the set itself is not empty
            st.write("Your current watch history:")
            # Sort the list for consistent display and use in multiselect for removal
            watched_list_for_df = sorted(list(st.session_state.watched_movies))
            
            # Display DataFrame
            watched_df = pd.DataFrame(watched_list_for_df, columns=['Title'])
            watched_df.index = range(1, len(watched_df) + 1) # 1-based indexing
            st.dataframe(watched_df, height=min(300, len(watched_df) * 40 + 40), use_container_width=True)

            # --- Add section to remove movies from watch history ---
            st.markdown("---") 
            st.subheader("Manage Your Watch History")
            
            if watched_list_for_df: # Check if there's anything to remove
                movies_to_remove_selection = st.multiselect(
                    "Select movies to remove from your watch history:",
                    options=watched_list_for_df, 
                    key="multiselect_remove_watched_movies_key" # Unique key for this multiselect
                )
                if st.button("Remove Selected from Watch History", key="remove_selected_from_watch_history_button_v3"): # New unique key
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
            # --- End of remove section ---
        else:
            st.info("Your watch history is currently empty. Add movies using the selection field above.")

        if st.button("Get Recommendations Based on Watch History", key="get_recs_watch_history_button_v3"): # New key
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
                            for i, row in recs_based_on_watched.reset_index(drop=True).iterrows(): # Numaralandırma
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
                        st.info("Could not find new recommendations based on your current watch history. Try adding more diverse movies!")

    # =================== UNWATCHED MOVIES ===================
    elif choice == MENU_ITEMS[5]:
        st.subheader("🕵️ Unwatched Movies")
        if 'watched_movies' not in st.session_state or not st.session_state['watched_movies']:
            st.info("Your watch history is empty. Watch some movies first!")
        else:
            if 'title' in movies.columns:
                unwatched_movies = movies[~movies['title'].isin(st.session_state['watched_movies'])]
                if not unwatched_movies.empty:
                    st.markdown("### Here are some movies you haven't watched yet:")
                    # Sonucu daha yönetilebilir kılmak için ilk 20'sini göster
                    display_unwatched = unwatched_movies[['title', 'genres']].head(20)
                    display_unwatched.index = range(1, len(display_unwatched) + 1) # Numaralandırma
                    if 'show_table' in globals() and callable(globals()['show_table']):
                        show_table(display_unwatched)
                    else:
                        st.dataframe(display_unwatched)
                else:
                    st.info("You've watched all the movies in our database!")
            else:
                st.warning("Movie titles are not available to determine unwatched movies.")


if __name__ == "__main__":
    main()