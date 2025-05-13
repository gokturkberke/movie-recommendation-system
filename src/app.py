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
from config import TMDB_API_KEY, MOOD_GENRE_MAP # TMDB_API_KEY'in buradan geldiğini varsayıyorum
from utils_data import (
    load_movies,
    load_ratings,
    load_tags,
    load_trained_surprise_model,
    clean_text
)

@st.cache_data # API çağrılarını önbelleğe almak için
def get_movie_details_from_tmdb(tmdb_id, api_key):
    """
    Verilen TMDB ID'si için film detaylarını (özellikle poster yolunu) TMDB API'sinden çeker.
    """
    if pd.isna(tmdb_id): # Eğer tmdb_id NaN ise boş string döndür veya None
        return None

    api_url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}&language=en-US"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        overview = data.get('overview', '')
        title = data.get('title', '')

        if poster_path:
            full_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return {
                "poster_url": full_poster_url,
                "overview": overview,
                "tmdb_title": title
            }
        else:
            return {
                "poster_url": None,
                "overview": overview,
                "tmdb_title": title
            }
    except requests.exceptions.RequestException as e:
        print(f"TMDB API isteği sırasında hata (tmdb_id: {tmdb_id}): {e}")
        return None
    except Exception as e:
        print(f"Film detayı işlenirken beklenmedik hata (tmdb_id: {tmdb_id}): {e}")
        return None

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
    all_movie_ids = movies_df['movieId'].unique()

    user_rated_movie_ids = []
    if ratings_df is not None and not ratings_df.empty:
        user_rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    else:
        print("Warning: Ratings data is not available or empty in get_user_recommendations.")

    predictions = []
    unrated_movie_ids = [mid for mid in all_movie_ids if mid not in user_rated_movie_ids]

    for movie_id in unrated_movie_ids:
        pred = surprise_model.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    num_candidates_to_fetch = top_n + (len(watched_titles) if watched_titles else 0) + 20
    candidate_movie_ids_ordered = [movie_id for movie_id, score in predictions[:num_candidates_to_fetch]]

    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_df.columns:
        cols_to_return.append('tmdbId')

    if not candidate_movie_ids_ordered:
        return pd.DataFrame(columns=cols_to_return)

    recommended_movies_df = movies_df[movies_df['movieId'].isin(candidate_movie_ids_ordered)][cols_to_return].copy()

    if watched_titles and not recommended_movies_df.empty:
        recommended_movies_df = recommended_movies_df[~recommended_movies_df['title'].isin(watched_titles)]

    if not recommended_movies_df.empty:
        final_candidate_ids_in_order = [
            mid for mid in candidate_movie_ids_ordered
            if mid in recommended_movies_df['movieId'].values
        ]
        if final_candidate_ids_in_order:
            recommended_movies_df = recommended_movies_df.set_index('movieId').loc[final_candidate_ids_in_order].reset_index()
            for col in cols_to_return:
                if col not in recommended_movies_df.columns:
                     recommended_movies_df[col] = pd.NA
        else:
             recommended_movies_df = pd.DataFrame(columns=cols_to_return)
    else:
        recommended_movies_df = pd.DataFrame(columns=cols_to_return)

    return recommended_movies_df[cols_to_return].head(top_n)


def get_filtered_svd_recommendations_for_persona(
    user_id,
    persona_target_genre_cols, # ['genre_comedy'], ['genre_action', 'genre_adventure'] gibi
    model,                     # surprise_model
    movies_data,               # one-hot encoded genre'ları içeren ana movies DataFrame'i
    ratings_data,              # tam ratings DataFrame'i (veya modelin eğitildiği veri)
    watched_titles,            # Kullanıcının genel izleme geçmişi (başlıklar)
    top_n_final=10,
    initial_candidate_pool_size=300
):
    """
    SVD önerilerini alır, belirtilen persona hedef türlerine göre filtreler
    ve izlenmiş filmleri çıkarır. Sonuç olarak bir DataFrame döndürür.
    """
    # Gerekli tür sütunlarının varlığını kontrol et
    for genre_col in persona_target_genre_cols:
        if genre_col not in movies_data.columns:
            st.error(f"Error: Column '{genre_col}' not found in movies_data DataFrame. "
                     f"Please check your 'movies_clean.csv' and preprocessing script to ensure "
                     f"one-hot encoded genre columns (e.g., 'genre_comedy') exist.")
            return pd.DataFrame()

    # 1. SVD'den ham tahminleri al
    all_movie_ids_in_moviedata = movies_data['movieId'].unique()
    
    user_rated_movie_ids = []
    if ratings_data is not None and not ratings_data.empty:
        user_rated_movie_ids = ratings_data[ratings_data['userId'] == user_id]['movieId'].unique()

    movies_to_predict_ids = [movie_id for movie_id in all_movie_ids_in_moviedata if movie_id not in user_rated_movie_ids]

    if not movies_to_predict_ids:
        print(f"User {user_id} has no new movies to predict for (they may have rated all movies).")
        return pd.DataFrame()

    predictions_list = [] 
    for movie_id_to_predict in movies_to_predict_ids:
        predicted_rating = model.predict(uid=user_id, iid=movie_id_to_predict).est # uid ve iid parametrelerini doğru kullandığınızdan emin olun
        predictions_list.append({'movieId': movie_id_to_predict, 'predicted_score': predicted_rating})
    
    if not predictions_list:
        return pd.DataFrame()

    predictions_df = pd.DataFrame(predictions_list)
    predictions_df.sort_values(by='predicted_score', ascending=False, inplace=True)

    # 2. Filtreleme için ilk adayları seç
    candidate_movies_df = predictions_df.head(initial_candidate_pool_size)

    # 3. Adayları film detayları (movieId, title, genres, tmdbId ve hedef tür sütunları) ile birleştir
    # movies_data'dan birleştirilecek sütunları belirle
    cols_to_bring_from_movies_data = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_data.columns:
        cols_to_bring_from_movies_data.append('tmdbId')
    
    # Hedef tür sütunlarını da ekle (varsa)
    for gc in persona_target_genre_cols:
        if gc in movies_data.columns and gc not in cols_to_bring_from_movies_data:
            cols_to_bring_from_movies_data.append(gc)
            
    candidate_movies_with_details = pd.merge(
        candidate_movies_df[['movieId', 'predicted_score']], 
        movies_data[cols_to_bring_from_movies_data], 
        on='movieId',
        how='left'
    )
    
    # Birleştirme sonrası NaN olan hedef tür sütunlarını 0 ile doldur
    for genre_col in persona_target_genre_cols:
        if genre_col in candidate_movies_with_details.columns: 
            candidate_movies_with_details[genre_col] = candidate_movies_with_details[genre_col].fillna(0).astype(int)
        # else: # Bu sütun movies_data'da yoksa, yukarıdaki kontrol zaten hata vermiştir.

    # 4. Hedef türlere göre filtrele
    if persona_target_genre_cols: 
        valid_persona_genre_cols = [col for col in persona_target_genre_cols if col in candidate_movies_with_details.columns]
        if not valid_persona_genre_cols:
             st.warning("No valid target genre columns found in candidate movies for filtering. Showing unfiltered SVD recommendations.")
             filtered_recommendations_df = candidate_movies_with_details.copy() # Filtresiz devam et
        else:
            filter_mask = candidate_movies_with_details[valid_persona_genre_cols].sum(axis=1) > 0
            filtered_recommendations_df = candidate_movies_with_details[filter_mask]
    else: 
        filtered_recommendations_df = candidate_movies_with_details.copy()

    if filtered_recommendations_df.empty:
        print(f"No movies found for User ID {user_id} matching persona genres from the initial SVD pool.")
        return pd.DataFrame()
        
    # 5. İzlenmiş filmleri (başlığa göre) çıkar
    if watched_titles and not filtered_recommendations_df.empty:
        if 'title' in filtered_recommendations_df.columns:
            filtered_recommendations_df = filtered_recommendations_df[
                ~filtered_recommendations_df['title'].isin(watched_titles)
            ]
        # else: # 'title' sütunu yoksa filtreleme yapılamaz, bu durum yukarıda merge'de ele alınmalı

    # 6. Sonuçları SVD tahmin skoruna göre sıralı tutarak top_n_final kadar al
    final_df_to_show = filtered_recommendations_df.head(top_n_final)
    
    # Gerekli sütunları döndür (predicted_score da faydalı olabilir)
    output_cols = ['movieId', 'title', 'genres', 'predicted_score']
    if 'tmdbId' in movies_data.columns and 'tmdbId' not in output_cols : 
        output_cols.append('tmdbId')

    final_df_to_show = final_df_to_show[[col for col in output_cols if col in final_df_to_show.columns]]

    return final_df_to_show.reset_index(drop=True)

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

def pick_random_movie(movies):
    return movies.sample(n=1).iloc[0]

def _extract_watched_movies_and_genres(watched_titles, movies_input_df, similarity_threshold=85):
    all_genres = set()
    final_watched_movies_df_list = []

    movies_df_copy = movies_input_df.copy()
    remaining_titles_for_fuzzy_match = list(watched_titles)
    movies_df_copy['title'] = movies_df_copy['title'].astype(str)

    for title_query in watched_titles:
        title_query_str = str(title_query)
        exact_matches = movies_df_copy[movies_df_copy['title'] == title_query_str]
        if not exact_matches.empty:
            final_watched_movies_df_list.append(exact_matches)
            if title_query_str in remaining_titles_for_fuzzy_match:
                remaining_titles_for_fuzzy_match.remove(title_query_str)

    if remaining_titles_for_fuzzy_match:
        cleaned_user_titles_for_fuzzy = [clean_text(t).lower() for t in remaining_titles_for_fuzzy_match if clean_text(t)]
        if 'title_for_matching' in movies_df_copy.columns and cleaned_user_titles_for_fuzzy:
            movies_df_copy['title_for_matching_fuzzy'] = movies_df_copy['title_for_matching'].fillna('').astype(str).apply(lambda x: clean_text(x).lower())
            already_added_movie_ids = set()
            if final_watched_movies_df_list:
                try:
                    temp_df_exact = pd.concat(final_watched_movies_df_list)
                    if not temp_df_exact.empty and 'movieId' in temp_df_exact.columns:
                        already_added_movie_ids.update(temp_df_exact['movieId'].unique())
                except ValueError:
                    pass

            for cleaned_title_query in cleaned_user_titles_for_fuzzy:
                if not cleaned_title_query: continue
                best_match_score = 0
                best_match_index = -1
                for index, row in movies_df_copy.iterrows():
                    if 'movieId' in row and row['movieId'] in already_added_movie_ids:
                        continue
                    score = fuzz.partial_ratio(cleaned_title_query, row['title_for_matching_fuzzy'])
                    if score > best_match_score:
                        best_match_score = score
                        best_match_index = index
                if best_match_score >= similarity_threshold and best_match_index != -1:
                    matched_movie_id = movies_df_copy.loc[best_match_index, 'movieId']
                    if matched_movie_id not in already_added_movie_ids:
                        final_watched_movies_df_list.append(movies_df_copy.loc[[best_match_index]])
                        already_added_movie_ids.add(matched_movie_id)

    if not final_watched_movies_df_list:
        final_watched_movies_df = pd.DataFrame()
    else:
        try:
            final_watched_movies_df = pd.concat(final_watched_movies_df_list)
            if not final_watched_movies_df.empty and 'movieId' in final_watched_movies_df.columns:
                final_watched_movies_df = final_watched_movies_df.drop_duplicates(subset=['movieId']).reset_index(drop=True)
            else:
                final_watched_movies_df = pd.DataFrame()
        except ValueError:
             final_watched_movies_df = pd.DataFrame()


    if not final_watched_movies_df.empty and 'genres' in final_watched_movies_df.columns:
        for genres_str in final_watched_movies_df['genres'].dropna().values:
            all_genres.update(str(genres_str).split('|'))

    return final_watched_movies_df, all_genres

def _get_genre_based_recommendations(movies_df, all_genres_set, watched_movie_ids, top_n):
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_df.columns:
        cols_to_return.append('tmdbId')

    if not all_genres_set:
        return pd.DataFrame(columns=cols_to_return)

    genre_matches = movies_df[movies_df['genres'].apply(
        lambda g: isinstance(g, str) and any(genre_item in g.split('|') for genre_item in all_genres_set)
    )]
    recommendations = genre_matches.copy()
    if watched_movie_ids is not None and not watched_movie_ids.empty:
        recommendations = recommendations[~recommendations['movieId'].isin(watched_movie_ids)]

    if recommendations.empty:
        return pd.DataFrame(columns=cols_to_return)

    num_to_return = min(top_n, len(recommendations))
    return recommendations[cols_to_return].head(num_to_return).reset_index(drop=True)

def _get_fallback_recommendations(movies_df, watched_movie_ids, top_n):
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_df.columns:
        cols_to_return.append('tmdbId')

    recommendations_pool = movies_df.copy()
    if watched_movie_ids is not None and not watched_movie_ids.empty:
        recommendations_pool = recommendations_pool[~recommendations_pool['movieId'].isin(watched_movie_ids)]

    if recommendations_pool.empty:
        return pd.DataFrame(columns=cols_to_return)

    num_to_sample = min(top_n, len(recommendations_pool))
    return recommendations_pool[cols_to_return].sample(n=num_to_sample, random_state=42).reset_index(drop=True)

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

    if matches.empty:
        best_fuzz_score = 0
        best_fuzz_idx = -1
        for idx_val, row_title in movies_with_content_for_tfidf['title_for_matching'].items():
            score = fuzz.ratio(cleaned_movie_title, row_title)
            if score > best_fuzz_score:
                best_fuzz_score = score
                best_fuzz_idx = idx_val

        if best_fuzz_score > 80:
            matches = movies_with_content_for_tfidf.loc[[best_fuzz_idx]]
        else:
            return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    idx = matches.index[0]
    matched_movie_id_from_tfidf_source = movies_with_content_for_tfidf.loc[idx, 'movieId']

    if 'movieId' not in movies_for_output_columns.columns or 'title' not in movies_for_output_columns.columns:
        st.error("Critical: 'movieId' or 'title' not in the DataFrame for output columns.")
        matched_movie_original_title = movies_with_content_for_tfidf.loc[idx, 'title']
    else:
        matched_movie_row_for_display = movies_for_output_columns[movies_for_output_columns['movieId'] == matched_movie_id_from_tfidf_source]
        if matched_movie_row_for_display.empty:
            matched_movie_original_title = movies_with_content_for_tfidf.loc[idx, 'title']
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

    menu = [
        "🎯 Content-Based Recommendation",
        "👥 Collaborative Filtering",
        "😊 Mood-Based Recommendation",
        "🎲 Random Movie",
        "📽️ Watch History & Recommendations",
        "🕵️ Unwatched Movies"
    ]
    choice = st.sidebar.radio("Choose a recommendation method:", menu, key="main_menu_choice")

    # =================== CONTENT-BASED ===================
    if choice == menu[0]:
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
    elif choice == menu[1]:
        st.success("**Collaborative Filtering Recommendation**")
        st.markdown("""
        This section utilizes our Singular Value Decomposition (SVD) model to provide
        personalized movie recommendations. Explore suggestions for pre-defined demo profiles
        (which will be filtered by their main genres after SVD) or enter a specific
        MovieLens User ID for general SVD recommendations.
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
                                top_n_final=10,
                                initial_candidate_pool_size=300 # Jupyter'daki gibi
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
    elif choice == menu[2]:
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
    elif choice == menu[3]:
        st.success("**Random Movie**")
        if st.button("Pick a Random Movie", key="random_movie_button_v2"): # Yeni key
            if not movies.empty:
                movie_picked = pick_random_movie(movies)
                st.info(f"**Title:** {movie_picked['title']}")
                st.info(f"**Genres:** {movie_picked['genres']}")

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
                        st.caption("Details not found on TMDB.")
                else:
                    st.caption("TMDB ID not found for this movie, so poster and overview cannot be displayed.")
            else:
                st.warning("No movies available to pick from.")

    # =================== WATCH HISTORY & RECOMMENDATIONS ===================
    elif choice == menu[4]:
        st.success("**Watch History & Personalized Recommendations**")

        if not movies.empty and 'title' in movies.columns:
            all_movie_titles = movies['title'].dropna().sort_values().unique().tolist()
            selectable_movies = [
                title for title in all_movie_titles
                if title not in st.session_state.get('watched_movies', set())
            ]
        else:
            selectable_movies = []

        if selectable_movies:
            st.multiselect(
                "Select movies to add to your watch history:",
                options=selectable_movies,
                key="add_selected_movies_multiselect_v2" # Yeni key
            )
            if st.button("Add Selected to Watch History", key="add_selected_to_watch_history_button_v2"): # Yeni key
                selected_movies_to_add = st.session_state.add_selected_movies_multiselect_v2 # Key'i burada da güncelle
                if selected_movies_to_add:
                    for movie_title_add in selected_movies_to_add:
                        st.session_state['watched_movies'].add(movie_title_add)
                    st.success(f"{len(selected_movies_to_add)} movie(s) added to your watch history.")
                    st.session_state.movies_added_to_watch_history_flag = True
                    st.rerun()
                else:
                    st.warning("Please select at least one movie to add.")
        elif not movies.empty and 'title' in movies.columns and not all_movie_titles:
             st.warning("Movie list is empty or contains no valid titles to select from.")
        elif movies.empty or 'title' not in movies.columns:
            st.warning("Movie list is not available to make selections.")
        else:
            st.info("No new movies available to add to watch history (either all are watched or the movie list is empty).")

        if st.session_state.get('watched_movies', set()):
            st.write("Your current watch history:")
            watched_df = pd.DataFrame(list(st.session_state['watched_movies']), columns=['Title'])
            watched_df.index = range(1, len(watched_df) + 1)
            st.dataframe(watched_df, height=min(300, len(watched_df) * 40))
        else:
            st.info("Your watch history is currently empty. Add movies using the selection field above.")

        if st.button("Get Recommendations Based on Watch History", key="get_recs_watch_history_button_v2"): # Yeni key
            watched_titles_set = st.session_state.get('watched_movies', set())
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
    elif choice == menu[5]:
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