# src/utils_data.py
import streamlit as st
import pandas as pd
import os
import re # clean_text için eklendi
from surprise import dump # load_trained_surprise_model için eklendi
import requests # get_movie_details_from_tmdb için eklendi
from thefuzz import fuzz # _extract_watched_movies_and_genres için eklendi

@st.cache_data
def load_cleaned_data(filename, data_path='cleaned_data'):
    """Loads a CSV file from the cleaned_data directory with error handling."""
    file_path = os.path.join(data_path, filename)
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            st.warning(f"Warning: {filename} is empty.")
        return df
    except FileNotFoundError:
        st.error(f"ERROR: File not found: {file_path}. Please ensure the file exists.")
        return pd.DataFrame() # Return empty DataFrame on error
    except pd.errors.EmptyDataError:
        st.error(f"ERROR: No data: {filename} is empty or corrupted.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"ERROR: An unexpected error occurred while loading {filename}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

@st.cache_data
def load_movies(data_path='cleaned_data'):
    return load_cleaned_data('movies_clean.csv', data_path)

@st.cache_data
def load_ratings(data_path='cleaned_data'):
    return load_cleaned_data('ratings_clean.csv', data_path)

@st.cache_resource # Model gibi kaynaklar için cache_resource daha uygun
def load_trained_surprise_model(model_filename="svd_trained_model.pkl"):
    # Bu fonksiyonun içindeki path'ler app.py'nin konumuna göreydi,
    # utils_data.py'nin konumuna göre düzeltilmesi gerekebilir.
    # Şimdilik app.py ile aynı src klasöründe olduğu için çalışacaktır.
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Bu utils_data.py'nin olduğu src klasörü olacak
    cleaned_data_dir = os.path.join(script_dir, '..', 'cleaned_data') # src'nin bir üstündeki cleaned_data
    model_path = os.path.join(cleaned_data_dir, model_filename)

    print(f"Önceden eğitilmiş Surprise modeli yükleniyor: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"HATA: Kayıtlı model dosyası bulunamadı: {model_path}. "
                 f"Lütfen önce train_save_model.py script'ini çalıştırın.")
        return None

    try:
        loaded_object = dump.load(model_path)
        model = loaded_object[1]
        print("Önceden eğitilmiş model başarıyla yüklendi.")
        return model
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        return None

@st.cache_data
def load_tags(data_path='cleaned_data'):
    return load_cleaned_data('tags_clean.csv', data_path)

# Centralized text cleaning function (consistent with preprocess_dataset.py)
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    # Remove year in parentheses (e.g., "(1995)") if present
    text = re.sub(r'\s*\(\d{4}\)', '', text) # Yıl parantezlerini eşleştirmek için \( ve \)
# ...
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # \s zaten boşluk karakterlerini ifade eder
    # Normalize whitespace (replace multiple spaces with a single space and strip)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Functions moved from app.py

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

def _get_raw_svd_predictions(user_id, surprise_model, movies_df, ratings_df, candidate_pool_size=None):
    """
    Helper function to get raw SVD predictions for a user.
    Returns a DataFrame with 'movieId' and 'predicted_score'.
    """
    all_movie_ids = movies_df['movieId'].unique()

    user_rated_movie_ids = []
    if ratings_df is not None and not ratings_df.empty:
        user_rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    else:
        print("Warning: Ratings data is not available or empty in _get_raw_svd_predictions.")

    movies_to_predict_ids = [mid for mid in all_movie_ids if mid not in user_rated_movie_ids]

    if not movies_to_predict_ids:
        print(f"User {user_id} has no new movies to predict for (they may have rated all movies).")
        return pd.DataFrame(columns=['movieId', 'predicted_score'])

    predictions_list = []
    for movie_id_to_predict in movies_to_predict_ids:
        predicted_rating = surprise_model.predict(uid=user_id, iid=movie_id_to_predict).est
        predictions_list.append({'movieId': movie_id_to_predict, 'predicted_score': predicted_rating})

    if not predictions_list:
        return pd.DataFrame(columns=['movieId', 'predicted_score'])

    predictions_df = pd.DataFrame(predictions_list)
    predictions_df.sort_values(by='predicted_score', ascending=False, inplace=True)
    
    if candidate_pool_size:
        return predictions_df.head(candidate_pool_size)
    return predictions_df

def pick_random_movie(movies_df):
    if movies_df.empty:
        return None # Return None if the DataFrame is empty
    return movies_df.sample(n=1).iloc[0]

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
