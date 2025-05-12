import streamlit as st
import pandas as pd
import os
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from surprise import dump # Ensure this import is present
from thefuzz import fuzz
import requests # Added requests import

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

@st.cache_data # API çağrılarını önbelleğe almak için
def get_movie_details_from_tmdb(tmdb_id, api_key):
    """
    Verilen TMDB ID'si için film detaylarını (özellikle poster yolunu) TMDB API'sinden çeker.
    """
    if pd.isna(tmdb_id): # Eğer tmdb_id NaN ise boş string döndür veya None
        return None

    # TMDB API'sinin film detayları için temel URL'si
    # Dökümantasyon: https://developer.themoviedb.org/reference/movie-details
    api_url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}&language=en-US"

    try:
        response = requests.get(api_url)
        response.raise_for_status() # HTTP_STATUS_CODE 200 değilse hata fırlatır (4xx, 5xx)
        data = response.json()

        poster_path = data.get('poster_path')
        overview = data.get('overview', '') # Özeti de alalım, belki sonra kullanırız
        title = data.get('title', '') # TMDB'deki başlığı da alabiliriz

        if poster_path:
            # Posterlerin tam URL'sini oluşturmak için temel resim URL'si
            # Farklı boyutlar için 'w500' kısmını değiştirebilirsin (örn: 'w200', 'w300', 'original')
            full_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return {
                "poster_url": full_poster_url,
                "overview": overview,
                "tmdb_title": title
            }
        else:
            return { # Poster olmasa bile diğer bilgileri döndürebiliriz
                "poster_url": None,
                "overview": overview,
                "tmdb_title": title
            }

    except requests.exceptions.RequestException as e:
        print(f"TMDB API isteği sırasında hata (tmdb_id: {tmdb_id}): {e}")
        # Consider logging this to Streamlit if running in that environment
        # st.warning(f"Film detayları (ID: {tmdb_id}) TMDB'den alınırken bir sorun oluştu.")
        return None # Hata durumunda None döndür
    except Exception as e:
        print(f"Film detayı işlenirken beklenmedik hata (tmdb_id: {tmdb_id}): {e}")
        return None

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
    # app.py'nin bulunduğu dizin (src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # cleaned_data klasörünün yolu (src'nin bir üst dizininde)
    cleaned_data_dir = os.path.join(script_dir, '..', 'cleaned_data')
    model_path = os.path.join(cleaned_data_dir, model_filename)

    print(f"Önceden eğitilmiş Surprise modeli yükleniyor: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"HATA: Kayıtlı model dosyası bulunamadı: {model_path}. "
                 f"Lütfen önce train_save_model.py script'ini çalıştırın.")
        return None # Model yüklenemezse None döndür

    try:
        # dump.load() bir tuple döndürür: (predictions, algo)
        # Biz sadece algo'yu (modeli) istiyoruz.
        loaded_object = dump.load(model_path)
        model = loaded_object[1] # Model (algo) tuple'ın ikinci elemanıdır
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
    text = re.sub(r'\s*\(\d{4}\)', '', text)
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Normalize whitespace (replace multiple spaces with a single space and strip)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_tfidf_matrix(movies, tags):
    if movies.empty:
        st.warning("Movies DataFrame is empty. Cannot generate TF-IDF matrix for content-based recommendations.")
        # Return None for matrix and vectorizer, and the original (empty) movies DataFrame
        return None, None, movies

    tags['tag'] = tags['tag'].fillna('').apply(clean_text) # Use centralized clean_text
    tags = tags.drop_duplicates(subset=['movieId', 'tag'])
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    movies = movies.merge(tags_grouped, on='movieId', how='left')
    
    # Ensure all components of 'content' are strings and NaNs are explicitly handled
    movies['title_for_matching'] = movies['title_for_matching'].fillna('').astype(str)
    movies['genres_for_matching'] = movies['genres_for_matching'].fillna('').astype(str)
    movies['tag'] = movies['tag'].fillna('').astype(str) # Ensure tags are also strings
    
    movies['content'] = movies['title_for_matching'] + ' ' + movies['genres_for_matching'] + ' ' + movies['tag']
    # Final check for NaNs in the content column itself, though previous steps should prevent this
    movies['content'] = movies['content'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])

    # Additional check: if no features were learned (e.g., all content was stop words or empty)
    if tfidf_matrix.shape[1] == 0:
        st.warning("No features were learned from movie content for TF-IDF. Content-based recommendations might be ineffective.")
        # Depending on desired behavior, could also return None, None, movies here
        # For now, proceeding with the matrix but warning the user.

    return tfidf_matrix, tfidf, movies # movies here is movies_with_tags

def get_user_recommendations(user_id, surprise_model, movies_df, ratings_df, watched_titles, top_n=10):
    """
    Generates movie recommendations for a user using a trained Surprise model.
    Filters out movies the user has already rated and movies in the watched_titles list.
    Returns movieId, title, genres, and tmdbId if available.
    """
    all_movie_ids = movies_df['movieId'].unique()
    user_rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()

    predictions = []
    unrated_movie_ids = [mid for mid in all_movie_ids if mid not in user_rated_movie_ids]

    for movie_id in unrated_movie_ids:
        pred = surprise_model.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))
            
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    num_candidates_to_fetch = top_n + (len(watched_titles) if watched_titles else 0) + 10 
    candidate_movie_ids_ordered = [movie_id for movie_id, score in predictions[:num_candidates_to_fetch]]
    
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_df.columns:
        cols_to_return.append('tmdbId')
    
    if not candidate_movie_ids_ordered:
        return pd.DataFrame(columns=cols_to_return)

    # Get movie details for the candidates, ensuring all necessary columns are selected
    recommended_movies_df = movies_df[movies_df['movieId'].isin(candidate_movie_ids_ordered)][cols_to_return].copy()
    
    # Filter out watched movies by title from the candidates
    if watched_titles and not recommended_movies_df.empty:
        recommended_movies_df = recommended_movies_df[~recommended_movies_df['title'].isin(watched_titles)]

    # Re-order the filtered candidates based on original prediction score and select top_n
    if not recommended_movies_df.empty:
        # Filter candidate_movie_ids_ordered to only those that are still in recommended_movies_df
        final_candidate_ids_in_order = [
            mid for mid in candidate_movie_ids_ordered 
            if mid in recommended_movies_df['movieId'].values
        ]
        if final_candidate_ids_in_order:
            # Set index to movieId to use .loc for reordering, then reset index
            # Ensure all cols_to_return are present after reordering
            recommended_movies_df = recommended_movies_df.set_index('movieId').loc[final_candidate_ids_in_order].reset_index()
            # Double check columns, though loc should preserve them if they were in recommended_movies_df
            for col in cols_to_return:
                if col not in recommended_movies_df.columns:
                     recommended_movies_df[col] = pd.NA # Should not happen if cols_to_return used for selection
        else: # All candidates were filtered out or didn't exist
             recommended_movies_df = pd.DataFrame(columns=cols_to_return)
    else: # recommended_movies_df is already empty or became empty after watched_titles filter
        recommended_movies_df = pd.DataFrame(columns=cols_to_return)
        
    return recommended_movies_df[cols_to_return].head(top_n)


def recommend_by_mood(mood, movies, watched_movies, top_n=10):
    """
    Recommends movies based on mood, filtering watched movies.
    Returns movieId, title, genres, and tmdbId if available.
    """
    genres_for_mood = MOOD_GENRE_MAP.get(mood.lower()) # Renamed variable for clarity
    
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies.columns:
        cols_to_return.append('tmdbId')

    if not genres_for_mood:
        return pd.DataFrame(columns=cols_to_return)

    movies_copy = movies.copy()
    movies_copy['genres'] = movies_copy['genres'].astype(str) # Ensure genres is string
    
    mask = movies_copy['genres'].apply(lambda g: any(genre_item in g for genre_item in genres_for_mood))
    filtered_movies = movies_copy[mask]
    
    if filtered_movies.empty:
        return pd.DataFrame(columns=cols_to_return)
    
    # Determine how many movies to sample initially
    num_to_sample = min(top_n + (len(watched_movies) if watched_movies else 0) + 5, len(filtered_movies))

    if num_to_sample <= 0:
        return pd.DataFrame(columns=cols_to_return)
        
    # Sample and select the required columns
    recommendations = filtered_movies.sample(n=num_to_sample, random_state=42)[cols_to_return].copy() # Ensure copy
    
    # Filter out watched movies
    if watched_movies and not recommendations.empty:
        recommendations = recommendations[~recommendations['title'].isin(watched_movies)]
        
    return recommendations.head(top_n).reset_index(drop=True)

def pick_random_movie(movies):
    return movies.sample(n=1).iloc[0]

def _extract_watched_movies_and_genres(watched_titles, movies_input_df, similarity_threshold=85):
    all_genres = set()
    final_watched_movies_df_list = [] # Use a list to append DataFrames

    movies_df_copy = movies_input_df.copy()

    # 1. Attempt exact matches on original 'title'
    remaining_titles_for_fuzzy_match = list(watched_titles)
    
    # Ensure 'title' column is string type for exact matching
    movies_df_copy['title'] = movies_df_copy['title'].astype(str)

    for title_query in watched_titles:
        # Ensure title_query is a string
        title_query_str = str(title_query)
        exact_matches = movies_df_copy[movies_df_copy['title'] == title_query_str]
        if not exact_matches.empty:
            final_watched_movies_df_list.append(exact_matches)
            if title_query_str in remaining_titles_for_fuzzy_match:
                remaining_titles_for_fuzzy_match.remove(title_query_str)

    # 2. For remaining titles, attempt fuzzy matches
    if remaining_titles_for_fuzzy_match:
        # Clean user titles once for fuzzy matching
        cleaned_user_titles_for_fuzzy = [clean_text(t).lower() for t in remaining_titles_for_fuzzy_match if clean_text(t)]

        # Ensure 'title_for_matching' exists and prepare it for fuzzy matching
        if 'title_for_matching' in movies_df_copy.columns and cleaned_user_titles_for_fuzzy:
            movies_df_copy['title_for_matching_fuzzy'] = movies_df_copy['title_for_matching'].fillna('').astype(str).apply(lambda x: clean_text(x).lower())
            
            # To keep track of movieIds already added by exact or previous fuzzy matches
            # Initialize with movieIds from exact matches if any
            already_added_movie_ids = set()
            if final_watched_movies_df_list:
                temp_df_exact = pd.concat(final_watched_movies_df_list)
                if not temp_df_exact.empty and 'movieId' in temp_df_exact.columns:
                    already_added_movie_ids.update(temp_df_exact['movieId'].unique())


            for cleaned_title_query in cleaned_user_titles_for_fuzzy:
                if not cleaned_title_query: continue
                
                best_match_score = 0
                best_match_index = -1
                
                # Iterate through movies to find the best fuzzy match for the current cleaned_title_query
                for index, row in movies_df_copy.iterrows():
                    # Skip if this movie's ID was already added
                    if 'movieId' in row and row['movieId'] in already_added_movie_ids:
                        continue

                    # Use fuzz.partial_ratio for flexibility
                    score = fuzz.partial_ratio(cleaned_title_query, row['title_for_matching_fuzzy'])
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_match_index = index
                
                # If a good enough match is found and not already added
                if best_match_score >= similarity_threshold and best_match_index != -1:
                    matched_movie_id = movies_df_copy.loc[best_match_index, 'movieId']
                    if matched_movie_id not in already_added_movie_ids:
                        final_watched_movies_df_list.append(movies_df_copy.loc[[best_match_index]])
                        already_added_movie_ids.add(matched_movie_id)
        
    # Consolidate and remove duplicates based on movieId
    if not final_watched_movies_df_list:
        final_watched_movies_df = pd.DataFrame()
    else:
        final_watched_movies_df = pd.concat(final_watched_movies_df_list)
        if not final_watched_movies_df.empty and 'movieId' in final_watched_movies_df.columns:
            final_watched_movies_df = final_watched_movies_df.drop_duplicates(subset=['movieId']).reset_index(drop=True)
        else: # Handle case where movieId might be missing or df is empty after concat
            final_watched_movies_df = pd.DataFrame()


    # 3. Extract genres from the identified watched movies
    if not final_watched_movies_df.empty and 'genres' in final_watched_movies_df.columns:
        for genres_str in final_watched_movies_df['genres'].dropna().values: 
            all_genres.update(str(genres_str).split('|')) # Ensure genres_str is string
            
    return final_watched_movies_df, all_genres

def _get_genre_based_recommendations(movies_df, all_genres_set, watched_movie_ids, top_n):
    """
    Helper to get recommendations based on a set of genres, excluding watched movies.
    Returns movieId, title, genres, and tmdbId if available.
    """
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_df.columns:
        cols_to_return.append('tmdbId')

    if not all_genres_set:
        return pd.DataFrame(columns=cols_to_return)

    # Find movies with at least one matching genre
    genre_matches = movies_df[movies_df['genres'].apply(
        lambda g: isinstance(g, str) and any(genre_item in g.split('|') for genre_item in all_genres_set)
    )]

    recommendations = genre_matches.copy() # Work on a copy
    # Remove watched movies from recommendations
    # Ensure watched_movie_ids is a Series or similar iterable for isin
    if watched_movie_ids is not None and not watched_movie_ids.empty:
        recommendations = recommendations[~recommendations['movieId'].isin(watched_movie_ids)]
    
    if recommendations.empty:
        return pd.DataFrame(columns=cols_to_return)

    num_to_return = min(top_n, len(recommendations))
    return recommendations[cols_to_return].head(num_to_return).reset_index(drop=True)

def _get_fallback_recommendations(movies_df, watched_movie_ids, top_n):
    """
    Helper to get fallback (random) recommendations, excluding watched movies.
    Returns movieId, title, genres, and tmdbId if available.
    """
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_df.columns:
        cols_to_return.append('tmdbId')

    recommendations_pool = movies_df.copy() # Work on a copy
    if watched_movie_ids is not None and not watched_movie_ids.empty:
        recommendations_pool = recommendations_pool[~recommendations_pool['movieId'].isin(watched_movie_ids)]
    
    if recommendations_pool.empty:
        return pd.DataFrame(columns=cols_to_return)
        
    num_to_sample = min(top_n, len(recommendations_pool))
    # Ensure sampling is done on the correct columns
    return recommendations_pool[cols_to_return].sample(n=num_to_sample, random_state=42).reset_index(drop=True)

def recommend_by_watched_genres(watched_titles, movies, top_n=10):
    """
    Recommends movies based on genres of watched titles.
    Returns movieId, title, genres, and tmdbId if available.
    """
    final_cols = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies.columns:
        final_cols.append('tmdbId')

    if not watched_titles: # No titles, no recommendations
        return pd.DataFrame(columns=final_cols)

    # Step 1: Extract watched movies and their genres
    # _extract_watched_movies_and_genres uses 'movies' (which has tmdbId)
    # and should ideally preserve tmdbId in watched_movies_df if present.
    watched_movies_df, all_genres = _extract_watched_movies_and_genres(watched_titles, movies.copy()) 
    
    # Step 2: Get IDs of watched movies for exclusion
    watched_movie_ids = pd.Series(dtype='int64') # Default to empty Series
    if not watched_movies_df.empty and 'movieId' in watched_movies_df.columns:
        watched_movie_ids = watched_movies_df['movieId']


    recommendations = pd.DataFrame(columns=final_cols) # Initialize with final_cols
    # Step 3: Try to get recommendations based on genres
    if all_genres:
        # Pass 'movies' which has tmdbId to helper
        recommendations = _get_genre_based_recommendations(movies, all_genres, watched_movie_ids, top_n)
    
    # Step 4: If no recommendations from genres (or all_genres was empty), get fallback
    if recommendations.empty:
        # Pass 'movies' which has tmdbId to helper
        recommendations = _get_fallback_recommendations(movies, watched_movie_ids, top_n)
            
    # Step 5: Ensure final result is not more than top_n and has a clean index and correct columns
    if recommendations.empty: # If still empty, return empty DF with correct columns
        return pd.DataFrame(columns=final_cols)

    # Ensure all final_cols are in recommendations, add if missing
    for col in final_cols:
        if col not in recommendations.columns:
            recommendations[col] = pd.NA 

    return recommendations[final_cols].head(top_n).reset_index(drop=True)

# The new, enhanced version of recommend_similar_movies_partial starts here
# The old version that was previously here has been removed.

def recommend_similar_movies_partial(
    movie_title, 
    movies_with_content_for_tfidf, # DataFrame used to build TF-IDF (movies_with_tags)
    tfidf_matrix, 
    movies_for_output_columns,    # DataFrame to get final movie details from (main \'movies\' df)
    watched_movie_titles_to_exclude, 
    top_n=10,                     # How many to return for this specific seed
    internal_candidate_count=20   # How many raw similar items to consider internally
):
    cols_to_return = ['movieId', 'title', 'genres']
    if 'tmdbId' in movies_for_output_columns.columns:
        cols_to_return.append('tmdbId')

    # Clean the input movie title for matching
    if not movie_title or not str(movie_title).strip():
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None # Return with score col
    
    cleaned_movie_title = clean_text(str(movie_title)).lower()
    if not cleaned_movie_title:
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    # Find matches for the input movie title using 'title_for_matching' from the TF-IDF source DataFrame
    # Ensure movies_with_content_for_tfidf has 'title_for_matching'
    if 'title_for_matching' not in movies_with_content_for_tfidf.columns:
        st.error("Critical: 'title_for_matching' not in DataFrame for TF-IDF. Cannot find movie.")
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None

    # Ensure it's string and handle potential NaNs before .str.contains
    movies_with_content_for_tfidf['title_for_matching'] = movies_with_content_for_tfidf['title_for_matching'].fillna('').astype(str)
    matches = movies_with_content_for_tfidf[movies_with_content_for_tfidf['title_for_matching'].str.contains(cleaned_movie_title, na=False)]
    
    if matches.empty:
        # Try a more fuzzy match if direct cleaned contains fails
        best_fuzz_score = 0
        best_fuzz_idx = -1
        # Use .items() for potentially non-unique indices or if index is not a simple range
        for idx_val, row_title in movies_with_content_for_tfidf['title_for_matching'].items():
            score = fuzz.ratio(cleaned_movie_title, row_title)
            if score > best_fuzz_score:
                best_fuzz_score = score
                best_fuzz_idx = idx_val # Store the actual index value
        
        if best_fuzz_score > 80: # Adjust threshold as needed
            matches = movies_with_content_for_tfidf.loc[[best_fuzz_idx]]
        else:
            return pd.DataFrame(columns=cols_to_return + ['similarity_score']), None


    # idx is the index in movies_with_content_for_tfidf (which aligns with tfidf_matrix)
    idx = matches.index[0]
    # Get the display title of the matched movie from the output DataFrame for consistency
    # We need its movieId to look up in movies_for_output_columns if it's different
    matched_movie_id_from_tfidf_source = movies_with_content_for_tfidf.loc[idx, 'movieId']
    
    # Ensure movies_for_output_columns has 'movieId' and 'title'
    if 'movieId' not in movies_for_output_columns.columns or 'title' not in movies_for_output_columns.columns:
        st.error("Critical: 'movieId' or 'title' not in the DataFrame for output columns.")
        # Fallback to title from tfidf source if essential columns are missing
        matched_movie_original_title = movies_with_content_for_tfidf.loc[idx, 'title']
    else:
        matched_movie_row_for_display = movies_for_output_columns[movies_for_output_columns['movieId'] == matched_movie_id_from_tfidf_source]
        if matched_movie_row_for_display.empty:
            # Fallback to title from tfidf source if not found in output df
            matched_movie_original_title = movies_with_content_for_tfidf.loc[idx, 'title']
        else:
            matched_movie_original_title = matched_movie_row_for_display['title'].iloc[0]
        
    cosine_sim_vector = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get indices of the (internal_candidate_count) most similar movies, EXCLUDING the movie itself.
    similar_indices_with_self = cosine_sim_vector.argsort()[-(internal_candidate_count + 1):][::-1]
    similar_indices_for_tfidf_df = [sim_idx for sim_idx in similar_indices_with_self if sim_idx != idx][:internal_candidate_count]

    if not similar_indices_for_tfidf_df:
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), matched_movie_original_title
    
    # Get movieIds and scores from the movies_with_content_for_tfidf DataFrame
    # This DataFrame's indices (similar_indices_for_tfidf_df) align with cosine_sim_vector
    
    # Temp DataFrame from TF-IDF source to get movieIds and scores
    # Ensure 'movieId' is present in movies_with_content_for_tfidf
    if 'movieId' not in movies_with_content_for_tfidf.columns:
        st.error("Critical: 'movieId' not in DataFrame for TF-IDF. Cannot create recommendations.")
        return pd.DataFrame(columns=cols_to_return + ['similarity_score']), matched_movie_original_title

    temp_recs_df = movies_with_content_for_tfidf.iloc[similar_indices_for_tfidf_df][['movieId']].copy()
    temp_recs_df['similarity_score'] = cosine_sim_vector[similar_indices_for_tfidf_df]
    
    # Now, use these movieIds to get full details from movies_for_output_columns
    # This ensures we have tmdbId and consistent titles/genres for output
    # Ensure 'movieId' is present in movies_for_output_columns for merging
    if 'movieId' not in movies_for_output_columns.columns:
        st.error("Critical: 'movieId' not in the DataFrame for output columns. Cannot merge recommendations.")
        # Fallback: return recommendations based on tfidf source if output df is problematic
        # This might lack tmdbId or other refined details.
        recommendations = movies_with_content_for_tfidf.iloc[similar_indices_for_tfidf_df][cols_to_return].copy()
        recommendations['similarity_score'] = cosine_sim_vector[similar_indices_for_tfidf_df]

    else:
        recommendations = movies_for_output_columns[
            movies_for_output_columns['movieId'].isin(temp_recs_df['movieId'])
        ].copy()
        
        # Merge the similarity scores back
        recommendations = recommendations.merge(
            temp_recs_df[['movieId', 'similarity_score']],
            on='movieId',
            how='left' 
        )
    
    # Filter out watched movies from the recommendations
    if watched_movie_titles_to_exclude and not recommendations.empty:
        # Ensure 'title' column exists in recommendations before filtering
        if 'title' in recommendations.columns:
             recommendations = recommendations[~recommendations['title'].isin(watched_movie_titles_to_exclude)]
        else:
            # If 'title' is somehow missing, we can't filter by it. Log or handle as appropriate.
            # For now, we proceed without this specific filtering if 'title' column is absent.
            pass 
        
    final_recommendations = recommendations.sort_values(by='similarity_score', ascending=False)
    
    # Ensure all target columns are present
    output_columns_with_score = cols_to_return + ['similarity_score']
    for col in output_columns_with_score:
        if col not in final_recommendations.columns:
            final_recommendations[col] = pd.NA # Add as NA if missing

    return final_recommendations[output_columns_with_score].head(top_n).reset_index(drop=True), matched_movie_original_title

def recommend_based_on_watch_history_content(
    watched_titles_list, 
    movies_with_tags_for_tfidf, # This is movies_with_tags from get_tfidf_matrix
    tfidf_matrix, 
    main_movies_df,             # This is the full 'movies' df with tmdbId etc.
    top_n=10
):
    if not watched_titles_list:
        return pd.DataFrame()

    all_recommendations_list = []
    
    # Get the actual movie DataFrame rows for watched titles to accurately exclude them
    # _extract_watched_movies_and_genres is good for matching input titles to DataFrame rows
    actual_watched_movies_df, _ = _extract_watched_movies_and_genres(watched_titles_list, main_movies_df.copy()) # Pass a copy
    
    watched_movie_titles_to_exclude = set()
    if not actual_watched_movies_df.empty and 'title' in actual_watched_movies_df.columns:
        watched_movie_titles_to_exclude = set(actual_watched_movies_df['title'].unique())
    else: # Fallback if titles could not be matched, use the input list directly
        watched_movie_titles_to_exclude = set(watched_titles_list)


    for movie_title_seed in watched_titles_list:
        recs_for_seed_df, matched_title = recommend_similar_movies_partial(
            movie_title=movie_title_seed,
            movies_with_content_for_tfidf=movies_with_tags_for_tfidf,
            tfidf_matrix=tfidf_matrix,
            movies_for_output_columns=main_movies_df,
            watched_movie_titles_to_exclude=watched_movie_titles_to_exclude, # Pass the set of titles
            top_n=top_n + 5, # Get a few extra per seed movie for better aggregation
            internal_candidate_count=top_n + 15 # Consider more candidates internally
        )
        
        if matched_title and not recs_for_seed_df.empty:
            # recs_for_seed_df should now contain 'similarity_score'
            all_recommendations_list.append(recs_for_seed_df)

    if not all_recommendations_list:
        st.info("Could not generate seed recommendations from watch history.")
        return pd.DataFrame()

    combined_recs_df = pd.concat(all_recommendations_list)
    
    if combined_recs_df.empty:
        st.info("Combined recommendations are empty before filtering duplicates.")
        return pd.DataFrame()

    # Sort by similarity score and remove duplicates, keeping the highest score for each movie
    combined_recs_df = combined_recs_df.sort_values(by='similarity_score', ascending=False)
    combined_recs_df = combined_recs_df.drop_duplicates(subset=['movieId'], keep='first')

    # Final filter for movies that might have been in watched_titles_list (if matching was imperfect)
    # This check is against the 'title' column of the recommended movies.
    final_recommendations_df = combined_recs_df[~combined_recs_df['title'].isin(watched_movie_titles_to_exclude)]
    
    # Define final columns based on main_movies_df
    final_output_cols = ['movieId', 'title', 'genres']
    if 'tmdbId' in main_movies_df.columns:
        final_output_cols.append('tmdbId')
    
    # Ensure all required output columns are present
    for col in final_output_cols:
        if col not in final_recommendations_df.columns:
            final_recommendations_df[col] = pd.NA 
            
    return final_recommendations_df[final_output_cols].head(top_n).reset_index(drop=True)

def show_table(df):
    if not df.empty:
        df = df.copy()
        df.index = range(1, len(df) + 1)
        st.dataframe(df)
    else:
        st.info("No data to display.")

def main():
    st.markdown("<h1 style='color:#1976d2;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("## 📋 Menu")

    # Verileri yükle
    base_dir_for_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    cleaned_data_path_in_app = os.path.join(base_dir_for_data, 'cleaned_data')

    movies = load_movies(data_path=cleaned_data_path_in_app)
    ratings = load_ratings(data_path=cleaned_data_path_in_app)
    tags = load_tags(data_path=cleaned_data_path_in_app)

    links_file_path = os.path.join(base_dir_for_data, 'data', 'links.csv') # data klasöründe olduğunu varsayıyoruz
    try:
        links_df = pd.read_csv(links_file_path)
        if links_df.empty:
            st.warning("Warning: links.csv is empty. Poster functionality might be affected.")
        # tmdbId sütunundaki NaN olmayan değerleri integer yapalım, API fonksiyonu int bekliyor.
        links_df = links_df[pd.notna(links_df['tmdbId'])].copy() # Önce NaN olanları atalım
        if not links_df.empty: # Check if links_df is not empty after dropping NaNs
            links_df['tmdbId'] = links_df['tmdbId'].astype(int)

    except FileNotFoundError:
        st.error(f"ERROR: links.csv not found at {links_file_path}. Poster functionality will be disabled.")
        links_df = pd.DataFrame(columns=['movieId', 'tmdbId']) # Hata durumunda boş DataFrame
    except Exception as e:
        st.error(f"ERROR: An unexpected error occurred while loading links.csv: {e}")
        links_df = pd.DataFrame(columns=['movieId', 'tmdbId']) # Hata durumunda boş DataFrame


    if movies.empty:
        st.error("Film verisi yüklenemedi. Uygulama devam edemiyor.")
        st.stop()
    if ratings.empty: # Assuming ratings are crucial for some parts, though not all.
        st.warning("Reyting verisi yüklenemedi. İşbirlikçi filtreleme gibi bazı özellikler çalışmayabilir.")
        # Do not st.stop() here if other features can work without ratings.
        # However, if collaborative filtering is selected and ratings are empty, it should be handled there.

    # --- START: Logic to reset multiselect after submission ---
    if st.session_state.get('movies_added_to_watch_history_flag', False):
        st.session_state.add_selected_movies_multiselect = []
        st.session_state.movies_added_to_watch_history_flag = False # Reset the flag
    # --- END: Logic to reset multiselect after submission ---

    # Generate TF-IDF matrix and related components
    tfidf_matrix, tfidf_vectorizer, movies_with_tags = get_tfidf_matrix(movies.copy(), tags.copy())

    # movies DataFrame'ine tmdbId'leri ekleyelim (merge edelim)
    if not movies.empty and not links_df.empty and 'tmdbId' in links_df.columns:
        movies = movies.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
    # Eğer içerik tabanlı önerilerde kullandığın movies_with_tags için de tmdbId gerekiyorsa:
    if not movies_with_tags.empty and not links_df.empty and 'tmdbId' in links_df.columns:
        movies_with_tags = movies_with_tags.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')


    # Check if TF-IDF matrix generation was successful for content-based features
    content_based_enabled = tfidf_matrix is not None and tfidf_vectorizer is not None and not movies_with_tags.empty
    if not content_based_enabled:
        st.warning(
            "TF-IDF matrix and related components could not be generated (e.g., due to empty movie data). "
            "Content-based recommendations will be disabled."
        )

    surprise_model = load_trained_surprise_model()

    if surprise_model is None:
        st.warning("İşbirlikçi filtreleme modeli yüklenemedi. Bu özellik kullanılamayabilir.")
    
    if 'watched_movies' not in st.session_state:
        st.session_state['watched_movies'] = set()
    if 'add_selected_movies_multiselect' not in st.session_state: # Key for the new multiselect
        st.session_state.add_selected_movies_multiselect = []
    if 'movies_added_to_watch_history_flag' not in st.session_state: # Initialize the new flag
        st.session_state.movies_added_to_watch_history_flag = False

    menu = [
        "🎯 Content-Based Recommendation",
        "👥 Collaborative Filtering",
        "😊 Mood-Based Recommendation",
        "🎲 Random Movie",
        "📽️ Watch History & Recommendations",
        "🕵️ Unwatched Movies"
    ]
    choice = st.sidebar.radio("Choose a recommendation method:", menu)

    if choice == menu[0]: # Content-Based Recommendation
        st.success("**Content-Based Recommendation**")
        if not content_based_enabled:
            st.error("Content-based recommendation is currently unavailable due to issues in data processing (e.g., TF-IDF matrix generation failed or movie data is empty).")
        else:
            movie_title = st.text_input("🎬 Enter a movie title you like (no need for year):")
            if st.button("Get Recommendations"):
                if not movie_title.strip():
                    st.warning("Please enter a movie title.")
                else:
                    recs_df, matched_title = recommend_similar_movies_partial(
                        movie_title=movie_title,
                        movies_with_content_for_tfidf=movies_with_tags, 
                        tfidf_matrix=tfidf_matrix,
                        movies_for_output_columns=movies, # Pass the main movies DataFrame
                        watched_movie_titles_to_exclude=st.session_state.get('watched_movies', set()),
                        top_n=10
                    )
                    if matched_title:
                        st.info(f"Showing recommendations based on: **{matched_title}**")

                    if not recs_df.empty:
                        with st.expander("See Recommendations", expanded=True): # Genişletilmiş olarak başlasın
                            if 'movieId' not in recs_df.columns and 'tmdbId' not in recs_df.columns:
                                st.warning("Recommendation data is missing 'movieId' or 'tmdbId' for poster lookup.")
                                # Fallback to old table display if essential IDs are missing
                                temp_display_df = recs_df.copy()
                                if 'title' not in temp_display_df.columns: temp_display_df['title'] = "N/A"
                                if 'genres' not in temp_display_df.columns: temp_display_df['genres'] = "N/A"
                                show_table(temp_display_df[['title', 'genres']])
                            else:
                                for index, row in recs_df.iterrows():
                                    # Ensure 'title' and 'genres' exist to prevent KeyError
                                    title_display = row.get('title', "Title not available")
                                    genres_display = row.get('genres', "Genres not available")

                                    st.subheader(f"{recs_df.index.get_loc(index) + 1}. {title_display}")
                                    st.write(f"**Genres:** {genres_display}")

                                    tmdb_id_to_fetch = None
                                    # Prefer tmdbId directly from recs_df if available and valid
                                    if 'tmdbId' in row and pd.notna(row['tmdbId']):
                                        tmdb_id_to_fetch = int(row['tmdbId']) # Ensure it's int for the API
                                    # Fallback: if tmdbId is not in recs_df or is NaN, try to find it using movieId from recs_df and links_df
                                    elif 'movieId' in row and pd.notna(row['movieId']) and not links_df.empty:
                                        link_info = links_df[links_df['movieId'] == row['movieId']]
                                        if not link_info.empty and 'tmdbId' in link_info.columns and pd.notna(link_info.iloc[0]['tmdbId']):
                                            tmdb_id_to_fetch = int(link_info.iloc[0]['tmdbId'])

                                    if tmdb_id_to_fetch:
                                        # Ensure TMDB_API_KEY is defined and accessible here
                                        # It should be defined globally or passed appropriately
                                        movie_details = get_movie_details_from_tmdb(tmdb_id_to_fetch, TMDB_API_KEY)
                                        if movie_details and movie_details.get("poster_url"):
                                            col1, col2 = st.columns([1, 3]) 
                                            with col1:
                                                st.image(movie_details["poster_url"], width=150) 
                                            with col2:
                                                if movie_details.get("overview"):
                                                    st.caption(f"Overview: {movie_details['overview']}")
                                                else:
                                                    st.caption("Overview not available.") # More specific message
                                        elif movie_details: # Details fetched but no poster
                                            st.caption("Poster not found on TMDB.")
                                            if movie_details.get("overview"):
                                                    st.caption(f"Overview: {movie_details['overview']}")
                                        else: # No details fetched at all
                                            st.caption("Details (including poster) not found on TMDB.")
                                    else:
                                        st.caption("TMDB ID not found for this movie, so poster cannot be displayed.")
                                    st.markdown("---") 
                    else:
                        st.warning("No recommendations found. Try a different title.")

    elif choice == menu[1]: # Collaborative Filtering
        st.success("**Collaborative Filtering Recommendation**")
        user_id_input = st.number_input("Enter your userId:", min_value=1, step=1, value=1)
        if st.button("Get Collaborative Recommendations"):
            if surprise_model is not None: 
                if user_id_input:
                    user_id = int(user_id_input)
                    # movies DataFrame passed here should have tmdbId after merge in main
                    recs_df = get_user_recommendations(
                        user_id,
                        surprise_model,
                        movies, 
                        ratings,
                        st.session_state.get('watched_movies', set()),
                        top_n=10
                    )
                    if not recs_df.empty:
                        with st.expander("See Recommendations", expanded=True):
                            # Poster display logic adapted for this section
                            if 'movieId' not in recs_df.columns and 'tmdbId' not in recs_df.columns:
                                st.warning("Recommendation data is missing 'movieId' or 'tmdbId' for poster lookup.")
                                temp_display_df = recs_df.copy()
                                if 'title' not in temp_display_df.columns: temp_display_df['title'] = "N/A"
                                if 'genres' not in temp_display_df.columns: temp_display_df['genres'] = "N/A"
                                show_table(temp_display_df[['title', 'genres']]) # Fallback to simple table
                            else:
                                for index, row in recs_df.iterrows():
                                    title_display = row.get('title', "Title not available")
                                    genres_display = row.get('genres', "Genres not available")
                                    st.subheader(f"{recs_df.index.get_loc(index) + 1}. {title_display}")
                                    st.write(f"**Genres:** {genres_display}")

                                    tmdb_id_to_fetch = None
                                    if 'tmdbId' in row and pd.notna(row['tmdbId']):
                                        tmdb_id_to_fetch = int(row['tmdbId'])
                                    elif 'movieId' in row and pd.notna(row['movieId']) and not links_df.empty:
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
                        st.warning("No recommendations found for this user. They might have rated all available movies, the user ID could be invalid, or all potential recommendations were already in your watch history.")
                else:
                    st.warning("Lütfen bir Kullanıcı ID'si girin.")
            else:
                st.error("İşbirlikçi filtreleme modeli şu anda kullanılamıyor.")

    elif choice == menu[2]: # Mood-Based Recommendation
        st.success("**Mood-Based Recommendation**")
        mood_selected = st.selectbox("Select your mood:", list(MOOD_GENRE_MAP.keys())) # Renamed variable
        if st.button("Get Mood-Based Recommendations"):
            # movies DataFrame passed here should have tmdbId
            recs_df = recommend_by_mood(
                mood_selected,
                movies, 
                st.session_state.get('watched_movies', set()),
                top_n=10
            )
            if not recs_df.empty:
                with st.expander("See Recommendations", expanded=True):
                    # Poster display logic adapted for this section
                    if 'movieId' not in recs_df.columns and 'tmdbId' not in recs_df.columns:
                        st.warning("Recommendation data is missing 'movieId' or 'tmdbId' for poster lookup.")
                        temp_display_df = recs_df.copy()
                        if 'title' not in temp_display_df.columns: temp_display_df['title'] = "N/A"
                        if 'genres' not in temp_display_df.columns: temp_display_df['genres'] = "N/A"
                        show_table(temp_display_df[['title', 'genres']]) # Fallback
                            
                    else:
                        for index, row in recs_df.iterrows():
                            title_display = row.get('title', "Title not available")
                            genres_display = row.get('genres', "Genres not available")
                            st.subheader(f"{recs_df.index.get_loc(index) + 1}. {title_display}")
                            st.write(f"**Genres:** {genres_display}")

                            tmdb_id_to_fetch = None
                            if 'tmdbId' in row and pd.notna(row['tmdbId']):
                                tmdb_id_to_fetch = int(row['tmdbId'])
                            elif 'movieId' in row and pd.notna(row['movieId']) and not links_df.empty:
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

    elif choice == menu[3]: # Random Movie
        st.success("**Random Movie**")
        if st.button("Pick a Random Movie"):
            if not movies.empty:
                # Ensure pick_random_movie gets the 'movies' DataFrame that potentially has 'tmdbId'
                movie = pick_random_movie(movies) 
                
                # Corrected f-strings
                st.info(f"**Title:** {movie['title']}")
                st.info(f"**Genres:** {movie['genres']}")

                tmdb_id_to_fetch = None
                # Attempt to get tmdbId directly from the movie series (if merged and not NaN)
                if 'tmdbId' in movie and pd.notna(movie['tmdbId']):
                    tmdb_id_to_fetch = int(movie['tmdbId'])
                # Fallback: if tmdbId is not in movie series or is NaN, 
                # try to find it using movieId from movie series and links_df
                elif 'movieId' in movie and pd.notna(movie['movieId']) and not links_df.empty:
                    link_info = links_df[links_df['movieId'] == movie['movieId']]
                    if not link_info.empty and 'tmdbId' in link_info.columns and pd.notna(link_info.iloc[0]['tmdbId']):
                        tmdb_id_to_fetch = int(link_info.iloc[0]['tmdbId'])

                if tmdb_id_to_fetch:
                    movie_details = get_movie_details_from_tmdb(tmdb_id_to_fetch, TMDB_API_KEY)
                    if movie_details and movie_details.get("poster_url"):
                        st.image(movie_details["poster_url"], width=200)
                    if movie_details and movie_details.get("overview"):
                        st.caption(f"Overview: {movie_details['overview']}") # Corrected f-string
                    elif movie_details: # Details fetched but no poster/overview
                        st.caption("Poster or overview not available on TMDB.")
                    else: # Failed to fetch details
                        st.caption("Details not found on TMDB.")
                else:
                    st.caption("TMDB ID not found for this movie, so poster and overview cannot be displayed.")
            else:
                st.warning("No movies available to pick from.")

    elif choice == menu[4]: # Watch History & Personalized Recommendations
        st.success("**Watch History & Personalized Recommendations**")

        # --- MODIFICATION START for adding movies to watch history ---
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
                key="add_selected_movies_multiselect"
            )
            if st.button("Add Selected to Watch History", key="add_selected_to_watch_history_button"):
                selected_movies_to_add = st.session_state.add_selected_movies_multiselect
                if selected_movies_to_add:
                    for movie_title in selected_movies_to_add:
                        st.session_state['watched_movies'].add(movie_title)
                    st.success(f"{len(selected_movies_to_add)} movie(s) added to your watch history.")
                    # Set the flag to clear multiselect on next rerun
                    st.session_state.movies_added_to_watch_history_flag = True
                    st.rerun() # Corrected from st.experimental_rerun()
                else:
                    st.warning("Please select at least one movie to add.")
        elif not movies.empty and 'title' in movies.columns and not all_movie_titles: # movies df exists but no titles
             st.warning("Movie list is empty or contains no valid titles to select from.")
        elif movies.empty or 'title' not in movies.columns: # movies df problematic
            st.warning("Movie list is not available to make selections.")
        else: # All selectable movies are already watched or no movies initially
            st.info("No new movies available to add to watch history (either all are watched or the movie list is empty).")
        # --- MODIFICATION END for adding movies to watch history ---
        
        # Display current watch history
        if st.session_state.get('watched_movies', set()):
            st.write("Your current watch history:")
            watched_df = pd.DataFrame(list(st.session_state['watched_movies']), columns=['Title'])
            watched_df.index = range(1, len(watched_df) + 1)
            st.dataframe(watched_df, height=min(300, len(watched_df) * 40))
        else:
            st.info("Your watch history is currently empty. Add movies using the selection field above.")

        if st.button("Get Recommendations Based on Watch History"):
            watched_titles_set = st.session_state.get('watched_movies', set())
            if not watched_titles_set:
                st.warning("Your watch history is empty. Please add some movies using the selection field above to get personalized suggestions.")
            else:
                if not content_based_enabled: # Check if tfidf_matrix etc. are available
                    st.error("Content-based components are not available for watch history recommendations. Please check data loading and TF-IDF generation.")
                    recs_based_on_watched = pd.DataFrame()
                else:
                    # Call the NEW function
                    recs_based_on_watched = recommend_based_on_watch_history_content(
                        watched_titles_list=list(watched_titles_set),
                        movies_with_tags_for_tfidf=movies_with_tags, # DF used for TF-IDF
                        tfidf_matrix=tfidf_matrix,                  # The TF-IDF matrix
                        main_movies_df=movies,                      # Main movies DF with all info (incl. tmdbId)
                        top_n=10
                    )

                if not recs_based_on_watched.empty:
                    st.subheader("Recommendations based on your watch history:")
                    with st.expander("See Recommendations", expanded=True):
                        # Poster display logic adapted for this section
                        if 'movieId' not in recs_based_on_watched.columns and 'tmdbId' not in recs_based_on_watched.columns:
                            st.warning("Recommendation data is missing 'movieId' or 'tmdbId' for poster lookup.")
                            temp_display_df = recs_based_on_watched.copy()
                            if 'title' not in temp_display_df.columns: temp_display_df['title'] = "N/A"
                            if 'genres' not in temp_display_df.columns: temp_display_df['genres'] = "N/A"
                            show_table(temp_display_df[['title', 'genres']]) # Fallback
                        else:
                            for index, row in recs_based_on_watched.iterrows():
                                title_display = row.get('title', "Title not available")
                                genres_display = row.get('genres', "Genres not available")
                                st.subheader(f"{recs_based_on_watched.index.get_loc(index) + 1}. {title_display}")
                                st.write(f"**Genres:** {genres_display}")

                                tmdb_id_to_fetch = None
                                if 'tmdbId' in row and pd.notna(row['tmdbId']):
                                    tmdb_id_to_fetch = int(row['tmdbId'])
                                elif 'movieId' in row and pd.notna(row['movieId']) and not links_df.empty:
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

    elif choice == menu[5]: # Unwatched Movies
        st.subheader("🕵️ Unwatched Movies")
        if 'watched_movies' not in st.session_state or not st.session_state['watched_movies']:
            st.info("Your watch history is empty. Watch some movies first!")
        else:
            unwatched_movies = movies[~movies['title'].isin(st.session_state['watched_movies'])]
            if not unwatched_movies.empty:
                st.markdown("### Here are some movies you haven't watched yet:")
                show_table(unwatched_movies[['title', 'genres']])
            else:
                st.info("You've watched all the movies in our database!")

if __name__ == "__main__":
    main()