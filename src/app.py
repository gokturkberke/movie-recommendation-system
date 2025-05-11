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
    """
    all_movie_ids = movies_df['movieId'].unique()
    user_rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()

    predictions = []
    # Consider only movies not yet rated by the user for prediction
    unrated_movie_ids = [mid for mid in all_movie_ids if mid not in user_rated_movie_ids]

    for movie_id in unrated_movie_ids:
        pred = surprise_model.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))
            
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Fetch more candidates initially to account for filtering watched_titles later
    # Buffer is added to increase chances of getting top_n results
    num_candidates_to_fetch = top_n + (len(watched_titles) if watched_titles else 0) + 10 
    
    candidate_movie_ids_ordered = [movie_id for movie_id, score in predictions[:num_candidates_to_fetch]]
    
    if not candidate_movie_ids_ordered: # No predictions or all rated
        return pd.DataFrame(columns=['title', 'genres'])

    # Get movie details for the candidates
    # Ensure we only try to fetch details for existing movieIds
    recommended_movies_df = movies_df[movies_df['movieId'].isin(candidate_movie_ids_ordered)].copy()
    
    # Filter out watched movies by title from the candidates
    if watched_titles and not recommended_movies_df.empty:
        recommended_movies_df = recommended_movies_df[~recommended_movies_df['title'].isin(watched_titles)]

    # Re-order the filtered candidates based on original prediction score and select top_n
    if not recommended_movies_df.empty:
        # Filter candidate_movie_ids_ordered to only those that are still in recommended_movies_df after watched_filter
        final_candidate_ids_in_order = [
            mid for mid in candidate_movie_ids_ordered 
            if mid in recommended_movies_df['movieId'].values
        ]
        if final_candidate_ids_in_order:
            recommended_movies_df = recommended_movies_df.set_index('movieId').loc[final_candidate_ids_in_order].reset_index()
        else: # All candidates were filtered out by watched_titles or didn't exist in movies_df
             recommended_movies_df = pd.DataFrame(columns=['title', 'genres', 'movieId'])
    # else: recommended_movies_df is already empty or became empty after watched_titles filter
        
    return recommended_movies_df[['title', 'genres']].head(top_n)


def recommend_by_mood(mood, movies, watched_movies, top_n=10):
    genres = MOOD_GENRE_MAP.get(mood.lower())
    if not genres:
        return pd.DataFrame()

    movies_copy = movies.copy()
    # Ensure 'genres' column is string type before applying string operations
    movies_copy['genres'] = movies_copy['genres'].astype(str)
    
    mask = movies_copy['genres'].apply(lambda g: any(genre in g for genre in genres))
    filtered_movies = movies_copy[mask]
    
    if filtered_movies.empty:
        return pd.DataFrame()
    
    # Determine how many movies to sample initially
    # Sample more to account for filtering watched_movies, but not more than available. Add a small buffer.
    num_to_sample = min(top_n + (len(watched_movies) if watched_movies else 0) + 5, len(filtered_movies))

    if num_to_sample <= 0: # If no movies to sample from (e.g., filtered_movies is small and watched_movies is large)
        return pd.DataFrame(columns=['title', 'genres'])
        
    recommendations = filtered_movies.sample(n=num_to_sample, random_state=42)[['title', 'genres']]
    
    # Filter out watched movies
    if watched_movies and not recommendations.empty:
        recommendations = recommendations[~recommendations['title'].isin(watched_movies)]
        
    return recommendations.head(top_n).reset_index(drop=True)

def pick_random_movie(movies):
    movie = movies.sample(n=1).iloc[0]
    return movie
"""
def recommend_by_watched_genres(watched_titles, movies, top_n=10):
    if not watched_titles:
        return pd.DataFrame()
    
    # Orijinal başlıkları tutan bir kopya oluştur
    original_movies = movies[['title', 'genres']].copy()

    # İzlenen başlıkları küçük harfe çevir ve temizle
    base_titles = [title.lower().strip() for title in watched_titles]
    filtered = original_movies[original_movies['title'].str.lower().apply(
        lambda title: any(base_title in title for base_title in base_titles)
    )]

    if filtered.empty:
        print("No matching titles found. Recommending by genres...")
        all_genres = set()
        for title in watched_titles:
            genres = movies[movies['title'].str.lower() == title.lower()]['genres'].values
            if len(genres) > 0:
                all_genres.update(genres[0].split('|'))
        filtered = original_movies[original_movies['genres'].apply(lambda g: any(genre in g for genre in all_genres))]

    # İzlenen filmleri öneri listesinden çıkar
    filtered = filtered[~filtered['title'].str.lower().isin(base_titles)]
    
    recommendations = filtered.head(top_n)
    return recommendations
"""
# Helper functions for recommend_by_watched_genres
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
    if not all_genres_set:
        return pd.DataFrame()

    # Find movies with at least one matching genre
    # Ensure 'genres' column is string and handle potential errors if a genre is not string
    genre_matches = movies_df[movies_df['genres'].apply(
        lambda g: isinstance(g, str) and any(genre_item in g.split('|') for genre_item in all_genres_set)
    )]

    recommendations = genre_matches
    # Remove watched movies from recommendations
    if watched_movie_ids is not None and not watched_movie_ids.empty:
        recommendations = genre_matches[~genre_matches['movieId'].isin(watched_movie_ids)]
    
    if recommendations.empty:
        return pd.DataFrame()

    # Return top N or fewer if less are available
    num_to_return = min(top_n, len(recommendations))
    return recommendations[['title', 'genres']].head(num_to_return).reset_index(drop=True)

def _get_fallback_recommendations(movies_df, watched_movie_ids, top_n):
    recommendations_pool = movies_df
    if watched_movie_ids is not None and not watched_movie_ids.empty:
        recommendations_pool = movies_df[~movies_df['movieId'].isin(watched_movie_ids)]
    
    if recommendations_pool.empty:
        return pd.DataFrame()
        
    # Sample safely
    num_to_sample = min(top_n, len(recommendations_pool))
    # Use random_state for reproducibility if desired, e.g., random_state=42
    return recommendations_pool[['title', 'genres']].sample(n=num_to_sample, random_state=42).reset_index(drop=True)

def recommend_by_watched_genres(watched_titles, movies, top_n=10):
    if not watched_titles:
        return pd.DataFrame()

    # Step 1: Extract watched movies and their genres
    watched_movies_df, all_genres = _extract_watched_movies_and_genres(watched_titles, movies)
    
    # Step 2: Get IDs of watched movies for exclusion
    # Ensure watched_movie_ids is a Series, even if empty, for consistent type handling
    watched_movie_ids = watched_movies_df['movieId'] if not watched_movies_df.empty else pd.Series(dtype='int64')

    recommendations = pd.DataFrame()
    # Step 3: Try to get recommendations based on genres
    if all_genres:
        recommendations = _get_genre_based_recommendations(movies, all_genres, watched_movie_ids, top_n)
    
    # Step 4: If no recommendations from genres (or all_genres was empty), get fallback recommendations
    if recommendations.empty:
        recommendations = _get_fallback_recommendations(movies, watched_movie_ids, top_n)
            
    # Step 5: Ensure final result is not more than top_n and has a clean index
    return recommendations.head(top_n).reset_index(drop=True)

def recommend_similar_movies_partial(movie_title, movies, tfidf_matrix, watched_movies, top_n=10):
    # movies argument is movies_with_tags, which was used to build tfidf_matrix
    movies_df = movies.copy() 
    # Ensure 'title_for_matching' (used for finding the input movie) is string and handles NaNs
    movies_df['title_for_matching'] = movies_df['title_for_matching'].fillna('').astype(str)

    # Validate movie_title input
    if not movie_title or not movie_title.strip():
        # st.warning("Please enter a movie title for content-based recommendations.") # Handled in main
        return pd.DataFrame(), None

    cleaned_movie_title = clean_text(movie_title).lower()
    if not cleaned_movie_title: # If cleaning results in an empty string
        # st.warning("Could not process the entered movie title.") # Handled in main
        return pd.DataFrame(), None

    # Find matches for the input movie title using the cleaned 'title_for_matching' column
    matches = movies_df[movies_df['title_for_matching'].str.contains(cleaned_movie_title, na=False)]
    if matches.empty:
        # st.warning(f"No movie found matching '{movie_title}'.") # Handled in main
        return pd.DataFrame(), None

    # idx is the index in the original DataFrame (movies_with_tags) from which tfidf_matrix was built
    idx = matches.index[0] 
    matched_movie_original_title = matches.iloc[0]['title'] # Get the display title of the matched movie
        
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Determine number of candidates to fetch: top_n + watched_count + buffer
    num_candidates_to_fetch = top_n + (len(watched_movies) if watched_movies else 0) + 5
    
    # Number of other movies available for recommendation (excluding the movie itself)
    num_available_others = len(cosine_sim) - 1
    if num_available_others <= 0:
        return pd.DataFrame(columns=['title', 'genres']), matched_movie_original_title

    # Fetch at most num_available_others
    actual_k_to_fetch = min(num_candidates_to_fetch, num_available_others)
    if actual_k_to_fetch <= 0:
        return pd.DataFrame(columns=['title', 'genres']), matched_movie_original_title

    # Get indices of the (actual_k_to_fetch) most similar movies, excluding the movie itself.
    # argsort()[-1] is the movie itself (highest similarity). We want others.
    similar_indices = cosine_sim.argsort()[-(actual_k_to_fetch + 1):-1][::-1]

    if not similar_indices.size: # No similar movies found (should be caught by actual_k_to_fetch <=0)
        return pd.DataFrame(columns=['title', 'genres']), matched_movie_original_title
    
    # Retrieve recommendations using .iloc on the original 'movies' (movies_with_tags) DataFrame
    recommendations = movies.iloc[similar_indices][['title', 'genres']].copy() # Use .copy() to avoid SettingWithCopyWarning later if needed
    
    # Filter out watched movies from the recommendations
    if watched_movies and not recommendations.empty:
        recommendations = recommendations[~recommendations['title'].isin(watched_movies)]
        
    # Return top_n recommendations and the title of the movie they were based on
    return recommendations.head(top_n).reset_index(drop=True), matched_movie_original_title

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

    if movies.empty:
        st.error("Film verisi yüklenemedi. Uygulama devam edemiyor.")
        st.stop()
    if ratings.empty: # Assuming ratings are crucial for some parts, though not all.
        st.warning("Reyting verisi yüklenemedi. İşbirlikçi filtreleme gibi bazı özellikler çalışmayabilir.")
        # Do not st.stop() here if other features can work without ratings.
        # However, if collaborative filtering is selected and ratings are empty, it should be handled there.

    # Generate TF-IDF matrix and related components
    tfidf_matrix, tfidf_vectorizer, movies_with_tags = get_tfidf_matrix(movies.copy(), tags.copy())

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
                    recs, matched_title = recommend_similar_movies_partial(
                        movie_title,
                        movies_with_tags, # This is movies DataFrame with 'content' and 'title_for_matching'
                        tfidf_matrix,
                        st.session_state.get('watched_movies', set()), # Pass watched movies
                        top_n=10
                    )
                    if matched_title:
                        st.info(f"Showing recommendations based on: **{matched_title}**")
                    if not recs.empty:
                        with st.expander("See Recommendations"):
                            show_table(recs)
                    else:
                        st.warning("No recommendations found. Try a different title.")

    elif choice == menu[1]:
        st.success("**Collaborative Filtering Recommendation**")
        user_id_input = st.number_input("Enter your userId:", min_value=1, step=1, value=1)
        if st.button("Get Collaborative Recommendations"):
            if surprise_model is not None: 
                if user_id_input:
                    user_id = int(user_id_input)
                    recs = get_user_recommendations(
                        user_id,
                        surprise_model,
                        movies, # Original movies DataFrame
                        ratings,
                        st.session_state.get('watched_movies', set()), # Pass watched movies
                        top_n=10
                    )
                    if not recs.empty:
                        with st.expander("See Recommendations"):
                            show_table(recs) 
                    else:
                        st.warning("No recommendations found for this user. They might have rated all available movies, the user ID could be invalid, or all potential recommendations were already in your watch history.")
                else:
                    st.warning("Lütfen bir Kullanıcı ID'si girin.")
            else:
                st.error("İşbirlikçi filtreleme modeli şu anda kullanılamıyor.")
    elif choice == menu[2]: # Mood-Based Recommendation
        st.success("**Mood-Based Recommendation**")
        mood = st.selectbox("Select your mood:", list(MOOD_GENRE_MAP.keys()))
        if st.button("Get Mood-Based Recommendations"):
            recs = recommend_by_mood(
                mood,
                movies,
                st.session_state.get('watched_movies', set()), # Pass watched movies
                top_n=10
            )
            if not recs.empty:
                with st.expander("See Recommendations"):
                    show_table(recs)
            else:
                st.warning("No movies found for this mood.")

    elif choice == menu[3]:
        st.success("**Random Movie**")
        if st.button("Pick a Random Movie"):
            movie = pick_random_movie(movies)
            st.info(f"**Title:** {movie['title']}")
            st.info(f"**Genres:** {movie['genres']}")

    elif choice == menu[4]:
        st.success("**Watch History & Personalized Recommendations**")
        all_titles = movies['title'].tolist()
        watched_titles = st.multiselect(
            "Select movies you've watched:",
            options=all_titles,
            default=[title for title in all_titles if title in st.session_state['watched_movies']]
        )
        st.session_state['watched_movies'] = set(watched_titles)
        st.info("You've watched:")
        watched_df = movies[movies['title'].isin(watched_titles)][['title', 'genres']]
        show_table(watched_df)
        if st.button("Recommend based on my watched movies"):
            if watched_titles:
                recs = recommend_by_watched_genres(
                    watched_titles, 
                    movies, 
                    top_n=10
                    # recommend_by_watched_genres inherently handles not re-recommending watched_titles
                )
                if not recs.empty:
                    with st.expander("See Recommendations based on Watched Genres"):
                        show_table(recs)
                else:
                    st.warning("No recommendations found based on your watched history. Try adding more movies to your watch history or explore other recommendation types.")
            else:
                st.info("Please select some movies you've watched to get recommendations.") # Added message for empty watched_titles

    elif choice == menu[5]:
        st.success("**Unwatched Movies**")
        all_titles = movies['title'].tolist()
        unwatched_titles = [title for title in all_titles if title not in st.session_state['watched_movies']]
        selected_unwatched = st.multiselect(
            "Select movies you want to mark as watched:",
            options=unwatched_titles
        )
        st.session_state['watched_movies'].update(selected_unwatched)
        st.info("Movies you haven't watched yet (showing 10):")
        unwatched_df = movies[movies['title'].isin(unwatched_titles)][['title', 'genres']].head(10)
        show_table(unwatched_df)

if __name__ == "__main__":
    main()