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
def load_movies(data_path='cleaned_data'):
    return pd.read_csv(os.path.join(data_path, 'movies_clean.csv'))

@st.cache_data
def load_ratings(data_path='cleaned_data'):
    return pd.read_csv(os.path.join(data_path, 'ratings_clean.csv'))

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
    return pd.read_csv(os.path.join(data_path, 'tags_clean.csv'))

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
    return tfidf_matrix, tfidf, movies

def get_user_recommendations(user_id, surprise_model, movies_df, ratings_df, top_n=10):
    """
    Generates movie recommendations for a user using a trained Surprise model.
    """
    # Get a list of all movie IDs
    all_movie_ids = movies_df['movieId'].unique()
    
    # Get a list of movies already rated by the user
    user_rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    
    # Predict ratings for movies not yet rated by the user
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in user_rated_movie_ids:
            # Surprise model's predict method returns a Prediction object
            # uid (user id), iid (item id), r_ui (true rating), est (estimated rating), details
            pred = surprise_model.predict(uid=user_id, iid=movie_id)
            predictions.append((movie_id, pred.est))
            
    # Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N movie IDs
    top_movie_ids = [movie_id for movie_id, score in predictions[:top_n]]
    
    # Get movie details for the recommended movies
    recommended_movies = movies_df[movies_df['movieId'].isin(top_movie_ids)][['title', 'genres']]
    
    # To maintain the order of recommendations
    recommended_movies = recommended_movies.set_index('movieId').loc[top_movie_ids].reset_index()
    return recommended_movies[['title', 'genres']]


def recommend_by_mood(mood, movies, top_n=10):
    genres = MOOD_GENRE_MAP.get(mood.lower())
    if not genres:
        return pd.DataFrame()
    mask = movies['genres'].apply(lambda g: any(genre in g for genre in genres))
    filtered = movies[mask]
    if filtered.empty:
        return pd.DataFrame()
    recommendations = filtered.sample(n=min(top_n, len(filtered)), random_state=42)[['title', 'genres']].reset_index(drop=True)
    return recommendations

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
"""
def recommend_similar_movies_partial(movie_title, movies, tfidf_matrix, top_n=10):
    matches = movies[movies['title'].str.lower().str.contains(movie_title.lower())]
    # NaN olanları kontrol ederek güvenli arama
    matches = movies[movies['title'].fillna('').str.lower().str.contains(movie_title.lower(), na=False)]
    
    if matches.empty:
        return pd.DataFrame(), None
    idx = matches.index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    recommendations = movies.iloc[similar_indices][['title', 'genres']].reset_index(drop=True)
    return recommendations, matches.iloc[0]['title']
"""
def recommend_similar_movies_partial(movie_title, movies, tfidf_matrix, top_n=10):
    # Ensure you're dropping NaN values from the correct columns
    # movies DataFrame here is actually movies_with_tags which includes 'title_for_matching'
    movies_df = movies.dropna(subset=['title', 'genres', 'title_for_matching']).copy()
    
    # If the movie title is empty, return an empty DataFrame
    if not movie_title:
        return pd.DataFrame(), None

    # Clean and normalize the user's input title
    cleaned_movie_title = clean_text(movie_title).lower()

    if not cleaned_movie_title: # If cleaning results in an empty string
        return pd.DataFrame(), None

    # Filter movies by matching the cleaned title against 'title_for_matching'
    # Assuming 'title_for_matching' is already cleaned and lowercased by preprocess_dataset.py
    matches = movies_df[movies_df['title_for_matching'].str.contains(cleaned_movie_title, na=False)]
    
    if matches.empty:
        return pd.DataFrame(), None

    # Get the index of the first matched movie
    # It's important that tfidf_matrix was built using the same indices as movies_df
    # or that movies_df is the same DataFrame (with same row order) used to build tfidf_matrix
    
    # Find the original index in the DataFrame that was used to create the TF-IDF matrix.
    # Assuming 'movies_with_tags' (passed as 'movies' argument) is the one used for TF-IDF.
    # And 'matches' is a filtered version of it.
    
    # Get the index from the original DataFrame that corresponds to the matched movie
    # This assumes that the 'movies' df passed to this function is the same one
    # that was used to generate the tfidf_matrix.
    # The 'matches' DataFrame will have indices from this original 'movies' DataFrame.
    
    idx = matches.index[0] # This index should correspond to the row in the original tfidf_matrix
        
    # Calculate cosine similarity between the matched movie and all other movies
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get the top N most similar movies
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    
    # Prepare the recommendations using the original 'title' and 'genres' for display
    # We use .iloc on the original 'movies' DataFrame (passed as argument) to get the correct rows
    recommendations = movies.iloc[similar_indices][['title', 'genres']].reset_index(drop=True)
    
    # Return the original title of the matched movie for display
    return recommendations, matches.iloc[0]['title']

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

    if movies.empty: # veya diğer DataFrame'ler için de kontrol
        st.error("Film verisi yüklenemedi. Uygulama devam edemiyor.")
        st.stop()
    if ratings.empty:
        st.error("Reyting verisi yüklenemedi. Uygulama devam edemiyor.")
        st.stop()


    tfidf_matrix, tfidf_vectorizer, movies_with_tags = get_tfidf_matrix(movies.copy(), tags.copy())

    # YENİ KOD: Kayıtlı modeli yükle
    surprise_model = load_trained_surprise_model() 

    if surprise_model is None:
        # Eğer model yüklenemediyse, kullanıcıya bilgi ver ve belki CF özelliğini devre dışı bırak
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

    if choice == menu[0]:
        st.success("**Content-Based Recommendation**")
        movie_title = st.text_input("🎬 Enter a movie title you like (no need for year):")
        if st.button("Get Recommendations"):
            recs, matched_title = recommend_similar_movies_partial(movie_title, movies_with_tags, tfidf_matrix, top_n=10)
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
            if surprise_model is not None: # Modelin yüklendiğinden emin ol
                if user_id_input:
                    user_id = int(user_id_input)
                    # Ensure ratings is passed if get_user_recommendations needs it
                    recs = get_user_recommendations(user_id, surprise_model, movies, ratings, top_n=10) 
                    if not recs.empty:
                        with st.expander("See Recommendations"):
                            show_table(recs)
                    else:
                        st.warning("Bu kullanıcı için öneri bulunamadı. Kullanıcı ID'sini kontrol edin veya kullanıcı tüm filmleri oylamış olabilir.")
                else:
                    st.warning("Lütfen bir Kullanıcı ID'si girin.")
            else:
                st.error("İşbirlikçi filtreleme modeli şu anda kullanılamıyor.")
    elif choice == menu[2]: # Mood-Based Recommendation
        st.success("**Mood-Based Recommendation**")
        mood = st.selectbox("Select your mood:", list(MOOD_GENRE_MAP.keys()))
        if st.button("Get Mood-Based Recommendations"):
            recs = recommend_by_mood(mood, movies, top_n=10)
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
                recs = recommend_by_watched_genres(watched_titles, movies, top_n=10)
                if not recs.empty:
                    with st.expander("See Recommendations"):
                        show_table(recs)
                else:
                    st.warning("No recommendations found based on your watched movies.")
            else:
                st.info("You haven't marked any movies as watched yet.")

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