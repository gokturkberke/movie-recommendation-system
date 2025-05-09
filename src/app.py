import streamlit as st
import pandas as pd
import os
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

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
@st.cache_data
def create_sparse_user_item_matrix(ratings):
    user_mapper = {user_id: idx for idx, user_id in enumerate(ratings['userId'].unique())}
    movie_mapper = {movie_id: idx for idx, movie_id in enumerate(ratings['movieId'].unique())}
    user_inv_mapper = {v: k for k, v in user_mapper.items()}
    movie_inv_mapper = {v: k for k, v in movie_mapper.items()}
    user_index = [user_mapper[i] for i in ratings['userId']]
    movie_index = [movie_mapper[i] for i in ratings['movieId']]
    
    # Use 'rating_z' (normalized ratings) instead of 'rating'
    ratings_matrix = csr_matrix((ratings['rating_z'], (user_index, movie_index)), 
                                shape=(len(user_mapper), len(movie_mapper)))
    return ratings_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

def get_user_recommendations(user_id, model_knn, ratings_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper, movies, ratings, top_n=10):
    if user_id not in user_mapper:
        return pd.DataFrame()
    user_idx = user_mapper[user_id]
    # model_knn = NearestNeighbors(metric='cosine', algorithm='brute') # Removed
    # model_knn.fit(ratings_matrix) # Removed
    distances, indices = model_knn.kneighbors(ratings_matrix[user_idx], n_neighbors=6)
    similar_users = indices.flatten()[1:]
    user_rated = set(ratings_matrix[user_idx].nonzero()[1])
    rec_scores = {}
    for sim_user_idx in similar_users:
        sim_user_ratings = ratings_matrix[sim_user_idx].toarray().flatten()
        for movie_idx, rating in enumerate(sim_user_ratings):
            if rating > 0 and movie_idx not in user_rated:
                rec_scores[movie_idx] = rec_scores.get(movie_idx, 0) + rating
    if not rec_scores:
        return pd.DataFrame()
    top_movies = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    movie_ids = [movie_inv_mapper[movie_idx] for movie_idx, _ in top_movies]
    recommended_movies = movies[movies['movieId'].isin(movie_ids)][['title', 'genres']].reset_index(drop=True)
    return recommended_movies

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
def _extract_watched_movies_and_genres(watched_titles, movies_input_df):
    all_genres = set()
    final_watched_movies_df = pd.DataFrame()
    # Work on a copy to avoid modifying the original DataFrame, especially if it's a slice
    movies_df_copy = movies_input_df.copy()

    # 1. Attempt exact matches on original 'title'
    list_of_exact_match_dfs = []
    for title_query in watched_titles:
        # Use movies_df_copy for consistent DataFrame source
        exact_matches = movies_df_copy[movies_df_copy['title'] == title_query]
        if not exact_matches.empty:
            list_of_exact_match_dfs.append(exact_matches)

    if list_of_exact_match_dfs:
        temp_watched_df = pd.concat(list_of_exact_match_dfs)
        final_watched_movies_df = temp_watched_df.drop_duplicates(subset=['movieId']).reset_index(drop=True)
    else:
        # 2. No exact matches, attempt fuzzy matches on cleaned titles against 'title_for_matching'
        list_of_similar_match_dfs = []
        # Clean user titles once
        cleaned_user_titles = [clean_text(t).lower() for t in watched_titles if clean_text(t)]

        # Ensure 'title_for_matching' exists and is suitable for matching
        # It's assumed to be pre-cleaned and lowercased by preprocess_dataset.py
        if 'title_for_matching' in movies_df_copy.columns:
            for cleaned_title_query in cleaned_user_titles:
                if not cleaned_title_query: continue # Skip empty strings after cleaning
                # Match on the 'title_for_matching' column
                similar_movies_matches = movies_df_copy[movies_df_copy['title_for_matching'].str.contains(cleaned_title_query, na=False)]
                if not similar_movies_matches.empty:
                    list_of_similar_match_dfs.append(similar_movies_matches)
        
        if list_of_similar_match_dfs:
            temp_watched_df = pd.concat(list_of_similar_match_dfs)
            final_watched_movies_df = temp_watched_df.drop_duplicates(subset=['movieId']).reset_index(drop=True)

    # 3. Extract genres from the identified watched movies
    if not final_watched_movies_df.empty:
        for genres_str in final_watched_movies_df['genres'].dropna().values: # Ensure NaNs in genres are skipped
            all_genres.update(genres_str.split('|'))
            
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
    movies = load_movies()
    ratings = load_ratings()
    tags = load_tags()
    tfidf_matrix, tfidf, movies_with_tags = get_tfidf_matrix(movies.copy(), tags.copy())
    ratings_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_sparse_user_item_matrix(ratings)

    # Fit k-NN model for collaborative filtering once
    model_knn_collaborative = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn_collaborative.fit(ratings_matrix)

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
        user_id = st.number_input("Enter your userId:", min_value=1, step=1)
        if st.button("Get Collaborative Recommendations"):
            recs = get_user_recommendations(int(user_id), model_knn_collaborative, ratings_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper, movies, ratings, top_n=10)
            if not recs.empty:
                with st.expander("See Recommendations"):
                    show_table(recs)
            else:
                st.warning("No recommendations found. Check the user ID.")

    elif choice == menu[2]:
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