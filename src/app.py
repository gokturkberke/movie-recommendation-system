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

def clean_text(s):
    if pd.isnull(s):
        return ""
    s = re.sub(r"\(\d{4}\)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_tfidf_matrix(movies, tags):
    tags['tag'] = tags['tag'].fillna('').apply(clean_text)
    tags = tags.drop_duplicates(subset=['movieId', 'tag'])
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies['title_clean'] = movies['title'].apply(clean_text)
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ').apply(clean_text)
    movies = movies.merge(tags_grouped, on='movieId', how='left')
    movies['tag'] = movies['tag'].fillna('')
    movies['content'] = movies['title_clean'] + ' ' + movies['genres_clean'] + ' ' + movies['tag']
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
    ratings_matrix = csr_matrix((ratings['rating'], (user_index, movie_index)), 
                                shape=(len(user_mapper), len(movie_mapper)))
    return ratings_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

def get_user_recommendations(user_id, ratings_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper, movies, ratings, top_n=10):
    if user_id not in user_mapper:
        return pd.DataFrame()
    user_idx = user_mapper[user_id]
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(ratings_matrix)
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
def recommend_by_watched_genres(watched_titles, movies, top_n=10):
    if not watched_titles:
        return pd.DataFrame()

    base_titles = [title.lower().strip() for title in watched_titles]
    
    filtered = movies[movies['title'].apply(
        lambda title: any(base_title in title for base_title in base_titles)
    )][['title_original', 'genres_original']].drop_duplicates()
    
    filtered.columns = ['title', 'genres']
    filtered = filtered[~filtered['title'].str.lower().isin(base_titles)]
    
    return filtered.head(top_n)
"""
def recommend_similar_movies_partial(movie_title, movies, tfidf_matrix, top_n=10):
    # NaN değerleri güvenli şekilde temizle
    movies = movies.dropna(subset=['title']).copy()
    movies['title'] = movies['title'].fillna('').astype(str).str.lower()

    if not movie_title:
        return pd.DataFrame(), None

    # Kullanıcı başlığını küçük harfe çevir ve boşlukları temizle
    movie_title = movie_title.lower().strip()

    # NaN olmayan ve başlıkla eşleşen filmleri filtrele
    matches = movies[movies['title'].str.contains(movie_title, na=False, case=False)]
    
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
    movies = movies.dropna(subset=['title', 'genres']).copy()
    
    # If the movie title is empty, return an empty DataFrame
    if not movie_title:
        return pd.DataFrame(), None

    # Normalize the title input
    movie_title = movie_title.lower().strip()

    # Filter movies by matching the title (case insensitive)
    matches = movies[movies['title'].str.contains(movie_title, na=False, case=False)]
    
    if matches.empty:
        return pd.DataFrame(), None

    # Get the index of the first matched movie
    idx = matches.index[0]
    
    # Calculate cosine similarity between the matched movie and all other movies
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get the top N most similar movies
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    
    # Prepare the recommendations
    recommendations = movies.iloc[similar_indices][['title', 'genres']].reset_index(drop=True)
    
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
            recs = get_user_recommendations(int(user_id), ratings_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper, movies, ratings, top_n=10)
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