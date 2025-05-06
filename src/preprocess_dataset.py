import pandas as pd
import os

# Find the data folder relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../data'))
CLEANED_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '../cleaned_data'))

os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

# --- 1. Clean movies.csv ---
movies = pd.read_csv(os.path.join(RAW_DATA_PATH, 'movies.csv'))

# Remove duplicates
movies = movies.drop_duplicates(subset=['movieId'])

# Handle missing values
movies = movies.dropna(subset=['title', 'genres'])

# One-hot encode genres
all_genres = set()
movies['genres'].str.split('|').apply(all_genres.update)
for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: int(genre in x.split('|')))

movies.to_csv(os.path.join(CLEANED_DATA_PATH, 'movies_clean.csv'), index=False)
print("movies_clean.csv saved.")

# --- 2. Clean ratings.csv ---
ratings = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ratings.csv'))

# Remove duplicates
ratings = ratings.drop_duplicates(subset=['userId', 'movieId', 'timestamp'])

# Handle missing values
ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])

# Remove users with <5 ratings
user_counts = ratings['userId'].value_counts()
ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= 5].index)]

# Remove movies with <5 ratings
movie_counts = ratings['movieId'].value_counts()
ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= 5].index)]

# Normalize ratings (z-score)
ratings['rating_z'] = (ratings['rating'] - ratings['rating'].mean()) / ratings['rating'].std()

ratings.to_csv(os.path.join(CLEANED_DATA_PATH, 'ratings_clean.csv'), index=False)
print("ratings_clean.csv saved.")

# --- 3. Clean tags.csv ---
tags = pd.read_csv(os.path.join(RAW_DATA_PATH, 'tags.csv'))

# Remove duplicates
tags = tags.drop_duplicates(subset=['userId', 'movieId', 'tag', 'timestamp'])

# Handle missing values
tags = tags.dropna(subset=['userId', 'movieId', 'tag'])

tags.to_csv(os.path.join(CLEANED_DATA_PATH, 'tags_clean.csv'), index=False)
print("tags_clean.csv saved.")

print("Preprocessing complete. Cleaned files are in the 'cleaned_data' folder.")