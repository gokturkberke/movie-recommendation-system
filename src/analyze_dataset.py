import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def find_data_folder():
    # Try current dir, parent dir, and src/../data
    possible_paths = [
        os.path.join(os.getcwd(), 'data'),
        os.path.join(os.getcwd(), '../data'),
        os.path.join(os.path.dirname(__file__), '../data'),
        os.path.join(os.path.dirname(__file__), '../../data'),
    ]
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'movies.csv')):
            return path
    print("Could not find 'data' folder with movies.csv. Please check your folder structure.")
    sys.exit(1)

def analyze_dataset():
    data_path = find_data_folder()

    print("=== MOVIES ===")
    movies = pd.read_csv(os.path.join(data_path, 'movies.csv'))
    print("Columns:", movies.columns.tolist())
    print("dtypes:\n", movies.dtypes)
    print("Number of rows:", len(movies))
    print("First 5 rows:\n", movies.head())
    # Process genres for visualization
    all_genres = movies['genres'].str.split('|').explode()
    genre_counts = all_genres.value_counts()
    # Filter out '(no genres listed)' if it exists and is not desired in the main plot
    if '(no genres listed)' in genre_counts:
        genre_counts = genre_counts.drop('(no genres listed)')

    print("Unique genres:", set(movies['genres'].str.cat(sep='|').split('|')))
    print()

    # Target 2.1: Distribution of Movie Genres (Top 20)
    plt.figure(figsize=(12, 8))
    top_n_genres = 20
    sns.barplot(x=genre_counts.head(top_n_genres).values, y=genre_counts.head(top_n_genres).index, palette='mako')
    plt.title(f'Top {top_n_genres} Movie Genres')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genres')
    plt.tight_layout() # Adjust layout to make room for labels
    plt.show()

    print("=== RATINGS ===")
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    print("Columns:", ratings.columns.tolist())
    print("dtypes:\n", ratings.dtypes)
    print("Number of rows:", len(ratings))
    print("First 5 rows:\n", ratings.head())
    rating_counts = ratings['rating'].value_counts().sort_index()
    print("Rating value counts:\n", rating_counts)
    print("Number of unique users:", ratings['userId'].nunique())
    print("Number of unique movies rated:", ratings['movieId'].nunique())
    print()

    # Target 1.1: Distribution of Ratings
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=ratings, palette='viridis')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Number of Ratings')
    plt.tight_layout()
    plt.show()

    # Target 1.2: Number of Ratings per Movie
    movie_rating_counts = ratings.groupby('movieId')['rating'].count()
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_rating_counts, bins=50, kde=False) # Using histplot
    plt.title('Distribution of Number of Ratings per Movie')
    plt.xlabel('Number of Ratings per Movie')
    plt.ylabel('Number of Movies')
    plt.xscale('log') # Apply log scale for better readability
    # plt.xlim(0, 500) # Alternative or additional: limit x-axis if needed, but log scale is often better for wide distributions
    plt.tight_layout()
    plt.show()

    print("=== TAGS ===")
    tags = pd.read_csv(os.path.join(data_path, 'tags.csv'))
    print("Columns:", tags.columns.tolist())
    print("dtypes:\n", tags.dtypes)
    print("Number of rows:", len(tags))
    print("First 5 rows:\n", tags.head())
    print("Number of unique tags:", tags['tag'].nunique())
    print("Number of unique users:", tags['userId'].nunique())
    print("Number of unique movies tagged:", tags['movieId'].nunique())
    print()

    # Target 3.1: Top N Most Frequent Tags
    plt.figure(figsize=(12, 10)) # Adjusted figure size for potentially many tags
    top_n_tags = 25
    tag_counts = tags['tag'].value_counts().head(top_n_tags)
    sns.barplot(x=tag_counts.values, y=tag_counts.index, palette='cubehelix')
    plt.title(f'Top {top_n_tags} Most Popular Tags')
    plt.xlabel('Frequency')
    plt.ylabel('Tags')
    plt.tight_layout() # Adjust layout to make room for labels
    plt.show()

if __name__ == "__main__":
    analyze_dataset()