import pandas as pd
import os
import sys

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
    print("Unique genres:", set('|'.join(movies['genres']).split('|')))
    print()

    print("=== RATINGS ===")
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    print("Columns:", ratings.columns.tolist())
    print("dtypes:\n", ratings.dtypes)
    print("Number of rows:", len(ratings))
    print("First 5 rows:\n", ratings.head())
    print("Rating value counts:\n", ratings['rating'].value_counts().sort_index())
    print("Number of unique users:", ratings['userId'].nunique())
    print("Number of unique movies rated:", ratings['movieId'].nunique())
    print()

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

if __name__ == "__main__":
    analyze_dataset()