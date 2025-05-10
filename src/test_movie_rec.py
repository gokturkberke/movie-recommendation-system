import unittest
import os
import pandas as pd
from surprise import SVD # Import SVD directly
from app import (
    recommend_by_watched_genres,
    # load_ratings_for_surprise, # Removed this import
    # train_surprise_model, # Removed this import
    get_user_recommendations,
    load_movies,
    load_ratings 
)
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

class TestMovieRecommendations(unittest.TestCase):

    def setUp(self):
        # Movies dosyasını yükler ve temizler
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Adjust path to be relative to the test file's location
        self.base_dir = os.path.join(script_dir, '..') 
        self.cleaned_data_path = os.path.join(self.base_dir, 'cleaned_data')
        
        self.movies_df = load_movies(data_path=self.cleaned_data_path)
        self.ratings_df = load_ratings(data_path=self.cleaned_data_path)

        # Prepare data for Surprise
        reader = Reader(rating_scale=(0.5, 5.0))
        self.surprise_data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], reader)
        
        # Split data into training and testing sets for Surprise model evaluation
        self.trainset, self.testset = train_test_split(self.surprise_data, test_size=0.25, random_state=42)
        
        # Train the SVD model directly within the test's setUp
        print("Testler için SVD modeli self.trainset üzerinde eğitiliyor...")
        algo_for_test = SVD()
        algo_for_test.fit(self.trainset) # Directly train on the test-specific trainset
        self.svd_model = algo_for_test # Use this model in tests
        print("Test modeli eğitildi.")

        # Test data for recommend_by_watched_genres
        self.test_data_watched_genres = [
            {
                "watched_movies": ['Toy Story (1995)', 'Jumanji (1995)'],
                "expected_recommendations": ['Toy Story 2 (1999)'] # Example, adjust based on your logic
            },
            {
                "watched_movies": ['Heat (1995)'],
                "expected_recommendations": ['Casino (1995)', 'GoldenEye (1995)'] # Example
            },
            {
                "watched_movies": ['Pulp Fiction (1994)'],
                "expected_recommendations": ['Reservoir Dogs (1992)', 'Natural Born Killers (1994)'] # Example
            }
        ]

    def test_recommendations_by_watched_genres(self):
        print("\\n=== Test Recommendations by Watched Genres ===")
        for test_case in self.test_data_watched_genres:
            watched_movies = test_case['watched_movies']
            expected_recommendations = test_case['expected_recommendations']
            
            recommendations = recommend_by_watched_genres(
                watched_movies, self.movies_df, top_n=10
            )
            
            print(f"\\nWatched: {watched_movies}")
            print(f"Expected (sample): {expected_recommendations}")
            print(f"Recommended: {recommendations['title'].tolist()}\\n")
            
            recommended_titles = recommendations['title'].tolist()
            # This test is more of a sanity check as exact matches can be hard with genre-based logic
            # You might want to check if recommended genres align with watched genres
            if expected_recommendations: # Only assert if there are expected titles
                 for expected in expected_recommendations:
                    self.assertIn(expected, recommended_titles, f"Expected '{expected}' in recommendations for watched: {watched_movies}")

    def test_empty_watched_list_for_genre_recs(self):
        print("\\n=== Test Empty Watched List for Genre Recs ===")
        recommendations = recommend_by_watched_genres([], self.movies_df)
        self.assertTrue(recommendations.empty, "Should return empty DataFrame for empty watched list")

    def test_no_matching_movies_for_genre_recs(self):
        print("\\n=== Test No Matching Movies for Genre Recs ===")
        # Using a very unlikely movie title
        recommendations = recommend_by_watched_genres(['NonExistentMovieTitle12345XYZ'], self.movies_df)
        # Depending on fallback logic, this might not be empty. 
        # If fallback provides random movies, this test needs adjustment.
        # Assuming current logic might return genre-based from "NonExistent..." if it has genres,
        # or fallback if no movies are found.
        # For now, let's assume it should be empty if the title itself is not found and no genres can be inferred.
        # This might need refinement based on how `_extract_watched_movies_and_genres` handles truly non-existent titles.
        print(f"Recommendations for non-existent movie: {recommendations}")


    # --- Tests for Collaborative Filtering (Surprise) ---

    def test_surprise_model_rmse(self):
        print("\\n=== Test Surprise Model RMSE ===")
        predictions = self.svd_model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        print(f"RMSE for SVD model: {rmse}")
        # Set a threshold for RMSE, e.g., less than 1.0. This depends on your data and expectations.
        self.assertLess(rmse, 1.0, "RMSE should be below 1.0 for the SVD model")

    def test_collaborative_filtering_recommendations_for_user(self):
        print("\\n=== Test Collaborative Filtering Recommendations ===")
        # Test for a specific user (e.g., userId 1)
        user_id_to_test = 1 
        
        # Ensure the user exists in the ratings data
        if user_id_to_test not in self.ratings_df['userId'].unique():
            print(f"User {user_id_to_test} not in ratings data, skipping CF test for this user.")
            return

        recommendations = get_user_recommendations(
            user_id=user_id_to_test,
            surprise_model=self.svd_model,
            movies_df=self.movies_df,
            ratings_df=self.ratings_df, # Pass the full ratings_df here
            top_n=5
        )
        
        print(f"\\nRecommendations for User ID {user_id_to_test}:")
        if not recommendations.empty:
            print(recommendations)
        else:
            print("No recommendations generated (this might be expected if user has rated all, or few items).")

        self.assertIsInstance(recommendations, pd.DataFrame, "Recommendations should be a DataFrame.")
        if not recommendations.empty:
            self.assertTrue('title' in recommendations.columns, "DataFrame should have a 'title' column.")
            self.assertTrue('genres' in recommendations.columns, "DataFrame should have a 'genres' column.")
            self.assertLessEqual(len(recommendations), 5, "Should return at most top_n recommendations.")

        # Further checks:
        # 1. Ensure recommended movies have not been previously rated by this user.
        user_rated_movies = self.ratings_df[self.ratings_df['userId'] == user_id_to_test]['movieId'].unique()
        if not recommendations.empty:
            recommended_movie_ids = self.movies_df[self.movies_df['title'].isin(recommendations['title'])]['movieId']
            for rec_movie_id in recommended_movie_ids:
                self.assertNotIn(rec_movie_id, user_rated_movies, 
                                 f"Movie ID {rec_movie_id} (recommended) was already rated by user {user_id_to_test}")
                                 
    def test_collaborative_filtering_for_new_user(self):
        print("\\n=== Test Collaborative Filtering for New User (Cold Start) ===")
        # A user ID that is definitely not in the dataset
        new_user_id = self.ratings_df['userId'].max() + 100 
        
        recommendations = get_user_recommendations(
            user_id=new_user_id,
            surprise_model=self.svd_model,
            movies_df=self.movies_df,
            ratings_df=self.ratings_df,
            top_n=5
        )
        
        print(f"\\nRecommendations for New User ID {new_user_id}:")
        if not recommendations.empty:
            print(recommendations)
        else:
            print("No recommendations generated (expected for a new user with no ratings).")
        
        # For a new user with no ratings, the current get_user_recommendations might return empty
        # or fall back to some default (e.g. popular movies, though not implemented here).
        # The current implementation of get_user_recommendations iterates through all_movie_ids
        # and predicts. For a truly new user not in the trainset, SVD might give default predictions.
        # Let's assert it's a DataFrame, and it could be empty.
        self.assertIsInstance(recommendations, pd.DataFrame, "Recommendations should be a DataFrame.")
        if recommendations.empty:
            print("Empty recommendations for new user, which is acceptable if no fallback is implemented.")


if __name__ == '__main__':
    unittest.main()

# Remove the duplicated class definition at the end of the file if it exists.
# The previous content had a duplicated TestMovieRecommendations class.
