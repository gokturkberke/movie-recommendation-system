import unittest
import pandas as pd
from app import recommend_by_watched_genres  # Öneri fonksiyonunun bulunduğu dosya

class TestMovieRecommendations(unittest.TestCase):

    def setUp(self):
        # Movies dosyasını yükler ve temizler
        self.movies = pd.read_csv('/Users/talya/Desktop/movie_recommendation - Kopya/cleaned_data/movies_clean.csv')
        self.movies.dropna(subset=['title', 'genres'], inplace=True)  # Güvenli temizlik
        self.movies['title'] = self.movies['title'].fillna('').astype(str).str.strip()
        self.movies['genres'] = self.movies['genres'].fillna('').astype(str).str.strip()
        
        # Test verisini ve beklenen sonuçları tanımlar
        self.test_data = [
            {
                "watched_movies": ['Toy Story (1995)', 'Jumanji (1995)'],
                "expected_recommendations": ['Toy Story 2 (1999)']
            },
            {
                "watched_movies": ['Heat (1995)'],
                "expected_recommendations": ['Casino (1995)', 'GoldenEye (1995)']
            },
            {
                "watched_movies": ['Pulp Fiction (1994)'],
                "expected_recommendations": ['Reservoir Dogs (1992)', 'Natural Born Killers (1994)']
            }
        ]

    def test_recommendations(self):
        print("\n=== Test Recommendations ===")
        for test_case in self.test_data:
            watched_movies = test_case['watched_movies']
            expected_recommendations = test_case['expected_recommendations']
            
            recommendations = recommend_by_watched_genres(
                watched_movies, self.movies, top_n=10
            )
            
            print(f"\nWatched: {watched_movies}")
            print(f"Expected: {expected_recommendations}")
            print(f"Recommended: {recommendations['title'].tolist()}\n")
            
            # Önerilen filmleri beklenenlerle karşılaştır
            recommended_titles = recommendations['title'].tolist()
            for expected in expected_recommendations:
                self.assertIn(expected, recommended_titles, f"Expected '{expected}' in recommendations for watched: {watched_movies}")

    def test_empty_watched_list(self):
        print("\n=== Test Empty Watched List ===")
        recommendations = recommend_by_watched_genres([], self.movies)
        self.assertTrue(recommendations.empty, "Should return empty DataFrame for empty watched list")

    def test_no_matching_movies(self):
        print("\n=== Test No Matching Movies ===")
        recommendations = recommend_by_watched_genres(['Non-Existent Movie'], self.movies)
        self.assertTrue(recommendations.empty, "Should return empty DataFrame for non-matching movies")

if __name__ == '__main__':
    unittest.main()


""""
import unittest
import pandas as pd
from app import recommend_by_watched_genres  # Öneri fonksiyonunun bulunduğu dosya

class TestMovieRecommendations(unittest.TestCase):

    def setUp(self):
        # Test verisini ve gerekli veriler manuel ekledim
        self.movies = pd.read_csv('/Users/talya/Desktop/movie_recommendation - Kopya/cleaned_data/movies_clean.csv') #dosya yolu göreceli olmalı
        self.test_data = [{"watched_movies": ['Toy Story (1995)', 'Jumanji (1995)'],
            'expected_recommendations': ['Toy Story 2 (1999)', 'Grumpier Old Men (1995)'] },
            {'watched_movies': ['Heat (1995)'],'expected_recommendations': ['Sudden Death (1995)', 'GoldenEye (1995)']}, 
            {'watched_movies': ['Pulp Fiction (1994)'],'expected_recommendations': []}
        ]

    def test_recommendations(self):
        # Her bir test senaryosu için öneriler
        for test_case in self.test_data:
            watched_movies = test_case['watched_movies']
            expected_recommendations = test_case['expected_recommendations']
            recommendations = recommend_by_watched_genres(watched_movies, self.movies, top_n=len(expected_recommendations)) #top_n doğru sayıda olmalı
            
            # Önerilen filmlerin beklenen filmlerle eşleşip eşleşmediğini kontrol 
            self.assertCountEqual(recommendations['title'].tolist(), expected_recommendations, f"Failed for watched movies: {watched_movies}")

    def test_empty_watched_list(self):
        # Kullanıcı hiç film izlemediyse boş liste dönmeli
        recommendations = recommend_by_watched_genres([], self.movies)
        self.assertTrue(recommendations.empty, "Should return empty DataFrame for empty watched list")

if __name__ == '__main__':
    unittest.main()
"""