# 🎬 Advanced Movie Recommendation System

This project is a comprehensive movie recommendation system that suggests personalized movie recommendations to users, utilizing the MovieLens 25M dataset. The system incorporates core data science methods such as data preprocessing, exploratory data analysis, and the implementation of various recommendation algorithms (content-based, collaborative filtering, mood-based, etc.). To enhance user interaction and experience, it features interactive elements and TMDB API integration for displaying movie posters and summaries.

## 🌟 Core Features

* **Diverse Recommendation Strategies:**
    * **Content-Based Recommendations:** Suggests similar movies by analyzing movie titles, genres, and tags (using TF-IDF and Cosine Similarity) based on a movie liked by the user or their watch history.
    * **Collaborative Filtering (SVD):** Provides personalized recommendations based on users' past rating behavior (using the Surprise library and SVD algorithm).
    * **Mood-Based Recommendations:** Offers random movie suggestions from genres uygun to the user's selected mood (e.g., Happy, Sad, Adventurous).
    * **Personalized Content-Based Recommendations from Watch History:** Recommends new movies similar in content (genre, tags, etc.) to movies the user has previously watched, ensuring not to re-recommend watched movies.
    * **Random Movie Picker:** Displays a random movie and its information from the dataset.
* **Interactive User Interface:**
    * A user-friendly web interface developed using [Streamlit](https://streamlit.io/).
    * Allows users to manage their watch history (add movies).
    * Displays recommended movies with their titles, genres, posters, and summaries fetched from TMDB.
* **Data Management and Preprocessing:**
    * Comprehensive cleaning and preprocessing of the MovieLens 25M dataset (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`).
    * TF-IDF vectorization for textual data.
    * Saving and reloading of the trained model and processed data.
* **TMDB API Integration:**
    * Dynamically fetches movie posters and summaries to provide a rich visual experience. (Requires a TMDB API Key)

## 🛠️ Technologies and Libraries Used

* **Programming Language:** Python 3.x
* **Data Analysis and Manipulation:** Pandas, NumPy
* **Machine Learning and Recommendation Algorithms:**
    * Scikit-learn (TF-IDF, Cosine Similarity)
    * Surprise (SVD algorithm, model training, and evaluation)
* **Text Similarity:** TheFuzz (FuzzyWuzzy)
* **Web Interface:** Streamlit
* **API Interaction:** Requests
* **Data Visualization (During Analysis Phase):** Matplotlib, Seaborn
* **Dataset:** [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/)

## 📂 Project Structure
.
├── cleaned_data/           # Preprocessed and cleaned CSV files & trained model (.pkl)
│   ├── movies_clean.csv
│   ├── ratings_clean.csv
│   ├── tags_clean.csv
│   └── svd_trained_model.pkl
├── data/                   # Raw MovieLens 25M dataset files
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
├── src/                    # Source code files
│   ├── app.py              # Main Streamlit application
│   ├── preprocess_dataset.py # Data preprocessing script
│   ├── train_save_model.py # Script for training and saving the SVD model
│   ├── utils_data.py       # Utility functions for data loading, etc.
│   ├── analyze_dataset.py  # Exploratory data analysis script (optional execution)
│   ├── test_movie_rec.py   # Unit tests
│   └── config.py           # Configuration (API key, mood-genre map)
└── README.md               # This file

## 🚀 Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/gokturkberke/movie-recommendation-system.git](https://github.com/gokturkberke/movie-recommendation-system.git)
    cd movie-recommendation-system
    ```

2.  **Install Required Libraries:**
    It's recommended to create a virtual environment first.
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/macOS
    # venv\Scripts\activate  # For Windows
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create the `requirements.txt` file based on your project's libraries, e.g., by running `pip freeze > requirements.txt` in your activated virtual environment.)*

3.  **Download the Dataset:**
    * Download the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/) (`ml-25m.zip`).
    * Extract the ZIP file and copy `links.csv`, `movies.csv`, `ratings.csv`, and `tags.csv` into the `data/` folder of your project.

4.  **Set Up Your TMDB API Key:**
    * Obtain a free API key from [The Movie Database (TMDB) API](https://www.themoviedb.org/documentation/api).
    * Open the `src/config.py` file and set your API key in the `TMDB_API_KEY` variable:
        ```python
        TMDB_API_KEY = "YOUR_TMDB_API_KEY"
        ```
    * **(Recommended Security Practice):** Instead of hardcoding your API key, consider using a `.env` file and the `python-dotenv` library to load it from environment variables. You would then need to update `config.py` and add `.env` to your `.gitignore` file.

5.  **Run the Data Preprocessing Script:**
    This script will clean the raw data and save it to the `cleaned_data/` folder.
    ```bash
    python src/preprocess_dataset.py
    ```

6.  **Train the Recommendation Model:**
    This script will train the SVD model and save it as `cleaned_data/svd_trained_model.pkl`.
    ```bash
    python src/train_save_model.py
    ```

7.  **Launch the Streamlit Application:**
    ```bash
    streamlit run src/app.py
    ```
    The application will typically open in your web browser at `http://localhost:8501`.

## 📖 Usage

Once the application is running:

1.  Select a recommendation method from the sidebar menu:
    * **Content-Based Recommendation:** Enter a movie title you like to find similar movies.
    * **Collaborative Filtering:** Enter your user ID to get personalized recommendations.
    * **Mood-Based Recommendation:** Get movie suggestions based on your current mood.
    * **Random Movie:** Discover a random movie.
    * **Watch History & Recommendations:** Add movies to your watch history and get recommendations based on it.
    * **Unwatched Movies:** List movies not in your watch history.
2.  Input the required information (movie title, user ID, etc.) in the respective fields and click the "Get Recommendations" (or similar) button.
3.  The results will be displayed, including movie titles, genres, posters, and summaries.

## 🧪 Tests (Optional)

Unit tests are available for some core functions of the project. To run them:
```bash
python -m unittest src/test_movie_rec.py
