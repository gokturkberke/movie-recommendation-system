# 🎬 Advanced Movie Recommendation System

This project is a comprehensive movie recommendation system that suggests personalized movie recommendations to users, utilizing the MovieLens latest (ml-latest) dataset. The system incorporates core data science methods such as data preprocessing, exploratory data analysis, and the implementation of various recommendation algorithms (content-based, collaborative filtering, mood-based, etc.). To enhance user interaction and experience, it features interactive elements and TMDB API integration for displaying movie posters and summaries.

## 🌟 Core Features

* **Diverse Recommendation Strategies:**
    * **Content-Based Recommendations:** Retrieves similar movies by analyzing titles, genres, and tags with TF-IDF/cosine similarity, then re-ranks candidates with Bayesian rating, popularity, and light diversity signals.
    * **Collaborative Filtering (SVD):** Provides personalized recommendations based on users' past rating behavior (using the Surprise library and SVD algorithm).
    * **Mood-Based Recommendations:** Offers random movie suggestions from genres mapped to the user's selected mood (e.g., Happy, Sad, Adventurous).
    * **Personalized Content-Based Recommendations from Watch History:** Stores watched movies by `movieId`, then recommends new movies similar in content while avoiding already watched IDs.
    * **Random Movie Picker:** Displays a random movie and its information from the dataset.
* **Interactive User Interface:**
    * A user-friendly web interface developed using [Streamlit](https://streamlit.io/).
    * Allows users to manage their watch history (add movies).
    * Displays recommended movies with their titles, genres, posters, and summaries fetched from TMDB.
* **Data Management and Preprocessing:**
    * Comprehensive cleaning and preprocessing of the MovieLens latest (ml-latest) dataset (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`).
    * Cleaned dataset snapshot: 79,477 movies, 33,703,215 ratings, and 2,328,298 tags.
    * Latest release year in the cleaned movies: 2023, with 553 movies.
    * TF-IDF vectorization for textual data.
    * Saving and reloading of the trained model and processed data.
* **TMDB API Integration:**
    * Dynamically fetches movie posters and summaries to provide a rich visual experience. (Requires a TMDB API Key)

## 🛠️ Technologies and Libraries Used

* **Programming Language:** Python 3.11
* **Data Analysis and Manipulation:** Pandas, NumPy
* **Machine Learning and Recommendation Algorithms:**
    * Scikit-learn (TF-IDF, Cosine Similarity)
    * Surprise (SVD algorithm, model training, and evaluation)
* **Text Similarity:** TheFuzz (FuzzyWuzzy)
* **Web Interface:** Streamlit
* **API Interaction:** Requests
* **Configuration:** YAML through `config/config.yaml`
* **Data Visualization (During Analysis Phase):** Matplotlib, Seaborn
* **Dataset:** [MovieLens Latest Dataset (ml-latest)](https://grouplens.org/datasets/movielens/latest/)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/gokturkberke/movie-recommendation-system.git
    cd movie-recommendation-system
    ```

2.  **Install Required Libraries:**
    It's recommended to create a virtual environment first.
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate  # For Linux/macOS
    # .venv\Scripts\activate  # For Windows
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the Dataset:**
    * Download the [MovieLens Latest Dataset](https://grouplens.org/datasets/movielens/latest/) (`ml-latest.zip`).
    * Extract the ZIP file and copy `links.csv`, `movies.csv`, `ratings.csv`, and `tags.csv` into the `data/` folder of your project.

4.  **Set Up Your TMDB API Key:**
    * Obtain a free API key from [The Movie Database (TMDB) API](https://www.themoviedb.org/documentation/api).
    * Set it as an environment variable before launching Streamlit:
        ```bash
        export TMDB_API_KEY="YOUR_TMDB_API_KEY"
        ```
    * Alternatively, place it in `.streamlit/secrets.toml`. If no key is configured, the app still runs but disables posters and overviews.

5.  **Run the Data Preprocessing Script:**
    This script will clean the raw data and save it to the `cleaned_data/` folder.
    ```bash
    .venv/bin/python scripts/preprocess_dataset.py
    ```

6.  **Train the Recommendation Model:**
    This script will train the SVD model and save it as `cleaned_data/svd_trained_model.pkl`.
    ```bash
    .venv/bin/python scripts/train_save_model.py
    ```

7.  **Launch the Streamlit Application:**
    ```bash
    .venv/bin/streamlit run src/app.py
    ```
    The application will typically open in your web browser at `http://localhost:8501`.

## Project Layout

* `src/`: Core Streamlit app, data access, TMDB client, recommendation logic, and evaluation helpers.
* `scripts/`: Command-line orchestrators that call core functions from `src/`.
* `tests/`: Unit and smoke tests.
* `config/config.yaml`: Runtime parameters such as paths, candidate pool sizes, hybrid weights, mood mappings, menu labels, and demo profiles.

To run a small offline baseline evaluation:
```bash
.venv/bin/python scripts/evaluate_baselines.py --max-users 5 --k 5 --example-count 1
```

## 📖 Usage

Once the application is running:

1.  Select a recommendation method from the sidebar menu:
    * **Content-Based Recommendation:** Enter a movie title or select a close match to find hybrid-ranked similar movies.
    * **Collaborative Filtering:** Enter your user ID to get personalized recommendations.
    * **Mood-Based Recommendation:** Get movie suggestions based on your current mood.
    * **Random Movie:** Discover a random movie.
    * **Watch History & Recommendations:** Add movies to your watch history and get recommendations based on their `movieId`.
    * **About & Help:** Explains recommendation methods and local setup requirements.
2.  Input the required information (movie title, user ID, etc.) in the respective fields and click the "Get Recommendations" (or similar) button.
3.  The results will be displayed, including movie titles, genres, posters, and summaries.

## 🧪 Tests (Optional)

Unit tests are available for some core functions of the project. To run them:
```bash
.venv/bin/python -m unittest discover -s tests
```

The legacy wrapper command is still supported:
```bash
.venv/bin/python -m unittest src/test_movie_rec.py
```
