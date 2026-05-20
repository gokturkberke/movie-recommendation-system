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

## Implemented vs. Experimental

Implemented in the Streamlit app:

* TF-IDF content-based recommendations with hybrid reranking.
* SBERT + FAISS semantic mode on the Content-Based page (requires prebuilt local index artifacts; see SBERT setup below).
* Surprise SVD collaborative filtering.
* Mood-based recommendations.
* Random movie picker.
* MovieId-based watch history recommendations.
* Optional TMDB poster and overview rendering.

Implemented for offline evaluation only:

* Popularity and random baselines.
* Pure TF-IDF content baseline.
* Hybrid watch-history content baseline.
* Semantic-LSA baseline (`--include-semantic`), built from TF-IDF + TruncatedSVD.
* SVD top-K and SVD rating prediction baselines.
* SBERT + FAISS semantic baseline (`--include-sbert-faiss`) using prebuilt local index artifacts.
* LightFM WARP baseline (`--include-lightfm`) using prebuilt local artifacts.

Future work:

* Implicit ALS / graph / sequence models.
* Larger, repeated evaluation runs before claiming model quality improvements.

## 🛠️ Technologies and Libraries Used

* **Programming Language:** Python 3.11
* **Data Analysis and Manipulation:** Pandas, NumPy
* **Machine Learning and Recommendation Algorithms:**
    * Scikit-learn (TF-IDF, Cosine Similarity)
    * Surprise (SVD algorithm, model training, and evaluation)
    * SentenceTransformers and FAISS for the optional SBERT semantic evaluation baseline
    * LightFM for the optional WARP offline evaluation baseline
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
    Then, install the dependencies and the project in editable mode:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```
    The editable install exposes `src/` modules as top-level imports (e.g., `from data_access import ...`) so tests and scripts no longer need `sys.path` workarounds.
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
* `docs/`: Project roadmap, evaluation design, and generated-result interpretation.
* `config/config.yaml`: Runtime parameters such as paths, candidate pool sizes, hybrid weights, mood mappings, menu labels, and demo profiles.
* `artifacts/`: Local generated evaluation reports, SBERT embeddings, and FAISS indexes. These are gitignored.

To run a small offline baseline evaluation:
```bash
.venv/bin/python scripts/evaluate_baselines.py --max-users 5 --k 5 --example-count 1
```

The same script can compare every available baseline (random, popularity, pure TF-IDF, hybrid TF-IDF, semantic-LSA, SVD top-K, SVD rating prediction):
```bash
.venv/bin/python scripts/evaluate_baselines.py \
  --max-users 25 --k 5,10,20 \
  --include-random --include-tfidf --include-content --include-semantic \
  --include-svd --include-svd-topk \
  --output-dir artifacts/evaluation
```
Results are written to `artifacts/evaluation/metrics_summary.{csv,json}` (gitignored). The Streamlit app is untouched.

To build and evaluate the real SBERT + FAISS semantic baseline, first create the local index artifacts:
```bash
.venv/bin/python scripts/build_sbert_faiss_index.py \
  --output-dir artifacts/indexes/sbert_faiss
```
For a quick smoke build, add `--sample-size 1000` and write to `/private/tmp/sbert_faiss_smoke`. Then evaluate the prebuilt index:
```bash
.venv/bin/python scripts/evaluate_baselines.py \
  --max-users 5 --k 5 \
  --include-sbert-faiss \
  --sbert-faiss-index-dir artifacts/indexes/sbert_faiss
```
This is evaluation-only and does not build embeddings during Streamlit startup.

To build and evaluate the LightFM WARP baseline, first create the local model artifacts:
```bash
.venv/bin/python scripts/train_lightfm_model.py \
  --output-dir artifacts/models/lightfm
```
Then evaluate the prebuilt artifact:
```bash
.venv/bin/python scripts/evaluate_baselines.py \
  --max-users 5 --k 5 \
  --include-lightfm \
  --lightfm-artifacts-dir artifacts/models/lightfm
```

Apple Silicon note: `lightfm==1.17` can fail during metadata generation on some Python 3.11/macOS arm64 environments with `AttributeError: 'dict' object has no attribute '__LIGHTFM_SETUP__'`. Retry with `pip install lightfm --no-build-isolation`. If that also fails, build from a temporary source checkout after replacing the setup sentinel with `import builtins; builtins.__LIGHTFM_SETUP__ = True`, or use a Linux/x86_64 host. Until LightFM is installed, the evaluation flag remains safe: the runner reports `lightfm_error` instead of crashing.

Current local findings are summarized in `docs/08_evaluation_results_report.md`. In short: popularity is a strong simple baseline at K=10, hybrid content showed the best K=20 ranking signal in the documented run, and hybrid latency was later reduced substantially by batching watch-history candidate generation. Treat these as local directional results, not final benchmark claims.

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
