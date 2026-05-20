- **Date:** 2026-05-20
- **Topic:** Full-catalog SBERT+FAISS run, post-optimization evaluation report refresh, and Streamlit UI integration
- **Motivation:** `docs/08_evaluation_results_report.md` "Follow-Up Checkpoints" section flags two leftover items: (a) the SBERT+FAISS baseline was added behind `--include-sbert-faiss` but only a 1,000-row smoke index was ever produced, so no full-catalog comparison exists, and (b) the main evaluation tables still carry the pre-optimization `hybrid_content` latency (13,491 ms mean) even though commit `c2793c4` reduced it to roughly 0.58 s per user. README `Future work` also lists "Streamlit UI integration for SBERT + FAISS" as still open. Baseline run id for comparison: `artifacts/evaluation/metrics_summary_2026-05-14T21-05-52Z.{csv,json}` (the file currently mirrored as `metrics_summary.{csv,json}` in the same directory).
- **Hypothesis:** With a full-catalog SBERT+FAISS index (`row_count` ≈ 79,477 movies, `embedding_dim` = 384), the `sbert_faiss_content` baseline will produce at least one hit at K=20 on the same 25-user / latest-1 holdout slice that `metrics_summary_2026-05-14T21-05-52Z` was built from, and the regenerated `hybrid_content` mean latency in the same run will be under 2,000 ms. We do not claim SBERT will beat hybrid; we only claim the comparison becomes possible and the report is no longer stale.
- **Preconditions:**
  - `requirements.txt` already lists `sentence-transformers` and `faiss-cpu` (commit `2c866f0`).
  - `src/experimental/sbert_faiss.py`, `scripts/build_sbert_faiss_index.py`, and the `--include-sbert-faiss` wiring in `scripts/evaluate_baselines.py` + `src/evaluation_runner.py` are committed (commit `9a14e12`).
  - Hybrid watch-history optimization is committed (commit `c2793c4`).
  - `cleaned_data/movies_clean.csv` (79,477 rows) and `cleaned_data/tags_clean.csv` (2,328,298 rows) are present locally.
  - `config/config.yaml:evaluation.sbert_faiss.index_dir = artifacts/indexes/sbert_faiss` is the agreed default location.
  - The execution order is strictly sequential: each numbered item below ships its own commit and smoke test before the next item starts.

## 1) Build the full-catalog SBERT+FAISS index

- **Goal:** Produce real (not sample-size) SBERT+FAISS artifacts under `artifacts/indexes/sbert_faiss/` so subsequent evaluation and UI work can rely on them.
- **Files:**
  - `scripts/build_sbert_faiss_index.py` (no code change expected; CLI-only invocation).
  - `src/experimental/sbert_faiss.py:build_sbert_faiss_artifacts` (existing function, no change expected).
  - `artifacts/indexes/sbert_faiss/sbert_faiss.index` (new, gitignored).
  - `artifacts/indexes/sbert_faiss/embeddings.npy` (new, gitignored).
  - `artifacts/indexes/sbert_faiss/movie_ids.csv` (new, gitignored).
  - `artifacts/indexes/sbert_faiss/metadata.json` (new, gitignored — the only artifact worth keeping in git history would be the row_count / dim / model_name metadata; left out of the commit unless `.gitignore` says otherwise).
- **Steps:**
  - Run `.venv/bin/python scripts/build_sbert_faiss_index.py --output-dir artifacts/indexes/sbert_faiss`. The defaults pull `model_name`, `batch_size`, and `index_dir` from `config/config.yaml:evaluation.sbert_faiss`.
  - Confirm the four output files exist and that `metadata.json` reports `row_count >= 70000` (full catalog) and `embedding_dim = 384`.
  - Run `.venv/bin/python -m unittest tests.test_sbert_faiss` to make sure the existing fixture-level tests still pass after the artifacts are in place (no test relies on the real index, but this is the unit-test smoke).
  - Run a small 5-user evaluation smoke: `.venv/bin/python scripts/evaluate_baselines.py --max-users 5 --k 5 --include-sbert-faiss --sbert-faiss-index-dir artifacts/indexes/sbert_faiss`. The JSON output must contain a `top_n.sbert_faiss_content` block and must not contain a `sbert_faiss_error` field.
  - Commit message: `feat(eval): build full-catalog SBERT+FAISS index artifacts`. Only the metadata pointer (this plan file's DONE marker) and any `.gitignore` adjustment travel with the commit; the binary `embeddings.npy` and `sbert_faiss.index` stay local under `artifacts/` per current repo policy.
- **Test / verification:**
  - `metadata.json:row_count` is approximately 79,477.
  - `metadata.json:embedding_dim` is 384.
  - 5-user smoke JSON has `top_n.sbert_faiss_content` populated.
  - `tests.test_sbert_faiss` exit code 0.
- **Expected outcome:** Full-catalog artifacts ready on disk; SBERT path is now exercisable by both the evaluation runner and (later) the Streamlit app. Decision criterion is binary: smoke produces `sbert_faiss_content` rows, or this item is reopened.
- **DONE (commit `<pending>`):** Full-catalog SBERT+FAISS artifacts built under `artifacts/indexes/sbert_faiss/`. Unit tests and a 5-user smoke evaluation both passed; the SBERT path is now exercisable end-to-end.
  - Metadata: `row_count = 79477`, `embedding_dim = 384`, `model_name = sentence-transformers/all-MiniLM-L6-v2`, `batch_size = 64`, `created_at = 2026-05-20T06:35:50Z`.
  - On-disk artifacts (all gitignored under `artifacts/`): `sbert_faiss.index` (~116 MB), `embeddings.npy` (~116 MB), `movie_ids.csv` (~515 KB), `metadata.json` (381 B).
  - Smoke: `tests.test_sbert_faiss` 2/2 OK; `evaluate_baselines.py --max-users 5 --k 5 --include-sbert-faiss` produced a populated `top_n.sbert_faiss_content` block (5 evaluated users, 22 unique items, no `sbert_faiss_error`).
  - Decision: shipped to Faz 2.

## 2) Refresh the post-optimization evaluation report

- **Goal:** Regenerate the canonical 25-user evaluation with all baselines (including SBERT+FAISS) and rewrite the user-facing report so its tables reflect the post-optimization latency and the new model.
- **Files:**
  - `docs/08_evaluation_results_report.md` (updated in place).
  - `artifacts/evaluation/metrics_summary.{csv,json}` and timestamped copies (regenerated locally, gitignored).
  - `artifacts/evaluation/run_config.json` (regenerated, gitignored).
- **Steps:**
  - Run the canonical command:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 25 --k 5,10,20 \
      --include-random --include-tfidf --include-content --include-semantic \
      --include-svd --include-svd-topk \
      --include-sbert-faiss \
      --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
      --output-dir artifacts/evaluation
    ```
  - Read the newly written `metrics_summary.csv` and verify 7 models × 3 K values + 1 SVD rating prediction row.
  - Rewrite the "Run Summary" block in `docs/08_evaluation_results_report.md` with the new date and the new command (semantic baseline note is unchanged; the SBERT+FAISS note moves from caveat to fact).
  - Rewrite the K=10 and K=20 tables to include `sbert_faiss_content` and to use the freshly produced numbers for every model.
  - Rewrite the "Latency Findings" table; the leading question is whether post-optimization `hybrid_content` mean latency is under 2,000 ms.
  - Trim the "Follow-Up Checkpoints" block so that the full-catalog SBERT+FAISS line is replaced with a single line pointing to this plan file as the audit trail; remove the "regenerate the post-optimization report" follow-up because this item closes that loop.
  - Adjust the "Conclusions" block to add a single short sentence comparing `hybrid_content` and `sbert_faiss_content` on K=20 NDCG; do not over-claim.
  - Commit message: `docs(eval): refresh evaluation results report with post-optimization + SBERT+FAISS run`.
- **Test / verification:**
  - `wc -l artifacts/evaluation/metrics_summary.csv` reports 22 lines (1 header + 7×3 model/K rows + 1 SVD rating prediction row).
  - In the new CSV, `hybrid_content,10,...,latency_mean_ms` is below 2000.
  - In the new CSV, `sbert_faiss_content` exists at K=5, 10, 20 with `evaluated_user_count >= 1`.
  - `docs/08_evaluation_results_report.md` no longer contains the literal `13,491.9 ms`.
- **Expected outcome:** A faithful, repeatable report that future agents can quote without needing the "actually that's stale" footnote. Decision criterion: every table reflects the new run; no number in the report contradicts `metrics_summary.csv`.
- **DONE / DROPPED:** (filled in after commit; include a small table with popularity / hybrid_content / sbert_faiss_content K=10 rows from the new run)

## 3) Wire SBERT semantic mode into the Streamlit Content-Based page

- **Goal:** Let users pick between the existing TF-IDF + hybrid recommender and the new SBERT semantic recommender on the existing Content-Based sidebar page, without adding a new sidebar menu item (CLAUDE.md scope rule forbids new pages).
- **Files:**
  - `src/app.py` (~480 lines): add a cached `cached_sbert_index()` resource keyed on the configured index dir; add a recommender-mode radio button inside `render_content_based_page()`; add a graceful-failure branch when artifacts are missing; the SBERT branch calls `experimental.sbert_faiss.sbert_faiss_recommendations_for_seed_ids` with the matched `movieId` from `find_movie_match_by_id`.
  - `src/recommenders/__init__.py`: no change required (Streamlit will import directly from `experimental.sbert_faiss` to honor the existing experimental boundary).
  - `README.md`: move the SBERT+FAISS line from `Future work` into `Implemented in the Streamlit app`, and add a short usage note that the user must run `scripts/build_sbert_faiss_index.py` before SBERT mode lights up.
  - `docs_untracked/05-ui-tmdb-and-runtime-flows.md`: if touched, note the new radio inside the Content-Based page section; otherwise leave it for a later docs sweep.
- **Steps:**
  - Inside `load_context()`, resolve the configured SBERT index dir from `EVALUATION_DEFAULTS["sbert_faiss"]["index_dir"]` via `project_path(...)`, attempt to load the index through `cached_sbert_index()` (returns `None` if artifacts are missing or imports fail), and add the result + the dir under new keys `sbert_index` and `sbert_index_dir` in the context dict.
  - Inside `render_content_based_page()`, render a `st.radio("Recommender", ("TF-IDF (hybrid)", "SBERT semantic"))`. Default is TF-IDF. SBERT option is `disabled=True` with a helper caption when `context["sbert_index"] is None`.
  - When SBERT is selected and a movie is matched (via the existing `suggest_movie_titles` + `find_movie_match_by_id` flow), call `sbert_faiss_recommendations_for_seed_ids([matched_movie_id], context["sbert_index"], movies_for_output, watched_movie_ids=st.session_state.watched_movie_ids, top_n=10)` and pass the result to `render_movie_list`.
  - In the TF-IDF branch, keep the existing flow byte-for-byte. No hybrid weights change.
  - Add a single-line caption under the radio explaining what SBERT mode does, citing the metadata.json model name when available.
  - Commit message: `feat(ui): add SBERT semantic mode to content-based recommendation page`.
- **Test / verification:**
  - `.venv/bin/python -m unittest discover -s tests` passes (no test should break; the experimental module is imported lazily through `cached_sbert_index`).
  - Manual: `.venv/bin/streamlit run src/app.py`. On the Content-Based page, the new radio appears, selecting "SBERT semantic" returns a list of 10 movies for a seed like "The Matrix", and TF-IDF mode still returns the existing hybrid output.
  - With `artifacts/indexes/sbert_faiss/` temporarily renamed to `_sbert_faiss_disabled`, restarting the app shows the SBERT option as disabled and a caption telling the user to run the build script. Restore the directory after the check.
- **Expected outcome:** Streamlit users can compare the two recommenders directly on the same seed without leaving the page. Decision criterion: manual smoke shows both branches return 10 rows on a real seed; missing-artifact branch shows the disabled state without crashing.
- **DONE / DROPPED:** (filled in after commit; record the seed movie used in the smoke and the elapsed wall time for the SBERT search)
