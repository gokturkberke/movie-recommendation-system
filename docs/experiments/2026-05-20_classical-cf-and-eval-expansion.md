- **Date:** 2026-05-20
- **Topic:** Classical CF expansion (LightFM + Implicit ALS) and wider evaluation baseline at 100 users
- **Motivation:** `docs/06_project_inventory_and_roadmap_en.md` Priority 3 and `docs/01_technical_comparison_report.md` §8 Phase 1 both call for moving past linear Surprise SVD to implicit-feedback classical CF models (LightFM with WARP/BPR ranking loss, Implicit ALS with Hu-Koren confidence weighting). At the same time, `docs/08_evaluation_results_report.md` (refreshed by commit `5612ff0`) carries a caveat that only 15 users had positive holdout items in the 25-user slice — too small to discriminate content-only baselines on relevance. The configured `evaluation.max_users` default of 100 was never run. This plan resolves both gaps in one sweep, then leaves graph (LightGCN) and sequential (SASRec / BERT4Rec) models as deferred follow-ups.
- **Hypothesis:** Running the existing 7 baselines at 100 users will produce a holdout slice with at least 60 positive-holdout users (vs. the previous 15), enough for `tfidf_content`, `semantic_content`, and `sbert_faiss_content` to register at least one relevance hit at K=20. Adding LightFM (WARP loss) and Implicit ALS (Hu-Koren confidence) on top of that baseline will land at top-N quality at least on par with the trained Surprise SVD top-K baseline (`svd_topk`), with mean per-user inference latency under 500 ms for LightFM and under 300 ms for ALS — well below `hybrid_content` (1,320.7 ms in the post-optimization run). We do not claim these models will beat the hybrid recommender; we claim they become legitimate, faster-to-serve alternatives in the comparison table.
- **Preconditions:**
  - The four prior commits from the previous round are in `main`: `3234442` (full-catalog SBERT+FAISS index), `5612ff0` (post-opt report refresh), `ad54232` (SBERT UI integration), `cc4ea21` (plan-file hash fix).
  - `cleaned_data/ratings_clean.csv` (~33.7M rows, gitignored) and `cleaned_data/movies_clean.csv` (79,477 rows) are present locally.
  - `cleaned_data/svd_trained_model.pkl` is present (~1.2 GB).
  - `artifacts/indexes/sbert_faiss/` is populated with the full-catalog index (`row_count = 79477`, `embedding_dim = 384`).
  - `requirements.txt` already lists `sentence-transformers` and `faiss-cpu`. The two new packages this plan adds — `lightfm` and `implicit` — are not yet installed.
  - Python 3.11, macOS arm64 venv at `.venv/`. On Apple Silicon, `lightfm` wheel-only installs sometimes need `pip install lightfm --no-build-isolation`; `implicit` ships native wheels for arm64.
  - CLAUDE.md §7 governs how DONE markers and commit hashes land in this file; do not fill `<hash>` placeholders before the corresponding commit is created.

## 1) Re-run the existing 7 baselines at 100 users

- **Goal:** Establish the wider baseline numbers (100 selected users, latest-1 holdout per user) for every existing model before LightFM and ALS join the comparison. This is the relevance / latency reference the next two items will compare against.
- **Files:**
  - `artifacts/evaluation/metrics_summary.{csv,json}` (regenerated, gitignored).
  - `artifacts/evaluation/run_config.json` (regenerated, gitignored).
  - `artifacts/evaluation/metrics_summary_<timestamp>.{csv,json}` (regenerated, gitignored).
  - `docs/08_evaluation_results_report.md` (updated in place to reflect the 100-user numbers).
- **Steps:**
  - Confirm the SBERT index dir is intact: `ls artifacts/indexes/sbert_faiss/metadata.json`.
  - Run the canonical wider evaluation:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 100 --k 5,10,20 \
      --include-random --include-tfidf --include-content --include-semantic \
      --include-svd --include-svd-topk \
      --include-sbert-faiss \
      --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
      --output-dir artifacts/evaluation
    ```
  - Read the new `metrics_summary.csv` and verify 7 models × 3 K values + 1 SVD rating prediction row (22 lines incl. header).
  - Update `docs/08_evaluation_results_report.md`:
    - "Run Summary" — change `Users selected` from 25 to 100; quote the new evaluated-user count from `data.evaluated_user_count` in the JSON.
    - K=10 and K=20 tables — replace every row with the new run's numbers (keep the 7 existing models; rows for LightFM and ALS are added in item 4).
    - "Latency Findings" — same: 7 rows, refreshed numbers.
    - "Caveats" — remove the line that pins the slice at 15 positive-holdout users; restate it with the new count.
    - "Conclusions" — keep the existing structure but refresh any sentence that quotes specific 25-user numbers.
  - Commit message: `docs(eval): expand evaluation run to 100 users for baseline comparison`. Use `git add -f docs/08_evaluation_results_report.md` because `/docs` is gitignored but tracked files still apply.
- **Test / verification:**
  - `wc -l artifacts/evaluation/metrics_summary.csv` returns 22.
  - `hybrid_content,10,...,latency_mean_ms` row reports a value below 2000 (post-optimization is around 1.3 s in the 25-user run; expect a similar or slightly higher mean at 100 users).
  - `evaluated_user_count` column for each `sbert_faiss_content,K` row is at least 60.
  - `docs/08_evaluation_results_report.md` does not contain the literal "Only 15 users had positive holdout items" line after the update.
- **Expected outcome:** A clean, 100-user reference table that future LightFM / ALS rows can be slotted into without re-running every model. Decision criterion: the report and the CSV agree row-by-row, and the new evaluated_user_count is large enough that the content-only baselines either produce hits or are unambiguously zero across a meaningful sample.
- **DONE (commit `5c6e5af`):** Re-ran the existing 7 top-N baselines plus SVD rating prediction at `--max-users 100`, then refreshed `docs/08_evaluation_results_report.md` with the wider-run numbers. The run is usable as the LightFM / ALS reference, but the positive-holdout count landed at 55 rather than the hypothesized 60.
  - Metric / result (K=10):
    | Model | Precision@10 | Recall@10 | NDCG@10 | Latency mean |
    |---|---:|---:|---:|---:|
    | popularity | 0.0091 | 0.0909 | 0.0322 | 87.9 ms |
    | hybrid_content | 0.0073 | 0.0727 | 0.0382 | 1,466.0 ms |
    | svd_topk | 0.0000 | 0.0000 | 0.0000 | 189.6 ms |
    | sbert_faiss_content | 0.0036 | 0.0364 | 0.0193 | 40.7 ms |
  - Run id: `artifacts/evaluation/metrics_summary_2026-05-20T07-20-53Z.{csv,json}`; `run_config.json` confirms `max_users = 100` and `include_sbert_faiss = true`.
  - Verification: `metrics_summary.csv` has 23 lines: 1 header + 21 top-N rows + 1 SVD rating-prediction row. This exposes an off-by-one in the authored plan text, which said 22 lines including the header.
  - Gate result: `hybrid_content` mean latency at K=10 is 1,465.99 ms, under the 2,000 ms gate. `sbert_faiss_content` evaluated-user count is 55 at K=5/10/20, so the original `>= 60` expectation did not hold.
  - Decision: proceed to Item 2 with 55 positive-holdout users documented as the actual 100-user reference slice.

## 2) Add a LightFM (WARP loss) baseline

- **Goal:** Introduce LightFM with WARP ranking loss as a measurable implicit-feedback alternative to Surprise SVD. The model is trained once offline, persisted under `artifacts/models/lightfm/`, and loaded by the evaluation runner via the same prebuilt-artifact pattern that `experimental.sbert_faiss` already uses.
- **Files:**
  - `requirements.txt`: append `lightfm` on its own line. If the macOS arm64 install fails with a Cython build error, retry with `pip install lightfm --no-build-isolation`; if that still fails, document in README under a new "Apple Silicon notes" subsection that the LightFM evaluation flag is x86_64 / Linux-friendly. Do not silently drop the dependency.
  - `src/experimental/lightfm_recommender.py` (new, target ~150 lines). Mirror the dataclass + load + recommend pattern from `src/experimental/sbert_faiss.py`. Public functions:
    - `build_interaction_matrix(ratings, positive_threshold=4.0)` returns `(scipy.sparse.csr_matrix, user_index, item_index)` where `user_index` and `item_index` map MovieLens `userId` / `movieId` to dense row / column positions. Filter `ratings` to rows where `rating >= positive_threshold` before construction.
    - `train_lightfm_model(interactions, no_components=64, loss="warp", epochs=20, num_threads=4)` returns a fitted `lightfm.LightFM`. Use `model.fit(interactions, epochs=epochs, num_threads=num_threads)`.
    - `save_lightfm_artifacts(model, user_index, item_index, output_dir, metadata=None)` writes:
      - `output_dir/lightfm_model.pkl` (joblib or `pickle.dump`),
      - `output_dir/user_index.csv` (`userId,position`),
      - `output_dir/item_index.csv` (`movieId,position`),
      - `output_dir/metadata.json` with `no_components`, `loss`, `epochs`, `row_count`, `created_at`.
    - `load_lightfm_artifacts(input_dir)` returns dataclass `LightfmArtifacts(model, user_index, item_index, metadata)`. Raise `FileNotFoundError` on any missing file (mirrors `load_sbert_faiss_index`).
    - `lightfm_recommendations_for_user(user_id, artifacts, movies_for_output, watched_movie_ids=None, top_n=10)`:
      - Map `user_id` to its row position via `artifacts.user_index`. If the user is not in the matrix, return `pd.DataFrame(columns=output_columns(movies_for_output) + ["similarity_score"])`.
      - Use `artifacts.model.predict(user_position, item_position_array)` over every column position in `item_index`.
      - Filter out positions whose `movieId` is in `watched_movie_ids` (use `filter_watched_movies` after merging on movieId).
      - Sort by predicted score descending, take `top_n`, attach `similarity_score = score`, run through `ensure_output_columns(...)`.
    - Helper `empty_lightfm_recommendations(movies)` mirroring `empty_sbert_faiss_recommendations`.
  - `scripts/train_lightfm_model.py` (new, ~40 lines). Argparse wrapper, mirror `scripts/build_sbert_faiss_index.py`:
    - Args: `--output-dir` (default from `EVALUATION_DEFAULTS["lightfm"]["artifacts_dir"]`), `--no-components`, `--loss`, `--epochs`, `--positive-threshold`, `--num-threads`.
    - Loads `cleaned_data/ratings_clean.csv` via `data_access.load_ratings()` (existing helper), builds the matrix, trains, saves artifacts. Print the metadata JSON at the end.
  - `scripts/evaluate_baselines.py`: add `--include-lightfm` (`store_true`) and `--lightfm-artifacts-dir` (default from config); thread both into `run_evaluation()`. Mirror the existing `--include-sbert-faiss` wiring exactly.
  - `src/evaluation_runner.py`: add `from experimental.lightfm_recommender import (load_lightfm_artifacts, lightfm_recommendations_for_user)`. Add `make_lightfm_per_user(artifacts, movies, train, max_k)` closure that returns a per-user recommend function aligned with the `evaluate_baseline(...)` signature (see `make_sbert_faiss_per_user` at lines 284-326 for the template). Add `include_lightfm`, `lightfm_artifacts_dir` parameters to `run_evaluation()`, plus the artifact-load `try/except (FileNotFoundError, ImportError, ValueError)` mirror of the SBERT path at lines 548-556 and 725-735. Record the model under the name `lightfm_warp`.
  - `config/config.yaml`: add a new block:
    ```yaml
    evaluation:
      lightfm:
        artifacts_dir: artifacts/models/lightfm
        no_components: 64
        loss: warp
        epochs: 20
        num_threads: 4
        positive_threshold: 4.0
    ```
    `src/config.py` already reads `evaluation.*` blocks via `EVALUATION_DEFAULTS`, so no code change there.
  - `tests/test_lightfm_recommender.py` (new). Use a `FakeLightfmModel` whose `predict(user_idx, item_idxs)` returns deterministic scores. Cover:
    - `build_interaction_matrix` filters out below-threshold ratings and produces aligned `user_index` / `item_index`.
    - `lightfm_recommendations_for_user` excludes watched movieIds and returns `BASE_OUTPUT_COLUMNS + ["similarity_score"]` columns.
    - Unknown user returns an empty frame.
  - `README.md`: under "Implemented for offline evaluation only", add a line for `LightFM WARP baseline (--include-lightfm) using prebuilt local artifacts`. Add a "Build the LightFM artifact" snippet immediately after the SBERT+FAISS snippet, structured the same way (`python scripts/train_lightfm_model.py --output-dir artifacts/models/lightfm`, then the evaluation flag).
- **Reuse:** `src/recommenders/common.py:filter_watched_movies, output_columns, ensure_output_columns, normalize_movie_ids`; the entire `experimental.sbert_faiss` dataclass + load pattern; the `evaluate_baseline` orchestration in `evaluation_runner.py`.
- **Steps:**
  - `pip install lightfm` (or the `--no-build-isolation` fallback). Verify with `.venv/bin/python -c "import lightfm; print(lightfm.__version__)"`.
  - `.venv/bin/python scripts/train_lightfm_model.py --output-dir artifacts/models/lightfm` — expect 5-15 min on CPU for the full 33M-rating matrix. The output should print metadata with `row_count` matching the positive-threshold filtered rating count and `no_components = 64`.
  - `.venv/bin/python -m unittest tests.test_lightfm_recommender` — must pass.
  - 5-user smoke:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 5 --k 5 \
      --include-lightfm \
      --lightfm-artifacts-dir artifacts/models/lightfm
    ```
    The JSON output must contain `top_n.lightfm_warp` with `evaluated_user_count >= 1` and no `lightfm_error` field. (If you choose a different `record(...)` name, keep it consistent with whatever you wire through `evaluate_baseline("...", ...)` and surface it in the report.)
  - Commit message: `feat(eval): add LightFM WARP baseline with prebuilt artifacts`. Use `git add -f docs/experiments/2026-05-20_classical-cf-and-eval-expansion.md` to include the DONE marker because `/docs` is gitignored.
- **Test / verification:**
  - `tests.test_lightfm_recommender` exit code 0.
  - `metadata.json:no_components = 64`, `metadata.json:loss = "warp"`.
  - 5-user smoke produces a populated `lightfm_warp` block.
  - Streamlit unit tests (`.venv/bin/python -m unittest discover -s tests`) still pass 100%.
- **Expected outcome:** LightFM joins the offline evaluation as a measurable, fast, implicit-feedback alternative to Surprise SVD. Decision criterion: a populated `lightfm_warp` row in the smoke output, with mean latency below 500 ms.
- **DONE (commit `65de01d`):** Added the LightFM WARP offline baseline with prebuilt artifact loading, `--include-lightfm` CLI wiring, config defaults, training script, README setup notes, and unit coverage. The runner now records `lightfm_warp` when artifacts load and returns `lightfm_error` instead of crashing when LightFM or artifact files are unavailable.
  - Install note: direct `pip install lightfm` and `pip install lightfm --no-build-isolation` both failed on this Python 3.11/macOS arm64 host with `AttributeError: 'dict' object has no attribute '__LIGHTFM_SETUP__'`. The package was installed by downloading `lightfm-1.17.tar.gz` to `/private/tmp`, patching setup to use `import builtins; builtins.__LIGHTFM_SETUP__ = True`, installing `wheel`, then running local source install. The built extension warns that OpenMP is unavailable, so training used one thread.
  - Train wall time: 499.20 seconds for the default artifact at `artifacts/models/lightfm/`.
  - Metadata: `row_count = 16863053`, `user_count = 305098`, `item_count = 40441`, `no_components = 64`, `loss = warp`, `epochs = 20`.
  - 5-user smoke (`--include-lightfm --lightfm-artifacts-dir artifacts/models/lightfm`): `precision_at_5 = 0.0800`, `ndcg_at_5 = 0.1635`, `latency_mean_ms = 43.7`; no `lightfm_error` field.
  - Verification: `tests.test_lightfm_recommender` 3/3 OK, `.venv/bin/python -m unittest discover -s tests` 53/53 OK, missing-artifact smoke returns a `lightfm_error` with the missing file list.
  - Decision: proceed to Item 3.

## 3) Add an Implicit ALS baseline

- **Goal:** Add Hu-Koren-style implicit ALS as a second classical CF alternative. Mirrors the LightFM file shape so both modules are easy to read side by side.
- **Files:**
  - `requirements.txt`: append `implicit` on its own line. Confirm wheel availability on macOS arm64; `implicit` ships official arm64 wheels for Python 3.11.
  - `src/experimental/als_recommender.py` (new, target ~140 lines). Public functions:
    - `build_confidence_matrix(ratings, positive_threshold=4.0, alpha=40.0)` returns `(scipy.sparse.csr_matrix, user_index, item_index)`. Apply `confidence = 1.0 + alpha * (rating - positive_threshold).clip(lower=0)` to rows where `rating >= positive_threshold`, and drop rows below threshold. `implicit` expects user-item interactions with values as confidence.
    - `train_als_model(confidence_matrix, factors=64, regularization=0.01, iterations=20, use_gpu=False)` returns `implicit.als.AlternatingLeastSquares(...).fit(confidence_matrix)`.
    - `save_als_artifacts(model, user_index, item_index, output_dir, metadata=None)` writes `als_model.npz` (use `model.save`), `user_index.csv`, `item_index.csv`, `metadata.json`.
    - `load_als_artifacts(input_dir)` returns dataclass `AlsArtifacts(model, user_index, item_index, metadata, user_items)` where `user_items` is the saved confidence matrix loaded back via `scipy.sparse.load_npz(...)`. Persist that matrix alongside the model (e.g., `user_items.npz`) so that `model.recommend(...)` has the per-user history at inference time.
    - `als_recommendations_for_user(user_id, artifacts, movies_for_output, watched_movie_ids=None, top_n=10)`:
      - Map `user_id` via `artifacts.user_index`. Unknown user → empty frame.
      - Call `model.recommend(user_position, artifacts.user_items[user_position], N=top_n + len(watched_movie_ids), filter_already_liked_items=True)`. The implicit API returns `(item_positions, scores)` as numpy arrays.
      - Translate positions back to `movieId` via the inverse of `artifacts.item_index`.
      - Drop any movieId in `watched_movie_ids` (implicit already filters liked items, but we still respect the explicit exclusion set passed from the runner).
      - Attach `similarity_score = score`, slice top_n, run through `ensure_output_columns(...)`.
    - Helper `empty_als_recommendations(movies)` mirroring the SBERT / LightFM versions.
  - `scripts/train_als_model.py` (new, ~40 lines). Argparse wrapper:
    - Args: `--output-dir`, `--factors`, `--regularization`, `--iterations`, `--alpha`, `--positive-threshold`.
    - Loads `cleaned_data/ratings_clean.csv`, builds the confidence matrix, trains, saves the model AND the `user_items.npz` companion matrix.
  - `scripts/evaluate_baselines.py`: add `--include-als` and `--als-artifacts-dir`; thread into `run_evaluation()`. Mirror the LightFM wiring.
  - `src/evaluation_runner.py`: add the load + closure + record path under the name `als_implicit`. Reuse the SBERT graceful-failure pattern for missing or unreadable artifacts.
  - `config/config.yaml`: add
    ```yaml
    evaluation:
      als:
        artifacts_dir: artifacts/models/als
        factors: 64
        regularization: 0.01
        iterations: 20
        alpha: 40.0
        positive_threshold: 4.0
    ```
  - `tests/test_als_recommender.py` (new). Use a `FakeAlsModel` whose `recommend(user_pos, user_items, N, filter_already_liked_items)` returns deterministic position lists. Cover:
    - `build_confidence_matrix` produces `confidence > 1` only for rows above the positive threshold.
    - `als_recommendations_for_user` excludes watched movieIds and returns the expected columns.
    - Unknown user returns an empty frame.
  - `README.md`: add a line under "Implemented for offline evaluation only" for `Implicit ALS baseline (--include-als)`. Add a "Build the ALS artifact" snippet next to LightFM's.
- **Reuse:** Same shared helpers from `src/recommenders/common.py`; reuse `build_interaction_matrix` conceptually but keep the confidence formula separate (do not over-share — the two recommenders' matrices have different value semantics, and a future LightFM hyperparameter sweep may diverge from ALS confidence).
- **Steps:**
  - `pip install implicit`. Verify with `.venv/bin/python -c "import implicit; print(implicit.__version__)"`.
  - `.venv/bin/python scripts/train_als_model.py --output-dir artifacts/models/als` — expect 3-10 min on CPU.
  - `.venv/bin/python -m unittest tests.test_als_recommender` — must pass.
  - 5-user smoke:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 5 --k 5 \
      --include-als \
      --als-artifacts-dir artifacts/models/als
    ```
    The JSON must contain `top_n.als_implicit` with `evaluated_user_count >= 1` and no `als_error` field.
  - Commit message: `feat(eval): add Implicit ALS baseline with prebuilt artifacts`. Use `git add -f` for the experiment plan file.
- **Test / verification:**
  - `tests.test_als_recommender` exit code 0.
  - `metadata.json:factors = 64`, `metadata.json:iterations = 20`.
  - 5-user smoke produces a populated `als_implicit` block with mean latency under 300 ms.
  - `.venv/bin/python -m unittest discover -s tests` still 100% green.
- **Expected outcome:** ALS joins the comparison table next to LightFM and SVD. Decision criterion: smoke produces `als_implicit` rows; no Streamlit regression.
- **DONE (commit `c956ea5`):** Added the Implicit ALS offline baseline with prebuilt artifact loading, `--include-als` CLI wiring, config defaults, training script, README setup notes, and unit coverage. The runner now records `als_implicit` when artifacts load and returns `als_error` instead of crashing when ALS artifact files are unavailable.
  - Install note: `pip install implicit` installed `implicit 0.7.3` from the macOS arm64 wheel.
  - Train wall time: 98.19 seconds for the default artifact at `artifacts/models/als/`.
  - Metadata: `row_count = 16863053`, `user_count = 305098`, `item_count = 40441`, `factors = 64`, `regularization = 0.01`, `iterations = 20`, `alpha = 40.0`, `use_gpu = false`.
  - 5-user smoke (`--include-als --als-artifacts-dir artifacts/models/als`): `precision_at_5 = 0.0000`, `ndcg_at_5 = 0.0000`, `latency_mean_ms = 7.1`; no `als_error` field.
  - Verification: `tests.test_als_recommender` 3/3 OK, `.venv/bin/python -m unittest discover -s tests` 56/56 OK, missing-artifact smoke returns an `als_error` with the missing file list.
  - Decision: proceed to Item 4.

## 4) Combined 100-user × 9-model evaluation and report refresh

- **Goal:** Re-run the canonical evaluation with every baseline including LightFM and ALS, then rewrite the report so the audit table reflects all 9 models on the same 100-user slice.
- **Files:**
  - `artifacts/evaluation/metrics_summary.{csv,json}` (regenerated, gitignored).
  - `artifacts/evaluation/run_config.json` (regenerated, gitignored).
  - `docs/08_evaluation_results_report.md` (rewritten to include LightFM + ALS).
- **Steps:**
  - Run the full command:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 100 --k 5,10,20 \
      --include-random --include-tfidf --include-content --include-semantic \
      --include-svd --include-svd-topk \
      --include-sbert-faiss --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
      --include-lightfm --lightfm-artifacts-dir artifacts/models/lightfm \
      --include-als --als-artifacts-dir artifacts/models/als \
      --output-dir artifacts/evaluation
    ```
  - `wc -l artifacts/evaluation/metrics_summary.csv` must return 28 lines (1 header + 9 × 3 rows + 1 SVD rating prediction).
  - Update `docs/08_evaluation_results_report.md`:
    - "Run Summary" — refresh the command block and configuration list with LightFM + ALS lines.
    - K=10 and K=20 tables — extend to 9 rows, sorted by NDCG (or whatever ordering the existing report uses for that K).
    - "Latency Findings" — extend to 9 rows; LightFM and ALS should slot into the fast tier near `sbert_faiss_content`.
    - "Conclusions" — add a short paragraph (2-3 sentences) framing the classical-CF triangle: Surprise SVD vs LightFM (WARP) vs Implicit ALS. State whether either new model beats `svd_topk` on NDCG@10 and on mean latency, without overclaiming on a single 100-user slice.
    - "Caveats" — keep the small-slice caveat but drop the "only 15 users had positive holdout" specifics; the slice is now larger and item 1's text already covered the new number.
  - Commit message: `docs(eval): refresh report with LightFM and Implicit ALS at 100 users`. Use `git add -f` for both the report and this plan file.
- **Test / verification:**
  - `metrics_summary.csv` row count is exactly 28.
  - Every model name on the table also appears in `metrics_summary.csv` — no contradictions.
  - Tests still pass: `.venv/bin/python -m unittest discover -s tests`.
  - Streamlit smoke (optional but cheap): `.venv/bin/python -m streamlit run src/app.py --server.headless true --server.port 8765 --browser.gatherUsageStats false` — confirm the Content-Based page still renders with both TF-IDF and SBERT modes (no Streamlit UI surface changed in this plan; this is a regression check only).
- **Expected outcome:** A single canonical report that ranks 9 models on the same 100-user slice. Decision criterion: report and CSV agree row-by-row, and the classical-CF conclusion paragraph is grounded in the new numbers.
- **DONE (commit `3f0ab0e`):** Re-ran the canonical 100-user evaluation with all 9 top-N models plus SVD rating prediction, then refreshed `docs/08_evaluation_results_report.md` with LightFM and ALS included in the command, run configuration, K=10/K=20 tables, latency findings, conclusions, and caveats.
  - Metric / result (K=10):
    | Model | Precision@10 | Recall@10 | NDCG@10 | Latency mean |
    |---|---:|---:|---:|---:|
    | lightfm_warp | 0.0255 | 0.2545 | 0.1427 | 43.6 ms |
    | hybrid_content | 0.0073 | 0.0727 | 0.0382 | 1,464.4 ms |
    | popularity | 0.0091 | 0.0909 | 0.0322 | 95.5 ms |
    | sbert_faiss_content | 0.0036 | 0.0364 | 0.0193 | 39.0 ms |
    | semantic_content | 0.0018 | 0.0182 | 0.0182 | 86.5 ms |
    | tfidf_content | 0.0036 | 0.0364 | 0.0139 | 46.3 ms |
    | als_implicit | 0.0000 | 0.0000 | 0.0000 | 7.8 ms |
    | random | 0.0000 | 0.0000 | 0.0000 | 12.5 ms |
    | svd_topk | 0.0000 | 0.0000 | 0.0000 | 189.4 ms |
  - Run id: `artifacts/evaluation/metrics_summary_2026-05-20T09-47-43Z.{csv,json}`; `run_config.json` confirms `include_lightfm = true`, `include_als = true`, `include_sbert_faiss = true`, and `max_users = 100`.
  - Verification: `metrics_summary.csv` has 29 lines: 1 header + 27 top-N rows + 1 SVD rating-prediction row. This exposes another off-by-one in the authored plan text, which said 28 lines while also describing 1 + 9 x 3 + 1 rows.
  - Error check: no `lightfm_error`, `als_error`, or `sbert_faiss_error` fields were present in the final JSON.
  - Tests: `.venv/bin/python -m unittest discover -s tests` 56/56 OK.
  - Decision: shipped. LightFM WARP is the strongest model on this slice; ALS is the fastest but produced no relevance hits.

## Deferred / Future (out of this plan)

- **LightGCN (graph CF):** `docs/01_technical_comparison_report.md` §8 Phase 2 calls for graph-based collaborative filtering via PyTorch Geometric or DGL. CPU is workable but slow at this catalog size, and the dependency footprint is heavy. Handle in a separate `docs/experiments/{date}_lightgcn-baseline.md` once classical CF results have stabilised.
- **SASRec / BERT4Rec (sequential):** Modelling watch order requires a full training pipeline (PyTorch, GPU strongly preferred) and a different per-user evaluation flow (sequence holdout rather than rating-positive holdout). Park separately.
- **Hyperparameter sweeps:** LightFM has `no_components × loss × epochs` to explore; ALS has `factors × regularization × alpha`. This plan ships a single configured baseline for each; a sweep is a follow-on once the single-point comparison is established.

## Critical Files (Reference)

- `src/evaluation_runner.py:476-735` — `run_evaluation()` orchestration; LightFM and ALS wiring should mirror the SBERT+FAISS path at lines 548-556 (artifact load) and 725-735 (closure registration).
- `src/experimental/sbert_faiss.py` — dataclass + load + recommend pattern to copy.
- `src/recommenders/common.py:filter_watched_movies, output_columns, ensure_output_columns, normalize_movie_ids` — output contract.
- `scripts/build_sbert_faiss_index.py` — argparse wrapper template for `train_lightfm_model.py` and `train_als_model.py`.
- `config/config.yaml:evaluation.sbert_faiss` — format template for the new `evaluation.lightfm` and `evaluation.als` blocks.
- `tests/test_sbert_faiss.py` — fixture pattern for the new LightFM and ALS unit tests.
- `docs/06_project_inventory_and_roadmap_en.md` Priority 3 — motivation source.
- `docs/01_technical_comparison_report.md` §8 Phase 1 — strategy source.
- `docs/08_evaluation_results_report.md` — refreshed in items 1 and 4.

## End-to-End Verification Sequence

1. Item 1 complete: `metrics_summary.csv` has 22 rows on the 100-user slice; `docs/08_evaluation_results_report.md` reflects the new run.
2. Item 2 complete: `tests.test_lightfm_recommender` passes; 5-user smoke produces `lightfm_warp`; the plan file's item-2 DONE marker carries a real commit hash and a metric triple.
3. Item 3 complete: `tests.test_als_recommender` passes; 5-user smoke produces `als_implicit`; the plan file's item-3 DONE marker carries a real commit hash and a metric triple.
4. Item 4 complete: 100-user × 9-model `metrics_summary.csv` has 28 rows; the report's K=10 / K=20 / latency tables all show 9 models; the conclusion paragraph addresses LightFM vs ALS vs SVD on this slice.
5. After every commit, record the real commit hash in this plan file; CLAUDE.md §7 forbids leaving temporary hash placeholders in place.

## Codex Execution Notes

- This file is the contract; CLAUDE.md and AGENTS.md in the repo root are the project-wide rules.
- The `/docs` path is gitignored, but tracked files (`docs/08_evaluation_results_report.md`, every file under `docs/experiments/`) still need `git add -f` to land in commits.
- Items are sequential: item 1 establishes the 100-user reference numbers, items 2 and 3 add the new models against that reference, item 4 produces the single integrated report. Skipping item 1 would re-mix sample-size effects into the LightFM / ALS comparison.
- Each commit should target a single item; do not bundle item 2 and item 3 into one commit even though they share a pattern.
- Apple Silicon: if `lightfm` install fails with a Cython / `numpy.distutils` error, retry with `pip install lightfm --no-build-isolation`. If that also fails, document the limitation in README ("LightFM baseline currently requires Linux or x86_64") and leave the `--include-lightfm` flag in place so the runner reports a graceful `lightfm_error` field on hosts where the import fails. Do not silently remove the flag or the wiring.
- `tests/test_movie_rec.py` already runs against the in-memory caches; verify it still passes after each commit. The Streamlit UI surface is not touched in this plan.
