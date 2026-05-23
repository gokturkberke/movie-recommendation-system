1. Business Requirements
This repository is a brownfield Streamlit movie recommendation application.

The primary product behavior is: - Load MovieLens data from local CSV files and cleaned artifacts. - Provide movie recommendations through the Streamlit UI. - Support content-based recommendations from a movie title or selected movieId. - Support collaborative filtering recommendations from a saved Surprise SVD model. - Support mood-based recommendations using the configured mood-to-genre map. - Support random movie picking with optional genre filters. - Support watch history stored by movieId and recommend unseen movies from watched-title content signals. - Fetch TMDB posters and overviews when TMDB_API_KEY is configured. - Keep the app usable when TMDB_API_KEY is missing by disabling poster and overview rendering only.

Recommendation inventory (production behavior): - Content-based: TF-IDF over title, genre, and tag text with cosine similarity. - Hybrid content ranking: similarity, Bayesian rating, popularity, and light diversity. - Collaborative filtering: Surprise SVD loaded from cleaned_data/svd_trained_model.pkl. - Mood-based: configured mood-to-genre map from config/config.yaml through src/config.py. - Watch history: movieId-based seed matching and watched movie exclusion. - Metadata enrichment: TMDB REST API for posters and overviews only.

Scope control is strict: - Do not add new Streamlit pages, menu items, or workflows unless explicitly requested. - Do not add new recommender families unless explicitly requested. - Do not add new data providers or external APIs unless explicitly requested. - Do not expand data artifacts, model artifact shapes, output columns, or user-facing fields unless explicitly requested. - Do not invent features because they seem useful. Brownfield rules apply: preserve existing intent and interfaces.

2. Technical Details
Package and runtime truth: - Package manager: pip - Primary runtime manifest: requirements.txt. - Primary app command: .venv/bin/python -m streamlit run src/app.py. - Primary test command: .venv/bin/python -m unittest discover -s tests. - Legacy test command: .venv/bin/python -m unittest src/test_movie_rec.py. - Treat requirements.txt and README.md as the most reliable runtime truth for execution decisions.

Architecture and ownership: - src/app.py is the Streamlit entrypoint, UI router, cache owner, and session state owner. - src/recommenders.py contains the core recommendation logic, matching, ranking, watch history aggregation, and output column control. - src/data_access.py loads movies, tags, ratings, links, SVD model artifacts, and merges TMDB ids. - src/tmdb_client.py handles TMDB id lookup and TMDB movie detail requests. - config/config.yaml is the central runtime parameter source; src/config.py is the compatibility facade for project paths, TMDB key lookup, mood mappings, menu items, demo profiles, and recommender constants. - scripts/preprocess_dataset.py converts raw MovieLens CSV files into cleaned_data artifacts. - scripts/train_save_model.py trains and saves the Surprise SVD model artifact. - scripts/evaluate_baselines.py runs offline baseline evaluation. - src/utils_data.py is a compatibility wrapper layer around current data access and recommender functions. - scripts/analyze_dataset.py is exploratory analysis space; it must not silently redefine production contracts. - tests/test_movie_rec.py contains unit and smoke coverage; src/test_movie_rec.py is a legacy wrapper.

Runtime flow (end-to-end): 1. Streamlit starts at src/app.py -> initialize_session_state() normalizes watched_movie_ids. 2. load_context() loads cleaned movies/tags, raw links, builds the TF-IDF matrix, merges tmdbId values, and resolves TMDB_API_KEY. 3. Content-based flow: title input -> suggest_movie_titles() disambiguation -> recommend_similar_movies() or recommend_similar_movies_by_id() -> cosine candidates -> watched exclusion -> hybrid rerank -> display. 4. Collaborative flow: load_surprise_model() and ratings_clean.csv -> demo profile or manual userId -> raw_svd_predictions() -> optional persona genre filtering -> watched exclusion -> display. 5. Mood flow: selected mood -> MOOD_GENRE_MAP genres -> deterministic sample with watched exclusion -> display. 6. Random flow: optional genre filter -> random sample from loaded movies -> display. 7. Watch history flow: add/remove movieIds in st.session_state -> recommend_based_on_watch_history_content() -> aggregate seed candidates -> watched exclusion -> hybrid rerank -> display. 8. Movie rendering uses tmdbId from merged movie rows or links.csv, then fetches TMDB metadata only when an API key is configured.

Repo drift and guardrails: - requirements.txt is canonical for runtime dependencies. - Preserve movie identity by movieId; do not fall back to title-based identity for watch history. - Preserve display titles with release years when title_original exists. - Preserve output columns such as movieId, title, genres, tmdbId, predicted_score, and HYBRID_SCORE_COLUMNS unless explicitly requested. - Preserve genre_* columns generated by preprocessing; demo profiles depend on these names. - Do not hardcode TMDB keys or local machine paths. - Keep Streamlit endpoints and UI state thin; keep recommendation behavior in src/recommenders.py and data loading in src/data_access.py.

3. Data Sources & Model Artifacts
* MovieLens is the primary dataset source.
* Raw data is expected under data/: movies.csv, ratings.csv, tags.csv, and links.csv.
* Cleaned data is expected under cleaned_data/: movies_clean.csv, ratings_clean.csv, and tags_clean.csv.
* The collaborative filtering model artifact is cleaned_data/svd_trained_model.pkl.
* Runtime data loading flow: src/app.py -> src/data_access.py -> data/ and cleaned_data/.
* Preprocessing flow: scripts/preprocess_dataset.py -> cleaned_data CSV artifacts.
* Training flow: scripts/train_save_model.py -> cleaned_data/svd_trained_model.pkl.
* TMDB metadata is optional and env-driven through TMDB_API_KEY or .streamlit/secrets.toml.
* If local cleaned data or model artifacts already exist, use them first; avoid unnecessary regeneration.
* Do not commit secrets, API keys, local absolute paths, or private dataset variants.
* Do not add cloud storage, remote artifact fetch, or automatic dataset download behavior unless explicitly requested.
* Large or generated MovieLens artifacts may be local-only; verify presence before assuming tests or training can run end-to-end.

4. Strategy
Every agent working in this repository must follow this loop exactly:
Plan -> Wait for Approval -> Code -> Test -> Fix
Plan requirements (minimum for approval): - Which files will change and why. - Current behavior vs. target behavior. - Validation path: how the change will be verified.
Execution rules: - Always read the relevant service, schema, and config files before proposing changes. - Preserve API contracts and field names unless explicitly requested. - Prefer small, local, architecture-preserving edits over broad rewrites. - Large or ambiguous changes require an explicit plan and approval first. - Every behavioral change must include at least one concrete validation path: - endpoint-level smoke test (/predict/ or /predict-url/), or - direct service-level check (ModelService / LLMService). - For fallback-related work, verify both branches with evidence: - primary success path - primary None/error path and fallback outcome.
5. Debugging Rules
When facing a bug, DO NOT GUESS. 1. Reproduce the problem. 2. Prove you reproduced it. 3. Find the root cause. 4. Fix it. 5. Prove you fixed it.
Repository-specific debugging rules: - Reproduce issues with tests, scripts, logs, stack traces, or artifact inspection. - Prove reproduction with concrete evidence before changing code. - Verify whether the failure belongs to: image loading/preprocessing, ONNX inference, LLM call (timeout/None/provider), config/env wiring, GCS artifact fetch, or API delivery. - Do not patch around failures by swallowing exceptions or returning silent defaults - prove the root cause first. - Prefer focused tests or targeted reproductions over speculative edits. - Keep fixes minimal and local to the proven fault line. - After a fix, prove the outcome with the same reproduction path or a tighter automated test.
6. Coding Standards
Non-negotiable rules: - No emojis ever in code, logs, commits, or generated documentation. - Never use Turkish characters in variables, functions, or comments. - Avoid over-defensive programming. Do not add unnecessary try/except blocks, wrappers, or fallback branches without evidence. - Keep communication and README-style documentation short, direct, and human-readable. - Do not generate AI slop.
Repo-safe engineering standards: - Preserve the existing module boundaries between src/, scripts/, tests/, and config/. - Do not silently rename model files, artifact files, registry names, endpoint paths, schema fields, or config keys. - Do not introduce broad refactors during feature work or bug work unless explicitly requested. - Prefer explicit, testable logic over abstraction-heavy wrappers and fallback-heavy control flow. - Keep Streamlit UI thin; keep recommendation behavior in src/recommenders.py and data loading in src/data_access.py. - If you discover drift or contradictions, document them clearly in the task output instead of masking them.

7. Experiment Planning And Execution Log
+ This section defines how every experiment / improvement plan in this project is documented and how its execution is marked. Goal: every plan and its outcome live in one place (`docs/experiments/`) in a commit-traceable way, so future references to an experiment are `grep`-able.
+ Where the plan file goes, and what it is named:
+ All new plans are saved under `docs/experiments/`.
+ Filename format: `{YYYY-MM-DD}_{plan-name}.md` (kebab-case plan name). Example: `2026-05-14_recency-weighting-sweep.md`, `2026-05-20_pri-transform-revisit.md`.
+ The date is the day the plan was **authored**, not the day it was executed. The filename stays fixed even if the plan spans multiple experiments over several days.
+ Do not touch code before the plan file exists. In the `Inspect -> Plan -> Code -> Test -> Fix` loop, the plan file is the artifact produced by the `Plan` step.
+ Required structure of the plan file:
+ A plan file is written **item by item**, **in logical order**, as a **narrative**: motivation -> hypothesis -> preconditions -> items -> expected outputs -> decision criteria.
+ Header block at the top of every plan file:
+   - **Date:** {YYYY-MM-DD}
+   - **Topic:** short title
+   - **Motivation:** which report section (`§X`) or which metric anomaly triggered this plan; link the baseline run id(s) so comparisons stay reproducible.
+   - **Hypothesis:** the proposition under test, expressed as a measurable claim (e.g. "`asinh` transform improves PRI Q90 coverage by at least 1pp").
+   - **Preconditions:** code / config / cache state that must already be in place before the plan starts.
+ Then a numbered list of items (`## 1) ...`, `## 2) ...`). Template for each item:
+   - **Goal:** what will change (code / config / sweep parameter).
+   - **Files:** paths to touch (with current line numbers or function names where useful).
+   - **Steps:** sub-bullets, one logical operation per bullet (e.g. "add `data.train_outlier_sigma.BidderTotalBids: 3.0` to config", "run the sweep runner with a single trial and capture the log").
+   - **Test / verification:** which unit test gets added or updated; which full-training output is compared against which metric table.
+   - **Expected outcome:** decision criterion (e.g. how big a Q50 MAE delta counts as meaningful, where coverage should land relative to the target band).
+   - **DONE / DROPPED:** empty at authoring time; filled in during execution (see below).
+ Items are ordered by the **narrative**: dependencies before dependents, independents in parallel. The flow "test the hypothesis with a single trial -> if positive, expand to a sweep -> production decision" must always be visible — random ordering is not acceptable.
+ Execution / marking contract:
+ Each time an item is executed, write the outcome **into the same file**, **immediately under that item**. Template:
+   ```
+   **DONE (commit `<hash>`):** {one or two sentences: what changed, which behavior was gained, any remaining side-effect.}
+   - Metric / result: {small baseline-vs-experiment table if relevant}
+   - Run id: {always include — found under `models/training/{target}/{run_id}/`}
+   - Sweep JSON: {if applicable, path under `docs/experiments/...sweep_...json`}
+   - Decision: {shipped to production, shelved, or fed into another experiment}
+   ```
+ The `<hash>` placeholder must never be left in place; do not write DONE before the commit lands. If the execution required multiple commits, list all of them in order, comma-separated.
+ For abandonment, the marker becomes `DROPPED ({date}):` followed by a one-paragraph reason. No item is left open; every item is closed as either DONE or DROPPED.
+ Outside the plan file, the audit report (`docs/experiments/10_05_2026_fully_report.md`) keeps cross-references across multiple plans. If a new plan resolves a specific audit item (e.g. §3.5 sample weighting), the plan file carries a `Corresponding audit item: §3.5` line, and the audit item is annotated with `see: docs/experiments/2026-05-14_recency-weighting-sweep.md`.
+ Any change landing in production config (`config/quantile_model_config.yaml`) as the outcome of a plan / experiment must carry an inline comment that points back to the plan file or report `§` (e.g. `# experiment 4 (report §14): DMBC sigma=4.0`). This is how a config reader finds the rationale behind a value.
+ A plan file does NOT contain:
+ Speculative "might also try" lists above and beyond the concrete intent. A plan is the contract for **work happening now**, not a wishlist.
+ Re-summaries of already-closed plans. A cross-reference link is enough.
+ Pasted code blocks. A plan file is prose + bullets; code changes live in the commit.
+ Pre-flight (before creating a new plan file):
+ `grep` under `docs/experiments/` for a half-open plan on the same topic. If one exists, append a new item to that plan file — do not create a new one.
+ Record the current benchmark / baseline run id before the plan starts (write it in the **Motivation** section). This is what later makes statements like "experiment X is +0.4 Q50 MAE vs baseline" reproducible.
