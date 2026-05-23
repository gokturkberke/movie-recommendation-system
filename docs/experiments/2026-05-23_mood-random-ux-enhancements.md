- **Date:** 2026-05-23
- **Topic:** Mood-Based and Random recommendation UX hardening
- **Motivation:** Inspection of the live Streamlit flows found that the Mood-Based page samples titles purely from a genre map without any quality signal, so movies with very low Bayesian rating can surface alongside well-rated picks. The Random page additionally does not apply the watch-history exclusion that every other recommender already honors, so a movie the user just added to their history can resurface on the next click. There is also no in-UI signal explaining why a mood recommendation was selected, which makes the feature feel opaque relative to the hybrid content explanations available elsewhere. This plan reuses existing utilities (`build_movie_stats`, `filter_watched_movies`, `render_movie`) instead of adding new recommender families or new data providers, in line with CLAUDE.md scope rules.
- **Hypothesis:** Threading the existing Bayesian rating into the mood pool and applying the existing watch-history filter to the random picker will raise the perceived quality of those two flows without changing public APIs, artifacts, or other recommender behavior. A small reason caption rendered alongside mood results will make the rule-based selection legible to the user.
- **Preconditions:**
  - `build_movie_stats(ratings)` already produces a `bayesian_rating` column and is wired through `cached_movie_stats()` in `src/app.py`.
  - `filter_watched_movies(df, watched_movie_ids)` already implements the canonical exclusion semantics used by the mood, content, and watch-history flows.
  - `recommend_by_mood`, `pick_random_movie`, `render_movie`, and `render_movie_list` accept additive optional parameters without breaking existing callers in the test suite.
  - Emoji policy exception: the project rule "No emojis ever in code, logs, commits, or generated documentation" stays in force for code identifiers, log lines, commit messages, and comments. This plan does not introduce any new emojis; existing UI strings in `config/config.yaml` (menu items, demo profiles) are unchanged.

## 1) Add `mood_min_bayesian_rating` configuration and facade constant

- **Goal:** Make the mood quality threshold a runtime-configurable value rather than a literal.
- **Files:**
  - `config/config.yaml` under `recommendations`.
  - `src/config.py` `DEFAULT_CONFIG["recommendations"]` and the module-level constants block.
- **Steps:**
  - Add `mood_min_bayesian_rating: 3.0` to `recommendations:` in the YAML.
  - Add the same key to `DEFAULT_CONFIG["recommendations"]` so the YAML and Python defaults stay aligned.
  - Expose `MOOD_MIN_BAYESIAN_RATING = float(config_value(["recommendations", "mood_min_bayesian_rating"], 3.0))` alongside `BAYESIAN_MIN_RATINGS`.
- **Test / verification:** `.venv/bin/python -c "from config import MOOD_MIN_BAYESIAN_RATING; print(MOOD_MIN_BAYESIAN_RATING)"` returns `3.0`. Existing `test_runtime_config_uses_yaml_backed_defaults` continues to pass.
- **Expected outcome:** Threshold value is sourced from `config.yaml` so future sweeps can re-tune it without code edits.
- **DONE (commit `2bf1da2`):** Added `mood_min_bayesian_rating: 3.0` to `config/config.yaml`, mirrored in `DEFAULT_CONFIG["recommendations"]`, exposed as `MOOD_MIN_BAYESIAN_RATING` in `src/config.py`. `from config import MOOD_MIN_BAYESIAN_RATING` returns `3.0`. No breakage in existing config-backed tests.

## 2) Extend `recommend_by_mood` with a Bayesian quality filter

- **Goal:** Drop movies below the threshold from the mood candidate pool, with a safe fallback to the unfiltered pool when the filter empties the pool entirely.
- **Files:** `src/recommenders/mood.py`.
- **Steps:**
  - Add optional `movie_stats` and `min_bayesian_rating` parameters to `recommend_by_mood`.
  - After the genre mask but before sampling, when both inputs are provided, intersect the filtered pool with `movie_stats[movie_stats["bayesian_rating"] >= min_bayesian_rating]["movieId"]`.
  - If the intersection is empty, keep the genre-filtered pool unchanged so sparse moods (for example `sad`) still return results.
- **Test / verification:**
  - New `test_mood_excludes_low_bayesian_rating`: synthetic stats with one movie below and one above 3.0; the below movie must not appear in results.
  - New `test_mood_falls_back_when_quality_filter_empties_pool`: stats with all movies below the threshold; result must remain non-empty.
  - Existing `test_mood_recommendations_filter_to_mapped_genres` continues to pass (default call without stats).
- **Expected outcome:** Mood results no longer surface movies the rating signal considers weak, while sparse moods still return picks.
- **DONE (commit `2bf1da2`):** Added `movie_stats` and `min_bayesian_rating` to `recommend_by_mood`; quality filter applied to the genre pool with fallback to the unfiltered pool when empty. **Followed up in item 8** because this ordering interacted incorrectly with the post-sample watched-exclusion in narrow scenarios. Both new tests passed against `2bf1da2`'s behavior with `watched=set()`.

## 3) Apply watch-history exclusion to `pick_random_movie`

- **Goal:** Stop recommending already-watched movies through the Random flow.
- **Files:** `src/recommenders/picker.py`.
- **Steps:**
  - Add an optional `watched_movie_ids` parameter and call `filter_watched_movies(filtered, watched_movie_ids)` after the genre filter and before sampling.
  - Preserve the existing "no candidates" semantics (`return None`).
- **Test / verification:**
  - New `test_random_movie_excludes_watched`: a two-movie pool with one id marked watched; ten consecutive calls always return the unwatched id.
  - New `test_random_movie_returns_none_when_all_watched`: every id in the pool watched; the call returns `None`.
  - Existing `test_random_movie_returns_none_for_empty_filter` continues to pass.
- **Expected outcome:** Random picks respect the watch history just like every other recommender family.
- **DONE (commit `2bf1da2`):** Added `watched_movie_ids` to `pick_random_movie`; `filter_watched_movies` applied after the genre mask. Both new tests passed. Item 7 extends this signature further with `excluded_movie_ids` for the re-roll path.

## 4) Surface mood reasoning through `render_movie` / `render_movie_list`

- **Goal:** Show a `Why: Mood: <mood> -> <genres>` caption beneath each mood recommendation without altering the content or collaborative pages.
- **Files:** `src/app.py`.
- **Steps:**
  - Add an optional `reason` parameter to `render_movie`. When present, render `st.caption(f"Why: {reason}")` between the genres line and the existing predicted-score caption.
  - Add an optional `reasons` list parameter to `render_movie_list` that maps element-wise into `render_movie`.
  - In `render_mood_page`, build `reason_text = f"Mood: {mood} -> {', '.join(mood_genres)}"` and pass `reasons=[reason_text] * len(recommendations)` to `render_movie_list`.
- **Test / verification:**
  - New `test_render_movie_signature_accepts_reason`: introspection asserts the new parameters exist and default to `None`.
  - Existing `test_app_import_smoke` continues to pass.
- **Expected outcome:** Users see a one-line explanation under each mood recommendation; non-mood pages are visually unchanged because `reasons` defaults to `None`.
- **DONE (commit `2bf1da2`):** Added `reason` to `render_movie` and `reasons` to `render_movie_list`; `render_mood_page` builds the per-result reason text and passes it through. Verify pass captured 10 `Why: Mood: happy -> Comedy, Family, Animation, Romance` captions under 10 mood recommendations.

## 5) Random page re-roll button and persistent pick state

- **Goal:** Let users request a different random pick without leaving the page, while keeping watch-history exclusion in force.
- **Files:** `src/app.py` `render_random_page`.
- **Steps:**
  - Store the current pick in `st.session_state["random_pick_movie"]` (dict view of the chosen movie row).
  - Expose a secondary "Pick Another" button that only appears once a pick exists; on click, draw a new pick, update session state, and `st.rerun()` to redraw.
  - Pass `watched_movie_ids=st.session_state.watched_movie_ids` into `pick_random_movie` from both buttons.
- **Test / verification:**
  - Existing `test_random_movie_returns_none_for_empty_filter` continues to pass (function-level contract unchanged).
  - Manual Streamlit smoke: after a successful first pick the "Pick Another" button appears; clicking it replaces the pick with a different movie that is not in the watch history.
- **Expected outcome:** Random discovery feels iterative instead of single-shot, and the watch history rule applies consistently.
- **DONE (commit `2bf1da2`):** Added `random_pick_movie` session state and the "Pick Another" button. Verify pass discovered that the button visibility check ran before the click handler that set the state (single-rerun model meant the button only appeared after a subsequent interaction). Restructured `render_random_page` so "Pick Another" renders after the saved-pick render, inside the branch that already requires a pick — included in the same commit. Item 9 layers stale-pick invalidation and true re-roll on top of this base.

## 6) Mood dropdown label clarity

- **Goal:** Make the dropdown values self-describing without changing the underlying keys, so the rule-based mapping is visible at a glance.
- **Files:** `src/app.py` `render_mood_page`.
- **Steps:**
  - Use `format_func=lambda key: f"{key.capitalize()} ({', '.join(MOOD_GENRE_MAP[key])})"` in the mood `st.selectbox` while keeping the option list as the existing keys.
- **Test / verification:** Manual Streamlit smoke. No automated test needed: this is a display-only change that does not affect any call signatures.
- **Expected outcome:** The dropdown shows entries like `Happy (Comedy, Family, Animation, Romance)` while the value passed into `recommend_by_mood` stays `"happy"`.
- **DONE (commit `2bf1da2`):** Added `format_func` to the mood `st.selectbox`. Verify pass captured options `['Happy (Comedy, Family, Animation, Romance)', 'Sad (Drama, Romance)', 'Adventurous (Action, Adventure, Thriller)', ...]` while the underlying value remained `'happy'`.

## Verify-phase finding: "Pick Another" visibility bug

The first implementation of item 5 rendered the "Pick Another" button conditional on `st.session_state.get("random_pick_movie") is not None`, but the check ran above the click handler that sets that state. Streamlit's single-rerun execution model meant the button only appeared on the *next* user interaction rather than in the same rerun that produced the first pick. AppTest captured this directly: after clicking `random_pick`, the button list was `['random_pick']` (no `random_pick_another`) and the user would have to interact again before the re-roll button became reachable. Unit tests would not have caught this because the function-level contract was intact; only driving the live script revealed it. Fix: restructure `render_random_page` so the "Pick Another" button is rendered *after* the saved-pick render, inside the branch that already requires the pick to exist (`src/app.py` `render_random_page`). The post-fix AppTest captures `['random_pick', 'random_pick_another']` immediately after the first pick.

## 7) `pick_random_movie` `excluded_movie_ids` parameter for true re-roll

- **Goal:** Make "Pick Another" actually re-roll. The first implementation could return the same `movieId` because the current pick was not removed from the candidate pool; statistically rare on the full MovieLens corpus but visible in narrow genre filters.
- **Files:** `src/recommenders/picker.py`.
- **Steps:**
  - Add an optional `excluded_movie_ids` parameter to `pick_random_movie`.
  - Union it with `watched_movie_ids` via `normalize_movie_ids`, then apply a single `filter_watched_movies` call. No second helper function is introduced; the existing exclusion semantics already cover the case.
  - `return None` semantics preserved for an empty post-filter pool.
- **Test / verification:**
  - New `test_pick_random_movie_excludes_current`: a two-movie pool with the current pick excluded; ten consecutive calls always return the other id.
  - New `test_pick_random_movie_returns_none_when_only_match_excluded`: a single-movie pool with that id excluded; the call returns `None`.
- **Expected outcome:** Pick Another is guaranteed to return a different movie when an alternative exists, or `None` when it does not.

## 8) Mood: reorder filter pipeline (watched exclusion before quality threshold)

- **Goal:** Fix a regression in item 2 where the Bayesian threshold was applied before watched-movie exclusion. Concrete scenario: one Comedy movie has `bayesian_rating=4.5` (watched) and another has `bayesian_rating=2.0` (unseen); the threshold filter kept only the watched film, sampling produced it, and `filter_watched_movies` then dropped it — yielding an empty result that should have been the low-rated unseen film via the fallback path.
- **Files:** `src/recommenders/mood.py`, `config/config.yaml`.
- **Steps:**
  - Reorder `recommend_by_mood`: genre mask → `filter_watched_movies` (unseen pool) → early return if unseen pool empty → quality threshold on unseen pool, with fallback to the unseen pool when the threshold empties it → sample.
  - Drop the now-redundant post-sample `filter_watched_movies` call (the sample pool is already unseen by construction).
  - Add an inline comment above `mood_min_bayesian_rating` in `config/config.yaml` clarifying the semantics: "preferred Bayesian rating for the unseen mood pool; if the threshold empties the pool, mood falls back to the unfiltered unseen pool".
- **Test / verification:**
  - New `test_mood_falls_back_when_only_high_rated_is_watched`: high-rated movie watched + low-rated movie unseen → returns the low-rated unseen movie via fallback.
  - New `test_mood_prefers_high_rated_unseen_over_low_rated_unseen`: both unseen, only the high-rated one passes the threshold → result excludes the low-rated one.
  - Existing `test_mood_excludes_low_bayesian_rating` and `test_mood_falls_back_when_quality_filter_empties_pool` continue to pass (both used `watched=set()` so the reorder is behaviorally equivalent).
- **Expected outcome:** The quality threshold is a *preference* applied over the unseen pool, never an unintended hard filter that interacts with watched exclusion to empty the result set.

## 9) Random: stale-pick invalidation and true re-roll wiring

- **Goal:** Two related fixes to `render_random_page`. (a) Stop rendering a saved pick after the user has moved it to watch history or changed the genre filter so it no longer matches. (b) Pass the current pick's id to `pick_random_movie` as `excluded_movie_ids` during re-roll, and on no-alternative produce a clear warning without silently re-rendering the same pick.
- **Files:** `src/app.py` `render_random_page`.
- **Steps:**
  - Inline helper `_pick_is_still_valid(pick_dict)` (scoped to `render_random_page`) checks: `movieId not in watched_movie_ids` and, when `selected_genres` is non-empty, that the pick's genre string contains at least one selected genre (case-insensitive substring match, mirroring the same matching `pick_random_movie` uses).
  - If a saved pick fails validation: clear `st.session_state.random_pick_movie`, emit a caption "Previous pick was filtered out by your watch history or genre selection.", and skip the render.
  - "Pick Another" handler: read `current_id` from the saved pick, pass `excluded_movie_ids={current_id}` to `pick_random_movie`. On `None` return: `st.warning("No other unseen movie matched the current filter.")` and intentionally do not clear state — the current pick stays visible.
- **Test / verification (AppTest, new `tests/test_app_integration.py`):**
  - `test_random_invalid_pick_cleared_when_watched_added`: real pick, then add its id to `watched_movie_ids`, rerun — session state cleared, no movie subheader, stale-pick caption present.
  - `test_random_invalid_pick_cleared_when_genre_changes`: real pick, then set `random_genres` multiselect to a genre not in the pick — session state cleared, no movie subheader.
  - `test_random_pick_another_excludes_current_id`: mock `recommenders.pick_random_movie` with `side_effect=[row_a, row_b, row_b]`; click Pick → state is row_a; click Pick Another → state is row_b; the mock's second call kwargs include `excluded_movie_ids={row_a.movieId}`.
  - `test_random_pick_another_warns_when_no_alternative`: mock with `side_effect=[row_a, None]`; after Pick Another, session state still holds `row_a`, a `st.warning` containing "No other unseen movie matched" is rendered, and the mock's second call still includes `excluded_movie_ids={row_a.movieId}`.
- **Expected outcome:** The Random page reflects the user's current filters at all times, re-roll never returns the same movie, and "no alternative" is communicated explicitly rather than by silent reuse.

## Verification (full)

- **Focused unit run:** `.venv/bin/python -m unittest tests.test_movie_rec` — `Ran 54 tests in 0.420s, OK`.
- **Full discover:** `.venv/bin/python -m unittest discover -s tests` — `Ran 86 tests in 11.408s, OK` after items 7-9 work. The 13 new tests across items 1-9 (5 from items 1-6, 2 from item 7, 2 from item 8, 4 from item 9) all pass; the wider repository suite (ALS, LightFM, SBERT FAISS, semantic embeddings, sweep, evaluation runner) is unaffected.
- **Streamlit boot smoke:** `.venv/bin/python -m streamlit run src/app.py --server.headless true --server.port 8765 --browser.gatherUsageStats false` followed by `curl http://localhost:8765/_stcore/health` returned `HTTP 200`. Process killed cleanly.
- **AppTest scenario sweep:** the four new `test_app_integration.py` tests double as automated AppTest captures for items 5, 7, and 9 (stale-by-watched, stale-by-genre, deterministic re-roll via mock + call-args, and the no-alternative state-preservation + warning path). The mood caption + enriched dropdown labels (items 4, 6) were captured during the verify-phase run earlier in the day (selectbox options `['Happy (Comedy, Family, Animation, Romance)', 'Sad (Drama, Romance)', ...]`, 10 `Why: Mood: happy -> Comedy, Family, Animation, Romance` captions under the 10 results) and remain valid because `render_mood_page` is unchanged after the items 7-9 round.
- **Regression:** content-based, collaborative, and watch-history pages render unchanged. `render_movie` / `render_movie_list` `reason`/`reasons` parameters default to `None`. The reordered mood pipeline keeps `recommend_by_mood`'s public signature and remains deterministic on `random_state=42`.

## Execution notes

Items 1-6 landed in commit `2bf1da2` ("feat(recommendations): add mood_min_bayesian_rating and update recommendation logic") on 2026-05-23. That commit included the Pick Another visibility fix — the bug was both discovered (via the verify-phase AppTest pass) and resolved before the commit landed; only one commit was needed for items 1-6.

Items 7-9 are a follow-up correction pass on items 1-6. Three items are still pending commit at the time of this update: extend `pick_random_movie` with `excluded_movie_ids`, reorder the mood pipeline so the Bayesian threshold runs over the unseen pool, and wire `render_random_page` for stale-pick invalidation, true re-roll, and explicit "no other movie" warning. The DONE markers for items 7-9 (and a matching narrative entry to this section) will be backfilled in a follow-up `docs(experiments): record items 7-9 DONE markers` commit once the implementation commit lands, mirroring the repository's existing pattern (for example commits `924ba31` and `8d19ea3`).

The implementation commit will use `git add -f docs/experiments/2026-05-23_mood-random-ux-enhancements.md` because `/docs` is gitignored. No artifact regeneration, no `requirements.txt`/`pyproject.toml` change, no SBERT loader change. Emoji policy exception remained scoped to UI strings; no emojis were introduced into code, logs, commit messages, comments, or this document.
