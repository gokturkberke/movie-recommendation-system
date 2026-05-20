- **Date:** 2026-05-21
- **Topic:** Investigate why `als_implicit` and `svd_topk` produced zero relevance hits in the 100-user evaluation, fix the wiring bug, and document the remaining algorithmic gap
- **Motivation:** The 9-model 100-user run committed in `3f0ab0e` (run id `artifacts/evaluation/metrics_summary_2026-05-20T09-47-43Z.{csv,json}`) shows `als_implicit` and `svd_topk` at 0.0000 on every relevance metric at K=10 and K=20, while `lightfm_warp` reaches `hit_rate@10 = 0.0727` and `NDCG@10 = 0.1427` on the same 55-user positive-holdout slice with the same train/holdout split. Two classical CF baselines collapsing to zero while a third clears the hurdle is asymmetric enough to be a wiring/data issue, at least for one of them. `docs/08_evaluation_results_report.md` Conclusions and Caveats sections currently state this as an open question. The audit trail for the wider run lives in `docs/experiments/2026-05-20_classical-cf-and-eval-expansion.md` item 4 DONE marker; this plan resolves the open question that marker left behind.
- **Hypothesis:**
  - **ALS:** `src/experimental/als_recommender.py:194-199` calls `model.recommend(user_pos, artifacts.user_items[user_pos], N=..., filter_already_liked_items=True)`. The `user_items` row was built from the full pre-split rating matrix at training time, so it includes the per-user holdout interaction (each evaluated user's latest rating with score >= 4.0). With `filter_already_liked_items=True`, ALS will systematically remove the holdout movieIds from its candidate set, making relevance hits impossible. Setting the flag to `False` and relying on the existing post-hoc `filter_watched_movies(...)` call (which only excludes train-derived movieIds) will yield `hit_rate@10 > 0` on the same 100-user slice, and bring ALS into the same evaluated-user count (55) as LightFM.
  - **SVD top-K:** `src/recommenders/svd.py:16-38` filters candidates by train-only `rated_movie_ids` and scores them with Surprise's RMSE-trained model. `catalog_coverage@10 = 0.0029` and `catalog_coverage@20 = 0.0055` indicate the model concentrates on a small popular-favorable set; the user's specific holdout is ranked outside the top 20. This is an explicit-rating SVD weakness, not a wiring bug. Verified by inspecting the rank of the holdout in the full predicted-score sort for sampled users: rank > 50 for at least 4 of 5 sampled users would settle the hypothesis.
- **Preconditions:**
  - All four items from `docs/experiments/2026-05-20_classical-cf-and-eval-expansion.md` are DONE on `main` through commit `3f0ab0e`.
  - `artifacts/evaluation/metrics_summary_2026-05-20T09-47-43Z.{csv,json}` and `run_config.json` are present locally as the broken baseline.
  - `artifacts/models/als/` carries `als_model.npz`, `user_items.npz`, `user_index.csv`, `item_index.csv`, `metadata.json` from commit `c956ea5` (305,098 users / 40,441 items / 16,863,053 nnz).
  - `artifacts/models/lightfm/` carries the WARP artifacts from commit `65de01d` for cross-check.
  - `cleaned_data/ratings_clean.csv`, `cleaned_data/movies_clean.csv`, and `cleaned_data/svd_trained_model.pkl` are present locally.
  - CLAUDE.md section 7 governs DONE marker structure; do not fill `<hash>` placeholders before the corresponding commit lands.

Corresponding audit item: `docs/08_evaluation_results_report.md` Conclusions ("`als_implicit` is the fastest model at 7.8 ms mean latency, but it produces no relevance hits in this run") and Caveats blocks.

## 1) Diagnostic reproduction script

- **Goal:** Produce concrete evidence for the two hypotheses above without changing any production code. The script is the artifact of the `Inspect -> Plan -> Code -> Test -> Fix` Inspect step: it has to print numbers, not opinions.
- **Files:**
  - `scripts/diagnose_als_svd_topk.py` (new, target ~180 lines). Argparse-driven, default behavior dumps 5 sample users from the same 100-user selection that the canonical evaluation uses (`select_evaluation_user_ids` from `src/evaluation_runner.py` with the existing config defaults).
- **Steps:**
  - Reuse:
    - `src/data_access.py:load_ratings, load_movies` for raw inputs.
    - `src/evaluation_runner.py:select_evaluation_user_ids, temporal_train_test_split` for the same selection and split logic the canonical run uses (do not duplicate the selection rules).
    - `src/experimental/als_recommender.py:load_als_artifacts` for the ALS artifact, plus a direct call to `artifacts.model.recommend` so the script can flip `filter_already_liked_items` between True and False for the same user.
    - `src/experimental/lightfm_recommender.py:load_lightfm_artifacts` for cross-check of which items live in the LightFM index versus the ALS index.
    - `src/data_access.py:load_surprise_model` for the SVD model handle.
  - Per sampled user, print:
    - `userId`, `holdout_movie_id`, `holdout_rating` (must be >= 4.0 to count for top-N; this is the positive-holdout slice).
    - Whether `holdout_movie_id` is in `als_artifacts.item_index` and in `lightfm_artifacts.item_index`.
    - Whether `holdout_movie_id` is in the user's `als_artifacts.user_items[user_position]` row (non-zero confidence). This is the smoking-gun check for ALS.
    - ALS top-20 with `filter_already_liked_items=True` (current behavior), printed as `(rank, movieId, score, is_holdout)`.
    - ALS top-20 with `filter_already_liked_items=False` (proposed behavior), printed the same way; then apply `filter_watched_movies` with the user's train movieIds and re-print.
    - SVD top-20 from `raw_svd_predictions` using only the train portion of the user's ratings, printed as `(rank, movieId, predicted_score, is_holdout)`.
    - The full-catalog rank of the SVD holdout: score every movie in `movies_clean.csv` via `model.predict(uid, iid)`, sort descending, find the holdout's 1-based rank.
  - Output: single human-readable text dump per user (heading line + bullet lines). Optionally accept `--output-path` for redirecting; default stdout. Do not write JSON in this script -- this is a debugging aid, not an artifact.
  - Run it locally: `.venv/bin/python scripts/diagnose_als_svd_topk.py --sample-size 5 --random-seed 42`.
  - Commit message: `chore(diagnose): add ALS / SVD top-K zero-hit reproduction script`. Use `git add -f docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md` to land this DONE marker alongside.
- **Test / verification:**
  - The script runs to completion without exception on the existing artifacts.
  - For at least 4 of the 5 sampled users: the holdout movieId appears as a non-zero entry in `als_artifacts.user_items[user_position]`. This is the wiring-bug proof.
  - The ALS top-20 with `filter_already_liked_items=True` contains the holdout for zero of the 5 users.
  - The ALS top-20 with `filter_already_liked_items=False` (then post-filtered by train watched ids) contains the holdout for at least 1 of the 5 users.
  - The SVD holdout rank in the full-catalog sort is greater than 50 for at least 4 of the 5 sampled users -- consistent with an algorithmic weakness, not a wiring filter.
- **Expected outcome:** Two cleanly proven hypotheses. Decision criterion: ALS shows the smoking-gun (holdout in user_items row) for at least 4 of 5 users; SVD shows a high holdout rank for at least 4 of 5 users.
- **DONE (commit `6df6a24`):** Added `scripts/diagnose_als_svd_topk.py` and ran it against the same 100-user / latest-1 / positive-threshold=4.0 slice the canonical evaluation uses. All four pre-stated verification criteria passed; both working hypotheses are confirmed.
  - Diagnostic summary across 5 sampled positive-holdout users (`random_seed=42`, sampled userIds=[5, 47, 68, 85, 108]):

    | Check | Result | Threshold |
    |---|---:|---:|
    | holdout in ALS `item_index` | 5 / 5 | >= 4 |
    | holdout in LightFM `item_index` (cross-check) | 5 / 5 | -- |
    | holdout in `user_items[user_position]` row (smoking gun for ALS) | 5 / 5 | >= 4 |
    | ALS top-20 with `filter_already_liked_items=True` contains holdout | 0 / 5 | == 0 |
    | ALS top-20 with `filter_already_liked_items=False` (post-filter train) contains holdout | 3 / 5 | >= 1 |
    | SVD full-catalog rank of holdout > 50 | 4 / 5 | >= 4 |

  - Per-user confidence values match the formula `1 + 40 * max(rating - 4.0, 0)`: userId 5 (rating 5.0) -> 41.0; userId 47 (rating 5.0) -> 41.0; userId 68 (rating 4.5) -> 21.0; userId 85 (rating 4.0) -> 1.0; userId 108 (rating 4.5) -> 21.0. The edge case at userId 85 (rating exactly equals threshold) leaves confidence=1.0, which weakens but does not zero the ALS signal.
  - SVD full-catalog ranks of the holdout for the same 5 users: 94, 20, 392, 14171, 10934. Consistent with the algorithmic-weakness hypothesis: RMSE-trained Surprise SVD concentrates on a narrow predicted-score-near-5.0 set, so user-specific holdout movies land outside the top 20 for 4 of 5 users in this sample.
  - Local dump path (not committed): `/private/tmp/diagnose_als_svd_topk_dump.txt`. Reproducible via `.venv/bin/python scripts/diagnose_als_svd_topk.py --sample-size 5 --random-seed 42 --output-path /private/tmp/diagnose_als_svd_topk_dump.txt`.
  - Decision: proceed to Item 2. The ALS fix is unambiguous (flip `filter_already_liked_items` to `False` and rely on the post-hoc train-only `filter_watched_movies`); the SVD top-K finding is now well-supported for Item 4's documentation.

## 2) ALS wiring fix (flip `filter_already_liked_items`)

- **Goal:** Land the minimal one-line behavior change in `als_recommendations_for_user` so the holdout movieIds are no longer filtered by Implicit's built-in exclude path; rely on the existing post-hoc `filter_watched_movies` (which uses only train-derived watched ids) for legitimate exclusion of the user's training history. This brings ALS into the same exclusion semantics as the LightFM path.
- **Files:**
  - `src/experimental/als_recommender.py`: change line 198 `filter_already_liked_items=True` to `filter_already_liked_items=False`. Add a single short comment above the call explaining the invariant -- the `user_items` row was built from pre-split data and includes the holdout, so train-only exclusion has to happen post-hoc via `filter_watched_movies`. Do not touch the function signature, the dataclass, or the artifact format.
  - `tests/test_als_recommender.py`: update the assertion that inspects `FakeAlsModel.last_call["filter_already_liked_items"]` so it expects `False`. Add one new unit test `test_als_recommendations_excludes_watched_via_post_filter` that builds a fake model returning 3 item positions (one of which maps to a movieId in `watched_movie_ids`), and asserts that the returned DataFrame does not contain that movieId. This proves the post-hoc filter still excludes the train history.
  - `requirements.txt`: no change.
  - `config/config.yaml`: no change.
  - `src/evaluation_runner.py`: no change. The `make_als_per_user` closure already passes `watched_by_user.get(user_id, [])` -- the train-only exclusion set -- and that path is the one the fix relies on.
- **Steps:**
  - Edit `src/experimental/als_recommender.py:198`.
  - Edit `tests/test_als_recommender.py` to flip the expected flag value and add the new test.
  - Run `.venv/bin/python -m unittest tests.test_als_recommender` -- must pass.
  - Run `.venv/bin/python -m unittest discover -s tests` -- must stay at 56/56.
  - 5-user smoke: `.venv/bin/python scripts/evaluate_baselines.py --max-users 5 --k 5 --include-als --als-artifacts-dir artifacts/models/als --output-dir artifacts/evaluation`. The JSON output must contain `top_n.als_implicit.5` with `hit_rate_at_k > 0` for at least one user (with random_seed=42, item 1's evidence sets the floor at one).
  - Commit message: `fix(als): rely on train-only filter_watched_movies for evaluation exclusion`. Use `git add -f docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md`.
- **Test / verification:**
  - `tests.test_als_recommender` exits 0; new test passes.
  - Full suite stays 56/56 (no regression).
  - 5-user smoke produces a populated `als_implicit` block with `hit_rate_at_k > 0` at K=5 -- the first non-zero ALS row in this project's history.
  - The smoke run's `recommended_item_count` for `als_implicit.5` is exactly `5 * evaluated_user_count` (no silent drops; LightFM passes this same check today).
- **Expected outcome:** The wiring bug is closed; ALS rejoins the relevance comparison. Decision criterion: 5-user smoke `hit_rate_at_k > 0` and the full test suite stays green.
- **DONE (commit `9ee777c`):** Flipped `filter_already_liked_items` to `False` in `src/experimental/als_recommender.py` (with a three-line comment naming the invariant) and updated `tests/test_als_recommender.py` to expect the new flag value plus a second exclusion test on a different watched movieId. The post-hoc `filter_watched_movies` call now does all train-history exclusion; the holdout movieIds, which previously sat in the pre-split `user_items` row, are no longer filtered out of ALS candidates.
  - 5-user smoke (`--max-users 5 --k 5 --include-als --als-artifacts-dir artifacts/models/als`):
    | Metric | Before fix | After fix |
    |---|---:|---:|
    | hit_rate@5 | 0.0000 | 0.4000 |
    | precision@5 | 0.0000 | 0.0800 |
    | recall@5 | 0.0000 | 0.4000 |
    | NDCG@5 | 0.0000 | 0.1723 |
    | evaluated_user_count | 5 | 5 |
    | recommended_item_count | 13 | 22 |
  - Unit tests: `.venv/bin/python -m unittest tests.test_als_recommender` 4/4 OK (was 3/3 before the added test); full suite `.venv/bin/python -m unittest discover -s tests` 57/57 OK (was 56/56).
  - The artifact directory `artifacts/models/als/` was not modified -- the change is purely in how the runtime consumes the artifact.
  - Decision: proceed to Item 3. The 5-user smoke validates the code path; the 100-user re-run in Item 3 will produce the canonical comparison row.

## 3) 100-user re-evaluation and report refresh

- **Goal:** Re-run the canonical 9-model evaluation with the fixed ALS and update `docs/08_evaluation_results_report.md` so the audit table reflects ALS post-fix on the same 100-user slice.
- **Files:**
  - `artifacts/evaluation/metrics_summary.{csv,json}` (regenerated, gitignored).
  - `artifacts/evaluation/run_config.json` (regenerated, gitignored).
  - `artifacts/evaluation/metrics_summary_<timestamp>.{csv,json}` (regenerated, gitignored).
  - `docs/08_evaluation_results_report.md` (updated in place: K=10 table, K=20 table, Latency Findings table, Conclusions paragraph, Caveats; no structural change).
- **Steps:**
  - Run the full canonical command from the previous plan (item 4 of `2026-05-20_classical-cf-and-eval-expansion.md`):
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
  - Verify `wc -l artifacts/evaluation/metrics_summary.csv` returns 29 (1 header + 9 models * 3 K + 1 SVD rating prediction row).
  - Confirm `als_implicit,10,...,hit_rate_at_k` in the new CSV is greater than zero.
  - Update `docs/08_evaluation_results_report.md`:
    - "Run Summary": no command change, but bump the date phrasing if used; otherwise leave intact.
    - K=10 and K=20 tables: replace the entire `als_implicit` row with the new numbers. Re-sort the K=10 table by NDCG@10 descending so the new ALS row lands in its correct position (likely between LightFM and the content baselines). Do the same for K=20.
    - "Latency Findings": replace ALS's row with the new latency (likely still under 50 ms; the flag flip should not noticeably affect latency).
    - "Conclusions": replace the sentence "`als_implicit` is the fastest model at 7.8 ms mean latency, but it produces no relevance hits in this run." with a 1-2 sentence comparison of ALS post-fix against LightFM WARP (which model leads NDCG, which leads latency); add a one-sentence reference to this plan as the audit trail for the change.
    - "Caveats": no change.
  - Commit message: `docs(eval): refresh report after ALS exclusion-semantics fix`. Use `git add -f` for both the report and this plan file.
- **Test / verification:**
  - `metrics_summary.csv` is exactly 29 lines.
  - `als_implicit` K=10 row in the CSV has `hit_rate_at_k > 0`, `evaluated_user_count = 55` (matches LightFM), and `precision_at_k > 0`.
  - The K=10 and K=20 tables in the report match the CSV row-for-row (no contradictions).
  - The report no longer claims ALS "produces no relevance hits in this run".
  - `.venv/bin/python -m unittest discover -s tests` stays 56/56.
- **Expected outcome:** A single canonical 100-user × 9-model table where ALS is no longer a silent zero. Decision criterion: report and CSV agree; new ALS row carries non-zero relevance; SVD top-K stays at zero (handled separately in item 4).
- **DONE / DROPPED:**

## 4) Document the SVD top-K zero-hit finding (no code change)

- **Goal:** Close the loop on the SVD top-K diagnosis in the same report, so a future agent does not re-investigate it. No fix is attempted here -- the fix belongs in a separate plan once we decide whether to keep explicit-rating SVD ranking at all.
- **Files:**
  - `docs/08_evaluation_results_report.md`: add a new subsection titled "Why SVD top-K stays at zero hits" between "Latency Findings" and "Conclusions". Three short paragraphs: (a) what `raw_svd_predictions` does -- candidate filter by train, score by Surprise's RMSE-trained predictor; (b) the observed evidence from the item 1 diagnostic dump -- holdout rank in the full-catalog predicted-score sort is greater than 50 for the sampled users; (c) why this is expected, not a bug -- Surprise SVD is optimized for RMSE on explicit ratings, not for ranking; ranking concentrates on a small popular-favorable set (`catalog_coverage@10 = 0.0029`).
  - This plan file: add the SVD top-K improvement to the "Deferred / Future" section so the next agent can pick it up.
- **Steps:**
  - Draft the three-paragraph subsection from the item 1 dump numbers. Do not invent numbers; quote the actual ranks the diagnostic script printed.
  - Add the deferred line at the bottom of this file.
  - Commit message: `docs(eval): explain SVD top-K zero-hit as expected algorithmic limitation`. Use `git add -f` for both files.
- **Test / verification:**
  - The new subsection exists with a heading line, three paragraphs, and at least one quoted rank from the diagnostic dump.
  - The "Deferred / Future" section of this plan file lists the SVD top-K ranking improvement.
  - No code file is modified by this item.
- **Expected outcome:** Future agents reading the report know SVD top-K's zero is expected and where to look if they want to fix it. Decision criterion: the subsection is in the report and the deferred line is in this plan file.
- **DONE / DROPPED:**

## Deferred / Future (out of this plan)

- **SVD top-K ranking improvement:** Document its known explicit-rating weakness here; do not fix in this plan. A future `docs/experiments/{date}_svd-topk-ranking-fix.md` can explore options such as predicted_score blended with log-popularity, learning-to-rank reranking over SVD candidates, or retiring `svd_topk` as a baseline once a learned reranker exists. Coverage and novelty trade-offs need to be measured in that plan, not asserted here.
- **Retrain ALS on a leave-one-out split:** The flag flip in item 2 is sufficient to unblock the metric on this slice. A stricter version of the experiment would retrain ALS with the holdout interactions explicitly held out from the user_items matrix. That is more expensive (a 100-user-specific artifact, or a leave-one-out training loop) and lives in a separate plan if we ever want a tighter offline evaluation guarantee.
- **LightFM hyperparameter sweep, eval slice expansion, cold-start segmentation:** Separate plans. Not in scope here.

## Critical Files (Reference)

- `src/experimental/als_recommender.py:175-235` -- `als_recommendations_for_user`. Item 2 changes line 198.
- `src/experimental/als_recommender.py:41-71` -- `build_confidence_matrix` and the confidence formula. No change; the formula is correct, the bug is downstream.
- `src/experimental/lightfm_recommender.py:175-178` and surrounding -- LightFM's recommend path, used as the "what works" reference shape during item 1 diagnostics.
- `src/evaluation_runner.py:374-407` -- `make_als_per_user` closure; this is where `watched_by_user` is built from `train` only, which is the invariant item 2 relies on.
- `src/evaluation_runner.py:559-610` -- `select_evaluation_user_ids` and `temporal_train_test_split`. Item 1's diagnostic script reuses these directly.
- `src/evaluation_runner.py:906-927` -- SVD top-K wiring; item 4 documents this as correct.
- `src/recommenders/svd.py:16-38` -- `raw_svd_predictions`; item 4 documents this as RMSE-trained, not ranking-trained.
- `src/recommenders/common.py:73-77` -- `filter_watched_movies`; the post-hoc filter that item 2's fix relies on.
- `tests/test_als_recommender.py:15-43` -- `FakeAlsModel` fixture; item 2 updates the expected flag and adds a new exclusion test.
- `artifacts/evaluation/metrics_summary_2026-05-20T09-47-43Z.{csv,json}` and `run_config.json` -- the broken baseline reference.
- `artifacts/models/als/metadata.json` -- artifact metadata; stays as-is, no retraining.
- `docs/08_evaluation_results_report.md` -- updated in items 3 and 4.
- `docs/experiments/2026-05-20_classical-cf-and-eval-expansion.md` -- prior plan; item 4 DONE marker is the audit pointer that this plan resolves.

## End-to-End Verification Sequence

1. Item 1 complete: `scripts/diagnose_als_svd_topk.py` exists and produced a 5-user dump that satisfies the four numeric checks in item 1's "Test / verification" block. DONE marker carries a real commit hash.
2. Item 2 complete: `tests.test_als_recommender` passes; full suite stays 56/56; 5-user smoke produces a non-zero `als_implicit` row. DONE marker carries a real commit hash and the 5-user smoke metrics.
3. Item 3 complete: 100-user × 9-model `metrics_summary.csv` has exactly 29 rows; `als_implicit` is non-zero on K=10 and K=20; the report's K-tables and Conclusions reflect the new numbers. DONE marker carries a real commit hash and a metric triple (precision/recall/NDCG at K=10).
4. Item 4 complete: report carries the "Why SVD top-K stays at zero hits" subsection; this plan's Deferred section lists the SVD top-K improvement. DONE marker carries a real commit hash.
5. After every commit, record the real commit hash in this plan file; CLAUDE.md section 7 forbids leaving temporary hash placeholders in place.

## Execution Notes

- This file is the contract; CLAUDE.md and AGENTS.md in the repo root are the project-wide rules.
- The `/docs` path is gitignored, but tracked files (`docs/08_evaluation_results_report.md`, every file under `docs/experiments/`) still need `git add -f` to land in commits.
- Items are sequential: item 1 produces the evidence, item 2 lands the code change, item 3 produces the canonical post-fix report, item 4 closes the SVD top-K loop. Skipping item 1 would mean the fix in item 2 is based on a guess, which CLAUDE.md section 5 explicitly forbids.
- One commit per item -- do not bundle items 2 and 3, even though they share a code path. The DONE marker for each item must reference a single commit hash.
