- **Date:** 2026-05-22
- **Topic:** Eval slice expansion and multi-seed variance bounds for the top-N comparison table
- **Motivation:** The 2026-05-21 ALS investigation landed in `cb5e4dc` with `als_implicit` at NDCG@10 = 0.1765, hit_rate@10 = 0.3091, and mean latency 7.1 ms -- the clear winner on every relevance metric and simultaneously the fastest. The same plan's Caveats section flags two known limitations: (a) 55 of 100 selected users had a positive holdout, still a small slice, and (b) the artifact-level training-set leakage shared between ALS and LightFM (both trained on the full rating matrix). The user explicitly picked "eval slice expansion" as the next phase to validate whether ALS leadership is a stable signal or a one-seed / small-N artifact. Phase 1 exploration also surfaced a structural issue: `select_evaluation_user_ids` is deterministic (sorted-userId first-N), and every artifact is pre-trained, so the existing `--random-seed` only varies the `random` baseline -- a naive multi-seed run would produce no variance for any informative model.
- **Hypothesis:**
  - **H1 (seed robustness):** Across `--user-sample-seed` in `{42, 7, 1337}` at `--max-users 100 --holdout-count 1`, the ordering `als_implicit > lightfm_warp > hybrid_content` on NDCG@10 holds in 3 of 3 runs, with `std(NDCG@10) / mean(NDCG@10)` for each of the three models below 0.30 (less than 30% relative variation across seeds).
  - **H2 (sample-size robustness):** At `--max-users 300 --user-sample-seed 42 --holdout-count 1`, the same `als_implicit > lightfm_warp > hybrid_content` ordering holds. Evaluated user count reaches at least 150.
  - **H3 (holdout-shape robustness):** Across `--user-sample-seed` in `{42, 7, 1337}` at `--max-users 300 --holdout-count 3`, the same ordering on NDCG@10 holds in 3 of 3 runs. We do not predict the absolute numbers because the recall denominator changes; we predict the ordering.
  - Refutation of any of these would not invalidate the previous plan's fix -- it would change how ALS leadership is framed in the report (from "the clear winner" to "the leading model on the canonical slice; ordering is sample-dependent").
- **Preconditions:**
  - All four items of `docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md` are DONE on `main` through commit `8d5d5d3`.
  - `artifacts/evaluation/metrics_summary_2026-05-20T18-47-21Z.{csv,json}` is the canonical 2026-05-21 reference; new runs sit alongside as timestamped copies.
  - `artifacts/models/{als,lightfm}/` and `artifacts/indexes/sbert_faiss/` are present locally with the same artifacts the 2026-05-21 plan used. No retraining in this plan.
  - `cleaned_data/ratings_clean.csv`, `cleaned_data/movies_clean.csv`, `cleaned_data/svd_trained_model.pkl` present.
  - CLAUDE.md section 7 governs DONE marker structure; do not fill `<hash>` placeholders before the corresponding commit lands.

## 1) Add user-sampling seed plumbing

- **Goal:** Make the user sampling step accept an optional random seed. Default behavior stays the deterministic first-N slice so every prior run reproduces.
- **Files:**
  - `src/evaluation_runner.py:86-101` (`select_evaluation_user_ids`): add `random_seed=None` keyword parameter. When `random_seed` is not None, sample `max_users` without replacement from the eligible-user list using `np.random.default_rng(random_seed).choice(...)`. When `None`, preserve the current `[:max_users]` first-N slice.
  - `src/evaluation_runner.py` (`run_evaluation` orchestration, the call to `select_evaluation_user_ids`): add `user_sample_seed` parameter to `run_evaluation()`, forward it into the call, and persist it in the run config block so the JSON / `run_config.json` artifact records it.
  - `scripts/evaluate_baselines.py`: add `--user-sample-seed` argument (separate from `--random-seed`, which stays for the `random` baseline). Default value: omit (None). Forward to `run_evaluation(user_sample_seed=...)`.
  - `tests/test_evaluation_runner.py` (new file, target ~60 lines):
    - `test_select_evaluation_user_ids_default_is_deterministic_first_n`: ratings with eligible userIds `[10, 20, 30, 40, 50]`, `max_users=3`, no seed -> returns `[10, 20, 30]`.
    - `test_select_evaluation_user_ids_with_seed_returns_stable_random_sample`: same ratings, `random_seed=42`, returns the same 3 userIds on two consecutive calls. The sample is a strict subset of the eligible pool, and is not the first-N slice (the test verifies the result is different from the deterministic one).
    - `test_select_evaluation_user_ids_seed_subsamples_within_eligible`: eligible pool of 5 users, `max_users=3`, `random_seed=7` -> returned set is a 3-element subset of the 5 eligible.
- **Reuse:** `np.random.default_rng(seed).choice(array, size, replace=False)` -- the same call shape used in `src/evaluation.py:417,430`. No new dependencies.
- **Steps:**
  - Make the function and CLI edits.
  - Add the test file (or extend an existing test file -- there is no `tests/test_evaluation_runner.py` yet; create it).
  - `.venv/bin/python -m unittest tests.test_evaluation_runner` -- must pass.
  - `.venv/bin/python -m unittest discover -s tests` -- must stay at 60+ (item adds 3 tests, the prior baseline was 57).
  - Quick CLI sanity:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py --max-users 5 --k 5 --include-random --user-sample-seed 42 --output-dir /private/tmp/seed_sanity
    ```
    Verify the JSON has `config.user_sample_seed == 42`.
  - Commit message: `feat(eval): add random user sampling for variance studies`. Use `git add -f docs/experiments/2026-05-22_eval-slice-expansion.md` to land the DONE marker.
- **Test / verification:**
  - All three new unit tests pass.
  - Full suite stays green.
  - 5-user CLI smoke produces a JSON with `config.user_sample_seed = 42`.
  - The 2026-05-21 canonical command (without `--user-sample-seed`) still produces the same 55 evaluated_user_count for the K=10 / als_implicit row -- backward compatibility check.
- **Expected outcome:** The eval flow can now produce real variance when seeded. Decision criterion: tests pass, default behavior unchanged.
- **DONE (commit `ed888fb`):** Extended `select_evaluation_user_ids` with an optional `random_seed` keyword; default `None` returns the deterministic first-N slice (backward compatible), seeded calls return a stable random sample via `np.random.default_rng(seed).choice(..., replace=False)` and sort the result for reproducibility. Threaded `user_sample_seed` through `run_evaluation()` and added the `--user-sample-seed` CLI flag (separate from the existing `--random-seed` for the random baseline). Recorded the value in `config.user_sample_seed` of every eval report.
  - Tests: `tests/test_evaluation_runner.py` 3/3 OK -- covers default first-N, seed stability across two calls, and different-seed sample divergence. Full suite `.venv/bin/python -m unittest discover -s tests` 60/60 OK (was 57/57, +3 new tests, zero regressions).
  - CLI sanity: `--max-users 5 --user-sample-seed 42 --include-random` produced `config.user_sample_seed = 42`, `config.random_seed = 42`, `data.selected_user_count = 5` in the output JSON.
  - Backward compatibility check: omitting `--user-sample-seed` keeps the prior deterministic first-N behavior; the 2026-05-21 canonical run still reproduces exactly.
  - Decision: proceed to Item 2 (multi-seed runs at 100 users).

## 2) Multi-seed run at 100 users / holdout=1

- **Goal:** Produce variance numbers for the top-3 ranking models on the same slice size as the current canonical reference.
- **Files:**
  - `artifacts/evaluation/metrics_summary_<timestamp>.{csv,json}` (3 new files, gitignored).
  - `artifacts/evaluation/metrics_summary.{csv,json}` and `run_config.json` are overwritten by each run; this is acceptable because the timestamped copies preserve history. After the third run, `metrics_summary.{csv,json}` points to the seed=1337 run.
- **Steps:**
  - For seed in `42, 7, 1337`, run:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 100 --k 5,10,20 \
      --include-random --include-tfidf --include-content --include-semantic \
      --include-svd --include-svd-topk \
      --include-sbert-faiss --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
      --include-lightfm --lightfm-artifacts-dir artifacts/models/lightfm \
      --include-als --als-artifacts-dir artifacts/models/als \
      --user-sample-seed <seed> \
      --output-dir artifacts/evaluation
    ```
  - After each run, capture the new timestamped JSON path from the artifacts directory listing.
  - Commit message: `chore(eval): capture 100-user 3-seed variance run artifacts`. Use `git add -f` for the plan file (the artifacts themselves are gitignored; the commit lands the DONE marker only).
- **Test / verification:**
  - Three timestamped JSONs exist under `artifacts/evaluation/`, each with `config.user_sample_seed` set to 42, 7, 1337 respectively.
  - For at least 2 of the 3 runs, `top_n.als_implicit.10.evaluated_user_count` differs from the canonical 55 -- proof the slice actually shifted.
  - The DONE marker captures, for each seed: `evaluated_user_count`, `als_implicit.10.{precision,recall,ndcg,hit_rate}`, `lightfm_warp.10.{ndcg,hit_rate}`, `hybrid_content.10.{ndcg,hit_rate}`.
- **Expected outcome:** A 3-run dataset for hypothesis H1. Decision criterion: artifacts exist; the three seed-specific evaluated_user_counts are not all identical.
- **DONE (commit `a16d88a`):** Ran the canonical 9-model evaluation three times with `--user-sample-seed` in `42, 7, 1337` at `--max-users 100 --holdout-count 1`. Each run took ~5 minutes wall time. The slice genuinely shifted across seeds: `evaluated_user_count` at K=10 landed at 54 / 52 / 54 versus the deterministic-first-N reference of 55.
  - Per-seed K=10 NDCG triples for the top-5 ranking models:

    | seed | evaluated_user_count | als_implicit | lightfm_warp | hybrid_content | popularity | sbert_faiss |
    |---:|---:|---:|---:|---:|---:|---:|
    | 42 | 54 | 0.1903 | 0.0763 | 0.0241 | 0.0260 | 0.0450 |
    | 7 | 52 | 0.2141 | 0.0921 | 0.0317 | 0.0453 | 0.0192 |
    | 1337 | 54 | 0.1676 | 0.0574 | 0.0117 | 0.0117 | 0.0058 |

  - H1 confirmation: `als_implicit > lightfm_warp > hybrid_content` ordering holds in 3 of 3 seeds. ALS NDCG@10 range 0.1676 to 0.2141 (std approx 0.024, ~12% relative variation against the mean of 0.1907). LightFM NDCG@10 range 0.0574 to 0.0921. Hybrid NDCG@10 range 0.0117 to 0.0317.
  - Run ids: `artifacts/evaluation/metrics_summary_2026-05-20T19-23-55Z.{csv,json}` (seed 42), `artifacts/evaluation/metrics_summary_2026-05-20T19-29-04Z.{csv,json}` (seed 7), `artifacts/evaluation/metrics_summary_2026-05-20T19-33-46Z.{csv,json}` (seed 1337). All gitignored.
  - Decision: proceed to Item 3. ALS leadership is seed-robust on this slice size; the next axis to verify is sample size.

## 3) Single-seed run at 300 users / holdout=1

- **Goal:** Isolate the "larger sample" effect from the "larger holdout" effect that item 4 introduces.
- **Files:**
  - `artifacts/evaluation/metrics_summary_<timestamp>.{csv,json}` (1 new file, gitignored).
- **Steps:**
  - Run:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 300 --k 5,10,20 \
      --include-random --include-tfidf --include-content --include-semantic \
      --include-svd --include-svd-topk \
      --include-sbert-faiss --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
      --include-lightfm --lightfm-artifacts-dir artifacts/models/lightfm \
      --include-als --als-artifacts-dir artifacts/models/als \
      --user-sample-seed 42 \
      --output-dir artifacts/evaluation
    ```
  - Capture the new timestamped JSON path.
  - Commit message: `chore(eval): capture 300-user single-seed run artifact`.
- **Test / verification:**
  - One timestamped JSON with `config.max_users = 300`, `config.user_sample_seed = 42`, `config.holdout_count = 1`.
  - `data.evaluated_user_count` reaches at least 150 (~ 55% of 300 expected based on the 100-user rate).
  - DONE marker captures evaluated_user_count and the same per-model K=10 metrics as item 2.
- **Expected outcome:** A reference point that separates "more users" from "more holdout". Decision criterion: ordering of als > lightfm > hybrid on NDCG@10 holds, or the deviation is documented.
- **DONE (commit `af00602`):** Ran the canonical 9-model evaluation at `--max-users 300 --user-sample-seed 42 --holdout-count 1`. Evaluated user count landed at 177 of 300 selected (~59% positive-holdout rate, in line with the 55% rate at 100 users). H2 confirmed: ALS > LightFM > hybrid ordering holds, and ALS leads by an even wider margin.
  - K=10 metrics:

    | Model | NDCG@10 | HitRate@10 | Precision@10 | Recall@10 | Latency mean |
    |---|---:|---:|---:|---:|---:|
    | als_implicit | 0.2196 | 0.3616 | 0.0362 | 0.3616 | 6.7 ms |
    | lightfm_warp | 0.1173 | 0.2147 | 0.0215 | 0.2147 | 37.3 ms |
    | popularity | 0.0467 | 0.0791 | 0.0079 | 0.0791 | 36.5 ms |
    | tfidf_content | 0.0389 | 0.0621 | 0.0062 | 0.0621 | 47.5 ms |
    | hybrid_content | 0.0383 | 0.0734 | 0.0073 | 0.0734 | 1,368.1 ms |
    | sbert_faiss_content | 0.0316 | 0.0621 | 0.0062 | 0.0621 | 42.9 ms |
    | semantic_content | 0.0184 | 0.0282 | 0.0028 | 0.0282 | 75.4 ms |
    | svd_topk | 0.0120 | 0.0339 | 0.0034 | 0.0339 | 182.3 ms |
    | random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 14.1 ms |

  - Notable shifts versus the 100-user multi-seed runs (item 2): ALS NDCG@10 rises to 0.2196 (vs 0.1676 - 0.2141 band at 100 users), reflecting more signal at the larger sample. SVD top-K, previously stuck at zero at K=10, now produces a small non-zero band (NDCG 0.0120) -- a few of the 177 evaluated users had a holdout that landed inside SVD's narrow popular-favorable top-K. At 300 users `popularity` slightly edged out `hybrid_content` on NDCG (0.0467 vs 0.0383) -- a fresh ordering wrinkle worth flagging in item 5's synthesis.
  - Run id: `artifacts/evaluation/metrics_summary_2026-05-20T19-46-01Z.{csv,json}` (gitignored).
  - Decision: proceed to Item 4. ALS > LightFM > hybrid is robust on both seed and sample-size axes at holdout=1.

## 4) Multi-seed run at 300 users / holdout=3

- **Goal:** Variance band on the most realistic offline eval shape: more users, more holdout, three seeds. This is the expensive one (~30-45 min per run).
- **Files:**
  - `artifacts/evaluation/metrics_summary_<timestamp>.{csv,json}` (3 new files, gitignored).
- **Steps:**
  - For seed in `42, 7, 1337`, run:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 300 --k 5,10,20 \
      --holdout-count 3 \
      --include-random --include-tfidf --include-content --include-semantic \
      --include-svd --include-svd-topk \
      --include-sbert-faiss --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
      --include-lightfm --lightfm-artifacts-dir artifacts/models/lightfm \
      --include-als --als-artifacts-dir artifacts/models/als \
      --user-sample-seed <seed> \
      --output-dir artifacts/evaluation
    ```
  - Commit message: `chore(eval): capture 300-user holdout=3 3-seed variance run artifacts`.
- **Test / verification:**
  - Three timestamped JSONs with `config.max_users = 300`, `config.holdout_count = 3`, distinct `user_sample_seed`.
  - `recall_at_10` numbers are mechanically smaller than items 2 and 3 (denominator is 3 instead of 1); the plan acknowledges this in item 5's paragraph.
  - DONE marker captures per-seed evaluated_user_count and the same per-model K=10 metrics for the three runs.
- **Expected outcome:** A 3-run dataset for hypothesis H3. Decision criterion: artifacts exist; the holdout=3 semantics is correctly noted in the synthesis.
- **DONE (commit to be backfilled):** Ran the 9-model evaluation three times at `--max-users 300 --holdout-count 3 --user-sample-seed` in `42, 7, 1337`. Each run took ~10 minutes wall time; chained as a single background bash for ~30 minutes total. H3 confirmed: ALS > LightFM > everything else holds in 3 of 3 seeds.
  - Per-seed K=10 NDCG:

    | seed | evaluated_user_count | als_implicit | lightfm_warp | hybrid_content | popularity | sbert_faiss | tfidf_content | svd_topk |
    |---:|---:|---:|---:|---:|---:|---:|---:|---:|
    | 42 | 259 | 0.2731 | 0.1447 | 0.0373 | 0.0442 | 0.0290 | 0.0418 | 0.0160 |
    | 7 | 245 | 0.2354 | 0.1087 | 0.0372 | 0.0464 | 0.0176 | 0.0371 | 0.0120 |
    | 1337 | 265 | 0.2142 | 0.1176 | 0.0231 | 0.0341 | 0.0180 | 0.0262 | 0.0213 |

  - Per-seed K=10 Recall (denominator is 3 holdout items per user, capped at 1.0):

    | seed | als_implicit | lightfm_warp | hybrid_content |
    |---:|---:|---:|---:|
    | 42 | 0.3861 | 0.2368 | 0.0611 |
    | 7 | 0.3156 | 0.1884 | 0.0565 |
    | 1337 | 0.2956 | 0.1899 | 0.0346 |

  - **Knife-edge finding:** `popularity` (NDCG 0.0341-0.0464) and `tfidf_content` (0.0262-0.0418) both beat `hybrid_content` (0.0231-0.0373) on NDCG@10 in 3 of 3 holdout=3 seeds. This contradicts the impression from the 2026-05-21 single-run report where hybrid was the third-best ranker. The hybrid recommender's third-place ranking is therefore **not robust** at the holdout=3 shape -- flag this in item 5.
  - **Recall semantics correction:** The plan's hypothesis text predicted recall numbers would be "mechanically smaller" at holdout=3. The opposite happened -- with 3 chances per user to land a hit in top-10 (vs 1), recall went up (ALS 0.36 -> 0.39 at seed 42). The recall metric is still in [0, 1] but the user-level distribution shifts upward. Item 5 will state this correctly.
  - Run ids: `artifacts/evaluation/metrics_summary_2026-05-20T19-57-47Z.{csv,json}` (seed 42), `metrics_summary_2026-05-20T20-07-31Z.{csv,json}` (seed 7), `metrics_summary_2026-05-20T20-26-25Z.{csv,json}` (seed 1337). All gitignored.
  - Decision: proceed to Item 5 (synthesis). All three hypotheses (H1, H2, H3) confirmed for ALS leadership; one new finding -- hybrid's #3 spot is unstable -- to surface in the report.

## 5) Synthesis -- "Variance Bounds" subsection in the evaluation report

- **Goal:** Translate the seven runs from items 2-4 into a readable summary block in `docs/08_evaluation_results_report.md`, then update Conclusions and Caveats to match.
- **Files:**
  - `docs/08_evaluation_results_report.md`: insert a new top-level subsection `## Variance Bounds (multi-seed slice studies)` between "Why SVD top-K stays at zero hits" and "Conclusions". Three sub-tables:
    - **Table A -- 100-user / holdout=1, 3 seeds:** for each of `als_implicit`, `lightfm_warp`, `hybrid_content`, `popularity`, `sbert_faiss_content`, columns are `NDCG@10 mean ± std`, `hit_rate@10 mean ± std`, `latency_mean_ms mean ± std`. One short paragraph after the table noting which models' orderings stay stable across seeds.
    - **Table B -- 300-user / holdout=1, single seed (42):** one row per model, columns `NDCG@10`, `hit_rate@10`, `latency_mean_ms`. One short paragraph comparing to Table A.
    - **Table C -- 300-user / holdout=3, 3 seeds:** same structure as Table A. One short paragraph noting the recall denominator change and which orderings still hold.
  - Update the existing Conclusions block:
    - Soften "the clear winner on every relevance metric" to something like "leads the canonical 100-user latest-1 slice and remains on top in <X> of <Y> multi-seed runs."
    - Add a sentence: "Audit: `docs/experiments/2026-05-22_eval-slice-expansion.md`."
  - Update Caveats: add a single line `Holdout = 3 expansion changes the recall denominator; recall numbers across holdout=1 and holdout=3 runs are not directly comparable.`
  - This plan file: fill DONE marker referencing the canonical synthesis commit.
- **Steps:**
  - Pull the seven JSON files' per-model `top_n.{model}.10.{ndcg_at_k, hit_rate_at_k}` and `latency.{model}.mean_ms` into a small Python one-liner to compute mean/std (do not commit the helper script; this is one-off arithmetic).
  - Write the three tables and the three short paragraphs.
  - Update Conclusions and Caveats per above.
  - Commit message: `docs(eval): add variance bounds subsection from multi-seed slice studies`. Use `git add -f` for both the report and this plan file.
- **Test / verification:**
  - The new subsection exists with three tables filled from real numbers.
  - For each of the three top-3 models, mean ± std appears in Tables A and C.
  - The Conclusions block's "clear winner" language is replaced with the variance-aware framing.
  - No code is touched in this item.
  - `.venv/bin/python -m unittest discover -s tests` still green (unaffected -- doc-only).
- **Expected outcome:** A defensible variance picture readable in 60 seconds. Decision criterion: the three tables agree with the seven run JSONs row-by-row.
- **DONE / DROPPED:**

## Deferred / Future (out of this plan)

- **Leave-one-out artifact retraining (leakage fix).** Both ALS and LightFM artifacts were trained on the full rating matrix including the holdout interactions, so the absolute NDCG numbers are inflated. A tighter offline evaluation would train per-fold artifacts that exclude the holdout. That is multi-hour compute and lives in a separate plan once variance bounds settle whether retraining is worth the cost.
- **Per-model hyperparameter sweeps.** LightFM `no_components x loss x epochs`; ALS `factors x regularization x alpha`. Plan once the variance work establishes whether the current single-point hyperparameters are near the Pareto.
- **Cold-start segmentation.** Metrics broken down by user history size (e.g., `<10`, `10-50`, `50-200`, `>200` train interactions). Natural follow-on after variance bounds are known.
- **UI explainability (Roadmap Priority 5).** Streamlit "why this movie?" feature on the Content-Based and Watch History pages. Separate plan, independent of eval track.
- **LightGCN / SASRec (Roadmap Priority 6).** Modern graph / sequential models. Plan once classical CF variance bounds are established.
- **SVD top-K ranking improvement.** Documented as known algorithmic limitation in `docs/08_evaluation_results_report.md`; a future plan can explore predicted_score blended with log-popularity, or a learning-to-rank reranker.

## Critical Files (Reference)

- `src/evaluation_runner.py:86-101` -- `select_evaluation_user_ids`; item 1 adds the optional `random_seed` parameter.
- `src/evaluation_runner.py:559-610` -- `run_evaluation` orchestration; item 1 forwards `user_sample_seed`.
- `scripts/evaluate_baselines.py` -- argparse + main; item 1 adds the new CLI flag.
- `tests/test_evaluation_runner.py` -- new file in item 1 (3 unit tests).
- `src/evaluation.py:417,430` -- existing `np.random.default_rng(seed).choice(...)` usage in `random_recommendations`; the new code mirrors this shape.
- `artifacts/evaluation/metrics_summary_2026-05-20T18-47-21Z.{csv,json}` -- 2026-05-21 canonical reference; items 2-4 sit alongside it.
- `artifacts/models/{als,lightfm}/` and `artifacts/indexes/sbert_faiss/` -- pre-trained artifacts; not modified.
- `docs/08_evaluation_results_report.md` -- item 5 adds the Variance Bounds subsection.
- `docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md` -- prior plan; this plan's motivation references its DONE markers.

## End-to-End Verification Sequence

1. Item 1 complete: `tests.test_evaluation_runner` passes 3/3; full suite stays >=60; a 5-user CLI smoke shows `config.user_sample_seed = 42` in the output JSON. DONE marker carries a real commit hash.
2. Item 2 complete: three timestamped JSONs with seeds 42, 7, 1337; at least 2 of 3 evaluated_user_counts differ from 55. DONE marker carries a real commit hash and the per-seed K=10 metric triples.
3. Item 3 complete: one timestamped JSON with `max_users=300`, `evaluated_user_count >= 150`. DONE marker carries a real commit hash and the K=10 metric triple.
4. Item 4 complete: three timestamped JSONs with `max_users=300`, `holdout_count=3`, distinct seeds. DONE marker carries a real commit hash and the per-seed K=10 metric triples.
5. Item 5 complete: `docs/08_evaluation_results_report.md` has the "Variance Bounds" subsection with three filled tables; Conclusions and Caveats reworded. DONE marker carries the synthesis commit hash.
6. After every commit, record the real commit hash in this plan file; CLAUDE.md section 7 forbids leaving temporary hash placeholders in place.

## Execution Notes

- This file is the contract; CLAUDE.md and AGENTS.md in the repo root are the project-wide rules.
- `/docs` is gitignored, but tracked files under `docs/` (including this plan) still need `git add -f` on commits.
- Items are sequential: item 1 is a hard prerequisite for items 2-4 (without the new flag, multi-seed runs produce no variance); items 2-4 are reads-only against pre-trained artifacts; item 5 synthesizes from items 2-4 outputs.
- One implementation commit per item, plus a follow-up `docs(experiments): record ...` commit that fills the DONE marker with the real hash. This matches the 2026-05-21 cadence.
- The artifact files under `artifacts/evaluation/metrics_summary_*` are gitignored; the commits land only the plan file and (for item 1 and item 5) source / doc changes.
