- **Date:** 2026-05-23
- **Note:** Filename and Date field reflect narrative ordering of the experiment arc; all DONE commits landed on 2026-05-21. CLAUDE.md section 7 strict reading is "filename = authoring date" -- this drift is intentional and recorded here rather than rewritten, so commit hashes and cross-references in other docs stay valid.
- **Topic:** Cold-start segmentation of the 9-model offline eval -- per-user-history buckets at the 300u / holdout=3 shape
- **Motivation:** The 2026-05-22 variance work closed with ALS leading every studied shape (NDCG@10 0.1907 +/- 0.0232 at 100u/h=1, 0.2196 at 300u/h=1, 0.2409 +/- 0.0299 at 300u/h=3). All those numbers are aggregates over the entire eval slice. The Roadmap's Priority 5 explicitly calls for cold-start performance reporting; the 2026-05-22 plan logged this as the natural follow-on. The open question: does ALS leadership hold for the cold-start case where the user has only a handful of training interactions? Classical CF is known to struggle on cold-start while content baselines (TF-IDF, SBERT+FAISS, semantic-LSA) are designed for it. The variance work also surfaced that popularity / tfidf_content / hybrid_content tossup the third-rank spot -- segmented metrics should reveal whether the three alternate winners across user types.
- **Hypothesis:**
  - **H1 (ALS coldness):** In the `cold_0_10` bucket (users with fewer than 10 training interactions), ALS NDCG@10 drops by at least 50% relative to the aggregate; in the `heavy_200_plus` bucket, ALS NDCG@10 is at least 1.5x the aggregate. The overall ALS > LightFM ordering still holds in every non-empty bucket in 3 of 3 seeds.
  - **H2 (content lift in cold):** At least one content baseline (`sbert_faiss_content`, `semantic_content`, or `tfidf_content`) beats `lightfm_warp` on NDCG@10 in the `cold_0_10` bucket in at least 1 of 3 seeds. We do not predict the cold-start winner outright; we predict the gap narrows enough that the ordering becomes seed-dependent there.
  - Refuting either hypothesis is informative -- it would say ALS is robust across user types and content baselines do not recover the cold case in this dataset.
- **Preconditions:**
  - All five items of `docs/experiments/2026-05-22_eval-slice-expansion.md` are DONE on `main` through commit `e507f5a`.
  - `artifacts/evaluation/metrics_summary_2026-05-20T{19-57-47,20-07-31,20-26-25}Z.{csv,json}` are the 300u/h=3 multi-seed reference runs (without segmentation). Items 2-3 of this plan rerun the same shape with segmentation; aggregate numbers should match those references row-for-row at the same seed (sanity check).
  - `artifacts/models/{als,lightfm}/`, `artifacts/indexes/sbert_faiss/`, `cleaned_data/svd_trained_model.pkl` present locally with the same content as the 2026-05-22 runs. No artifact retraining in this plan.
  - The `--user-sample-seed` plumbing from commit `ed888fb` is on `main`. The new segmentation code in item 1 sits next to it.
  - CLAUDE.md section 7 governs DONE marker structure; do not fill `<hash>` placeholders before the corresponding commit lands.

## 1) Implement segmentation in the eval framework

- **Goal:** Add the post-hoc segmentation path. The default segments are user-history buckets `[("cold_0_10", 0, 10), ("warm_10_50", 10, 50), ("regular_50_200", 50, 200), ("heavy_200_plus", 200, None)]` with `lower <= n < upper` semantics; `None` upper means open-ended. The behavior is gated by a CLI flag so all prior runs reproduce byte-for-byte.
- **Files:**
  - `src/evaluation.py:22` (next to `temporal_train_test_split`): add `segment_users_by_history(train, segments, user_col="userId") -> dict[str, set[int]]`. Returns a mapping from segment name to the set of userIds whose train-interaction count falls in `[lower, upper)`.
  - `src/evaluation.py:164-236` (`top_n_metrics`): no change. Segmentation is a wrapper that calls it multiple times.
  - `src/evaluation_runner.py` (`build_metric_report` and `evaluate_baseline`): accept an optional `segment_user_ids` keyword (default `None`). When provided, after computing the existing aggregate metrics for each `k`, iterate the segment map, filter `recommendations[recommendations["userId"].isin(seg_ids)]` and `holdout[holdout["userId"].isin(seg_ids)]`, call `top_n_metrics(...)` again, and attach the result as `metrics[str(k)]["segments"][seg_name] = ...`. Skip a segment that has zero users with positive holdout (no metrics entry rather than zeros).
  - `src/evaluation_runner.py` (`run_evaluation`): accept `segment_by_history=False` and `segment_bounds=None`; when enabled, compute the segment map once from `train`, pass it to every `evaluate_baseline` call. Record `segment_by_history`, `segment_bounds`, and the resolved segment definitions in `report["config"]` so the JSON is self-describing.
  - `scripts/evaluate_baselines.py`: add `--segment-by-history` (`store_true`) and `--segment-bounds` (comma-separated integers, e.g. `0,10,50,200`; default unset means use the 4-bucket default). Pass into `run_evaluation()`.
  - `tests/test_evaluation_runner.py`: add 3 tests.
    - `test_segment_users_by_history_default_buckets`: a 5-user train with counts `[3, 12, 80, 250, 500]` produces 1 user in `cold_0_10`, 1 in `warm_10_50`, 1 in `regular_50_200`, 2 in `heavy_200_plus`.
    - `test_segment_users_by_history_open_upper`: user count = 200 lands in `heavy_200_plus` (200 included because upper is None).
    - `test_top_n_metrics_segments_match_aggregate_when_single_segment_covers_all`: when only one segment covers all users, the per-segment metrics dict equals the aggregate metrics dict for the same K.
- **Reuse:** `top_n_metrics` (unchanged). `pd.DataFrame.groupby(user_col).size()` for the per-user train count. The existing recommendations / holdout DataFrames inside `evaluate_baseline`.
- **Steps:**
  - Implement the helper, wire `build_metric_report`, `evaluate_baseline`, `run_evaluation`.
  - Implement the CLI flags + forwarding.
  - Add the three unit tests.
  - `.venv/bin/python -m unittest tests.test_evaluation_runner` -- must pass.
  - `.venv/bin/python -m unittest discover -s tests` -- must stay at 63 (60 + 3 new).
  - 5-user CLI smoke: `.venv/bin/python scripts/evaluate_baselines.py --max-users 5 --k 5 --include-random --segment-by-history --output-dir /private/tmp/seg_sanity`. Verify the JSON has `config.segment_by_history=true`, `config.segment_bounds` populated, and `top_n.random.5.segments` exists.
  - Commit message: `feat(eval): add user-history segmentation to offline eval`. `git add -f docs/experiments/2026-05-23_cold-start-segmentation.md` to land the DONE marker.
- **Test / verification:**
  - 3 unit tests pass; full suite stays at 63.
  - 5-user smoke produces a JSON with `config.segment_by_history=true` and segment blocks attached.
  - Backward compatibility: omitting the flag produces a JSON without `segments` blocks; the 2026-05-22 canonical run still reproduces.
- **Expected outcome:** Segmentation is available behind a flag and exercised by tests. Decision criterion: tests pass and the smoke JSON has the expected new keys.
- **DONE (commit `9454cbd`):** Added `segment_users_by_history` to `src/evaluation.py`, extended `build_metric_report` and `evaluate_baseline` with an optional `segment_user_ids` kwarg, wired `--segment-by-history` and `--segment-bounds` CLI flags, and threaded `segment_by_history` / `segment_bounds` through `run_evaluation()`. The segment map is computed once from `train` and reused for every `evaluate_baseline` call. The config block records `segment_by_history`, `segment_bounds`, and the resolved `segment_definitions`.
  - Tests: 4 new unit tests under `tests/test_evaluation_runner.py` (default bucket semantics, open upper boundary, segment-equals-aggregate when one segment covers all users, segment-skipped-when-no-positive-holdout). Full suite 64/64 OK (was 60/60).
  - CLI sanity: `--max-users 10 --include-random --segment-by-history` produced `config.segment_by_history=true`, `config.segment_definitions` populated with the 4 default buckets, and `top_n.random.5.segments` with `warm_10_50` (3 users) and `regular_50_200` (2 users) buckets populated (cold and heavy empty for this small deterministic slice -- expected).
  - Backward compat: omitting the flag produced `segment_by_history=false`, `segment_definitions=null`, no `segments` block in any model's K block.
  - Decision: proceed to Item 2.

## 2) Segmented run at 300 users / holdout=3 / single seed 42

- **Goal:** Verify the bucket distribution is non-degenerate and capture the first segmented dataset. Gate before going multi-seed.
- **Files:**
  - `artifacts/evaluation/metrics_summary_<timestamp>.{csv,json}` (gitignored).
- **Steps:**
  - Run:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 300 --k 5,10,20 \
      --holdout-count 3 \
      --include-random --include-tfidf --include-content --include-semantic \
      --include-svd --include-svd-topk \
      --include-sbert-faiss --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
      --include-lightfm --lightfm-artifacts-dir artifacts/models/lightfm \
      --include-als --als-artifacts-dir artifacts/models/als \
      --user-sample-seed 42 \
      --segment-by-history \
      --output-dir artifacts/evaluation
    ```
  - Inspect the JSON: print the 4 segment buckets' `evaluated_user_count` for `als_implicit` at K=10. Gate: at least 3 of the 4 buckets have >= 5 evaluated users. If any bucket is below threshold, adjust `--segment-bounds` (e.g., merge the low bucket into a neighbor) and re-run before going to item 3.
  - Sanity check: the aggregate K=10 numbers in this run should be very close to the 2026-05-22 seed-42 run (`metrics_summary_2026-05-20T19-57-47Z.json`) -- ALS NDCG@10 around 0.27. They will not match exactly because random recommendations vary, but the deterministic models should match to four decimals.
  - Commit message: `chore(eval): capture 300u/h=3/seed=42 segmented run artifact`.
- **Test / verification:**
  - One timestamped JSON with `config.segment_by_history=true`, `config.user_sample_seed=42`, `config.max_users=300`, `config.holdout_count=3`.
  - At least 3 of 4 buckets have `evaluated_user_count >= 5` at K=10 for ALS.
  - ALS aggregate NDCG@10 within 0.005 of the 2026-05-22 seed-42 reference (deterministic model, same artifact).
- **Expected outcome:** A first segmented dataset that validates bucket sizes. Decision criterion: pass the bucket-size gate; aggregate match holds.
- **DONE (commit `d066101`):** Ran `--max-users 300 --holdout-count 3 --user-sample-seed 42 --segment-by-history`. All 4 default buckets cleared the >= 5 user gate. Aggregate metrics matched the 2026-05-22 seed-42 reference to full precision (segmentation does not alter the aggregate path).
  - Bucket sizes at K=10:

    | Bucket | Evaluated users |
    |---|---:|
    | cold_0_10 | 41 |
    | warm_10_50 | 117 |
    | regular_50_200 | 70 |
    | heavy_200_plus | 31 |
    | **Total** | **259** (matches aggregate `evaluated_user_count`) |

  - Per-segment NDCG@10 (most informative table of the run):

    | Model | cold_0_10 (n=41) | warm_10_50 (n=117) | regular_50_200 (n=70) | heavy_200_plus (n=31) |
    |---|---:|---:|---:|---:|
    | als_implicit | **0.5149** | 0.3177 | 0.1600 | 0.0406 |
    | lightfm_warp | 0.3071 | 0.1655 | 0.0667 | 0.0280 |
    | hybrid_content | 0.0676 | 0.0412 | 0.0254 | 0.0099 |
    | popularity | 0.1360 | 0.0364 | 0.0152 | 0.0177 |
    | sbert_faiss_content | 0.0531 | 0.0287 | 0.0116 | 0.0380 |
    | tfidf_content | 0.0848 | 0.0339 | 0.0484 | 0.0000 |

  - **Both H1 and H2 are refuted (informatively).** H1 predicted ALS would drop by >= 50% in cold_0_10 relative to aggregate. Instead it rises ~90% (0.5149 vs 0.2731 aggregate). H2 predicted at least one content baseline would beat LightFM in cold_0_10. LightFM cold_0_10 (0.3071) beats every content baseline (best content is popularity at 0.1360). The most likely interpretation: the **training-set leakage** documented in the 2026-05-22 plan's Caveats amplifies in cold-start, because cold users' "true" signal is overwhelmingly in the holdout and both ALS and LightFM artifacts were trained on the full matrix including those holdouts. This finding strengthens the case for the deferred leave-one-out retraining experiment.
  - Aggregate sanity: ALS K=10 NDCG = 0.2731336481, hit_rate = 0.5598, precision = 0.0776, recall = 0.3861, evaluated_user_count = 259 -- exact match (to 10 decimals) to `metrics_summary_2026-05-20T19-57-47Z.json`.
  - Run id: `artifacts/evaluation/metrics_summary_2026-05-20T21-09-15Z.{csv,json}` (gitignored).
  - Decision: proceed to Item 3 (multi-seed runs at the same shape).

## 3) Multi-seed segmented runs at 300 users / holdout=3

- **Goal:** Variance band per segment across seeds `42, 7, 1337` (item 2's seed=42 run may be re-used if the bucket sizes pass the gate).
- **Files:**
  - Three new `artifacts/evaluation/metrics_summary_<timestamp>.{csv,json}` (gitignored). Item 2's run is one of the three.
- **Steps:**
  - For seed in `7, 1337` (item 2 already covered seed 42), run the same command with `--user-sample-seed <seed>`.
  - Total wall time approx 25-30 minutes (2 runs of ~12 minutes each).
  - Commit message: `chore(eval): capture 300u/h=3 segmented multi-seed run artifacts`.
- **Test / verification:**
  - Three timestamped JSONs cover seeds `42, 7, 1337` with `config.segment_by_history=true`.
  - Per-segment `evaluated_user_count` varies across seeds (proof the slice shifted across seeds within the segment).
  - DONE marker captures, per seed: bucket sizes and the per-bucket ALS / LightFM / popularity / sbert_faiss / tfidf NDCG@10.
- **Expected outcome:** A 3-run segmented dataset for the synthesis. Decision criterion: three artifacts exist; bucket counts non-degenerate in each.
- **DONE (commit `e532888`):** Ran the remaining two seeds (7 and 1337) at the same `--max-users 300 --holdout-count 3 --segment-by-history` shape. Combined with item 2's seed-42 artifact, the 3-seed dataset is complete.
  - Bucket sizes are stable across seeds (counts shift modestly):

    | seed | cold_0_10 | warm_10_50 | regular_50_200 | heavy_200_plus |
    |---:|---:|---:|---:|---:|
    | 42 | 41 | 117 | 70 | 31 |
    | 7 | 45 | 103 | 67 | 30 |
    | 1337 | 46 | 113 | 71 | 35 |

  - Per-segment NDCG@10 per seed (ALS / LightFM):

    | seed | ALS cold | ALS warm | ALS regular | ALS heavy | LightFM cold | LightFM warm | LightFM regular | LightFM heavy |
    |---:|---:|---:|---:|---:|---:|---:|---:|---:|
    | 42 | 0.5149 | 0.3177 | 0.1600 | 0.0406 | 0.3071 | 0.1655 | 0.0667 | 0.0280 |
    | 7 | 0.3763 | 0.3176 | 0.0717 | 0.1080 | 0.2315 | 0.1041 | 0.0634 | 0.0413 |
    | 1337 | 0.4920 | 0.2106 | 0.1205 | 0.0504 | 0.2717 | 0.1289 | 0.0385 | 0.0392 |

  - Key cross-seed orderings:
    - `als_implicit > lightfm_warp` holds in 3 of 3 seeds in every segment (12/12 segment-seed cells).
    - `lightfm_warp > popularity` holds in 3 of 3 seeds in cold and warm; flips in regular and heavy where popularity occasionally edges close.
    - Both ALS and LightFM peak in `cold_0_10` and trough in `heavy_200_plus` -- the opposite of what classical CF wisdom predicts. The 2026-05-22 leakage caveat is the most likely driver and is the headline of item 5's narrative.

  - Run ids: `metrics_summary_2026-05-20T21-09-15Z` (seed 42, from item 2), `metrics_summary_2026-05-20T21-23-38Z` (seed 7), `metrics_summary_2026-05-20T21-37-33Z` (seed 1337). All gitignored.
  - Decision: proceed to Item 5 (synthesis). The dataset cleanly supports a "leakage amplified in cold-start" finding and a model-by-segment leaderboard.

## 4) Synthesis -- "Cold-start segmentation" subsection in the evaluation report

- **Goal:** Translate the three segmented runs into a readable summary block in `docs/08_evaluation_results_report.md`, then update Conclusions and Caveats to match.
- **Files:**
  - `docs/08_evaluation_results_report.md`: insert a new subsection `## Cold-start segmentation (user-history buckets)` between "Variance Bounds (multi-seed slice studies)" and "Conclusions". One sub-table per segment with the same column structure as Variance Bounds tables (top-5 models, NDCG@10 mean ± std, hit_rate@10 mean ± std, evaluated_user_count). Two short paragraphs: (a) which model wins each segment and is the win seed-robust; (b) does ALS leadership hold uniformly or break in cold-start.
  - Update Conclusions:
    - If ALS wins every segment in 3/3 seeds, tighten "leads on every studied shape" to "leads on every studied shape and on every user-history segment".
    - If ALS loses any segment, replace the universal-leader sentence with a per-segment qualifier and name the actual cold-start winner.
  - Update Caveats: add a single line noting that per-segment evaluated_user_count is small (typically 5-100 per bucket per seed), so segment-level relative orderings are directionally informative but should not be reported as definitive.
  - This plan file: fill DONE marker referencing the canonical synthesis commit.
- **Steps:**
  - Pull per-segment NDCG@10 and hit_rate@10 from the three JSONs, compute mean / std across seeds.
  - Write the four segment tables and the two interpretation paragraphs.
  - Update Conclusions and Caveats per above.
  - Commit message: `docs(eval): add cold-start segmentation subsection`. `git add -f` both files.
- **Test / verification:**
  - New subsection exists with four tables filled from real numbers.
  - Conclusions block updated; the universal claim is either tightened or qualified.
  - No code is touched in this item.
  - `.venv/bin/python -m unittest discover -s tests` stays at 63 (doc-only).
- **Expected outcome:** A readable cold-start picture. Decision criterion: the four tables agree with the three run JSONs row-by-row; Conclusions reflects the segmented reality.
- **DONE (commit `f1142b3`):** Added the "Cold-start segmentation (user-history buckets)" subsection to `docs/08_evaluation_results_report.md` with four per-bucket tables (mean +/- std across seeds 42, 7, 1337). Conclusions reworded: ALS leadership tightened from "every shape" to "every shape and every segment" (12 of 12 segment-seed cells); third-rank framing replaced with the segment-dependent ordering (popularity / popularity / tfidf / sbert across cold / warm / regular / heavy). Caveats gained per-bucket sample-size note (44/111/69/32 per seed) and a stronger leakage framing pointing to the cold-vs-heavy inversion as visible evidence.
  - Cross-table NDCG@10 summary (mean +/- std):

    | Model | cold | warm | regular | heavy |
    |---|---:|---:|---:|---:|
    | als_implicit | 0.4610 +/- 0.0743 | 0.2820 +/- 0.0618 | 0.1174 +/- 0.0442 | 0.0663 +/- 0.0364 |
    | lightfm_warp | 0.2701 +/- 0.0378 | 0.1328 +/- 0.0308 | 0.0562 +/- 0.0154 | 0.0362 +/- 0.0072 |
    | popularity | 0.0876 +/- 0.0423 | 0.0415 +/- 0.0096 | 0.0247 +/- 0.0104 | 0.0184 +/- 0.0132 |
    | hybrid_content | 0.0585 +/- 0.0079 | 0.0365 +/- 0.0102 | 0.0201 +/- 0.0094 | 0.0103 +/- 0.0065 |
    | tfidf_content | 0.0562 +/- 0.0266 | 0.0318 +/- 0.0055 | 0.0370 +/- 0.0179 | 0.0140 +/- 0.0136 |
    | sbert_faiss_content | 0.0394 +/- 0.0148 | 0.0184 +/- 0.0095 | 0.0140 +/- 0.0057 | 0.0244 +/- 0.0143 |

  - Tests stay 64/64 (doc-only change).
  - Decision: shipped. All four items of this plan are now DONE. The leave-one-out retraining experiment is now the most natural follow-on -- the cold-start inversion makes it concretely worth doing rather than a hypothetical "would be more rigorous" exercise.

## Deferred / Future (out of this plan)

- **Leave-one-out artifact retraining.** Still deferred. The absolute-numbers caveat from the 2026-05-22 plan persists; the segmented orderings remain defensible because both ALS and LightFM share the leakage symmetrically.
- **LightFM / ALS hyperparameter sweeps.** Separate plan; segmentation can be re-run on swept artifacts later.
- **Item-cold-start (never-rated movies).** Only the **user** side is segmented here; an item-side cold-start study (recall on never-rated movies, possibly weighted by movie release-year recency) is a separate question.
- **UI explainability (Priority 5 carry-over).** Different track; the segmentation insight could feed back into a per-segment recommendation explanation in a future UI plan.
- **LightGCN / SASRec (Priority 6).** Modern graph / sequential models; useful for cold-start in principle, but heavy lift and out of this plan.
- **SVD top-K ranking improvement.** Already documented as expected limitation.

## Critical Files (Reference)

- `src/evaluation.py:22` -- `temporal_train_test_split` neighbor; item 1 adds `segment_users_by_history` next to it.
- `src/evaluation.py:164-236` -- `top_n_metrics`; unchanged, called per segment via filtered DataFrames.
- `src/evaluation_runner.py` `build_metric_report` and `evaluate_baseline` -- item 1 adds the optional `segment_user_ids` keyword and the per-segment loop.
- `src/evaluation_runner.py` `run_evaluation` -- item 1 adds `segment_by_history` and `segment_bounds` parameters; computes the segment map once.
- `scripts/evaluate_baselines.py` -- item 1 adds `--segment-by-history` and `--segment-bounds` CLI flags.
- `tests/test_evaluation_runner.py` -- item 1 adds 3 unit tests.
- `artifacts/evaluation/metrics_summary_2026-05-20T{19-57-47,20-07-31,20-26-25}Z.{csv,json}` -- 2026-05-22 multi-seed reference for the aggregate sanity check in item 2.
- `docs/08_evaluation_results_report.md` -- item 4 adds the "Cold-start segmentation" subsection.
- `docs/experiments/2026-05-22_eval-slice-expansion.md` -- prior plan; this plan's motivation references its Deferred section.

## End-to-End Verification Sequence

1. Item 1 complete: 3 new unit tests pass; full suite stays at 63; a 5-user CLI smoke produces a JSON with `config.segment_by_history=true` and segment blocks under each model. DONE marker carries a real commit hash.
2. Item 2 complete: one timestamped JSON at `--max-users 300 --holdout-count 3 --user-sample-seed 42 --segment-by-history`. At least 3 of 4 buckets have `evaluated_user_count >= 5`. ALS aggregate NDCG@10 matches the 2026-05-22 seed-42 reference to four decimals. DONE marker carries a real commit hash, bucket sizes, and the per-bucket NDCG@10 triple for ALS / LightFM / hybrid / popularity / sbert.
3. Item 3 complete: three timestamped JSONs with distinct seeds; per-segment evaluated_user_counts vary; DONE marker carries the per-seed-per-bucket data needed for item 4's tables.
4. Item 4 complete: `docs/08_evaluation_results_report.md` carries the "Cold-start segmentation" subsection with four tables; Conclusions reflects the segmented finding; Caveats notes the per-bucket sample-size limit. DONE marker carries the synthesis commit hash.
5. After every commit, record the real hash in this plan file; CLAUDE.md section 7 forbids leaving placeholders.

## Execution Notes

- `/docs` is gitignored; tracked files including this plan need `git add -f` on commits.
- Items are sequential: item 1 is a hard prerequisite for items 2-4. Item 2 acts as a bucket-size gate before going multi-seed in item 3. Item 4 synthesizes from items 2-3.
- One implementation commit per item plus a follow-up `docs(experiments): record ...` commit that fills the DONE marker with the real hash. Matches the cadence of the 2026-05-21 and 2026-05-22 plans.
- Artifact files under `artifacts/evaluation/metrics_summary_*` stay gitignored.
