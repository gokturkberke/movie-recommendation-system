- **Date:** 2026-05-24
- **Topic:** Leave-one-out leakage fix for ALS / LightFM artifacts -- retrain with eval holdouts excluded and quantify the cold-start advantage that was attributable to leakage
- **Motivation:** The 2026-05-23 cold-start segmentation exposed a direct visible signature of the training-set leakage that earlier plans had only flagged in Caveats. Both `als_implicit` and `lightfm_warp` scored monotonically highest on the bucket with the **least** training behavior and lowest on the bucket with the **most**. ALS NDCG@10: cold 0.4610 +/- 0.0743 -> warm 0.2820 -> regular 0.1174 -> heavy 0.0663. LightFM NDCG@10: 0.2701 -> 0.1328 -> 0.0562 -> 0.0362. This inverts classical CF wisdom; the most plausible driver is that both artifacts were trained on the full rating matrix including every eval user's holdout (positive-threshold >= 4.0) interaction. For cold users that holdout IS most of their history, so the artifact has memorised exactly what the eval will hold out. Every prior plan's Conclusions section carries a leakage caveat; this plan closes that loop by retraining with the holdouts excluded and quantifying the delta.
- **Hypothesis:**
  - **H1 (leakage attenuates cold inflation):** After LOO retraining, ALS NDCG@10 in `cold_0_10` drops by at least 30% relative to the leaked baseline (0.4610 -> at most 0.323). Same direction for LightFM.
  - **H2 (cold > heavy inversion partially or fully resolved):** Either ALS LOO NDCG@10 in `cold_0_10` falls below `warm_10_50` (textbook-aligned), or the gap narrows substantially (cold-to-heavy ratio drops from ~7x leaked to <=2x LOO).
  - **H3 (ALS > LightFM ordering preserved):** ALS still leads LightFM on aggregate NDCG@10 in 3 of 3 seeds post-LOO, because the leakage is symmetric across both classical CF models.
  - Refutation: if H1 fails (cold stays inflated), the leakage hypothesis is wrong and something else is driving the inversion. If H3 fails, the leadership conclusion in the report becomes a leaked artifact and must be rewritten.
- **Preconditions:**
  - All four items of `docs/experiments/2026-05-23_cold-start-segmentation.md` are DONE on `main` through commit `9327536`.
  - `artifacts/models/lightfm/` (commit `65de01d`) and `artifacts/models/als/` (commit `c956ea5`) are present as the leaked baselines; they stay on disk and are NOT modified. New LOO artifacts go to new directories.
  - `cleaned_data/ratings_clean.csv` (~33.7M rows), `cleaned_data/movies_clean.csv` (79,477 rows), `cleaned_data/svd_trained_model.pkl` present locally.
  - `--user-sample-seed` plumbing (commit `ed888fb`) and `--segment-by-history` plumbing (commit `9454cbd`) are on `main`.
  - `artifacts/holdouts/` directory may or may not exist; the extraction script in item 2 creates it.
  - CLAUDE.md section 7 governs DONE markers; do not fill `<hash>` placeholders before the commit lands.

Corresponding audit item: `docs/08_evaluation_results_report.md` Caveats block (training-set leakage line) and the "Cold-start segmentation" subsection's interpretation paragraph.

## 1) Add `exclude_pairs` plumbing to matrix builders + train script CLI flags

- **Goal:** Make holdout exclusion a first-class capability of the LightFM and ALS training paths. Default behavior unchanged when no exclusion is given. CLI accepts a CSV path.
- **Files:**
  - `src/experimental/lightfm_recommender.py:37-67`: add `exclude_pairs=None` keyword to `build_interaction_matrix`. After the `rating >= positive_threshold` filter, drop rows where `(userId, movieId)` is in the exclusion set. Use `pd.DataFrame.set_index([...]).index.isin(...)` for vectorized filtering.
  - `src/experimental/als_recommender.py:41-71`: same parameter and same filtering pattern in `build_confidence_matrix`.
  - `scripts/train_lightfm_model.py`: add `--exclude-holdout-pairs <path>` argparse argument. When provided, read the CSV with `pd.read_csv(path)`, build a set of `(userId, movieId)` tuples from its rows, and pass to the builder. The CSV must have `userId,movieId` columns.
  - `scripts/train_als_model.py`: same flag and same loading pattern.
  - Metadata (saved JSON in each artifact dir): if exclusion is in effect, record `excluded_pair_count` and `exclude_pairs_path` next to the existing fields. When exclusion is `None`, omit these keys (backward compatible).
  - `tests/test_lightfm_recommender.py`: add `test_build_interaction_matrix_excludes_specified_pairs` -- a 4-rating fixture for 2 users x 2 movies; pass `exclude_pairs={(10, 1)}`; assert that the resulting matrix has nnz = 3 (not 4) and `user_index` / `item_index` do not contain orphaned rows. Add `test_build_interaction_matrix_none_exclude_is_no_op` -- same fixture without exclude_pairs returns the same shape as `exclude_pairs=set()` and as `exclude_pairs=None`.
  - `tests/test_als_recommender.py`: mirror tests for `build_confidence_matrix`.
- **Reuse:**
  - `pd.DataFrame.set_index + .index.isin` -- standard pandas vectorized filter.
  - Existing argparse defaults from `scripts/train_lightfm_model.py` and `scripts/train_als_model.py`.
- **Steps:**
  - Edit `build_interaction_matrix` and `build_confidence_matrix`.
  - Edit both train scripts.
  - Add the 4 unit tests.
  - `.venv/bin/python -m unittest tests.test_lightfm_recommender tests.test_als_recommender` -- both pass.
  - `.venv/bin/python -m unittest discover -s tests` -- full suite 68/68 (was 64/64, +4).
  - 5-row CLI smoke for ALS:
    ```bash
    mkdir -p /private/tmp/loo_smoke
    echo "userId,movieId" > /private/tmp/loo_smoke/excl.csv
    printf "1,1\n2,1\n" >> /private/tmp/loo_smoke/excl.csv
    .venv/bin/python scripts/train_als_model.py --output-dir /private/tmp/loo_smoke/als --exclude-holdout-pairs /private/tmp/loo_smoke/excl.csv
    ```
    Verify `metadata.json:excluded_pair_count = 2`. Compare `row_count` to a parallel run without `--exclude-holdout-pairs` and confirm it drops by exactly the number of high-rating excluded pairs (could be 0-2 depending on data; the count is the upper bound).
  - Commit message: `feat(eval): add holdout exclusion to LightFM and ALS matrix builders`. `git add -f docs/experiments/2026-05-24_leave-one-out-leakage-fix.md`.
- **Test / verification:**
  - 4 new unit tests pass; full suite 68/68.
  - CLI smoke produces a metadata file with `excluded_pair_count` reflecting the excluded CSV row count (lower-bounded by 0 because some excluded pairs may have been below the rating threshold anyway).
  - Backward compat: omitting `--exclude-holdout-pairs` reproduces the prior artifact byte-for-byte modulo timestamp (i.e., `row_count`, `user_count`, `item_count` match the committed `artifacts/models/{lightfm,als}/metadata.json`).
- **Expected outcome:** Both training paths can leakage-correct via a single CSV. Decision criterion: tests pass; backward compat holds.
- **DONE (commit `0aad222`):** Extended `build_interaction_matrix` (LightFM) and `build_confidence_matrix` (ALS) with optional `exclude_pairs` keyword that drops the matching `(userId, movieId)` rows via `pd.MultiIndex.isin`. Added `--exclude-holdout-pairs <CSV>` to both train scripts; CSV is read once, converted to a set of integer tuples, and threaded through. Metadata gains `excluded_pair_count` and `exclude_pairs_path` only when the flag is provided -- backward compatible.
  - Tests: 4 new unit tests (2 per builder) pass. Full suite 68/68 OK (was 64/64).
  - CLI smoke: trained ALS with a 2-row exclusion CSV. Leaked baseline `row_count = 16,863,053`; LOO smoke `row_count = 16,863,051` -- exact delta of 2 matching the excluded count. `metadata.json` carries `excluded_pair_count = 2` and `exclude_pairs_path` pointing at the input CSV.
  - Backward compat: omitting the flag returns the prior code path; the existing tests on `build_interaction_matrix` and `build_confidence_matrix` that do not pass `exclude_pairs` still pass byte-for-byte.
  - Decision: proceed to Item 2 (extract union holdouts + retrain LOO artifacts).

## 2) Extract union holdout set + retrain LOO artifacts

- **Goal:** Produce the `artifacts/holdouts/...csv` and retrain LightFM / ALS once with the exclusion applied.
- **Files:**
  - `scripts/extract_holdout_pairs.py` (new, ~80-100 lines). Argparse-driven. Loads `cleaned_data/ratings_clean.csv` via `load_ratings()`. For each `--user-sample-seed` value in the comma-separated list (default `42,7,1337`), reuses `select_evaluation_user_ids` and `temporal_train_test_split` from `src/evaluation_runner.py` and `src/evaluation.py` with the canonical eval defaults (`max_users=300`, `holdout_count=3`, `min_interactions=5`, `positive_threshold=4.0`). Filters the holdout DataFrame to `rating >= positive_threshold`. Accumulates `(userId, movieId)` pairs across seeds. Writes the deduplicated union to the output CSV (default `artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv`).
  - `artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv` -- new, gitignored output.
  - `artifacts/models/lightfm_loo/` -- new artifact dir produced by retrained LightFM.
  - `artifacts/models/als_loo/` -- new artifact dir produced by retrained ALS.
- **Steps:**
  - `.venv/bin/python scripts/extract_holdout_pairs.py --user-sample-seeds 42,7,1337 --max-users 300 --holdout-count 3 --output-path artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv`. Expected row count: each seed has ~260 positive-holdout users * up to 3 holdout items each; union across 3 seeds with overlapping users probably ~1500-2500 unique pairs. Confirm via `wc -l`.
  - `.venv/bin/python scripts/train_lightfm_model.py --output-dir artifacts/models/lightfm_loo --exclude-holdout-pairs artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv`. ~8 minutes wall time.
  - `.venv/bin/python scripts/train_als_model.py --output-dir artifacts/models/als_loo --exclude-holdout-pairs artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv`. ~2 minutes wall time.
  - Inspect `artifacts/models/{lightfm,als}_loo/metadata.json`:
    - `row_count` should equal the leaked artifact's `row_count` minus the count of excluded pairs that passed the positive-threshold filter (close to the CSV row count for canonical 4.0+ holdouts).
    - `excluded_pair_count` matches the CSV row count.
    - `user_count` and `item_count` should be within 0-1% of the leaked artifact's (some users / items may disappear if all their high-rating interactions were excluded -- expected for very cold users).
  - Commit message: `chore(eval): extract holdout union and train LOO LightFM/ALS artifacts`.
- **Test / verification:**
  - `artifacts/holdouts/...csv` has `userId,movieId` columns and >= 500 rows (loose lower bound; expect ~2000).
  - Both LOO artifacts' `metadata.json` carry the new `excluded_pair_count` and `exclude_pairs_path` keys.
  - `row_count` of each LOO artifact is strictly less than the leaked baseline's `row_count` by exactly the excluded-pair count (after threshold filtering).
  - The DONE marker records: row_count_leaked, row_count_loo, excluded_pair_count, train_seconds for each model.
- **Expected outcome:** Leakage-corrected artifacts on disk, ready for re-evaluation. Decision criterion: artifact files exist, metadata is internally consistent, row counts shift as expected.
- **DONE (commit `df131f5`):** Added `scripts/extract_holdout_pairs.py` and ran the full extract + retrain pipeline.
  - Holdout extraction across seeds {42, 7, 1337} at max_users=300, holdout_count=3, positive_threshold=4.0:

    | seed | positive-holdout pairs |
    |---:|---:|
    | 42 | 535 |
    | 7 | 484 |
    | 1337 | 536 |
    | **Union (deduplicated)** | **1555** |

    The union equals the sum -- zero seed overlap because random sampling 300 users from ~300K eligible produces negligible user collisions across seeds.

  - LOO artifacts vs leaked baselines:

    | Field | LightFM leaked | LightFM LOO | ALS leaked | ALS LOO |
    |---|---:|---:|---:|---:|
    | row_count | 16,863,053 | 16,861,498 | 16,863,053 | 16,861,498 |
    | user_count | 305,098 | 305,087 | 305,098 | 305,087 |
    | item_count | 40,441 | 40,440 | 40,441 | 40,440 |
    | train_seconds | 499.2 | 406.2 | 98.2 | 92.0 |
    | excluded_pair_count | (none) | 1,555 | (none) | 1,555 |

  - `row_count` delta is exactly 1555 for both models (matches the excluded count). `user_count` drops by 11 -- those users had ALL their `>=4.0` ratings in the eval holdout, so the LOO matrix has zero training rows for them. `item_count` drops by 1 -- one movie had no remaining `>=4.0` ratings after exclusion. This is the expected leave-one-out behavior.
  - Holdout CSV: `artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv` (1555 rows + header, gitignored).
  - Leaked baseline artifacts at `artifacts/models/{lightfm,als}/` were not modified.
  - Decision: proceed to Item 3 (re-run 3-seed segmented eval against the LOO artifacts).

## 3) Re-run 3-seed segmented eval against the LOO artifacts

- **Goal:** Three new `metrics_summary_*.json` artifacts that mirror the 2026-05-23 segmented runs but use the LOO artifacts.
- **Files:**
  - `artifacts/evaluation/metrics_summary_<timestamp>.{csv,json}` -- 3 new gitignored outputs.
- **Steps:**
  - For seed in `42, 7, 1337`, run:
    ```bash
    .venv/bin/python scripts/evaluate_baselines.py \
      --max-users 300 --k 5,10,20 \
      --holdout-count 3 \
      --include-random --include-tfidf --include-content --include-semantic \
      --include-svd --include-svd-topk \
      --include-sbert-faiss --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
      --include-lightfm --lightfm-artifacts-dir artifacts/models/lightfm_loo \
      --include-als --als-artifacts-dir artifacts/models/als_loo \
      --user-sample-seed <seed> \
      --segment-by-history \
      --output-dir artifacts/evaluation
    ```
  - Chain in background; ~30-40 minutes wall time total.
  - Sanity check after the seed-42 run: the deterministic non-CF baselines (`popularity`, `tfidf_content`, `hybrid_content`, `sbert_faiss_content`, `semantic_content`, `svd_topk`, `random`) should report numbers identical to the 2026-05-23 seed-42 run (`metrics_summary_2026-05-20T21-09-15Z.json`). Only `lightfm_warp` and `als_implicit` should change.
  - Commit message: `chore(eval): capture LOO 3-seed segmented run artifacts`.
- **Test / verification:**
  - Three timestamped JSONs. `config.lightfm_artifacts_dir` ends with `lightfm_loo`, `config.als_artifacts_dir` ends with `als_loo`.
  - Per-seed `popularity.10.ndcg_at_k` matches the 2026-05-23 reference at the same seed exactly (deterministic, no artifact dependency).
  - Per-seed `als_implicit.10.evaluated_user_count` equals the 2026-05-23 reference's (same eval slice, same user split).
  - DONE marker records the per-seed K=10 NDCG@10 for `als_implicit` and `lightfm_warp` aggregate and per-segment.
- **Expected outcome:** A 3-seed LOO dataset ready for the synthesis. Decision criterion: three artifacts exist; non-CF baselines unchanged.
- **DONE (commit to be backfilled):** Ran the 9-model segmented eval three times against the LOO artifacts at `--max-users 300 --holdout-count 3 --segment-by-history --user-sample-seed` in `42, 7, 1337`. Total wall time ~38 minutes.
  - Sanity check passed cleanly: every deterministic non-CF baseline (popularity, tfidf_content, hybrid_content, sbert_faiss_content, semantic_content, svd_topk, random) produces NDCG@10 values that are **byte-for-byte identical** to the 2026-05-23 leaked-baseline runs at the same seed. Only `als_implicit` and `lightfm_warp` shifted, which is exactly the expected behavior of swapping in LOO artifacts.
  - **Aggregate NDCG@10 collapse:**

    | seed | ALS leaked | ALS LOO | delta | LightFM leaked | LightFM LOO | delta |
    |---:|---:|---:|---:|---:|---:|---:|
    | 42 | 0.2731 | 0.0863 | **-0.1868** | 0.1447 | 0.0883 | -0.0564 |
    | 7 | 0.2354 | 0.0607 | **-0.1747** | 0.1087 | 0.0548 | -0.0539 |
    | 1337 | 0.2142 | 0.0681 | -0.1460 | 0.1176 | 0.0601 | -0.0575 |

    ALS lost ~70% of its NDCG@10 across seeds; LightFM lost ~40-50%. The asymmetry says ALS was the bigger beneficiary of the leakage.

  - **ALS leadership is no longer universal at the aggregate level:** at seed 42, LightFM LOO (0.0883) very slightly **beats** ALS LOO (0.0863). At seeds 7 and 1337, ALS still leads (0.0607 vs 0.0548; 0.0681 vs 0.0601) but by margins inside the seed variance band. The previous "ALS > LightFM in every shape" claim is now within noise rather than a clear gap.

  - **Per-segment ALS NDCG@10 (mean across 3 seeds):**

    | Segment | Leaked mean | LOO mean | delta | ratio |
    |---|---:|---:|---:|---:|
    | cold_0_10 | 0.4610 | 0.1272 | -0.3338 | 0.28x |
    | warm_10_50 | 0.2820 | 0.0700 | -0.2120 | 0.25x |
    | regular_50_200 | 0.1174 | 0.0579 | -0.0595 | 0.49x |
    | heavy_200_plus | 0.0663 | 0.0338 | -0.0325 | 0.51x |

    The cold and warm buckets lost ~72-75% of their NDCG; regular and heavy lost ~50%. The leakage was strongest exactly where cold users' "true" signal was in the holdout the artifact had memorized.

  - **Per-segment LightFM NDCG@10 (mean across 3 seeds):**

    | Segment | Leaked mean | LOO mean | delta | ratio |
    |---|---:|---:|---:|---:|
    | cold_0_10 | 0.2701 | 0.0847 | -0.1854 | 0.31x |
    | warm_10_50 | 0.1328 | 0.0740 | -0.0588 | 0.56x |
    | regular_50_200 | 0.0562 | 0.0549 | -0.0013 | 0.98x |
    | heavy_200_plus | 0.0362 | 0.0524 | **+0.0162** | 1.45x |

    LightFM cold loses 69% (similar to ALS). But LightFM **heavy IMPROVED** under LOO -- the LOO matrix is slightly tighter (fewer noisy positives) so the model's signal on heavy users actually got cleaner.

  - **Cold > heavy inversion attenuates but does not flip.** For ALS, LOO cold-to-heavy ratio drops from 6.96x to 3.76x; still inverted versus textbook expectation, but much closer to flat. For LightFM, LOO ratio is 1.62x (was 7.46x leaked) -- nearly flat, the inversion is essentially resolved.

  - **New per-segment leaderboard under LOO:**
    - `cold_0_10`: ALS leads in 3/3 seeds.
    - `warm_10_50`: LightFM leads in 2/3 seeds (42 and 1337); ALS leads in 1 (seed 7).
    - `regular_50_200`: ALS leads in 2/3 seeds (7 and 1337); LightFM leads in 1 (seed 42).
    - `heavy_200_plus`: **LightFM leads in 3/3 seeds** (0.051 / 0.055 / 0.052 vs ALS 0.031 / 0.031 / 0.039).

  - Run ids: `metrics_summary_2026-05-21T07-58-22Z` (seed 42), `metrics_summary_2026-05-21T08-11-13Z` (seed 7), `metrics_summary_2026-05-21T08-25-04Z` (seed 1337). All gitignored.
  - Decision: proceed to Item 4. The synthesis needs to clearly restate the conclusion: ALS is the cold-start leader but **not** the universal leader once leakage is removed; LightFM is more robust on long-history users.

## 4) Synthesis: "Leakage-corrected (leave-one-out) re-evaluation" subsection

- **Goal:** Translate the three LOO runs + the three leaked baselines into a readable comparison block in `docs/08_evaluation_results_report.md`.
- **Files:**
  - `docs/08_evaluation_results_report.md`: insert a new subsection `## Leakage-corrected (leave-one-out) re-evaluation` between "Cold-start segmentation (user-history buckets)" and "Conclusions". Two tables:
    - **Table A -- Aggregate at 300u/h=3 across seeds {42, 7, 1337}:** rows are `als_implicit`, `lightfm_warp`, `hybrid_content`, `popularity`; columns are NDCG@10 leaked, NDCG@10 LOO, HitRate@10 leaked, HitRate@10 LOO. All entries `mean +/- std` across seeds.
    - **Table B -- Per-segment NDCG@10 for ALS and LightFM at 300u/h=3:** rows are the four buckets x two models = 8 rows; columns are leaked vs LOO mean +/- std.
  - Two paragraphs:
    - (i) Aggregate deltas with signs. By how much did ALS NDCG@10 drop? Did the ALS > LightFM ordering survive? Did the non-CF baselines stay unchanged (sanity)?
    - (ii) Whether the cold > heavy inversion survived, attenuated, or flipped. Numbers, then verbal interpretation.
  - Update Conclusions: tighten or qualify the universal-leader statement based on the LOO results. The cold-start segmentation paragraph's "leakage signature" framing now points readers to this subsection.
  - Update Caveats: replace "leakage caveat -- open" wording with "leakage caveat -- measured; see Leakage-corrected subsection".
  - This plan file: fill DONE marker.
- **Steps:**
  - Pull aggregate and per-segment NDCG@10 / HitRate@10 from the three LOO JSONs and the three leaked-baseline JSONs (`metrics_summary_2026-05-20T21-{09-15,23-38,37-33}Z.json`). Compute mean / std.
  - Write the two tables and the two paragraphs.
  - Update Conclusions and Caveats.
  - Commit message: `docs(eval): add leakage-corrected leave-one-out subsection`. `git add -f` both files.
- **Test / verification:**
  - New subsection exists with both tables and both paragraphs.
  - Conclusions reflects the LOO findings (either ALS still wins on every shape and segment, or the claim is qualified).
  - Caveats no longer treats leakage as "open" -- it cross-references the new subsection.
  - Full test suite stays >=68 (doc-only change).
- **Expected outcome:** A defensible leakage-corrected picture. Decision criterion: the two tables agree with the six JSONs row-by-row; the universal-leader claim is either preserved with new evidence or honestly qualified.
- **DONE / DROPPED:**

## Deferred / Future (out of this plan)

- **Per-seed LOO artifacts.** Union exclusion is sufficient for this demonstration. Per-seed would tighten the rigor at 5x compute cost; reopen if the union LOO numbers prove fragile.
- **Item-side cold-start retraining.** User-side leakage is the immediate issue; item-side (recall on never-rated movies, with deliberately held-out item embeddings) is a separate question and lives in a future plan.
- **Hyperparameter sweeps on LOO artifacts.** Once LOO numbers are settled, a LightFM `no_components x loss x epochs` and ALS `factors x regularization x alpha` sweep against the cleanest baseline becomes valuable. Separate plan.
- **UI explainability (Roadmap Priority 5).** Independent track, no eval dependency.
- **LightGCN / SASRec (Roadmap Priority 6).** Modern graph / sequential models; in principle they share the same leakage and should benefit from the LOO methodology, but training cost is much higher. Separate plan.
- **SVD top-K leakage / ranking improvement.** Surprise SVD inherits the same leakage in principle, but the model already produces near-zero relevance at K=10 / K=20 in every shape; correcting its leakage would not change the model's ordering relative to ALS / LightFM. Documented separately in the "Why SVD top-K stays at zero hits" subsection.

## Critical Files (Reference)

- `src/experimental/lightfm_recommender.py:37-67` -- `build_interaction_matrix`; item 1 adds `exclude_pairs=None`.
- `src/experimental/als_recommender.py:41-71` -- `build_confidence_matrix`; same pattern.
- `scripts/train_lightfm_model.py` -- item 1 adds `--exclude-holdout-pairs` CLI flag.
- `scripts/train_als_model.py` -- same.
- `scripts/extract_holdout_pairs.py` -- new in item 2.
- `tests/test_lightfm_recommender.py` and `tests/test_als_recommender.py` -- 2 new tests each in item 1.
- `src/evaluation_runner.py:86-101` (`select_evaluation_user_ids`) and `src/evaluation.py:22+` (`temporal_train_test_split`) -- unchanged, reused by the extraction script.
- `artifacts/models/lightfm/`, `artifacts/models/als/` -- existing leaked baselines; not modified.
- `artifacts/models/lightfm_loo/`, `artifacts/models/als_loo/` -- new LOO artifact dirs (gitignored).
- `artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv` -- new (gitignored).
- `artifacts/evaluation/metrics_summary_2026-05-20T21-{09-15,23-38,37-33}Z.{csv,json}` -- 2026-05-23 segmented baseline references (leaked).
- `docs/08_evaluation_results_report.md` -- item 4 inserts the "Leakage-corrected (leave-one-out) re-evaluation" subsection.
- `docs/experiments/2026-05-23_cold-start-segmentation.md` -- prior plan; this plan's motivation references its DONE markers.

## End-to-End Verification Sequence

1. Item 1 complete: 4 new unit tests pass; full suite >=68; CLI smoke for ALS produces a `metadata.json` with `excluded_pair_count` equal to the CSV row count after threshold filtering. DONE marker has a real commit hash.
2. Item 2 complete: `artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv` exists with >=500 rows; both `artifacts/models/{lightfm,als}_loo/metadata.json` exist with `excluded_pair_count > 0`; `row_count` shrinks by exactly the excluded amount. DONE marker records both models' train wall time, row counts, and the CSV row count.
3. Item 3 complete: three timestamped JSONs with `config.{lightfm,als}_artifacts_dir` ending in `_loo`. Non-CF baselines match 2026-05-23 references exactly. DONE marker captures per-seed K=10 NDCG@10 for ALS and LightFM (aggregate + per-segment).
4. Item 4 complete: `docs/08_evaluation_results_report.md` carries the new subsection with two filled tables and two interpretation paragraphs; Conclusions reflects LOO findings; Caveats no longer flags leakage as "open". DONE marker has the synthesis commit hash.
5. After every commit, record the real hash in this plan file; CLAUDE.md section 7 forbids leaving placeholders.

## Execution Notes

- `/docs` is gitignored; tracked files including this plan need `git add -f` on commits.
- Items are sequential: item 1 unlocks item 2, item 2 produces inputs for item 3, item 3 produces inputs for item 4.
- One implementation commit per item plus a follow-up `docs(experiments): record ...` commit that fills the DONE marker with the real hash. Matches the 2026-05-21 / 22 / 23 cadence.
- Artifact files under `artifacts/{models,holdouts,evaluation}/` stay gitignored.
- The leaked artifacts at `artifacts/models/{lightfm,als}/` stay on disk untouched. The LOO artifacts live next to them in `_loo` directories. The 2026-05-23 segmented runs are the leaked baseline; item 4 compares against them.
