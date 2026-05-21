# Evaluation Results Report

## Run Summary

**Status (2026-05-21):** The headline numbers in the Run Summary and Model Comparison sections below come from the initial 100-user single-run and are now **superseded** by the leakage-corrected leave-one-out + hyperparameter-sweep subsections later in this document. Current strongest classical-CF baseline: tuned ALS `factors=64, regularization=0.1` (aggregate NDCG@10 = 0.0787 +/- 0.0115 over 3 seeds). See `docs/experiments/2026-05-24_leave-one-out-leakage-fix.md` and `docs/experiments/2026-05-25_hyperparam-sweep-loo.md` for the audit trail.

This report summarizes the latest local offline evaluation run:

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

Run configuration:

- Users selected: 100
- Users with positive holdout items evaluated for top-N metrics: 55
- Holdout policy: latest 1 interaction per eligible user
- Positive threshold: rating >= 4.0
- Movie catalog rows: 79,477
- Ratings rows: 33,703,215
- Semantic baseline: TF-IDF + TruncatedSVD LSA (`--include-semantic`)
- SBERT+FAISS baseline: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, full-catalog index under `artifacts/indexes/sbert_faiss/` (`row_count = 79,477`)
- LightFM WARP baseline: prebuilt local artifact under `artifacts/models/lightfm/` (`row_count = 16,863,053`, `no_components = 64`, `epochs = 20`)
- Implicit ALS baseline: prebuilt local artifact under `artifacts/models/als/` (`row_count = 16,863,053`, `factors = 64`, `iterations = 20`)

Generated local artifacts:

- `artifacts/evaluation/metrics_summary.csv`
- `artifacts/evaluation/metrics_summary.json`
- timestamped CSV/JSON copies
- `artifacts/evaluation/run_config.json`

These artifacts are local/generated and are not tracked by git.

The audit trail lives in `docs/experiments/2026-05-20_classical-cf-and-eval-expansion.md` (initial 9-model build) and `docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md` (the ALS exclusion-semantics fix applied here in item 3, SVD top-K diagnosis in item 4).

## Metric Meanings

- `precision_at_k`: share of the top K recommendations that were relevant.
- `recall_at_k`: share of relevant holdout items recovered in the top K.
- `hit_rate_at_k`: whether at least one relevant holdout item appeared in the top K.
- `ndcg_at_k`: ranking quality that rewards relevant hits near the top.
- `map_at_k`: average precision up to K; rewards relevant items appearing early and multiple relevant hits.
- `mrr_at_k`: reciprocal rank of the first relevant hit up to K.
- `catalog_coverage`: share of the movie catalog reached by recommendations.
- `user_coverage`: share of evaluated users that received recommendations.
- `diversity`: genre-based intra-list diversity.
- `novelty`: higher values indicate less popular, less obvious recommendations.
- `serendipity`: relevant hits not already produced by the popularity baseline.
- `latency_mean_ms` / `latency_p95_ms`: per-user recommendation latency.
- `rmse` / `mae`: SVD explicit rating prediction error on holdout ratings.

## Model Comparison

### Top-N at K=10

| Model | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 | MAP@10 | MRR@10 | Coverage | Diversity | Novelty | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| als_implicit | 0.0309 | 0.3091 | 0.3091 | 0.1765 | 0.1372 | 0.1372 | 0.0042 | 0.7576 | 10.2125 | 7.1 ms |
| lightfm_warp | 0.0255 | 0.2545 | 0.2545 | 0.1427 | 0.1100 | 0.1100 | 0.0032 | 0.7540 | 9.8075 | 40.4 ms |
| hybrid_content | 0.0073 | 0.0727 | 0.0727 | 0.0382 | 0.0283 | 0.0283 | 0.0019 | 0.6902 | 10.0856 | 1,369.4 ms |
| popularity | 0.0091 | 0.0909 | 0.0909 | 0.0322 | 0.0150 | 0.0150 | 0.0005 | 0.7936 | 8.6046 | 83.5 ms |
| sbert_faiss_content | 0.0036 | 0.0364 | 0.0364 | 0.0193 | 0.0136 | 0.0136 | 0.0039 | 0.7165 | 11.6010 | 34.6 ms |
| semantic_content | 0.0018 | 0.0182 | 0.0182 | 0.0182 | 0.0182 | 0.0182 | 0.0045 | 0.5549 | 12.4005 | 75.6 ms |
| tfidf_content | 0.0036 | 0.0364 | 0.0364 | 0.0139 | 0.0071 | 0.0071 | 0.0020 | 0.6826 | 10.0908 | 43.1 ms |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0069 | 0.8579 | 13.5810 | 11.3 ms |
| svd_topk | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0029 | 0.7756 | 12.3858 | 174.2 ms |

At K=10, `als_implicit` (post-fix) leads every relevance metric and is simultaneously the fastest model at 7.1 ms mean latency -- the cleanest win in this run. `lightfm_warp` is the next-best ranker, also in the fast tier. `hybrid_content` stays the third-best ranker but remains the slowest baseline. `svd_topk` is still at zero hits and is discussed in its own subsection below.

### Top-N at K=20

| Model | Precision@20 | Recall@20 | HitRate@20 | NDCG@20 | MAP@20 | MRR@20 | Coverage | Diversity | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| als_implicit | 0.0191 | 0.3818 | 0.3818 | 0.1942 | 0.1417 | 0.1417 | 0.0068 | 0.7589 | 10.3834 |
| lightfm_warp | 0.0173 | 0.3455 | 0.3455 | 0.1657 | 0.1163 | 0.1163 | 0.0054 | 0.7672 | 10.0121 |
| hybrid_content | 0.0064 | 0.1273 | 0.1273 | 0.0525 | 0.0324 | 0.0324 | 0.0033 | 0.7050 | 10.4092 |
| popularity | 0.0055 | 0.1091 | 0.1091 | 0.0367 | 0.0162 | 0.0162 | 0.0008 | 0.7941 | 8.8305 |
| tfidf_content | 0.0045 | 0.0909 | 0.0909 | 0.0280 | 0.0111 | 0.0111 | 0.0035 | 0.6833 | 10.3564 |
| sbert_faiss_content | 0.0027 | 0.0545 | 0.0545 | 0.0236 | 0.0146 | 0.0146 | 0.0071 | 0.7395 | 11.7631 |
| semantic_content | 0.0018 | 0.0364 | 0.0364 | 0.0230 | 0.0196 | 0.0196 | 0.0082 | 0.5573 | 12.4960 |
| svd_topk | 0.0036 | 0.0727 | 0.0727 | 0.0178 | 0.0046 | 0.0046 | 0.0055 | 0.7849 | 12.5488 |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0138 | 0.8484 | 13.5795 |

At K=20, `als_implicit` continues to lead every relevance metric, with `lightfm_warp` in second and `hybrid_content` in third. `svd_topk` recovers a small non-zero band at K=20 (HitRate 0.0727, NDCG 0.0178) but remains far behind every other model that produces hits.

### SVD Rating Prediction

The SVD explicit rating prediction baseline was evaluated separately from top-N ranking:

| Metric | Value |
|---|---:|
| RMSE | 0.7558 |
| MAE | 0.5706 |
| Prediction count | 100 |

These values measure rating prediction error, not recommendation list quality.

## Latency Findings

Per-user latency from fastest to slowest:

| Model | Mean latency | p95 latency |
|---|---:|---:|
| als_implicit | 7.1 ms | 11.2 ms |
| random | 11.3 ms | 12.2 ms |
| sbert_faiss_content | 34.6 ms | 75.1 ms |
| lightfm_warp | 40.4 ms | 46.4 ms |
| tfidf_content | 43.1 ms | 80.1 ms |
| semantic_content | 75.6 ms | 104.6 ms |
| popularity | 83.5 ms | 39.3 ms |
| svd_topk | 174.2 ms | 178.2 ms |
| hybrid_content | 1,369.4 ms | 4,110.7 ms |

`als_implicit` is both the fastest model in this run and the leader on every relevance metric -- the only baseline that wins both axes simultaneously. `lightfm_warp` is the next-best on relevance at 40.4 ms mean. `hybrid_content` stays under the 2,000 ms mean-latency gate but remains the slowest baseline.

## Why SVD top-K stays at zero hits

The `svd_topk` baseline calls `raw_svd_predictions` in `src/recommenders/svd.py`, which scores every movie not in the user's train ratings using the Surprise SVD model's `predict(uid, iid)` method, then returns the top-K by predicted rating. The same Surprise model also produces the RMSE / MAE values in the "SVD Rating Prediction" section above (0.7558 / 0.5706), so the predictor itself is sound on its training objective.

The item 1 diagnostic in `docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md` ranked each sampled user's positive holdout movieId in the SVD full-catalog predicted-score sort. The five sampled users' holdouts landed at ranks 94, 20, 392, 14171, and 10934 -- four of five fell outside the top 50, and four of five outside the top 100. The user whose holdout was at rank 20 is the only K=20 hit in the canonical table.

This is the expected weakness of explicit-rating SVD for top-K ranking, not a wiring bug. Surprise SVD optimizes mean squared error on observed (user, item) ratings; its top-ranked candidates are therefore items whose mean predicted rating crowds the upper bound (4.7 - 5.0), a narrow popular-favorable set -- catalog coverage at K=10 is only 0.0029. Improving this would mean blending the predicted score with a ranking signal (popularity log, recency) or replacing the baseline with a ranker trained on a ranking objective (LightFM WARP is already in the same table). That work is deferred and logged at the bottom of the audit plan.

## Variance Bounds (multi-seed slice studies)

The K=10 / K=20 tables above are a single 100-user / latest-1 / first-N slice. To check whether that snapshot generalises, `docs/experiments/2026-05-22_eval-slice-expansion.md` ran the same canonical command across three axes -- multi-seed user sampling at 100 users, a single-seed expansion to 300 users, and multi-seed at 300 users with `holdout_count=3`. Numbers below are NDCG@10 / HitRate@10 / mean per-user latency. For the multi-seed groups, the entry format is `mean +/- standard deviation` across the three seeds (42, 7, 1337). The single-seed group reports the point value only.

### Table A -- 100 users, holdout=1, seeds {42, 7, 1337}

Evaluated user counts (positive holdout) per seed: 54, 52, 54.

| Model | NDCG@10 | HitRate@10 | Mean latency |
|---|---:|---:|---:|
| als_implicit | 0.1907 +/- 0.0232 | 0.3196 +/- 0.0628 | 7.5 +/- 0.1 ms |
| lightfm_warp | 0.0753 +/- 0.0173 | 0.1752 +/- 0.0399 | 41.4 +/- 0.6 ms |
| hybrid_content | 0.0225 +/- 0.0101 | 0.0439 +/- 0.0119 | 1,326.8 +/- 155.8 ms |
| popularity | 0.0277 +/- 0.0169 | 0.0503 +/- 0.0296 | 82.1 +/- 0.8 ms |
| sbert_faiss_content | 0.0234 +/- 0.0199 | 0.0311 +/- 0.0212 | 39.2 +/- 2.8 ms |

`als_implicit > lightfm_warp` holds in 3 of 3 seeds (relative variation of ALS NDCG@10 is about 12% of the mean). The third spot is genuinely a tossup at this slice size: `popularity`, `sbert_faiss_content`, and `hybrid_content` overlap each other within one standard deviation on both NDCG and hit rate. Hybrid's "third-best" framing from the previous single-run table is not robust at this sample size.

### Table B -- 300 users, holdout=1, single seed 42

Evaluated user count: 177.

| Model | NDCG@10 | HitRate@10 | Mean latency |
|---|---:|---:|---:|
| als_implicit | 0.2196 | 0.3616 | 6.7 ms |
| lightfm_warp | 0.1173 | 0.2147 | 37.3 ms |
| hybrid_content | 0.0383 | 0.0734 | 1,368.1 ms |
| popularity | 0.0467 | 0.0791 | 36.5 ms |
| sbert_faiss_content | 0.0316 | 0.0621 | 42.9 ms |

The larger sample tightens every number toward the bigger one in Table A's variance band. ALS pulls further ahead (NDCG@10 0.2196 vs 0.19 mean at 100 users), confirming the leadership is not a small-sample artifact. At the 300-user shape `popularity` already edges `hybrid_content` on NDCG@10 (0.0467 vs 0.0383); the lower tier is still noisy but the ordering is starting to settle.

### Table C -- 300 users, holdout=3, seeds {42, 7, 1337}

Evaluated user counts: 259, 245, 265. Recall denominators change at holdout=3 (each user has 3 positive items, capped at 1.0 -- hit rates rise because there are 3 chances to land a top-10 hit per user).

| Model | NDCG@10 | HitRate@10 | Mean latency |
|---|---:|---:|---:|
| als_implicit | 0.2409 +/- 0.0299 | 0.4996 +/- 0.0560 | 11.6 +/- 7.9 ms |
| lightfm_warp | 0.1237 +/- 0.0188 | 0.3133 +/- 0.0505 | 37.5 +/- 0.8 ms |
| hybrid_content | 0.0325 +/- 0.0082 | 0.0965 +/- 0.0317 | 1,325.1 +/- 126.4 ms |
| popularity | 0.0416 +/- 0.0065 | 0.1144 +/- 0.0115 | 37.7 +/- 0.9 ms |
| sbert_faiss_content | 0.0216 +/- 0.0065 | 0.0677 +/- 0.0200 | 44.1 +/- 11.1 ms |

`als_implicit > lightfm_warp` holds again in 3 of 3 seeds. ALS HitRate@10 nearly doubles (0.32 -> 0.50 mean) because the holdout window widened from 1 to 3; the ratio between models in this column stays stable. The hybrid recommender's #3 placement does not survive -- `popularity` and `tfidf_content` (not shown in this table but easily checked in the artifacts) both beat `hybrid_content` on NDCG@10 in all three seeds. This is the most reliable shape in the study and the place where `hybrid_content` is most clearly outranked.

## Cold-start segmentation (user-history buckets)

Building on the 300-user / holdout=3 multi-seed runs above, `docs/experiments/2026-05-23_cold-start-segmentation.md` re-ran the same three seeds with `--segment-by-history`, partitioning each run's evaluated users into four buckets by their train-interaction count: `cold_0_10` (n < 10), `warm_10_50` (10 <= n < 50), `regular_50_200` (50 <= n < 200), `heavy_200_plus` (n >= 200). Bucket sizes were stable across seeds (cold ~44, warm ~111, regular ~69, heavy ~32 per seed). Entries below are NDCG@10 / HitRate@10 reported as `mean +/- std` across seeds {42, 7, 1337}.

### Cold (n < 10, mean ~44 users per seed)

| Model | NDCG@10 | HitRate@10 |
|---|---:|---:|
| als_implicit | 0.4610 +/- 0.0743 | 0.7577 +/- 0.0194 |
| lightfm_warp | 0.2701 +/- 0.0378 | 0.5161 +/- 0.0328 |
| popularity | 0.0876 +/- 0.0423 | 0.2072 +/- 0.0803 |
| hybrid_content | 0.0585 +/- 0.0079 | 0.1473 +/- 0.0845 |
| tfidf_content | 0.0562 +/- 0.0266 | 0.1244 +/- 0.0830 |
| sbert_faiss_content | 0.0394 +/- 0.0148 | 0.1077 +/- 0.0560 |

### Warm (10 <= n < 50, mean ~111 users per seed)

| Model | NDCG@10 | HitRate@10 |
|---|---:|---:|
| als_implicit | 0.2820 +/- 0.0618 | 0.5880 +/- 0.0950 |
| lightfm_warp | 0.1328 +/- 0.0308 | 0.3531 +/- 0.1123 |
| popularity | 0.0415 +/- 0.0096 | 0.1202 +/- 0.0122 |
| hybrid_content | 0.0365 +/- 0.0102 | 0.1111 +/- 0.0204 |
| tfidf_content | 0.0318 +/- 0.0055 | 0.1055 +/- 0.0099 |
| sbert_faiss_content | 0.0184 +/- 0.0095 | 0.0682 +/- 0.0234 |

### Regular (50 <= n < 200, mean ~69 users per seed)

| Model | NDCG@10 | HitRate@10 |
|---|---:|---:|
| als_implicit | 0.1174 +/- 0.0442 | 0.3351 +/- 0.0949 |
| lightfm_warp | 0.0562 +/- 0.0154 | 0.2074 +/- 0.0458 |
| tfidf_content | 0.0370 +/- 0.0179 | 0.0967 +/- 0.0350 |
| popularity | 0.0247 +/- 0.0104 | 0.0721 +/- 0.0139 |
| hybrid_content | 0.0201 +/- 0.0094 | 0.0676 +/- 0.0364 |
| sbert_faiss_content | 0.0140 +/- 0.0057 | 0.0481 +/- 0.0080 |

### Heavy (n >= 200, mean ~32 users per seed)

| Model | NDCG@10 | HitRate@10 |
|---|---:|---:|
| als_implicit | 0.0663 +/- 0.0364 | 0.1887 +/- 0.0390 |
| lightfm_warp | 0.0362 +/- 0.0072 | 0.1152 +/- 0.0511 |
| sbert_faiss_content | 0.0244 +/- 0.0143 | 0.0533 +/- 0.0214 |
| popularity | 0.0184 +/- 0.0132 | 0.0644 +/- 0.0357 |
| tfidf_content | 0.0140 +/- 0.0136 | 0.0413 +/- 0.0361 |
| hybrid_content | 0.0103 +/- 0.0065 | 0.0425 +/- 0.0210 |

**Who wins each segment.** `als_implicit > lightfm_warp` in every segment in 3 of 3 seeds (12 of 12 segment-seed cells). The third-rank position is segment-dependent: `popularity` is third in cold; `popularity` and `hybrid_content` tie within one std in warm; `tfidf_content` rises to third in regular; `sbert_faiss_content` rises to third in heavy. So the "third spot is a tossup" finding from the Variance Bounds section above sharpens here -- the actual third-place model **changes by user-history bucket**.

**Why both CF models invert classical wisdom (the leakage signature).** Textbook expectation is that ALS and LightFM are weak at cold-start and strong on heavy users (where they have the most behavior signal). The opposite happens here: both models' NDCG@10 falls monotonically from cold to heavy (ALS 0.46 -> 0.07; LightFM 0.27 -> 0.04). The most likely explanation is the artifact-level training-set leakage that prior plans noted in Caveats: both artifacts were trained on the full rating matrix including each eval user's holdout interactions. For cold users, their "true" preference signal lives almost entirely in the holdout, so the artifact has effectively memorized exactly the items the eval will hold out. The "Leakage-corrected (leave-one-out) re-evaluation" subsection below measures the size of this effect directly.

## Leakage-corrected (leave-one-out) re-evaluation

The 2026-05-24 audit (`docs/experiments/2026-05-24_leave-one-out-leakage-fix.md`) retrained both classical CF artifacts with the union of holdout `(userId, movieId)` pairs across seeds {42, 7, 1337} removed from the training matrix -- 1,555 pairs total. The leaked artifacts at `artifacts/models/{lightfm,als}/` were untouched; the leakage-corrected pair lives at `artifacts/models/{lightfm,als}_loo/`. All three 300u/h=3 segmented seeds were re-run against the LOO artifacts. Numbers below are `mean +/- std` across the 3 seeds.

**Sanity check.** Every deterministic non-CF baseline (popularity, tfidf_content, hybrid_content, sbert_faiss_content, semantic_content, svd_topk, random) returns NDCG@10 identical to its leaked-run value at the same seed. Only `als_implicit` and `lightfm_warp` shift. This isolates the leakage effect to exactly the two artifacts that were retrained.

### Aggregate NDCG@10 / HitRate@10 at 300u/h=3

| Model | NDCG@10 leaked | NDCG@10 LOO | HitRate@10 leaked | HitRate@10 LOO |
|---|---:|---:|---:|---:|
| als_implicit | 0.2409 +/- 0.0299 | **0.0717 +/- 0.0132** | 0.4996 +/- 0.0560 | 0.2170 +/- 0.0363 |
| lightfm_warp | 0.1237 +/- 0.0188 | **0.0678 +/- 0.0180** | 0.3133 +/- 0.0505 | 0.1788 +/- 0.0516 |
| hybrid_content | 0.0325 +/- 0.0082 | 0.0325 +/- 0.0082 | 0.0965 +/- 0.0317 | 0.0965 +/- 0.0317 |
| popularity | 0.0416 +/- 0.0065 | 0.0416 +/- 0.0065 | 0.1144 +/- 0.0115 | 0.1144 +/- 0.0115 |

ALS lost approximately 70% of its aggregate NDCG@10 (0.241 -> 0.072); LightFM lost approximately 45% (0.124 -> 0.068). The asymmetry says ALS was the bigger beneficiary of the training-set leakage. At the aggregate level the two CF models are now within one standard deviation of each other, and at seed 42 specifically, LightFM LOO (0.0883) very slightly beats ALS LOO (0.0863). The previous canonical statement that "ALS leads on every studied shape" is no longer supported once the leakage is removed.

### Per-segment NDCG@10 for ALS and LightFM

| Segment | ALS leaked | ALS LOO | LightFM leaked | LightFM LOO |
|---|---:|---:|---:|---:|
| cold_0_10 | 0.4610 +/- 0.0743 | 0.1272 +/- 0.0569 | 0.2701 +/- 0.0378 | 0.0846 +/- 0.0500 |
| warm_10_50 | 0.2820 +/- 0.0618 | 0.0700 +/- 0.0112 | 0.1328 +/- 0.0308 | 0.0740 +/- 0.0181 |
| regular_50_200 | 0.1174 +/- 0.0442 | 0.0579 +/- 0.0009 | 0.0562 +/- 0.0154 | 0.0549 +/- 0.0096 |
| heavy_200_plus | 0.0663 +/- 0.0364 | 0.0338 +/- 0.0047 | 0.0362 +/- 0.0072 | **0.0524 +/- 0.0020** |

**How much of the cold inflation was leakage.** For ALS, cold lost 72% of its leaked NDCG@10 (0.461 -> 0.127); warm lost 75%; regular lost 51%; heavy lost 49%. For LightFM the cold loss is 69%; warm 44%; regular ~3%; heavy actually **rose** by 45% (0.036 -> 0.052) because the tighter LOO training matrix removed some noisy positives that were dragging the heavy-user signal down. The cold > heavy NDCG inversion attenuates but does not fully flip: for ALS the cold-to-heavy ratio drops from 6.96x leaked to 3.76x LOO; for LightFM it drops from 7.46x leaked to 1.62x LOO -- essentially flat.

**The corrected leaderboard is segment-dependent.** Under LOO at 300u/h=3:

- `cold_0_10`: ALS leads in 3 of 3 seeds. The "ALS is the cold-start king" finding survives the correction (the absolute numbers shrink, but the relative gap to LightFM holds).
- `warm_10_50`: LightFM leads in 2 of 3 seeds (42 and 1337); ALS leads in 1 (seed 7).
- `regular_50_200`: ALS leads in 2 of 3 seeds (7 and 1337); LightFM leads in 1 (seed 42). Mixed -- within seed variance.
- `heavy_200_plus`: **LightFM leads in 3 of 3 seeds** with a comfortable margin (LightFM 0.0524 +/- 0.0020 vs ALS 0.0338 +/- 0.0047).

The headline post-LOO: ALS still owns cold-start; LightFM owns long-history users; the middle two buckets are a tossup. This is the textbook-aligned outcome that classical CF wisdom predicted, which the leaked artifacts had concealed.

## Hyperparameter sweep on LOO artifacts

The 2026-05-25 audit (`docs/experiments/2026-05-25_hyperparam-sweep-loo.md`) ran a 12-artifact sweep -- LightFM over `no_components in {32, 64, 128} x loss in {warp, bpr}` (6 combos, epochs=20) and ALS over `factors in {32, 64, 128} x regularization in {0.01, 0.1}` (6 combos, alpha=40, iterations=20). Every artifact was retrained with the same 1,555-row LOO exclusion CSV used in the 2026-05-24 plan; aggregate eval ran at seed=42 with segmentation enabled. Winners were then put through a full 3-seed segmented eval.

### Sweep results at seed=42 (all 12 artifacts)

LightFM (sorted by aggregate NDCG@10):

| Slug | Aggregate | Cold | Warm | Regular | Heavy |
|---|---:|---:|---:|---:|---:|
| n64_lwarp_e20 | **0.0738** | 0.1266 | 0.0801 | 0.0549 | 0.0234 |
| n64_lbpr_e20 | 0.0727 | 0.1330 | 0.0801 | 0.0458 | 0.0259 |
| n32_lwarp_e20 | 0.0718 | 0.1090 | 0.0767 | 0.0623 | 0.0255 |
| n128_lwarp_e20 | 0.0711 | 0.0895 | 0.0975 | 0.0318 | 0.0359 |
| n32_lbpr_e20 | 0.0704 | 0.1390 | 0.0792 | 0.0425 | 0.0099 |
| n128_lbpr_e20 | 0.0700 | 0.1354 | 0.0812 | 0.0410 | 0.0066 |

ALS (sorted by aggregate NDCG@10):

| Slug | Aggregate | Cold | Warm | Regular | Heavy |
|---|---:|---:|---:|---:|---:|
| f64_r0.1_a40_i20 | **0.0919** | 0.1795 | 0.0844 | 0.0764 | 0.0396 |
| f64_r0.01_a40_i20 | 0.0886 | 0.1787 | 0.0848 | 0.0765 | 0.0113 |
| f128_r0.01_a40_i20 | 0.0835 | 0.1732 | 0.0837 | 0.0590 | 0.0192 |
| f128_r0.1_a40_i20 | 0.0808 | 0.1580 | 0.0886 | 0.0481 | 0.0236 |
| f32_r0.01_a40_i20 | 0.0775 | 0.1720 | 0.0823 | 0.0371 | 0.0260 |
| f32_r0.1_a40_i20 | 0.0722 | 0.1551 | 0.0708 | 0.0496 | 0.0190 |

The cleanest hyperparam win is ALS `regularization=0.1` over `0.01` at `factors=64`: aggregate NDCG@10 rises +3.7% (0.0886 -> 0.0919) and the heavy-segment number jumps ~3.5x (0.0113 -> 0.0396) at the single seed. For LightFM the best aggregate is exactly the baseline shape (`n64_lwarp`, 0.0738); BPR never beats WARP at the same `no_components`. Cold-start lead stays with ALS in every combo (best ALS cold 0.1795 vs best LightFM cold 0.1390).

### Winners vs single-point LOO at 3 seeds

The top-1 LightFM (`n64_lwarp_e20`) and top-1 ALS (`f64_r0.1_a40_i20`) were run through the full 9-model 3-seed segmented eval. Entries below are mean +/- std across seeds {42, 7, 1337}.

| Model + variant | Aggregate NDCG@10 | Cold NDCG@10 | Heavy NDCG@10 |
|---|---:|---:|---:|
| LightFM single-point LOO (n64_lwarp, original train) | 0.0678 +/- 0.0180 | 0.0846 +/- 0.0500 | 0.0524 +/- 0.0020 |
| LightFM sweep winner (n64_lwarp, fresh train) | 0.0631 +/- 0.0104 | 0.0803 +/- 0.0411 | 0.0516 +/- 0.0257 |
| ALS single-point LOO (f64_r0.01) | 0.0717 +/- 0.0132 | 0.1272 +/- 0.0569 | 0.0338 +/- 0.0047 |
| ALS sweep winner (f64_r0.1) | **0.0787 +/- 0.0115** | 0.1335 +/- 0.0486 | **0.0485 +/- 0.0141** |

For LightFM the two rows have the same hyperparameters; the gap (0.0678 vs 0.0631 aggregate) measures the training-noise floor at WARP / `no_components=64`, since `train_lightfm_model.py` does not plumb a `random_state` and each retrain reseeds internally. The 0.0047 gap is inside the std band -- no meaningful tuning gain for LightFM in the swept grid.

For ALS the regularization=0.1 winner lifts aggregate NDCG@10 by +0.0070 (+10%), heavy-segment NDCG@10 by +0.0147 (+43%), and pulls cold NDCG@10 up by +0.0063. Per-seed: tuned ALS beats single-point ALS on aggregate in 3 of 3 seeds; on heavy in 3 of 3 seeds. This is a real, seed-robust hyperparam gain.

### Updated per-seed leaderboard with tuned ALS

Per-seed heavy-segment NDCG@10 (tuned ALS vs sweep-winner LightFM): seed 42 -> ALS 0.0396 vs LightFM 0.0234 (ALS); seed 7 -> ALS 0.0647 vs LightFM 0.0735 (LightFM); seed 1337 -> ALS 0.0411 vs LightFM 0.0579 (LightFM). LightFM still wins heavy in 2 of 3 seeds, but the gap has shrunk: pre-sweep ALS heavy was 0.034 vs LightFM 0.052 (clear LightFM lead); post-sweep ALS heavy is 0.049 vs LightFM 0.052 (within one std). On the aggregate, tuned ALS now beats LightFM in 3 of 3 seeds.

The hyperparameter pass tightens the 2026-05-24 leaderboard. ALS keeps cold ownership and now reclaims aggregate leadership decisively; LightFM's heavy advantage narrows from "clear" to "marginal within seed variance".

## Conclusions

- The evaluation flow covers 9 top-N models. The ALS exclusion-semantics fix from `docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md` item 2 is applied in every run.
- `als_implicit` leads on the **leaked-artifact** baselines (every shape and every segment in the original 2026-05-22 / 2026-05-23 runs) but the picture shifts twice under correction. (a) Under LOO at single-point hyperparameters, ALS aggregate NDCG@10 falls to 0.0717 +/- 0.0132 and LightFM to 0.0678 +/- 0.0180 -- within one std at the aggregate. (b) Under LOO plus the hyperparameter sweep, ALS at `factors=64, regularization=0.1` lifts aggregate NDCG@10 to 0.0787 +/- 0.0115 and beats LightFM on aggregate in 3 of 3 seeds. ALS stays ~7 ms latency throughout.
- Per-segment under LOO + sweep, the leaderboard is: ALS owns `cold_0_10` (3 of 3 seeds, ~0.13 mean); ALS owns aggregate; LightFM still owns `heavy_200_plus` but only in 2 of 3 seeds (0.052 vs 0.049 mean -- within one std now versus a clear gap pre-sweep); the middle two buckets remain mixed within seed variance.
- The third-rank position is **not stable** and depends on the segment: `popularity` is third in the cold and warm buckets under both leaked and LOO; `tfidf_content` is third in regular under leaked; `sbert_faiss_content` rises near the top of `heavy_200_plus` once ALS / LightFM are leakage-corrected.
- Among classical CF: ALS > LightFM > SVD top-K on NDCG@10 and on mean latency, on every studied shape.
- `svd_topk` produces zero hits at K=10 on the small canonical slice and a small non-zero band at K=20 and at 300 users; the "Why SVD top-K stays at zero hits" subsection above explains this as an expected algorithmic limitation of RMSE-trained Surprise SVD, not a wiring bug.
- SVD rating prediction works and has RMSE 0.7558 / MAE 0.5706 on the sampled holdout.
- The previously-noted training-set leakage has now been **measured and corrected**: the "Leakage-corrected (leave-one-out) re-evaluation" subsection above quantifies its effect (ALS aggregate NDCG@10 loses ~70%; LightFM ~45%; the universal-leader claim does not survive). The corrected (LOO) results are the better basis for any forward-looking quality claim.

## Caveats

- This is a small local run, not a final benchmark.
- The canonical K=10 / K=20 tables at the top of this report come from a single 100-user deterministic-first-N latest-1 slice (run id `2026-05-20T18-47-21Z`). The Variance Bounds subsection covers seven additional runs across two slice sizes, two holdout shapes, and three seeds.
- 55 of the 100 selected users had positive holdout items in the canonical run; the multi-seed 100-user runs landed at 54 / 52 / 54.
- The holdout=3 expansion changes recall semantics: each user has three positive items in the denominator, capped at 1.0. Recall values across `holdout=1` and `holdout=3` runs are not directly comparable; the variance subsection treats them as separate tables.
- The training-set leakage caveat is now **measured rather than open**. The canonical K=10 / K=20 tables and the Variance Bounds / Cold-start segmentation subsections all use the leaked artifacts at `artifacts/models/{lightfm,als}/`; the absolute NDCG numbers there are inflated (ALS by ~70%, LightFM by ~45% at the 300u/h=3 shape). The Leakage-corrected and Hyperparameter sweep subsections above provide the deflated reference. Forward-looking quality claims should rely on the tuned LOO numbers (ALS f64_r0.1 aggregate 0.0787 +/- 0.0115, LightFM n64_lwarp 0.0631 +/- 0.0104), not the leaked ones.
- LightFM training is stochastic (the train script does not plumb a `random_state` to `LightFM.__init__`), so reruns at the same hyperparameters drift by roughly +/- 0.0145 NDCG@10. The reported LightFM numbers should be read with that noise floor in mind. ALS is more deterministic (drift ~0.003).
- The cold-start segmentation per-bucket sample sizes are small (cold ~44, warm ~111, regular ~69, heavy ~32 users per seed). Per-segment orderings are directionally informative across 3 seeds but should not be reported as definitive without a larger eval slice or a leave-one-out comparison.
- Generated artifacts are intentionally local and should be regenerated when code or data changes.
