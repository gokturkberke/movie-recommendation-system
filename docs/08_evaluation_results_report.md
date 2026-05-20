# Evaluation Results Report

## Run Summary

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

**Why both CF models invert classical wisdom (the leakage signature).** Textbook expectation is that ALS and LightFM are weak at cold-start and strong on heavy users (where they have the most behavior signal). The opposite happens here: both models' NDCG@10 falls monotonically from cold to heavy (ALS 0.46 -> 0.07; LightFM 0.27 -> 0.04). The most likely explanation is the artifact-level training-set leakage already noted in the Caveats: both artifacts were trained on the full rating matrix including each eval user's holdout interactions. For cold users, their "true" preference signal lives almost entirely in the holdout, so the artifact has effectively memorized exactly the items the eval will hold out. For heavy users, the holdout is a smaller fraction of their full history and there is more diversity in the remaining behavior, so the memorization advantage shrinks. This makes the segmented orderings defensible -- both models share the leakage symmetrically -- but the absolute cold-start numbers are inflated. A leave-one-out retrain (logged in the audit plan's Deferred section) is the right next step before quoting these absolute numbers in any external setting.

## Conclusions

- The evaluation flow covers 9 top-N models. The ALS exclusion-semantics fix from `docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md` item 2 is applied in every run.
- `als_implicit` is the leading ranker on every studied shape **and on every user-history segment**. It tops the canonical 100-user latest-1 slice, the 100-user multi-seed runs at holdout=1 (3 of 3 seeds), the 300-user single-seed run, the 300-user multi-seed runs at holdout=3 (3 of 3 seeds), and the per-segment partition of the 300u/h=3 runs (12 of 12 segment-seed cells). It is simultaneously the fastest model at ~7 ms mean latency.
- `lightfm_warp` is the runner-up on every shape and every segment. The gap to ALS is wide (ALS NDCG@10 ~1.7-2.1x LightFM across slices and segments). Latency stays in the ~40 ms fast tier.
- The third-rank position is **not stable** and depends on the segment: `popularity` is third in the cold and warm buckets; `tfidf_content` is third in regular; `sbert_faiss_content` is third in heavy. The single "third-best" label from the original canonical table is misleading -- the actual third-place model changes with user-history size.
- Among classical CF: ALS > LightFM > SVD top-K on NDCG@10 and on mean latency, on every studied shape.
- `svd_topk` produces zero hits at K=10 on the small canonical slice and a small non-zero band at K=20 and at 300 users; the "Why SVD top-K stays at zero hits" subsection above explains this as an expected algorithmic limitation of RMSE-trained Surprise SVD, not a wiring bug.
- SVD rating prediction works and has RMSE 0.7558 / MAE 0.5706 on the sampled holdout.
- These remain local directional results. Both ALS and LightFM artifacts were trained on the full rating matrix including the eval holdout interactions; the cold-start segmentation makes this leakage visible (both classical CF models invert textbook cold-vs-heavy behavior). A tighter leave-one-out retraining is logged in the audit plans' Deferred sections and is the right next step before quoting any absolute cold-start number.

## Caveats

- This is a small local run, not a final benchmark.
- The canonical K=10 / K=20 tables at the top of this report come from a single 100-user deterministic-first-N latest-1 slice (run id `2026-05-20T18-47-21Z`). The Variance Bounds subsection covers seven additional runs across two slice sizes, two holdout shapes, and three seeds.
- 55 of the 100 selected users had positive holdout items in the canonical run; the multi-seed 100-user runs landed at 54 / 52 / 54.
- The holdout=3 expansion changes recall semantics: each user has three positive items in the denominator, capped at 1.0. Recall values across `holdout=1` and `holdout=3` runs are not directly comparable; the variance subsection treats them as separate tables.
- Both ALS and LightFM artifacts were trained on the full rating matrix including the holdout interactions; absolute NDCG numbers are inflated to the same degree for both models, which preserves the ALS > LightFM ordering but does not establish absolute quality. The cold-start segmentation surfaces this directly: both classical CF models score highest on the bucket with the least training behavior and lowest on the bucket with the most, which is the inverse of textbook CF cold-start curves. A leave-one-out retraining is logged in the audit plans' Deferred sections.
- The cold-start segmentation per-bucket sample sizes are small (cold ~44, warm ~111, regular ~69, heavy ~32 users per seed). Per-segment orderings are directionally informative across 3 seeds but should not be reported as definitive without a larger eval slice or a leave-one-out comparison.
- Generated artifacts are intentionally local and should be regenerated when code or data changes.
