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

## Conclusions

- The evaluation flow covers 9 top-N models on the same 100-user slice. The ALS exclusion-semantics fix from `docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md` item 2 is applied in this run.
- `als_implicit` (post-fix) is now the strongest model at K=10 and K=20 by every relevance metric -- precision, recall, hit rate, NDCG, MAP, MRR -- and simultaneously the fastest at 7.1 ms mean latency. The fix flipped `filter_already_liked_items` to False; the artifact itself was not retrained.
- `lightfm_warp` is the runner-up on every relevance metric and is also in the fast tier (40.4 ms mean).
- `hybrid_content` is the third-best ranker by NDCG but remains the slowest baseline.
- `popularity` is still a useful simple baseline at K=10 (Precision 0.0091, Recall 0.0909) but is below both classical CF leaders on every rank-sensitive metric.
- Among classical CF: ALS > LightFM > SVD top-K on NDCG@10 and on mean latency.
- `svd_topk` produces zero hits at K=10 and a small non-zero band at K=20; item 1 of the audit plan diagnosed this as an expected algorithmic limitation of RMSE-trained Surprise SVD, not a wiring bug.
- SVD rating prediction works and has RMSE 0.7558 / MAE 0.5706 on the sampled holdout.
- These are still local directional results from one 100-user latest-1 holdout run, not a final benchmark claim.

## Caveats

- This is a small local run, not a final benchmark.
- 55 of the 100 selected users had positive holdout items for top-N evaluation; the plan's target of at least 60 was not met, but the wider run still produced content-baseline hits that the 25-user slice missed.
- The holdout size is one latest interaction per user.
- Generated artifacts are intentionally local and should be regenerated when code or data changes.
