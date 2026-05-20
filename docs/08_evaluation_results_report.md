# Evaluation Results Report

## Run Summary

This report summarizes the latest local offline evaluation run:

```bash
.venv/bin/python scripts/evaluate_baselines.py \
  --max-users 25 --k 5,10,20 \
  --include-random --include-tfidf --include-content --include-semantic \
  --include-svd --include-svd-topk \
  --include-sbert-faiss \
  --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
  --output-dir artifacts/evaluation
```

Run configuration:

- Users selected: 25
- Users with positive holdout items evaluated for top-N metrics: 15
- Holdout policy: latest 1 interaction per eligible user
- Positive threshold: rating >= 4.0
- Movie catalog rows: 79,477
- Ratings rows: 33,703,215
- Semantic baseline: TF-IDF + TruncatedSVD LSA (`--include-semantic`)
- SBERT+FAISS baseline: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, full-catalog index under `artifacts/indexes/sbert_faiss/` (`row_count = 79,477`)

Generated local artifacts:

- `artifacts/evaluation/metrics_summary.csv`
- `artifacts/evaluation/metrics_summary.json`
- timestamped CSV/JSON copies
- `artifacts/evaluation/run_config.json`

These artifacts are local/generated and are not tracked by git.

The audit trail for this run lives in `docs/experiments/2026-05-20_sbert-faiss-full-run.md`.

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
| popularity | 0.0133 | 0.1333 | 0.1333 | 0.0430 | 0.0178 | 0.0178 | 0.00036 | 0.7952 | 8.5845 | 38.9 ms |
| hybrid_content | 0.0133 | 0.1333 | 0.1333 | 0.0401 | 0.0148 | 0.0148 | 0.00094 | 0.6897 | 9.7544 | 1,320.7 ms |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00189 | 0.8774 | 11.9178 | 10.8 ms |
| sbert_faiss_content | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00126 | 0.7297 | 10.9657 | 36.1 ms |
| semantic_content | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00135 | 0.5642 | 11.2597 | 117.8 ms |
| svd_topk | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00122 | 0.7800 | 11.0585 | 194.2 ms |
| tfidf_content | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00096 | 0.6380 | 10.0830 | 43.4 ms |

At K=10, `popularity` and `hybrid_content` tie on precision, recall, and hit rate. Popularity has a slight NDCG edge because its single hit lands at a higher rank. The other five models produce no hits at K=10 in this 15-user slice.

### Top-N at K=20

| Model | Precision@20 | Recall@20 | HitRate@20 | NDCG@20 | MAP@20 | MRR@20 | Coverage | Diversity | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_content | 0.0133 | 0.2667 | 0.2667 | 0.0744 | 0.0246 | 0.0246 | 0.00157 | 0.6721 | 10.0125 |
| popularity | 0.0100 | 0.2000 | 0.2000 | 0.0582 | 0.0211 | 0.0211 | 0.00062 | 0.8055 | 8.8027 |
| svd_topk | 0.0067 | 0.1333 | 0.1333 | 0.0331 | 0.0087 | 0.0087 | 0.00243 | 0.7865 | 11.1930 |
| tfidf_content | 0.0033 | 0.0667 | 0.0667 | 0.0167 | 0.0044 | 0.0044 | 0.00170 | 0.6598 | 10.2603 |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00377 | 0.8581 | 11.8972 |
| sbert_faiss_content | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00247 | 0.7355 | 11.0031 |
| semantic_content | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00244 | 0.5672 | 11.3028 |

At K=20, `hybrid_content` is the strongest top-N model in this run by recall, hit rate, NDCG, MAP, and MRR. The margin over popularity is small and based on only 15 users with positive holdout items, so this should be treated as directional rather than conclusive.

### SVD Rating Prediction

The SVD explicit rating prediction baseline was evaluated separately from top-N ranking:

| Metric | Value |
|---|---:|
| RMSE | 0.7241 |
| MAE | 0.5097 |
| Prediction count | 25 |

These values measure rating prediction error, not recommendation list quality.

## Latency Findings

Watch-history hybrid is no longer the latency outlier:

| Model | Mean latency | p95 latency |
|---|---:|---:|
| random | 10.8 ms | 12.0 ms |
| sbert_faiss_content | 36.1 ms | 74.5 ms |
| popularity | 38.9 ms | 44.9 ms |
| tfidf_content | 43.4 ms | 83.2 ms |
| semantic_content | 117.8 ms | 160.3 ms |
| svd_topk | 194.2 ms | 214.6 ms |
| hybrid_content | 1,320.7 ms | 4,007.3 ms |

`hybrid_content` mean latency dropped from 13,491.9 ms in the pre-optimization run (commit `c2793c4` batched watch-history seed similarity and deferred hybrid/diversity rerank) to 1,320.7 ms here, an order-of-magnitude improvement on the same 25-user slice. SBERT+FAISS is the fastest semantic-aware option at 36.1 ms mean — comparable to popularity and faster than TF-IDF content scoring.

## Conclusions

- The evaluation flow now covers every available baseline, including the real SBERT+FAISS full-catalog index (79,477 movies, 384-dim).
- `hybrid_content` is the strongest top-N model at K=20 by recall, hit rate, NDCG, MAP, and MRR.
- `popularity` remains a strong simple baseline at K=10 and slightly edges hybrid on NDCG@10 due to rank position.
- `sbert_faiss_content` reaches broader catalog coverage and higher novelty than TF-IDF content, but does not produce relevance hits on this 15-user holdout slice — same outcome as `tfidf_content` and `semantic_content`, which all rely on content seeds rather than user history.
- `svd_topk` produced one K=20 hit pattern but did not beat hybrid or popularity.
- SVD rating prediction works and has RMSE 0.7241 / MAE 0.5097 on the sampled holdout.
- The next investigation should widen the user slice and the holdout to make the content-only baselines comparable on relevance, not just on cost and coverage.

## Caveats

- This is a small local run, not a final benchmark.
- Only 15 users had positive holdout items for top-N evaluation.
- The holdout size is one latest interaction per user.
- Generated artifacts are intentionally local and should be regenerated when code or data changes.
