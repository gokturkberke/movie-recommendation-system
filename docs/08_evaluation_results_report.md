# Evaluation Results Report

## Run Summary

This report summarizes the latest local offline evaluation run:

```bash
.venv/bin/python scripts/evaluate_baselines.py \
  --max-users 25 --k 5,10,20 \
  --include-random --include-tfidf --include-content --include-semantic \
  --include-svd-topk --include-svd \
  --output-dir artifacts/evaluation
```

Run configuration:

- Users selected: 25
- Users with positive holdout items evaluated for top-N metrics: 15
- Holdout policy: latest 1 interaction per eligible user
- Positive threshold: rating >= 4.0
- Movie catalog rows: 79,477
- Ratings rows: 33,703,215
- Semantic baseline: TF-IDF + TruncatedSVD LSA, not SBERT/FAISS

Generated local artifacts:

- `artifacts/evaluation/metrics_summary.csv`
- `artifacts/evaluation/metrics_summary.json`
- timestamped CSV/JSON copies
- `artifacts/evaluation/run_config.json`

These artifacts are local/generated and are not tracked by git.

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

| Model | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 | Coverage | Diversity | Novelty | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0133 | 0.1333 | 0.1333 | 0.0430 | 0.00036 | 0.7952 | 8.5845 | 39.9 ms |
| hybrid_content | 0.0133 | 0.1333 | 0.1333 | 0.0401 | 0.00094 | 0.6897 | 9.7544 | 13,491.9 ms |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00189 | 0.8774 | 11.9178 | 11.7 ms |
| semantic_content | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00135 | 0.5642 | 11.2597 | 110.5 ms |
| svd_topk | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00122 | 0.7800 | 11.0585 | 186.3 ms |
| tfidf_content | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00096 | 0.6380 | 10.0830 | 51.0 ms |

At K=10, `hybrid_content` ties `popularity` on precision, recall, and hit rate, but has slightly lower NDCG. Hybrid recommends a broader and more novel set than popularity, but it does not currently show a clear relevance win in this run.

### Top-N at K=20

| Model | Precision@20 | Recall@20 | HitRate@20 | NDCG@20 | Coverage | Diversity | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid_content | 0.0133 | 0.2667 | 0.2667 | 0.0744 | 0.00157 | 0.6721 | 10.0125 |
| popularity | 0.0100 | 0.2000 | 0.2000 | 0.0582 | 0.00062 | 0.8055 | 8.8027 |
| svd_topk | 0.0067 | 0.1333 | 0.1333 | 0.0331 | 0.00243 | 0.7865 | 11.1930 |
| tfidf_content | 0.0033 | 0.0667 | 0.0667 | 0.0167 | 0.00170 | 0.6598 | 10.2603 |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00377 | 0.8581 | 11.8972 |
| semantic_content | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00244 | 0.5672 | 11.3028 |

At K=20, `hybrid_content` is the strongest top-N model in this run by recall, hit rate, and NDCG. The gain is small and based on only 15 users with positive holdout items, so this should be treated as directional rather than conclusive.

### SVD Rating Prediction

The SVD explicit rating prediction baseline was evaluated separately from top-N ranking:

| Metric | Value |
|---|---:|
| RMSE | 0.7241 |
| MAE | 0.5097 |
| Prediction count | 25 |

These values measure rating prediction error, not recommendation list quality.

## Latency Findings

`hybrid_content` is the clear latency problem:

| Model | Mean latency | p95 latency |
|---|---:|---:|
| random | 11.7 ms | 12.8 ms |
| popularity | 39.9 ms | 51.8 ms |
| tfidf_content | 51.0 ms | 104.9 ms |
| semantic_content | 110.5 ms | 155.0 ms |
| svd_topk | 186.3 ms | 190.7 ms |
| hybrid_content | 13,491.9 ms | 43,753.9 ms |

The hybrid path is orders of magnitude slower than the other baselines. This should be profiled before optimization. The likely cause is repeated per-seed content recommendation work in the watch-history hybrid flow.

## Follow-Up Checkpoints

After this report was first generated, the evaluation CSV/JSON schema was extended with `map_at_k` and `mrr_at_k`. The watch-history hybrid path was also optimized by batching seed similarity computation and deferring hybrid/diversity reranking until after candidate aggregation.

Local five-user profiling before the optimization showed roughly 6.9 seconds mean per-user hybrid latency. After the optimization, the same profiler showed roughly 0.58 seconds mean per-user latency. A ten-user content-only evaluation smoke showed `hybrid_content` mean latency around 664 ms.

A real SBERT + FAISS evaluation baseline was also added behind `--include-sbert-faiss`. A 1,000-row smoke index produced `sbert_faiss_content` rows successfully, but this is not a full-catalog benchmark. Build full SBERT + FAISS artifacts before comparing it against the other baselines.

## Conclusions

- The evaluation flow now covers every currently available baseline.
- `hybrid_content` only clearly beats popularity at K=20 in this run.
- `popularity` remains a strong simple baseline at K=10.
- `semantic_content` currently means semantic-LSA, not real SBERT/FAISS, and did not produce hits in this run.
- `svd_topk` produced one K=20 hit pattern but did not beat hybrid or popularity.
- SVD rating prediction works and has RMSE 0.7241 / MAE 0.5097 on the sampled holdout.
- The next technical priority should be regenerating a full post-optimization report before making model-quality claims.

## Caveats

- This is a small local run, not a final benchmark.
- Only 15 users had positive holdout items for top-N evaluation.
- The holdout size is one latest interaction per user.
- Generated artifacts are intentionally local and should be regenerated when code or data changes.
