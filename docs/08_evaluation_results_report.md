# Evaluation Results Report

## Run Summary

This report summarizes the latest local offline evaluation run:

```bash
.venv/bin/python scripts/evaluate_baselines.py \
  --max-users 100 --k 5,10,20 \
  --include-random --include-tfidf --include-content --include-semantic \
  --include-svd --include-svd-topk \
  --include-sbert-faiss \
  --sbert-faiss-index-dir artifacts/indexes/sbert_faiss \
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

Generated local artifacts:

- `artifacts/evaluation/metrics_summary.csv`
- `artifacts/evaluation/metrics_summary.json`
- timestamped CSV/JSON copies
- `artifacts/evaluation/run_config.json`

These artifacts are local/generated and are not tracked by git.

The audit trail for this wider run lives in `docs/experiments/2026-05-20_classical-cf-and-eval-expansion.md`.

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
| hybrid_content | 0.0073 | 0.0727 | 0.0727 | 0.0382 | 0.0283 | 0.0283 | 0.0019 | 0.6902 | 10.0856 | 1,466.0 ms |
| popularity | 0.0091 | 0.0909 | 0.0909 | 0.0322 | 0.0150 | 0.0150 | 0.0005 | 0.7936 | 8.6046 | 87.9 ms |
| sbert_faiss_content | 0.0036 | 0.0364 | 0.0364 | 0.0193 | 0.0136 | 0.0136 | 0.0039 | 0.7165 | 11.6010 | 40.7 ms |
| semantic_content | 0.0018 | 0.0182 | 0.0182 | 0.0182 | 0.0182 | 0.0182 | 0.0045 | 0.5549 | 12.4005 | 99.1 ms |
| tfidf_content | 0.0036 | 0.0364 | 0.0364 | 0.0139 | 0.0071 | 0.0071 | 0.0020 | 0.6826 | 10.0908 | 46.7 ms |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0069 | 0.8579 | 13.5810 | 11.7 ms |
| svd_topk | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0029 | 0.7756 | 12.3858 | 189.6 ms |

At K=10, `hybrid_content` leads by NDCG, MAP, and MRR even though `popularity` has the highest precision, recall, and hit rate. `sbert_faiss_content`, `semantic_content`, and `tfidf_content` now all register hits on the wider slice, which was not true in the 25-user run.

### Top-N at K=20

| Model | Precision@20 | Recall@20 | HitRate@20 | NDCG@20 | MAP@20 | MRR@20 | Coverage | Diversity | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_content | 0.0064 | 0.1273 | 0.1273 | 0.0525 | 0.0324 | 0.0324 | 0.0033 | 0.7050 | 10.4092 |
| popularity | 0.0055 | 0.1091 | 0.1091 | 0.0367 | 0.0162 | 0.0162 | 0.0008 | 0.7941 | 8.8305 |
| tfidf_content | 0.0045 | 0.0909 | 0.0909 | 0.0280 | 0.0111 | 0.0111 | 0.0035 | 0.6833 | 10.3564 |
| sbert_faiss_content | 0.0027 | 0.0545 | 0.0545 | 0.0236 | 0.0146 | 0.0146 | 0.0071 | 0.7395 | 11.7631 |
| semantic_content | 0.0018 | 0.0364 | 0.0364 | 0.0230 | 0.0196 | 0.0196 | 0.0082 | 0.5573 | 12.4960 |
| svd_topk | 0.0036 | 0.0727 | 0.0727 | 0.0178 | 0.0046 | 0.0046 | 0.0055 | 0.7849 | 12.5488 |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0138 | 0.8484 | 13.5795 |

At K=20, `hybrid_content` is still the strongest top-N model by NDCG, MAP, and MRR. `popularity` remains second on NDCG, while TF-IDF content has the next-best recall and hit rate among the non-popularity baselines.

### SVD Rating Prediction

The SVD explicit rating prediction baseline was evaluated separately from top-N ranking:

| Metric | Value |
|---|---:|
| RMSE | 0.7558 |
| MAE | 0.5706 |
| Prediction count | 100 |

These values measure rating prediction error, not recommendation list quality.

## Latency Findings

Watch-history hybrid is no longer the latency outlier:

| Model | Mean latency | p95 latency |
|---|---:|---:|
| random | 11.7 ms | 12.2 ms |
| sbert_faiss_content | 40.7 ms | 83.7 ms |
| tfidf_content | 46.7 ms | 84.5 ms |
| popularity | 87.9 ms | 41.6 ms |
| semantic_content | 99.1 ms | 134.4 ms |
| svd_topk | 189.6 ms | 192.4 ms |
| hybrid_content | 1,466.0 ms | 4,362.4 ms |

`hybrid_content` stays under the 2,000 ms mean-latency gate on the wider 100-user run, but remains the slowest baseline. SBERT+FAISS is still the fastest semantic-aware option at 40.7 ms mean, close to the pure TF-IDF content path and materially faster than semantic-LSA.

## Conclusions

- The evaluation flow now covers every pre-classical-CF baseline on a 100-user slice, including the real SBERT+FAISS full-catalog index (79,477 movies, 384-dim).
- `hybrid_content` is the strongest top-N model by NDCG at K=10 and K=20, and it remains under the 2,000 ms mean latency gate.
- `popularity` remains a strong simple baseline; it leads precision, recall, and hit rate at K=10, but trails hybrid on rank-sensitive metrics.
- `sbert_faiss_content`, `tfidf_content`, and `semantic_content` now all produce relevance hits on the wider run, giving LightFM and ALS a more meaningful reference point than the 25-user report.
- `svd_topk` still does not beat hybrid or popularity in top-N ranking on this slice.
- SVD rating prediction works and has RMSE 0.7558 / MAE 0.5706 on the sampled holdout.
- The next step is to add LightFM and Implicit ALS, then regenerate the final 9-model report on this same 100-user protocol.

## Caveats

- This is a small local run, not a final benchmark.
- 55 of the 100 selected users had positive holdout items for top-N evaluation; the plan's target of at least 60 was not met, but the wider run still produced content-baseline hits that the 25-user slice missed.
- The holdout size is one latest interaction per user.
- Generated artifacts are intentionally local and should be regenerated when code or data changes.
