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
| lightfm_warp | 0.0255 | 0.2545 | 0.2545 | 0.1427 | 0.1100 | 0.1100 | 0.0032 | 0.7540 | 9.8075 | 43.6 ms |
| hybrid_content | 0.0073 | 0.0727 | 0.0727 | 0.0382 | 0.0283 | 0.0283 | 0.0019 | 0.6902 | 10.0856 | 1,464.4 ms |
| popularity | 0.0091 | 0.0909 | 0.0909 | 0.0322 | 0.0150 | 0.0150 | 0.0005 | 0.7936 | 8.6046 | 95.5 ms |
| sbert_faiss_content | 0.0036 | 0.0364 | 0.0364 | 0.0193 | 0.0136 | 0.0136 | 0.0039 | 0.7165 | 11.6010 | 39.0 ms |
| semantic_content | 0.0018 | 0.0182 | 0.0182 | 0.0182 | 0.0182 | 0.0182 | 0.0045 | 0.5549 | 12.4005 | 86.5 ms |
| tfidf_content | 0.0036 | 0.0364 | 0.0364 | 0.0139 | 0.0071 | 0.0071 | 0.0020 | 0.6826 | 10.0908 | 46.3 ms |
| als_implicit | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0041 | 0.7639 | 10.2227 | 7.8 ms |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0069 | 0.8579 | 13.5810 | 12.5 ms |
| svd_topk | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0029 | 0.7756 | 12.3858 | 189.4 ms |

At K=10, `lightfm_warp` is the clear ranking leader by precision, recall, hit rate, NDCG, MAP, and MRR. It also serves in the same latency tier as SBERT+FAISS and TF-IDF content scoring, while `hybrid_content` remains far slower.

### Top-N at K=20

| Model | Precision@20 | Recall@20 | HitRate@20 | NDCG@20 | MAP@20 | MRR@20 | Coverage | Diversity | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lightfm_warp | 0.0173 | 0.3455 | 0.3455 | 0.1657 | 0.1163 | 0.1163 | 0.0054 | 0.7672 | 10.0121 |
| hybrid_content | 0.0064 | 0.1273 | 0.1273 | 0.0525 | 0.0324 | 0.0324 | 0.0033 | 0.7050 | 10.4092 |
| popularity | 0.0055 | 0.1091 | 0.1091 | 0.0367 | 0.0162 | 0.0162 | 0.0008 | 0.7941 | 8.8305 |
| tfidf_content | 0.0045 | 0.0909 | 0.0909 | 0.0280 | 0.0111 | 0.0111 | 0.0035 | 0.6833 | 10.3564 |
| sbert_faiss_content | 0.0027 | 0.0545 | 0.0545 | 0.0236 | 0.0146 | 0.0146 | 0.0071 | 0.7395 | 11.7631 |
| semantic_content | 0.0018 | 0.0364 | 0.0364 | 0.0230 | 0.0196 | 0.0196 | 0.0082 | 0.5573 | 12.4960 |
| svd_topk | 0.0036 | 0.0727 | 0.0727 | 0.0178 | 0.0046 | 0.0046 | 0.0055 | 0.7849 | 12.5488 |
| als_implicit | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0068 | 0.7597 | 10.3882 |
| random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0138 | 0.8484 | 13.5795 |

At K=20, `lightfm_warp` remains the strongest model by every relevance metric. `hybrid_content` is the next-best ranking model, while `als_implicit` is fast but produces no relevance hits on this slice.

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
| als_implicit | 7.8 ms | 11.7 ms |
| random | 12.5 ms | 13.7 ms |
| sbert_faiss_content | 39.0 ms | 80.2 ms |
| lightfm_warp | 43.6 ms | 47.6 ms |
| tfidf_content | 46.3 ms | 91.7 ms |
| semantic_content | 86.5 ms | 119.0 ms |
| popularity | 95.5 ms | 40.5 ms |
| svd_topk | 189.4 ms | 200.1 ms |
| hybrid_content | 1,464.4 ms | 4,369.9 ms |

`als_implicit` is the fastest model in this run, followed by random sampling. `lightfm_warp` lands in the fast semantic/content tier at 43.6 ms mean while also leading the relevance tables. `hybrid_content` stays under the 2,000 ms mean-latency gate, but remains the slowest baseline.

## Conclusions

- The evaluation flow now covers 9 top-N models on the same 100-user slice, including LightFM WARP and Implicit ALS.
- `lightfm_warp` is the strongest model at K=10 and K=20 by precision, recall, hit rate, NDCG, MAP, and MRR.
- `hybrid_content` is now the second-best model by NDCG at K=10 and K=20, but it is far slower than LightFM.
- `popularity` remains a useful simple baseline, but it no longer leads any rank-sensitive table after LightFM is added.
- `als_implicit` is the fastest model at 7.8 ms mean latency, but it produces no relevance hits in this run.
- Among the classical CF models, LightFM WARP beats both Surprise SVD top-K and Implicit ALS on NDCG@10 and mean latency; ALS beats SVD on latency but not relevance.
- SVD rating prediction works and has RMSE 0.7558 / MAE 0.5706 on the sampled holdout.
- These are still local directional results from one 100-user latest-1 holdout run, not a final benchmark claim.

## Caveats

- This is a small local run, not a final benchmark.
- 55 of the 100 selected users had positive holdout items for top-N evaluation; the plan's target of at least 60 was not met, but the wider run still produced content-baseline hits that the 25-user slice missed.
- The holdout size is one latest interaction per user.
- Generated artifacts are intentionally local and should be regenerated when code or data changes.
