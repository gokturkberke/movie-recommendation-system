import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from config import EVALUATION_DEFAULTS, EVALUATION_OUTPUT_DIR
from evaluation_runner import parse_k_values, run_evaluation


_SEMANTIC_DEFAULTS = EVALUATION_DEFAULTS.get("semantic") or {}
_SBERT_FAISS_DEFAULTS = EVALUATION_DEFAULTS.get("sbert_faiss") or {}
_LIGHTFM_DEFAULTS = EVALUATION_DEFAULTS.get("lightfm") or {}
_ALS_DEFAULTS = EVALUATION_DEFAULTS.get("als") or {}
_DEFAULT_K_VALUES = EVALUATION_DEFAULTS.get("k_values") or [10]
_DEFAULT_K_STR = ",".join(str(int(value)) for value in _DEFAULT_K_VALUES)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate offline recommendation baselines.")
    parser.add_argument("--max-users", type=int, default=int(EVALUATION_DEFAULTS.get("max_users", 100)), help="Maximum eligible users to evaluate. Use 0 for all.")
    parser.add_argument("--k", default=_DEFAULT_K_STR, help="Comma-separated top-N cutoffs, for example 5,10,20.")
    parser.add_argument("--holdout-count", type=int, default=int(EVALUATION_DEFAULTS.get("holdout_count", 1)), help="Latest interactions held out per user.")
    parser.add_argument("--min-interactions", type=int, default=int(EVALUATION_DEFAULTS.get("min_interactions", 5)), help="Minimum interactions required per user.")
    parser.add_argument("--positive-threshold", type=float, default=float(EVALUATION_DEFAULTS.get("positive_threshold", 4.0)), help="Rating threshold treated as positive.")
    parser.add_argument("--include-random", action="store_true", help="Evaluate the random baseline.")
    parser.add_argument("--include-tfidf", action="store_true", help="Evaluate the pure TF-IDF content baseline (no hybrid rerank).")
    parser.add_argument("--include-content", action="store_true", help="Evaluate the watch-history hybrid (TF-IDF + Bayesian + popularity + diversity).")
    parser.add_argument("--include-semantic", action="store_true", help="Evaluate the semantic content baseline (TF-IDF + TruncatedSVD LSA, watch-history seeds, max-similarity aggregation).")
    parser.add_argument("--semantic-components", type=int, default=int(_SEMANTIC_DEFAULTS.get("components", 64)), help="Latent dimensions for the semantic embedding index (TruncatedSVD).")
    parser.add_argument("--semantic-random-state", type=int, default=int(_SEMANTIC_DEFAULTS.get("random_state", 42)), help="Random state used by the semantic TruncatedSVD fit.")
    parser.add_argument("--include-sbert-faiss", action="store_true", help="Evaluate the prebuilt SBERT+FAISS content baseline.")
    parser.add_argument("--sbert-faiss-index-dir", default=_SBERT_FAISS_DEFAULTS.get("index_dir", "artifacts/indexes/sbert_faiss"), help="Directory containing prebuilt SBERT+FAISS artifacts.")
    parser.add_argument("--include-lightfm", action="store_true", help="Evaluate the prebuilt LightFM WARP baseline.")
    parser.add_argument("--lightfm-artifacts-dir", default=_LIGHTFM_DEFAULTS.get("artifacts_dir", "artifacts/models/lightfm"), help="Directory containing prebuilt LightFM artifacts.")
    parser.add_argument("--include-als", action="store_true", help="Evaluate the prebuilt Implicit ALS baseline.")
    parser.add_argument("--als-artifacts-dir", default=_ALS_DEFAULTS.get("artifacts_dir", "artifacts/models/als"), help="Directory containing prebuilt ALS artifacts.")
    parser.add_argument("--include-svd-topk", action="store_true", help="Evaluate SVD top-K recommendations from the trained Surprise model.")
    parser.add_argument("--include-svd", action="store_true", help="Evaluate SVD holdout rating prediction (RMSE/MAE).")
    parser.add_argument("--no-measure-latency", action="store_true", help="Disable per-user latency measurement.")
    parser.add_argument("--random-seed", type=int, default=int(EVALUATION_DEFAULTS.get("random_seed", 42)), help="Seed used by the random baseline.")
    parser.add_argument("--user-sample-seed", type=int, default=None, help="Optional seed for random user-pool sampling. When unset, the eligible-user list is taken first-N (deterministic, backward compatible).")
    parser.add_argument("--output-dir", default=str(EVALUATION_OUTPUT_DIR), help="Directory for metrics_summary.json/csv. Use empty string to disable saving.")
    parser.add_argument("--example-count", type=int, default=0, help="Include this many recommendation examples.")
    parser.add_argument("--include-reasons", action="store_true", help="Include hybrid explanation text in examples.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir else None
    report = run_evaluation(
        max_users=args.max_users,
        k_values=parse_k_values(args.k),
        holdout_count=args.holdout_count,
        min_interactions=args.min_interactions,
        positive_threshold=args.positive_threshold,
        include_random=args.include_random,
        include_tfidf=args.include_tfidf,
        include_content=args.include_content,
        include_semantic=args.include_semantic,
        include_sbert_faiss=args.include_sbert_faiss,
        include_lightfm=args.include_lightfm,
        include_als=args.include_als,
        include_svd_topk=args.include_svd_topk,
        include_svd=args.include_svd,
        measure_latency=not args.no_measure_latency,
        output_dir=output_dir,
        random_seed=args.random_seed,
        user_sample_seed=args.user_sample_seed,
        semantic_components=args.semantic_components,
        semantic_random_state=args.semantic_random_state,
        sbert_faiss_index_dir=args.sbert_faiss_index_dir,
        lightfm_artifacts_dir=args.lightfm_artifacts_dir,
        als_artifacts_dir=args.als_artifacts_dir,
        example_count=args.example_count,
        include_reasons=args.include_reasons,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
