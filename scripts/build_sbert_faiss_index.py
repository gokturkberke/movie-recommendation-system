import argparse
import json

from config import EVALUATION_DEFAULTS, project_path
from data_access import load_movies, load_tags
from experimental.sbert_faiss import build_sbert_faiss_artifacts


_SBERT_DEFAULTS = EVALUATION_DEFAULTS.get("sbert_faiss") or {}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build SBERT embeddings and a FAISS index for MovieLens movies.")
    parser.add_argument("--output-dir", default=str(project_path(_SBERT_DEFAULTS.get("index_dir", "artifacts/indexes/sbert_faiss"))), help="Directory for FAISS index artifacts.")
    parser.add_argument("--model-name", default=_SBERT_DEFAULTS.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"), help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=int(_SBERT_DEFAULTS.get("batch_size", 64)), help="SBERT encode batch size.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional first-N movie sample for smoke builds.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    artifacts = build_sbert_faiss_artifacts(
        load_movies(),
        load_tags(),
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
    )
    print(json.dumps(artifacts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
