"""Diagnostic dump for ALS and SVD top-K zero-hit behavior.

For N sampled positive-holdout users from the canonical evaluation slice,
prints whether the holdout movieId lives in the ALS item_index / LightFM
item_index / per-user user_items row, the ALS recommend output with
filter_already_liked_items in both True and False modes, the SVD top-K
predictions over the train-excluded catalog, and the SVD full-catalog
rank of the holdout. Output is human-readable text.

Audit trail: docs/experiments/2026-05-21_als-svd-zero-hit-investigation.md
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from config import EVALUATION_DEFAULTS, project_path
from data_access import load_movies, load_ratings, load_surprise_model
from evaluation import temporal_train_test_split
from evaluation_runner import filter_to_users, select_evaluation_user_ids
from experimental.als_recommender import load_als_artifacts
from experimental.lightfm_recommender import load_lightfm_artifacts


def parse_args():
    defaults = EVALUATION_DEFAULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=int(defaults.get("random_seed", 42)))
    parser.add_argument("--max-users", type=int, default=int(defaults.get("max_users", 100)))
    parser.add_argument("--min-interactions", type=int, default=int(defaults.get("min_interactions", 5)))
    parser.add_argument("--holdout-count", type=int, default=int(defaults.get("holdout_count", 1)))
    parser.add_argument("--positive-threshold", type=float, default=float(defaults.get("positive_threshold", 4.0)))
    parser.add_argument(
        "--als-artifacts-dir",
        type=str,
        default=str(project_path(defaults["als"]["artifacts_dir"])),
    )
    parser.add_argument(
        "--lightfm-artifacts-dir",
        type=str,
        default=str(project_path(defaults["lightfm"]["artifacts_dir"])),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-path", type=str, default=None)
    return parser.parse_args()


def als_raw_recommend(artifacts, user_position, requested_n, filter_already_liked):
    item_positions, scores = artifacts.model.recommend(
        int(user_position),
        artifacts.user_items[int(user_position)],
        N=int(requested_n),
        filter_already_liked_items=bool(filter_already_liked),
    )
    item_positions = np.asarray(item_positions).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    movie_id_by_position = {
        int(position): int(movie_id)
        for movie_id, position in artifacts.item_index.items()
    }
    rows = []
    for position, score in zip(item_positions, scores):
        pos_int = int(position)
        if pos_int not in movie_id_by_position:
            continue
        rows.append((movie_id_by_position[pos_int], float(score)))
    frame = pd.DataFrame(rows, columns=["movieId", "score"])
    if not frame.empty:
        frame.insert(0, "rank", frame.index + 1)
    return frame


def svd_train_excluded_predictions(svd_model, user_id, catalog_movie_ids, user_train_movie_ids):
    rated = set(int(mid) for mid in user_train_movie_ids)
    rows = []
    for movie_id in catalog_movie_ids:
        if int(movie_id) in rated:
            continue
        prediction = svd_model.predict(uid=int(user_id), iid=int(movie_id))
        rows.append((int(movie_id), float(prediction.est)))
    frame = pd.DataFrame(rows, columns=["movieId", "predicted_score"])
    frame = frame.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", frame.index + 1)
    return frame


def format_rows(frame, holdout_movie_id, score_col):
    lines = []
    for _, row in frame.iterrows():
        marker = "  <-- HOLDOUT" if int(row["movieId"]) == int(holdout_movie_id) else ""
        lines.append(
            f"    rank={int(row['rank']):>3d}  movieId={int(row['movieId']):>7d}  {score_col}={row[score_col]:>9.4f}{marker}"
        )
    return lines


def main():
    args = parse_args()

    print("Loading ratings, movies, models...", file=sys.stderr)
    ratings = load_ratings()
    movies = load_movies()
    als_artifacts = load_als_artifacts(args.als_artifacts_dir)
    lightfm_artifacts = load_lightfm_artifacts(args.lightfm_artifacts_dir)
    svd_model, svd_error = load_surprise_model()
    if svd_error:
        print(f"WARNING: SVD model unavailable: {svd_error}", file=sys.stderr)

    print("Selecting evaluation users and splitting...", file=sys.stderr)
    selected_user_ids = select_evaluation_user_ids(
        ratings,
        max_users=args.max_users,
        min_interactions=args.min_interactions,
        holdout_count=args.holdout_count,
    )
    sampled_ratings = filter_to_users(ratings, selected_user_ids)
    train, holdout = temporal_train_test_split(
        sampled_ratings,
        holdout_count=args.holdout_count,
        min_interactions=args.min_interactions,
    )
    positive_holdout = holdout[holdout["rating"] >= args.positive_threshold].copy()
    positive_user_ids = positive_holdout["userId"].astype(int).drop_duplicates().tolist()

    rng = np.random.default_rng(args.random_seed)
    sample_size = min(args.sample_size, len(positive_user_ids))
    sampled = sorted(
        int(uid)
        for uid in rng.choice(positive_user_ids, size=sample_size, replace=False).tolist()
    )

    catalog_movie_ids = movies["movieId"].dropna().astype(int).unique().tolist()

    lines = [
        "# ALS / SVD top-K zero-hit diagnostic",
        f"selected_users={len(selected_user_ids)}  positive_holdout_users={len(positive_user_ids)}  sampled={sampled}",
        f"als_artifact_users={len(als_artifacts.user_index)}  als_artifact_items={len(als_artifacts.item_index)}",
        f"lightfm_artifact_users={len(lightfm_artifacts.user_index)}  lightfm_artifact_items={len(lightfm_artifacts.item_index)}",
        f"catalog_movie_count={len(catalog_movie_ids)}",
        "",
    ]

    counters = {
        "holdout_in_als_index": 0,
        "holdout_in_lightfm_index": 0,
        "holdout_in_user_items": 0,
        "als_filter_true_hits": 0,
        "als_filter_false_hits": 0,
        "svd_topk_hits": 0,
        "svd_rank_over_50": 0,
    }

    for user_id in sampled:
        user_holdout = positive_holdout[positive_holdout["userId"].astype(int) == user_id].iloc[0]
        holdout_movie_id = int(user_holdout["movieId"])
        holdout_rating = float(user_holdout["rating"])

        in_als_index = holdout_movie_id in als_artifacts.item_index
        in_lightfm_index = holdout_movie_id in lightfm_artifacts.item_index
        counters["holdout_in_als_index"] += int(in_als_index)
        counters["holdout_in_lightfm_index"] += int(in_lightfm_index)

        user_position = als_artifacts.user_index.get(user_id)
        user_items_confidence = None
        if user_position is not None and in_als_index:
            holdout_item_position = als_artifacts.item_index[holdout_movie_id]
            user_items_confidence = float(als_artifacts.user_items[user_position, holdout_item_position])
            if user_items_confidence > 0.0:
                counters["holdout_in_user_items"] += 1

        user_train_watched = train.loc[train["userId"].astype(int) == user_id, "movieId"].dropna().astype(int).tolist()

        als_true = pd.DataFrame()
        als_false = pd.DataFrame()
        if user_position is not None:
            als_true = als_raw_recommend(als_artifacts, user_position, args.top_k, filter_already_liked=True)
            if not als_true.empty and holdout_movie_id in als_true["movieId"].values:
                counters["als_filter_true_hits"] += 1

            requested_n = min(len(als_artifacts.item_index), args.top_k + len(user_train_watched))
            als_false_full = als_raw_recommend(als_artifacts, user_position, requested_n, filter_already_liked=False)
            als_false = als_false_full[~als_false_full["movieId"].isin(user_train_watched)].head(args.top_k).copy()
            if not als_false.empty:
                als_false["rank"] = np.arange(1, len(als_false) + 1)
                if holdout_movie_id in als_false["movieId"].values:
                    counters["als_filter_false_hits"] += 1

        svd_topk = pd.DataFrame()
        svd_holdout_rank = None
        if svd_model is not None:
            svd_full = svd_train_excluded_predictions(svd_model, user_id, catalog_movie_ids, user_train_watched)
            svd_topk = svd_full.head(args.top_k)
            if holdout_movie_id in svd_topk["movieId"].values:
                counters["svd_topk_hits"] += 1
            holdout_rows = svd_full.loc[svd_full["movieId"] == holdout_movie_id]
            if not holdout_rows.empty:
                svd_holdout_rank = int(holdout_rows["rank"].iloc[0])
                if svd_holdout_rank > 50:
                    counters["svd_rank_over_50"] += 1

        lines.append(f"## userId={user_id}")
        lines.append(f"- holdout_movie_id={holdout_movie_id}  holdout_rating={holdout_rating}")
        lines.append(f"- in_als_item_index={in_als_index}  in_lightfm_item_index={in_lightfm_index}")
        if user_items_confidence is None:
            lines.append("- als_user_items_confidence=N/A (user or item not in ALS index)")
        else:
            lines.append(f"- als_user_items_confidence={user_items_confidence:.4f}")
        lines.append(f"- train_watched_count={len(user_train_watched)}")
        lines.append("")
        lines.append(f"  ALS top-{args.top_k} with filter_already_liked_items=True:")
        lines.extend(format_rows(als_true, holdout_movie_id, "score") if not als_true.empty else ["    (empty)"])
        lines.append("")
        lines.append(f"  ALS top-{args.top_k} with filter_already_liked_items=False (post-filter train watched):")
        lines.extend(format_rows(als_false, holdout_movie_id, "score") if not als_false.empty else ["    (empty)"])
        lines.append("")
        lines.append(f"  SVD top-{args.top_k} (raw_svd_predictions equivalent, train excluded):")
        lines.extend(format_rows(svd_topk, holdout_movie_id, "predicted_score") if not svd_topk.empty else ["    (empty)"])
        if svd_holdout_rank is None:
            lines.append("  SVD full-catalog rank of holdout: N/A")
        else:
            lines.append(f"  SVD full-catalog rank of holdout: {svd_holdout_rank}")
        lines.append("")
        lines.append("")

    n = len(sampled)
    lines.append(f"# Summary across {n} users")
    lines.append(f"- holdout in ALS item_index: {counters['holdout_in_als_index']} / {n}")
    lines.append(f"- holdout in LightFM item_index: {counters['holdout_in_lightfm_index']} / {n}")
    lines.append(f"- holdout in user_items row (smoking gun for ALS): {counters['holdout_in_user_items']} / {n}")
    lines.append(f"- ALS top-K with filter=True contains holdout: {counters['als_filter_true_hits']} / {n} (expect 0)")
    lines.append(f"- ALS top-K with filter=False (post-filter train) contains holdout: {counters['als_filter_false_hits']} / {n} (expect >= 1)")
    lines.append(f"- SVD top-K (train excluded) contains holdout: {counters['svd_topk_hits']} / {n}")
    lines.append(f"- SVD full-catalog rank of holdout > 50: {counters['svd_rank_over_50']} / {n} (expect >= 4)")

    output_text = "\n".join(lines) + "\n"
    if args.output_path:
        Path(args.output_path).write_text(output_text)
        print(f"Wrote diagnostic dump to {args.output_path}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
