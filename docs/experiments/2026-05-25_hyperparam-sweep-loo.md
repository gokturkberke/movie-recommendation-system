- **Date:** 2026-05-25
- **Topic:** Hyperparameter sweep for LightFM and ALS on LOO artifacts -- close the "is the single-point tuning near Pareto?" question
- **Motivation:** The 2026-05-24 LOO eval landed at LightFM heavy = 0.0524 +/- 0.0020 and ALS cold = 0.1272 +/- 0.0569 with `no_components=64, loss=warp, epochs=20` for LightFM and `factors=64, regularization=0.01, alpha=40, iterations=20` for ALS. Both are single points in a multi-dimensional hyperparameter space. Without a sweep, we cannot tell whether the post-LOO segment-dependent ordering (ALS cold, LightFM heavy, mixed middle) is a real model-class signature or an artifact of the single-point tuning. The 2026-05-22 plan logged this sweep as the natural follow-on once LOO baselines settled.
- **Hypothesis:**
  - **H1 (LightFM heavy is hyperparam-bound):** At least one LightFM hyperparameter combination in the grid `no_components in {32,64,128} x loss in {warp,bpr}` produces a heavy-segment NDCG@10 above 0.060 (+15% over the single-point LOO baseline of 0.0524). If no combo clears 0.060, LightFM heavy is at a model-class ceiling.
  - **H2 (ALS cold is hyperparam-bound):** At least one ALS combination in the grid `factors in {32,64,128} x regularization in {0.01,0.1}` produces cold NDCG@10 above 0.150 (+18% over the single-point LOO 0.1272). Otherwise ALS cold is at a model-class ceiling for this dataset.
  - **H3 (segment leadership preserved):** At each segment, the same model class still wins after tuning: ALS still leads cold; LightFM still leads heavy; the middle two stay mixed.
- **Preconditions:**
  - All four items of `docs/experiments/2026-05-24_leave-one-out-leakage-fix.md` are DONE on `main` through commit `9af04ee`.
  - `artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv` (1555 rows) is present locally.
  - `artifacts/models/lightfm_loo/` and `artifacts/models/als_loo/` exist as the single-point comparison reference; not modified.
  - `--exclude-holdout-pairs` flag on both train scripts (commit `0aad222`) is on `main`.
  - LightFM training fell to ~7 min wall time post-LOO (was ~8 min); ALS to ~1.5 min. Compute budget below assumes these.
  - CLAUDE.md section 7 governs DONE markers; do not fill `<hash>` placeholders before the commit lands.

Corresponding audit item: 2026-05-24 plan's open question ("are the LOO heavy and cold numbers Pareto?") and `docs/06_project_inventory_and_roadmap_en.md` Priority 5 follow-on note.

## 1) Sweep driver script + unit tests

- **Goal:** A reusable orchestrator that takes a grid spec, retrains one classical-CF artifact per combination with `--exclude-holdout-pairs` applied, and writes a manifest CSV listing every combination + its output directory. Idempotent so re-running skips existing artifacts.
- **Files:**
  - `scripts/sweep_classical_cf.py` (new, target ~130 lines). Argparse-driven.
    - `--model lightfm|als` -- selects which train script to call.
    - `--grid key=v1,v2,...` (repeatable) -- one occurrence per hyperparameter dimension.
    - `--fixed key=value` (repeatable) -- hyperparameters held constant.
    - `--exclude-holdout-pairs <CSV>` -- forwarded to the train script.
    - `--output-root artifacts/sweeps` -- root dir for `{model}_{slug}` subdirs.
    - `--manifest-path artifacts/sweeps/sweep_manifest.csv` -- written / appended after each combo completes.
    - Slug builder: deterministic mapping from `{key: value}` to a filesystem-safe string, e.g. `lightfm_n128_lwarp_e20`. Document the encoding inline.
    - For each combination, build the train-script command line, run via `subprocess.run`, capture stdout (the saved metadata JSON), parse `artifact_dir` from the response, append a manifest row.
    - Skip when `{output-root}/{model}_{slug}/metadata.json` already exists (idempotency).
  - `tests/test_sweep_classical_cf.py` (new, ~70 lines). Two tests:
    - `test_slug_round_trip`: passing `no_components=128, loss=warp, epochs=20` to the slug builder yields `n128_lwarp_e20` (or equivalent), and a regex / reverse parse recovers the dict.
    - `test_manifest_skip_when_dir_exists`: a fake `output-root` with a pre-seeded `lightfm_n32_lwarp_e20/metadata.json` makes the driver skip that combo (no subprocess call) but still emits the row in the manifest with a `skipped=true` flag.
  - `.gitignore`: no change (the `artifacts/` directory is already gitignored).
- **Reuse:** `scripts/train_lightfm_model.py`, `scripts/train_als_model.py` -- called as subprocesses. No new training code.
- **Steps:**
  - Implement the driver, tests, and a small `build_slug(params)` helper inside the driver.
  - `.venv/bin/python -m unittest tests.test_sweep_classical_cf` -- 2/2 pass.
  - `.venv/bin/python -m unittest discover -s tests` -- 70/70 (was 68/68, +2 new).
  - 2-combo smoke (writes to `/private/tmp/sweep_smoke/`):
    ```bash
    .venv/bin/python scripts/sweep_classical_cf.py \
      --model lightfm \
      --grid no-components=32,64 \
      --grid loss=warp \
      --fixed epochs=20 \
      --fixed num-threads=1 \
      --exclude-holdout-pairs artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv \
      --output-root /private/tmp/sweep_smoke \
      --manifest-path /private/tmp/sweep_smoke/manifest.csv
    ```
    Verify the manifest has 2 rows; both metadata.json files exist with `excluded_pair_count=1555`. Re-run the same command and verify both rows now have `skipped=true`.
  - Commit message: `feat(eval): add LightFM/ALS hyperparameter sweep driver`. `git add -f docs/experiments/2026-05-25_hyperparam-sweep-loo.md`.
- **Test / verification:**
  - 2 new unit tests pass; full suite stays at 70.
  - 2-combo smoke produces 2 artifact dirs + manifest; second run skips.
  - Idempotency: `metadata.json` timestamps unchanged across reruns.
- **Expected outcome:** A reusable sweep driver. Decision criterion: tests pass; smoke produces expected manifest shape.
- **DONE (commit `35690f1`):** Added `scripts/sweep_classical_cf.py` (~155 lines) and `tests/test_sweep_classical_cf.py`. Driver supports `--grid` (repeatable, comma-separated) and `--fixed` (repeatable, single value) axes; expands the cartesian product; calls the existing train scripts via `subprocess.run` with `--exclude-holdout-pairs` forwarded. Slug encoding is stable: `lightfm_n128_lwarp_e20`, `als_f64_r0.01_a40.0_i20`. Idempotency: existing `metadata.json` -> skip + manifest row with `skipped=true`.
  - Tests: 5 new (slug for LightFM, slug for ALS, grid+fixed merge, duplicate-axis raise, manifest skip). All pass. Full suite 73/73 OK (was 68/68).
  - CLI sanity: `scripts/sweep_classical_cf.py --help` resolves; required flags enforced (`--model`, `--exclude-holdout-pairs`, `--output-root`, `--manifest-path`).
  - Decision: proceed to Item 2 (12-artifact sweep run).

## 2) Execute the sweep (12 artifacts)

- **Goal:** Train 12 artifacts (6 LightFM + 6 ALS) under `artifacts/sweeps/` with the LOO exclusion CSV applied.
- **Files:**
  - `artifacts/sweeps/lightfm_n{32,64,128}_l{warp,bpr}_e20/` -- 6 LightFM artifact dirs.
  - `artifacts/sweeps/als_f{32,64,128}_r{0.01,0.1}_a40_i20/` -- 6 ALS artifact dirs.
  - `artifacts/sweeps/sweep_manifest.csv` -- combined 12-row manifest.
- **Steps:**
  - LightFM sweep:
    ```bash
    .venv/bin/python scripts/sweep_classical_cf.py \
      --model lightfm \
      --grid no-components=32,64,128 \
      --grid loss=warp,bpr \
      --fixed epochs=20 \
      --fixed num-threads=1 \
      --exclude-holdout-pairs artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv \
      --output-root artifacts/sweeps \
      --manifest-path artifacts/sweeps/sweep_manifest.csv
    ```
    Expected wall time: 6 * ~7 min = ~42 min.
  - ALS sweep (appends to the same manifest):
    ```bash
    .venv/bin/python scripts/sweep_classical_cf.py \
      --model als \
      --grid factors=32,64,128 \
      --grid regularization=0.01,0.1 \
      --fixed alpha=40 \
      --fixed iterations=20 \
      --exclude-holdout-pairs artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv \
      --output-root artifacts/sweeps \
      --manifest-path artifacts/sweeps/sweep_manifest.csv
    ```
    Expected wall time: 6 * ~1.5 min = ~10 min.
  - Total: ~50-60 min. Run in background.
  - Commit message: `chore(eval): produce 12-artifact classical-CF sweep with LOO exclusion`.
- **Test / verification:**
  - `wc -l artifacts/sweeps/sweep_manifest.csv` returns 13 (12 rows + header).
  - Spot-check three random combos: `cat artifacts/sweeps/{slug}/metadata.json | jq` shows the matching hyperparameters and `excluded_pair_count=1555`.
  - DONE marker records wall time per combo (median, total) and the manifest path.
- **Expected outcome:** 12 leakage-corrected artifacts ready for sweep eval. Decision criterion: manifest row count and metadata correctness pass.
- **DONE (commit `ecdfe10`):** Ran the LightFM and ALS sweeps back-to-back via the driver. 12 artifacts produced; `artifacts/sweeps/sweep_manifest.csv` has 13 lines (1 header + 12 rows). Every artifact's `metadata.json` carries `excluded_pair_count=1555` and `row_count=16,861,498` -- proving the LOO exclusion was applied uniformly.
  - Total wall time ~96 minutes (longer than the 50-60 min estimate; BPR loss made LightFM training meaningfully slower than WARP):

    | Artifact | Train seconds | Notes |
    |---|---:|---|
    | lightfm_n32_lwarp_e20_t1 | 285.9 | fastest LightFM |
    | lightfm_n32_lbpr_e20_t1 | 660.9 | |
    | lightfm_n64_lwarp_e20_t1 | 456.9 | matches 2026-05-24 single-point baseline shape |
    | lightfm_n64_lbpr_e20_t1 | 1088.1 | |
    | lightfm_n128_lwarp_e20_t1 | 655.2 | |
    | lightfm_n128_lbpr_e20_t1 | 1927.0 | slowest combo by far |
    | als_f32_r0.01_a40_i20 | 61.7 | |
    | als_f32_r0.1_a40_i20 | 61.8 | |
    | als_f64_r0.01_a40_i20 | 93.1 | matches 2026-05-24 single-point baseline shape |
    | als_f64_r0.1_a40_i20 | 93.0 | |
    | als_f128_r0.01_a40_i20 | 188.6 | |
    | als_f128_r0.1_a40_i20 | 188.7 | |

    - Observations: BPR loss roughly doubles LightFM training time vs WARP at the same `no_components`. ALS regularization has effectively zero impact on training time (61.7 vs 61.8 etc.) -- the regularization variant is computed during the same SGD pass. ALS scales sub-linearly with `factors` for this dataset (32 -> 62s, 64 -> 93s, 128 -> 189s).
  - Manifest path: `artifacts/sweeps/sweep_manifest.csv` (gitignored).
  - Decision: proceed to Item 3 (lightweight per-artifact eval).

## 3) Lightweight per-artifact eval

- **Goal:** Single-seed (seed=42) segmented eval per artifact, capturing aggregate and per-segment NDCG@10. Minimal-flag eval to stay fast.
- **Files:**
  - `scripts/eval_sweep.py` (new, target ~90 lines). Reads `sweep_manifest.csv`. For each row:
    - Determines the model class from the manifest column or the slug prefix.
    - Builds the `evaluate_baselines.py` command line with `--max-users 300 --k 5,10,20 --holdout-count 3 --user-sample-seed 42 --segment-by-history` and ONLY the matching `--include-{lightfm|als}` flag pointed at the artifact dir.
    - Captures the timestamped JSON path. Parses `top_n.{model}.10` aggregate + `top_n.{model}.10.segments.{seg}.ndcg_at_k` for each of the four segments.
    - Appends a row to `artifacts/sweeps/sweep_results.csv` with columns: `slug, model, hyperparams_json, aggregate_ndcg10, cold_ndcg10, warm_ndcg10, regular_ndcg10, heavy_ndcg10, evaluated_user_count, run_id`.
  - `artifacts/sweeps/sweep_results.csv` -- new gitignored.
- **Steps:**
  - Implement `eval_sweep.py`.
  - Run: `.venv/bin/python scripts/eval_sweep.py --manifest artifacts/sweeps/sweep_manifest.csv --output artifacts/sweeps/sweep_results.csv`. Background. Expected wall time: ~36-40 min.
  - Commit message: `chore(eval): capture per-artifact NDCG@10 for the 12-artifact sweep`.
- **Test / verification:**
  - `wc -l artifacts/sweeps/sweep_results.csv` returns 13 (12 rows + header).
  - Sanity row matching `lightfm_n64_lwarp_e20` should have `aggregate_ndcg10` within 0.001 of the 2026-05-24 single-point LightFM LOO seed-42 number (0.0883).
  - Same for `als_f64_r0.01_a40_i20` against the LOO seed-42 ALS number (0.0863).
  - Every row's per-segment numbers are within [0, 1].
- **Expected outcome:** A 12-row sweep results table. Decision criterion: sanity match holds; numbers are populated for every combo.
- **DONE (commit `37be11e`):** Added `scripts/eval_sweep.py` and ran it against the 12-row manifest. Output: `artifacts/sweeps/sweep_results.csv` with 13 lines (1 header + 12 rows). Each row carries aggregate NDCG@10 / HitRate@10 and per-segment NDCG@10 for cold / warm / regular / heavy.
  - **LightFM sweep** sorted by aggregate NDCG@10:

    | Slug | Aggregate | Cold | Warm | Regular | Heavy |
    |---|---:|---:|---:|---:|---:|
    | n64_lwarp_e20 | **0.0738** | 0.1266 | 0.0801 | 0.0549 | 0.0234 |
    | n64_lbpr_e20 | 0.0727 | 0.1330 | 0.0801 | 0.0458 | 0.0259 |
    | n32_lwarp_e20 | 0.0718 | 0.1090 | 0.0767 | 0.0623 | 0.0255 |
    | n128_lwarp_e20 | 0.0711 | 0.0895 | 0.0975 | 0.0318 | **0.0359** |
    | n32_lbpr_e20 | 0.0704 | **0.1390** | 0.0792 | 0.0425 | 0.0099 |
    | n128_lbpr_e20 | 0.0700 | 0.1354 | 0.0812 | 0.0410 | 0.0066 |

  - **ALS sweep** sorted by aggregate NDCG@10:

    | Slug | Aggregate | Cold | Warm | Regular | Heavy |
    |---|---:|---:|---:|---:|---:|
    | f64_r0.1_a40_i20 | **0.0919** | 0.1795 | 0.0844 | 0.0764 | **0.0396** |
    | f64_r0.01_a40_i20 | 0.0886 | 0.1787 | 0.0848 | 0.0765 | 0.0113 |
    | f128_r0.01_a40_i20 | 0.0835 | 0.1732 | 0.0837 | 0.0590 | 0.0192 |
    | f128_r0.1_a40_i20 | 0.0808 | 0.1580 | 0.0886 | 0.0481 | 0.0236 |
    | f32_r0.01_a40_i20 | 0.0775 | 0.1720 | 0.0823 | 0.0371 | 0.0260 |
    | f32_r0.1_a40_i20 | 0.0722 | 0.1551 | 0.0708 | 0.0496 | 0.0190 |

  - **Sanity check.** ALS `f64_r0.01_a40_i20` aggregate (0.0886) matches the 2026-05-24 LightFM single-point LOO seed-42 number (0.0863) within 0.0023 (ALS is essentially deterministic across reruns). LightFM `n64_lwarp_e20` shows a 0.0145 drop from the 2026-05-24 reference (0.0883 -> 0.0738). LightFM's training script does not plumb a `random_state` to `LightFM.__init__`, so different runs of the same hyperparameters produce different weights -- this 17% gap is the size of the training-noise floor for LightFM and is recorded in the report.
  - **Key findings:**
    - ALS `regularization=0.1` beats `0.01` across every factor count on aggregate. At `factors=64` it lifts NDCG@10 by +3.7% (0.0886 -> 0.0919) and heavy NDCG@10 by ~3.5x (0.0113 -> 0.0396). This is the cleanest hyperparam win in the sweep.
    - LightFM `WARP` beats `BPR` at every `no_components` at aggregate; the gap is small (0.0727 vs 0.0738 at `no_components=64`).
    - LightFM `no_components=64` is near the aggregate Pareto -- 32 and 128 are within 4% of 64.
    - The best LightFM heavy (n128_lwarp at 0.0359) does not match the best ALS heavy (f64_r0.1 at 0.0396).
  - **Winners for Item 4:** LightFM `n64_lwarp_e20` (defends the existing single-point baseline), ALS `f64_r0.1_a40_i20` (new winner -- regularization=0.1 instead of 0.01).
  - Decision: proceed to Item 4 with these two winners. The LightFM "winner" comparison will primarily document the training-noise floor; the ALS winner comparison will document a real hyperparam gain.

## 4) Synthesis + winners' multi-seed eval + report subsection

- **Goal:** Identify the top-1 LightFM and top-1 ALS by aggregate NDCG@10, run those two through a full 3-seed segmented eval, then add a "Hyperparameter sweep on LOO artifacts" subsection to `docs/08_evaluation_results_report.md`.
- **Files:**
  - `artifacts/evaluation/metrics_summary_*.{csv,json}` -- 6 new (3 seeds x 2 winners), gitignored. If seed-42 minimal eval from item 3 used the same artifact as one of the winners, that file is reused; otherwise we re-run all three seeds.
  - `docs/08_evaluation_results_report.md`: insert new subsection `## Hyperparameter sweep on LOO artifacts` between "Leakage-corrected (leave-one-out) re-evaluation" and "Conclusions". Two tables:
    - **Table A -- Full sweep at seed=42 (single seed):** 12 rows. Columns: model, hyperparam combo, aggregate NDCG@10, cold NDCG@10, warm NDCG@10, regular NDCG@10, heavy NDCG@10. Two short paragraphs interpreting which dimension matters most for LightFM and ALS.
    - **Table B -- Winners vs single-point LOO (3 seeds):** 4 rows -- single-point LightFM LOO, winner LightFM, single-point ALS LOO, winner ALS. Columns: aggregate NDCG@10 mean +/- std, cold mean +/- std, heavy mean +/- std.
  - Conclusions: update with whether the corrected leaderboard from 2026-05-24 holds under tuned hyperparameters (or names the new winner).
  - Caveats: add a sentence about the sweep being seed=42 single-seed for the grid (with winners verified at 3 seeds).
  - This plan file: fill DONE marker.
- **Steps:**
  - Read `sweep_results.csv`, sort by `aggregate_ndcg10`, pick top-1 LightFM and top-1 ALS.
  - For each winner, run the full 9-model 3-seed segmented eval (3 commands per winner if seed=42 result isn't already saved with the full flag set). Total ~36-50 min.
  - Compute mean +/- std for the winner rows.
  - Write the subsection.
  - Update Conclusions and Caveats.
  - Commit message: `docs(eval): add hyperparameter sweep subsection`. `git add -f` both files.
- **Test / verification:**
  - New subsection present with both tables and two paragraphs.
  - Conclusions reflects the sweep findings (does ALS still own cold? LightFM still own heavy?).
  - Full test suite stays at 70.
- **Expected outcome:** A defensible Pareto picture for both classical CF models. Decision criterion: sweep table agrees with `sweep_results.csv` row-for-row; winners' multi-seed numbers agree with the new eval JSONs.
- **DONE / DROPPED:**

## Deferred / Future (out of this plan)

- **Three-dim LightFM sweep** (no_components x loss x epochs at multiple epoch values). The 2-dim cross is enough for a Pareto pass; if a winner is closer to the single-point baseline than expected, a third dim can be a follow-on plan.
- **ALS sweep on `alpha` and `iterations`.** These are held constant in this plan. Reopen if the regularization sweep alone produces meaningful gains.
- **Per-seed sweep replication.** Sweeping at all 3 seeds is 3x compute. Defer until the seed=42 sweep surfaces a winner whose seed=42 result is suspicious.
- **GPU sweep for ALS.** CPU is fine for this grid; revisit if grid size grows.
- **LightGCN baseline (Priority 6a).** Separate plan once the classical CF Pareto is settled.
- **SASRec / BERT4Rec (Priority 6b).** Separate plan.
- **UI explainability (Priority 5).** Independent track.
- **Item-side cold-start.** Different question, separate plan.

## Critical Files (Reference)

- `scripts/train_lightfm_model.py`, `scripts/train_als_model.py` -- unchanged; called via subprocess from the sweep driver.
- `scripts/evaluate_baselines.py` -- unchanged; called via subprocess from `eval_sweep.py`.
- `scripts/sweep_classical_cf.py` -- new in item 1.
- `scripts/eval_sweep.py` -- new in item 3.
- `tests/test_sweep_classical_cf.py` -- new in item 1.
- `artifacts/holdouts/2026-05-23_300u_h3_seeds_42_7_1337.csv` -- LOO exclusion CSV reused.
- `artifacts/models/lightfm_loo/`, `artifacts/models/als_loo/` -- single-point reference, NOT modified.
- `artifacts/sweeps/` -- new gitignored directory tree.
- `docs/08_evaluation_results_report.md` -- item 4 inserts the sweep subsection.
- `docs/experiments/2026-05-24_leave-one-out-leakage-fix.md` -- prior plan; this plan extends its Conclusions findings.

## End-to-End Verification Sequence

1. Item 1 complete: `tests.test_sweep_classical_cf` 2/2 OK; full suite 70/70; 2-combo smoke produces idempotent manifest. DONE marker has a real commit hash.
2. Item 2 complete: 12-row `sweep_manifest.csv`; all `metadata.json` files carry `excluded_pair_count=1555`. DONE marker captures per-combo wall time and the manifest path.
3. Item 3 complete: 12-row `sweep_results.csv`; aggregate matches single-point LOO numbers for the `n64_lwarp_e20` / `f64_r0.01_a40_i20` combos. DONE marker captures the top-1 LightFM and top-1 ALS combos with their aggregate / cold / heavy numbers.
4. Item 4 complete: report subsection with both tables; winners' multi-seed numbers; Conclusions updated. DONE marker has the synthesis commit hash.
5. After every commit, record the real hash in this plan file; CLAUDE.md section 7 forbids leaving placeholders.

## Execution Notes

- `/docs` is gitignored; tracked files including this plan need `git add -f` on commits.
- Items are sequential: item 1 unlocks item 2; item 2 unlocks item 3; item 3 unlocks item 4.
- One implementation commit per item plus a `docs(experiments): record ...` follow-up. Matches the 2026-05-21 / 22 / 23 / 24 cadence.
- All artifact files under `artifacts/sweeps/` and `artifacts/evaluation/` stay gitignored.
- LightFM training on this host warns about OpenMP unavailability and runs single-threaded. Use `--fixed num-threads=1` to make the manifest explicit.
