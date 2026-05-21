- **Date:** 2026-05-21
- **Topic:** Finalization docs cleanup -- align README and the evaluation report with the post-LOO + post-sweep ground truth, and record the future-filename drift on the four prior plan files
- **Motivation:** The 8-commit LOO + hyperparameter-sweep arc landed on 2026-05-21 (commits `0aad222` through `fe8542c`). The 12-artifact sweep recorded in `docs/experiments/2026-05-25_hyperparam-sweep-loo.md` (item 4 synthesis) establishes tuned ALS `factors=64, regularization=0.1` as the strongest classical-CF baseline by aggregate NDCG@10 = 0.0787 +/- 0.0115 (3/3 seed wins), with cold 0.1335 and heavy 0.0485. LightFM remains slightly ahead on heavy but the gap sits inside the LightFM training-noise floor (~0.0145, attributable to the missing `random_state`). Three artifacts now misrepresent that state:
  - `README.md:186` still reads `LightFM WARP leads every ranking metric at K=10 and K=20 (NDCG@10 = 0.1427)` and `Implicit ALS and SVD top-K produced no relevance hits on this slice`. Both sentences are leakage + single-point artifacts and have been superseded.
  - `README.md:49-52` Future work is a two-bullet stub that does not match the close-out list the user has now committed to.
  - `docs/08_evaluation_results_report.md:3-40` Run Summary still presents the 100-user single-run as headline; the LOO + sweep subsections below it now supersede those numbers but there is no banner saying so.
  - The plan files `2026-05-22_*.md`, `2026-05-23_*.md`, `2026-05-24_*.md`, `2026-05-25_*.md` are filename-dated up to four days into the future relative to 2026-05-21, even though every DONE commit hash in them landed on 2026-05-21. CLAUDE.md section 7 reads "filename = authoring date"; the drift is documented here rather than rewritten so the existing commit hashes and cross-references stay valid.
- **Hypothesis:** After this cleanup, a first-time reader of `README.md` can identify in under one minute (a) the current product baseline, (b) the strongest offline classical-CF baseline with its hyperparameters and numeric result, and (c) the open future-work list, without having to open `docs/experiments/`. A reader of `docs/08_evaluation_results_report.md` reaches a "headline numbers below are superseded" banner before the now-stale 100-user findings. The four future-dated plan files carry an explicit note acknowledging the filename drift.
- **Preconditions:**
  - All four items of `docs/experiments/2026-05-25_hyperparam-sweep-loo.md` are DONE on `main` through commit `fe8542c`.
  - The Streamlit product app continues to run on Surprise SVD only; no `src/` or `scripts/` change in this plan.
  - CLAUDE.md section 7 governs DONE markers; the `<hash>` placeholder must never be left in place after the commit lands.
- **Corresponding audit item:** project finalization / readme freshness; user-driven close-out, not an audit `section` reference.

## 1) Refresh README "Current local findings" paragraph

- **Goal:** Replace the stale LightFM-leads / ALS-zero-hit sentence at `README.md:186` with a paragraph that reports the tuned-ALS aggregate winner under LOO, names the LightFM heavy caveat, and explicitly retires the earlier 100-user single-run headline.
- **Files:** `README.md` -- the single paragraph beginning `Current local findings are summarized in...`.
- **Steps:**
  - Identify the paragraph by its anchor sentence `Current local findings are summarized in \`docs/08_evaluation_results_report.md\`.`.
  - Replace the body of that paragraph (keep the leading sentence pointing at the report) with: the tuned-ALS aggregate result (0.0787 +/- 0.0115, 3/3 seed wins, cold 0.1335), the LightFM heavy-segment caveat tied to the LightFM training-noise floor (~0.0145), and an explicit "superseded" note for the earlier 100-user single-run headline. Keep the closing "Treat these as local directional results, not final benchmark claims." sentence.
- **Test / verification:**
  - `grep -n "LightFM WARP leads every ranking metric" README.md` returns empty.
  - `grep -n "tuned ALS" README.md` returns at least one match in the new paragraph.
  - `grep -n "superseded" README.md` returns at least one match in the new paragraph.
- **Expected outcome:** A reader pointed at `README.md` from the repo root learns the current strongest offline classical-CF baseline, its hyperparameters, and that the older headline has been retired -- without needing to open the experiment plans.
- **DONE (commit `86383a1`):** Replaced the `README.md:186` paragraph in place. Stale `LightFM WARP leads every ranking metric at K=10 and K=20 (NDCG@10 = 0.1427)` sentence is gone; new paragraph names tuned ALS aggregate (0.0787 +/- 0.0115, 3/3 seed wins, cold 0.1335) and explicitly retires the 100-user single-run headline as superseded.
  - Verification: `grep "LightFM WARP leads every ranking metric" README.md` returns empty; `grep "tuned ALS" README.md` returns 3 matches (Status block, Future work, Current local findings paragraph); `grep "superseded" README.md` matches in the Current local findings paragraph.
  - Decision: shipped to production README.

## 2) Add a top-of-README status block

- **Goal:** Place a 1-3 line "Status" block near the top of `README.md` so the current recommended baseline and the product/offline split are visible without scrolling.
- **Files:** `README.md` -- inserted between the H1 (`# Advanced Movie Recommendation System` line) and the existing project-overview paragraph, OR as a fenced status block directly above the `## Core Features` heading -- whichever preserves the existing flow more cleanly. Implementation choice: directly above `## Core Features` keeps the H1 + overview untouched.
- **Steps:**
  - Insert the Status block above `## Core Features`. Contents:
    - Product app: TF-IDF hybrid + Surprise SVD + optional SBERT.
    - Offline-best classical CF under LOO: tuned ALS `factors=64, regularization=0.1` (see `docs/08_evaluation_results_report.md`).
    - LightFM and ALS live in the offline-eval hat only; the Streamlit UI does not call them by design (intentional scope control).
- **Test / verification:** `grep -n "Status (2026-05-21)" README.md` returns one match.
- **Expected outcome:** The "what is this repo running right now" question has a one-glance answer near the top of README.
- **DONE (commit `86383a1`):** Inserted `## Status (2026-05-21)` block directly above `## Core Features` (README.md:26). Three bullets: product app stack, offline-best classical-CF baseline (tuned ALS f64 r0.1 with the numeric NDCG@10), and the explicit "LightFM and ALS are offline-eval only -- the Streamlit UI does not call them by design" scope-control statement.
  - Verification: `grep "Status (2026-05-21)" README.md` returns one match at line 26.
  - Decision: shipped to production README.

## 3) Slim README Future work list

- **Goal:** Replace the two-bullet stub at `README.md:49-52` with the user's close-out list of five concrete future-work items.
- **Files:** `README.md` -- the `Future work:` block immediately under the `Implemented vs. Experimental` section.
- **Steps:**
  - Replace the two existing bullets (`Graph / sequence models.` and `Larger, repeated evaluation runs before claiming model quality improvements.`) with five bullets:
    - Surface explainability in the Streamlit UI (why-this-was-recommended panel).
    - Optional product integration of tuned ALS / LightFM behind a feature flag (today they are offline-eval only).
    - Modern sequence / graph recommenders (LightGCN, SASRec, BERT4Rec) on the offline-eval hat.
    - One-click artifact reproducibility (LightFM + ALS artifacts are gitignored; expose a single scripted recipe).
    - LightFM `random_state` seeding to remove the ~0.0145 training-noise floor for benchmark stability.
- **Test / verification:** `grep -n "random_state" README.md` returns at least one match. `grep -n "LightGCN" README.md` returns at least one match.
- **Expected outcome:** The Future work list now reflects the genuine close-out items; the implementation of any of these would be a separate plan.
- **DONE (commit `86383a1`):** Replaced the two-bullet stub at `README.md:55` (post-shift due to the inserted Status block) with the five close-out bullets: UI explainability, ALS/LightFM product integration behind a flag, modern sequence/graph recommenders, one-click artifact reproducibility, and LightFM `random_state` seeding.
  - Verification: `grep "random_state" README.md` matches in the Future work list (line 61) and again in the Current local findings paragraph; `grep "LightGCN" README.md` matches the modern-recommenders bullet (line 59).
  - Decision: shipped to production README; each bullet is a candidate for a follow-up plan in its own right.

## 4) Annotate the four future-dated plan files

- **Goal:** Add a single `Note:` line directly under the `Date:` header in each of the four future-dated plan files so the filename drift is acknowledged on the file itself.
- **Files:**
  - `docs/experiments/2026-05-22_eval-slice-expansion.md`
  - `docs/experiments/2026-05-23_cold-start-segmentation.md`
  - `docs/experiments/2026-05-24_leave-one-out-leakage-fix.md`
  - `docs/experiments/2026-05-25_hyperparam-sweep-loo.md`
- **Steps:**
  - Insert one line directly after the existing `- **Date:** 2026-05-2X` header line:
    `- **Note:** Filename and Date field reflect narrative ordering of the experiment arc; all DONE commits landed on 2026-05-21. CLAUDE.md section 7 strict reading is "filename = authoring date" -- this drift is intentional and recorded here rather than rewritten, so commit hashes and cross-references in other docs stay valid.`
  - No other body change; do not rename files, do not adjust the `Date:` field, do not touch commit hashes.
- **Test / verification:** `grep -n "narrative ordering" docs/experiments/2026-05-2[2345]_*.md` returns one match per file (four matches total).
- **Expected outcome:** Audit trail integrity preserved (filename ordering + commit hashes intact) and the drift is explicitly documented at the source file rather than buried in a separate notes doc.
- **DONE (commit `86383a1`):** Inserted the identical `Note:` line on line 2 of each of the four files, directly under the `Date:` header. Filenames, `Date:` fields, and DONE commit hashes elsewhere in those files are untouched.
  - Verification: `grep "narrative ordering" docs/experiments/2026-05-2[2345]_*.md` returns one match per file (four matches total).
  - Decision: shipped; the drift is now explicit at the source file rather than relying on a separate notes doc.

## 5) Stale-banner on the evaluation report

- **Goal:** Add a single banner paragraph between the `## Run Summary` heading and the existing body of `docs/08_evaluation_results_report.md`, warning the reader that the headline numbers in the Run Summary and Model Comparison sections are superseded by the LOO + sweep subsections later in the same document.
- **Files:** `docs/08_evaluation_results_report.md` -- between line 3 (`## Run Summary`) and line 5 (start of the existing paragraph).
- **Steps:**
  - Insert one paragraph. Contents: a `Status (2026-05-21):` lead-in, naming the tuned-ALS aggregate result (NDCG@10 = 0.0787 +/- 0.0115), pointing readers to `docs/experiments/2026-05-24_leave-one-out-leakage-fix.md` and `docs/experiments/2026-05-25_hyperparam-sweep-loo.md` for the audit trail. No body rewrite below.
- **Test / verification:** `grep -n "Status (2026-05-21)" docs/08_evaluation_results_report.md` returns one match.
- **Expected outcome:** A reader entering the eval report sees the supersession banner before reaching the original 100-user run-summary block.
- **DONE (commit `86383a1`):** Inserted the `Status (2026-05-21):` banner paragraph at `docs/08_evaluation_results_report.md:5`, between `## Run Summary` and the existing body paragraph. The body of the report (Run configuration, Model Comparison, latency tables) is untouched.
  - Verification: `grep "Status (2026-05-21)" docs/08_evaluation_results_report.md` returns one match at line 5.
  - Decision: shipped; body rewrite explicitly out of scope per the plan's `Out of scope` section.

## Verification (whole cleanup, post-commit)

- `git diff` against the previous commit shows changes only in the seven files listed under "Files touched" in this plan. No `src/` or `scripts/` lines move.
- `.venv/bin/python -m unittest discover -s tests` -- sanity check that no test relies on the changed prose. Expected: identical pre/post-commit pass count (docs-only commit).
- `grep -n "LightFM WARP leads every ranking metric" README.md` is empty.
- `grep -n "Status (2026-05-21)" README.md docs/08_evaluation_results_report.md` returns one match per file.
- `grep -n "narrative ordering" docs/experiments/2026-05-2[2345]_*.md` returns one match per file.
- After the commit lands, fill `DONE (commit <hash>)` lines for each item above; the `<hash>` placeholder must never be left in place.
