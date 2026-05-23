- **Date:** 2026-05-23
- **Topic:** Runtime manifest and documented Streamlit command alignment
- **Motivation:** Repository inspection after the SBERT runtime-loader fix found two documentation/install drifts. `requirements.txt` contains the supported offline-evaluation dependencies `lightfm` and `implicit`, while `pyproject.toml` does not, so `pip install -e .` does not express all README-supported evaluation paths. The documented primary app command uses the virtual environment's console-script launcher, but that local launcher has a stale interpreter path while `.venv/bin/python -m streamlit run src/app.py` works. This is a packaging and documentation consistency correction; no benchmark run id applies.
- **Hypothesis:** Aligning editable-install dependencies with `requirements.txt` and standardizing the documented Streamlit command will remove manifest and command-reference drift without changing recommendation behavior or runtime code.
- **Preconditions:**
  - `requirements.txt` remains the canonical dependency manifest and already includes `sentence-transformers`, `faiss-cpu`, `lightfm`, and `implicit`.
  - The SBERT runtime-loader fix is committed and has already shown that `torchvision` is not required for loading the prebuilt semantic index.
  - No new recommendation workflow, artifact generation, or configuration change is part of this correction.

## 1) Align package metadata and runtime documentation

- **Goal:** Make editable installation expose the supported offline dependencies and make all documented app startup references use the working Python-module invocation.
- **Files:**
  - `pyproject.toml:[project].dependencies`.
  - `README.md` dependency description and Streamlit launch command.
  - `AGENTS.md` and `CLAUDE.md` primary app command declarations.
  - `docs/experiments/2026-05-20_sbert-faiss-full-run.md` historical command reference, updated at the user's request for repository-wide command consistency.
- **Steps:**
  - Add `lightfm` and `implicit` to `pyproject.toml` without changing `requirements.txt`.
  - Clarify in README that SBERT semantic mode is available in the app and offline evaluation, with `sentence-transformers` used to build embeddings and `faiss-cpu` used to load/search a prebuilt index.
  - Replace remaining console-script launcher references with `.venv/bin/python -m streamlit run src/app.py`.
  - Refresh the editable install so installed metadata reflects the updated manifest.
- **Test / verification:**
  - Search README, AGENTS, CLAUDE, and docs for stale console-script launcher references; none should remain.
  - Compare normalized `requirements.txt` and `pyproject.toml` dependencies; both difference sets should be empty.
  - Run `.venv/bin/python -m pip install -e .` and check that installed `movie-rec` metadata includes the aligned dependencies.
  - Run `.venv/bin/python -m pip check` and `.venv/bin/python -m unittest discover -s tests`.
  - Start the app using `.venv/bin/python -m streamlit run src/app.py` and confirm the Content-Based page renders.
- **Expected outcome:** Supported package-install paths and documented startup commands agree with each other while product behavior, artifacts, and configuration remain unchanged.
- **DONE / DROPPED:**
