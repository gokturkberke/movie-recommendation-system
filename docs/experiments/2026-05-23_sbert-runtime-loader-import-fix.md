- **Date:** 2026-05-23
- **Topic:** Avoid SBERT model imports while loading prebuilt FAISS artifacts
- **Motivation:** The Streamlit SBERT UI path added in `docs/experiments/2026-05-20_sbert-faiss-full-run.md` item 3 loads an existing full-catalog index during app startup. With that index present, Streamlit's module watcher examines lazy `transformers` image modules and prints repeated `ModuleNotFoundError: No module named 'torchvision'` traces, although SBERT recommendations still work. This is a runtime defect investigation, not a metric experiment; the baseline artifact is the existing `artifacts/indexes/sbert_faiss/metadata.json` index (`row_count = 79477`, `embedding_dim = 384`, created in the 2026-05-20 SBERT run).
- **Hypothesis:** Loading a prebuilt SBERT+FAISS index without importing `sentence_transformers` removes the Streamlit watcher traceback sequence while preserving successful loading of all 79,477 indexed movies and unchanged semantic recommendations.
- **Preconditions:**
  - `src/experimental/sbert_faiss.py` currently uses one dependency helper for both offline index building and runtime index loading.
  - `requirements.txt` already contains `sentence-transformers` and `faiss-cpu`; this fix does not add `torchvision`.
  - The local full-catalog artifacts remain at `artifacts/indexes/sbert_faiss/`; this fix does not rebuild or reshape them.

## 1) Separate runtime FAISS loading from offline embedding dependencies

- **Goal:** Keep `SentenceTransformer` required for artifact creation, while making existing-index loading require only FAISS and artifact readers.
- **Files:**
  - `src/experimental/sbert_faiss.py:require_sbert_faiss_dependencies`, `build_sbert_faiss_artifacts`, and `load_sbert_faiss_index`.
  - `tests/test_sbert_faiss.py:TestSbertFaiss`.
- **Steps:**
  - Add a small internal FAISS-only import helper with the existing FAISS missing-dependency error.
  - Preserve the current build helper contract and embedding-generation behavior by having it still return both `SentenceTransformer` and FAISS.
  - Change `load_sbert_faiss_index()` to use the FAISS-only helper and leave its artifact file contract unchanged.
  - Add a unit test that loads temporary index artifacts through a fake FAISS reader while making `sentence_transformers` unavailable; loading must succeed.
- **Test / verification:**
  - Run `.venv/bin/python -m unittest tests.test_sbert_faiss`.
  - Run `.venv/bin/python -m unittest discover -s tests`.
  - Load the existing real index through `.venv/bin/python` and verify its metadata reports 79,477 rows and 384-dimensional embeddings.
  - Start Streamlit with file watching enabled and select the SBERT path; recommendations must still work and the prior missing-`torchvision` watcher traceback sequence must not appear.
- **Expected outcome:** Runtime index loading no longer imports `sentence_transformers` or `transformers`; SBERT artifact build behavior, artifact shapes, application UI, and recommendation output contracts remain unchanged.
- **DONE / DROPPED:**
