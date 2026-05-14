# Recommended Agent Context — Movie Recommendation Modernization

## Use this bundle instead of feeding every report to agents

The full report set has heavy overlap. For coding and planning agents, use this smaller English context pack to keep the agent focused on implementation, evaluation, and architecture decisions.

## Recommended files to provide

### Minimum coding context

1. `06_project_inventory_and_roadmap_en.md`  
   Use first. It explains the current repository baseline, implemented modules, known gaps, and the practical modernization roadmap.

2. `07_performance_evaluation_analysis.md`  
   Use for any task that changes recommendation logic. It defines metrics, evaluation splits, benchmark baselines, and the suggested evaluation module structure.

3. `08_evaluation_results_report.md`  
   Use when interpreting current local baseline results, latency findings, and caveats.

### Add only when needed

4. `02_strategic_architecture_design.md`  
   Add this when asking the agent to redesign the system into a retrieval → ranking → re-ranking architecture.

5. `01_technical_comparison_report.md`  
   Add this when asking the agent to compare model options such as SBERT + FAISS, LightFM, LightGCN, SASRec, Two-Tower models, MMoE, or LLM-based explanation.

## Do not include by default

Do not feed the Turkish overview reports or long industrial background reports to implementation agents by default. They repeat many of the same concepts and may push the agent toward vague strategy instead of concrete code changes.

## Recommended agent prompts

### Evaluation agent

Use:

- `06_project_inventory_and_roadmap_en.md`
- `07_performance_evaluation_analysis.md`

Prompt idea:

> Implement a reproducible offline evaluation module for the existing MovieLens recommendation project. Start with temporal/user-aware train-test splitting, top-K ranking metrics, beyond-accuracy metrics, and a CLI that compares Popularity, TF-IDF content-based, SVD, and Hybrid recommenders.

### SBERT + FAISS implementation agent

Use:

- `06_project_inventory_and_roadmap_en.md`
- `07_performance_evaluation_analysis.md`
- `01_technical_comparison_report.md`

Prompt idea:

> Extend the existing evaluation layer with Sentence-BERT embeddings and FAISS indexing. Keep the TF-IDF and semantic-LSA baselines intact, use prebuilt local artifacts, and report Precision@10, Recall@10, MAP@10, MRR@10, NDCG@10, coverage, diversity, novelty, and latency.

### Architecture refactor agent

Use:

- `06_project_inventory_and_roadmap_en.md`
- `07_performance_evaluation_analysis.md`
- `02_strategic_architecture_design.md`

Prompt idea:

> Refactor the project into a modular recommendation pipeline with candidate generation, ranking, re-ranking, evaluation, and UI layers. Do not over-engineer; keep the current Streamlit app functional while making the recommender layer testable.
