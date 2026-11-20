# Documentation

This directory collects the **technical specifications, experiment protocols, architecture notes, and project deliverables** for the Amazon Electronics Big Data Recommender.

The project uses Amazon Electronics ratings and product metadata to build a **Top-10 recommendation pipeline**, comparing a **Popularity baseline** with **Spark ALS**, measuring recommendation quality and coverage, and studying scalability and GCP resource–cost trade-offs.

## Documentation map

The following paths are **planned deliverables from the issue backlog**. Some may not exist yet; create or update them as their tickets are completed.

| Area | Document / location | What it covers | Ticket(s) |
| --- | --- | --- | --- |
| Data | `dataset_sources.md`, `schema.md` | Dataset references, raw file formats, field definitions, and quality notes. | T01 |
| Local setup | `local_setup.md` | Python, Java, PySpark, virtual environment, and reproducible smoke-test instructions. | T03 |
| Architecture | `architecture/`, `proposal_outline.md` | Pipeline diagram, scope, research questions, and system design. | T04 |
| Exploration | `data_5v.md`, `subset_method.md` | Evidence-based 5V assessment and reproducible subset strategy. | T08–T09 |
| Splitting | `split_protocol.md` | Temporal train/validation/test boundaries and leakage prevention. | T16 |
| Evaluation | `evaluation_protocol.md` | Relevance policy, Top-10 candidate rules, warm/cold-start cohorts, and comparison rules. | T18 |
| Cloud | `gcp_setup.md`, `gcs_layout.md` | IAM, budget controls, bucket zones, and data layout. | T29–T30 |
| Experiments | `model_findings.md`, `benchmark_findings.md`, `billing_method.md` | ALS vs Popularity results, benchmark interpretation, and verified billing methodology. | T28, T35–T36 |
| Reporting | `reports/`, `final_report.docx` or `.pdf`, `presentation.pptx` | Drafts, experimental evidence, final English report, and slides. | T38–T39 |

The repository may also contain epic-specific overview files such as `E2_README.md` and `E3_README.md`. Keep these as navigational summaries and link to the detailed documents above.

## Recommended reading order

1. [Project README](../README.md) — overall project scope and repository structure.
2. Dataset source and schema documents — understand ratings, metadata, and identifier fields.
3. Local setup and architecture documents — reproduce the environment and understand the workflow.
4. Split and evaluation protocols — understand the shared rules before comparing models.
5. Model findings and benchmark reports — review measured results and limitations.
6. Final report and slides — consolidated project outcomes.

## Key protocol decisions

- Use **temporal** train/validation/test splits to reduce future-information leakage.
- Keep **one consistent evaluation protocol** for ALS and Popularity: shared split, candidate universe, relevance criteria, and seen-item filtering.
- The proposed binary relevance criterion is **rating >= 4**, to be finalized before test evaluation.
- Track **cold-start users/items** explicitly; use a Popularity fallback when ALS cannot serve a user, and report coverage before/after fallback.
- Report **Recall@10, NDCG@10, user coverage**, and relevant runtime measures.
- Distinguish measured cloud billing costs from estimates. Use a pilot/subsets when full-scale processing is impractical.

These are project requirements and planned conventions; they are **not claims that experiments have already been completed**.

## Writing and evidence guidelines

- Write final deliverables in **English** and use consistent terminology throughout.
- Document the relevant ticket ID, inputs, output paths, date/run ID, environment/configuration, and limitations.
- Support reported numerical findings with reproducible logs, tables, or charts in `results/`.
- Avoid committing raw/large datasets and sensitive cloud credentials to Git.
- Update documentation when changing schemas, data splits, metrics, or model configuration.
- Prefer relative links to repository files so documentation works after cloning.

## Where the code and outputs live

- [`../src/etl/`](../src/etl/) — dataset profiling, cleaning, metadata, and Parquet processing.
- [`../src/splitting/`](../src/splitting/) and [`../src/evaluation/`](../src/evaluation/) — split logic and evaluation functions, according to the implemented layout.
- [`../src/models/`](../src/models/) — Popularity, ALS, Top-10 generation, and fallback.
- [`../scripts/`](../scripts/) — pipeline entry points and automation.
- [`../deployment/`](../deployment/) — GCP jobs and deployment instructions.
- [`../results/`](../results/) — lightweight experiment results and figures.

For implementation progress, consult the GitHub issues rather than assuming every planned file is available.
