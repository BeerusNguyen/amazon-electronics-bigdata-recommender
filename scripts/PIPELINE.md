# Scripts

This directory contains **entry-point scripts and automation helpers** for the Amazon Electronics Big Data Recommender project. Core implementation code belongs in `src/`; scripts here provide convenient ways to run, validate, or orchestrate those modules.

## Purpose

- Run individual pipeline stages or the complete workflow.
- Perform environment and Spark smoke tests.
- Automate repeatable local and cloud tasks when applicable.
- Keep command-line entry points separate from the ETL, recommendation, and evaluation logic in `src/`.

## Planned files

| File | Purpose | Related ticket |
| --- | --- | --- |
| `run_pipeline.py` | Orchestrate the end-to-end pipeline and expose reproducible CLI options. | T37 |
| `smoke_test.py` | Verify that the configured local PySpark environment starts and can execute a simple job. | T03 |
| Other helper scripts | Add only when needed, with a clear purpose and usage instructions. | As applicable |

> **Status:** The table describes planned or specified entry points, not a guarantee that all files have already been implemented. Refer to the current directory and GitHub issues for the actual status.

## Pipeline overview

```text
Ratings CSV + Metadata JSONL
             |
             v
       Cleaning & ETL
             |
             v
       Parquet datasets
             |
             v
 Temporal train / val / test split
             |
             v
     Popularity + Spark ALS
             |
             v
      Top-10 recommendations
             |
             v
 Recall@10 / NDCG@10 / Coverage
             |
             v
   Results + demo / GCP runs
```

## How to use

Run scripts **from the repository root** so relative paths resolve consistently. After completing the setup described in the root `README.md` and `docs/local_setup.md`, commands will typically follow this pattern:

```powershell
# Example only — run after these entry points have been implemented
python scripts/smoke_test.py
python scripts/run_pipeline.py --help
```

Check each script's `--help` output or module documentation for the supported arguments. The project may also use scripts in `deployment/` for GCP-specific jobs.

## Conventions

1. Avoid hard-coded machine-specific paths; use CLI arguments or versioned configuration files in `configs/`.
2. Keep reusable processing and model logic in `src/`, not duplicated in scripts.
3. Record dataset version, configuration, run ID, and results locations when running experiments.
4. Do not commit raw datasets, large generated artifacts, credentials, tokens, or service-account keys.
5. Include a brief usage note and link the relevant ticket in new scripts or pull requests.

## Related directories

- [`../src/`](../src/) — reusable ETL, splitting, model, evaluation, and serving code.
- [`../configs/`](../configs/) — experiment and runtime configuration.
- [`../deployment/`](../deployment/) — cloud deployment assets.
- [`../results/`](../results/) — small reports, metrics, and figures.
- [`../docs/`](../docs/) — setup notes, protocols, architecture, and reports.

For project scope and instructions, start with the [root README](../README.md).
