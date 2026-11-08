# Amazon Electronics Big Data Recommender

**Project:** Scalable and Cost-Efficient Product Recommendation System Using Apache Spark on Amazon Electronics Data

This repository is a shared workspace for a Big Data final project. The goal is to compare Popularity and Apache Spark MLlib ALS for Top-10 product recommendations, evaluate ranking quality, and benchmark scalability and cost using Google Cloud.

## Dataset

Source: [UCSD Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html#subsets).

- Electronics Ratings Only: approximately 20.99 million records; headerless CSV with `product_id,user_id,rating,timestamp`.
- Electronics Metadata: JSON Lines with `asin`, `title`, `brand`, `category`, and optional fields. The join key is Ratings `product_id` = Metadata `asin`.
- Metadata enriches recommendation display; ALS trains from clean user-item ratings, not text or product price.

**Data files are not committed to Git.** Put original input files in `data/raw/`, samples in `data/subsets/`, and publish large shared datasets to Google Cloud Storage as appropriate.

## Structure and Epic ownership

| Directory | Usage |
| --- | --- |
| `configs/` | Spark configuration examples |
| `data/raw/`, `data/subsets/` | Local input data (gitignored) |
| `src/exploration/` | E2 — exploratory data analysis and 5V |
| `src/etl/` | E3 — data validation, cleaning, metadata deduplication and Parquet |
| `src/splitting/` | E4 — time-based split and user/item mapping |
| `src/models/` | E5 — Popularity, ALS and cold-start fallback |
| `src/evaluation/` | E6 — Recall@10, NDCG@10, coverage |
| `deployment/gcp/` | E7 — cloud batch jobs and deployment |
| `scripts/`, `tests/` | Entry points, automation and tests |
| `results/` | Small shareable summaries and charts |
| `docs/architecture/`, `docs/reports/` | Design documentation and report drafts |
| `src/serving/` | E9 — lookup of Top-10 recommendations |

## Development environment

Recommended initial local configuration: Python 3.10, JDK 17 and PySpark 4.0.5. Verify compatibility with the selected Google Cloud Spark runtime before deploying.

Windows PowerShell (when Python 3.10 is installed):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
java -version
.\.venv\Scripts\python.exe tests/test_spark_smoke.py
```

Set `JAVA_HOME` to the location of your installed JDK 17 as needed. If Spark write operations fail because of missing Windows Hadoop native utilities, run the file I/O tests in WSL2 Ubuntu or a GCP Linux runtime.

## Collaboration workflow

1. Clone this repository and start from an up-to-date `main` branch.
2. Use a ticket-specific branch, e.g. `feature/e3-etl`.
3. Keep data files, virtual environments and credentials out of commits.
4. Open a Pull Request and have another member review before merging.
5. Explain input and output schemas, running instructions, and testing evidence in each PR.

## Current status

This is the **T02 repository scaffold**, not a completed recommender. E3 and other Epic implementation will be committed separately by their owners.
