# EDA ? Amazon Electronics Ratings + Product Metadata

## Input files

- Ratings CSV: `C:\UIT\Khoa_2023_Nam_4_HK1\DuLieuLonTrenNenTangDamMay\Final_Project\data\raw\Electronics.csv`
- Metadata JSON: `C:\UIT\Khoa_2023_Nam_4_HK1\DuLieuLonTrenNenTangDamMay\Final_Project\data\raw\meta_Electronics.json`

Input: `C:\UIT\Khoa_2023_Nam_4_HK1\DuLieuLonTrenNenTangDamMay\Final_Project\data\raw\Electronics.csv`

- Total parsed rows: 20,994,353
- Valid rows: 20,994,353; invalid rows: 0
- Users: 9,838,676; items: 756,489
- Distinct user–item pairs: 20,553,480
- Sparsity: 99.99972385%
- Exact duplicate excess: 426,262
- Repeated-pair excess (includes exact duplicates): 440,873
- Historical period (UTC): 1997-12-04T00:00:00+00:00 → 2018-10-05T00:00:00+00:00

## Method notes

- Quality flags overlap; do not sum them.
- Blank lines are ignored by Spark CSV.
- No deduplication before rating/time distributions; use valid original rows.
- Sparsity and activity use distinct user-item pairs.
- Historical batch data; timestamps do not establish ingestion velocity.
- Remote/glob input requires input-version and an external file manifest.

## 5V evidence

- **Volume:** see counts above; local file bytes = 876247561. Compressed file size, if applicable, differs from decoded size.
- **Variety:** ratings come from `Electronics.csv`; product fields and join coverage come from `meta_Electronics.json`/`.gz`.
- **Velocity:** `monthly_activity.csv` and `figures/time_activity.png` show historical event frequency, not real-time ingestion speed.
- **Veracity:** `ratings_profile.json` gives missing/invalid counts and duplication. Error categories overlap. Excluded rows are not silently repaired.
- **Value:** supports testing Top-10 recommendations using Popularity and ALS; EDA does not prove recommendation quality.

## Interpretation

Sparsity = 1 − distinct(user,item) / (distinct users × distinct items). A large user–item matrix can be sparse even when the row count is large. Inspect the single-interaction entity counts and activity plots before designing evaluation.

Rating/time plots use all valid rows. Interaction histograms count distinct counterpart IDs, so duplicate ratings cannot inflate these histograms. No activity threshold is applied. No train/test split is created by this EDA; any later model preprocessing must fit its rules on training data only.

All sanity checks: True. Valid-row sample is sorted, deterministic, and not statistically representative. Inspect it against the source manually.
