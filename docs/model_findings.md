# Model findings (E6, T28)

Scope: Amazon Electronics, E4 Main cohort (5% of users, seed 42), chronological split (Validation from 2017‑02‑01, Test from 2017‑10‑01). K = 10, relevant = held‑out rating ≥ 4, seen items excluded, candidate catalog = history items. Test history = Train + Validation. The ALS configuration evaluated on Test (rank 8, maxIter 5, regParam 0.05, explicit feedback, seed 42) was locked on Validation by E5 before Test was read. Every number below is traceable to a file under `results/e6/`.

## Main result (Test, 75,009 target users, 55,347 users with a relevant target)

| Method | Recall@10 (95% CI) | NDCG@10 (95% CI) | HitRate@10 | User coverage | Catalog coverage |
| --- | --- | --- | ---: | ---: | ---: |
| Random (floor) | 0.000018 (0–0.000054) | 0.000009 (0–0.000027) | 0.000018 | 100% | 97.4% |
| Popularity | **0.008577** (0.007873–0.009297) | **0.005936** (0.005406–0.006496) | 0.011148 | 100% | 0.007% |
| ALS only | 0.000018 (0–0.000054) | 0.000006 (0–0.000019) | 0.000018 | 38.6% | 5.0% |
| ALS + Popularity fallback | 0.005552 (0.004988–0.006147) | 0.003783 (0.003370–0.004241) | 0.006685 | 100% | 5.0% |

Source: `results/e6/model_comparison.csv`. CIs: user-level bootstrap, 2,000 resamples.

## Findings

1. **The Popularity baseline is the best method on both Validation and Test.** Paired NDCG@10 difference Popularity − (ALS + Popularity) on Test = +0.00215 (95% CI 0.00183–0.00248), sign-flip test with Holm correction p ≤ 0.001; 248 users are better served by Popularity and 1 user by the hybrid. All five paired conclusions agree between Validation and Test (Popularity > ALS + Popularity > ALS only ≈ Random; ALS only and Random swap places by mean but are statistically tied in both phases), and the same order holds at K = 1, 3, 5, 10 and with relevance ≥ 5 instead of ≥ 4.
2. **Explicit ALS is statistically indistinguishable from random recommendations.** Validation: p_Holm = 0.43; Test: each method hits exactly one user, p_Holm = 1.0.
3. **The fallback carries the hybrid.** 61.4% of Test target users have no history, so ALS serves only 38.6% of users; the Popularity fallback raises user coverage to 100%. ALS never returns a short list, so every hybrid list is either all-ALS (warm users, Recall@10 = 0.000045) or all-Popularity (cold users, Recall@10 = 0.009306). The hybrid loses to Popularity because ALS replaces Popularity for warm users.
4. **Root cause: the explicit-rating objective on very sparse users.** On warm held-out pairs, ALS rating predictions have RMSE 3.07 (clipped to 1–5), worse than a constant global mean (1.37). Items ranked first by ALS have a median of 5 history ratings (Popularity: 741 on Test) and a mean predicted score above 9 on a 1–5 scale. A Validation-only diagnostic ablation shows that raising regParam to 0.5 or 1.0 does not help, while `implicitPrefs=True` raises ALS-only Recall@10 about 34-fold (0.000045 → 0.001537) and hybrid NDCG@10 from 0.001465 to 0.002218, still below Popularity (0.002353). The ablation was not used to choose the Test model.
5. **The Validation tuning grid did not separate configurations.** The spread of hybrid NDCG@10 across the four E5 trials (1.5e‑5) is of the same order as the spread across three seeds of the selected configuration (8.8e‑6).
6. **Accuracy and diversity pull in opposite directions.** Popularity recommends only 14 distinct items to all 75,009 users (personalization 0.005, Gini 1.00, 100% head items). ALS covers 5.0% of the catalog with near-unique lists (personalization 0.997) but almost never hits. The hybrid sits in between (personalization 0.623, 61% head items).
7. **Absolute values are low because the protocol is strict, not because the measurement is wrong.** Users have 1.41 relevant targets on average among 203,850 catalog items, and 14.5% of targets are items absent from history. The reachable ceiling for any history-catalog recommender is Recall@10 = 0.854.

## Reliability evidence

- All four recommendation tables pass integrity checks in both phases: no duplicates, contiguous ranks, ≤ 10 items, no seen items, no items outside the history catalog, no held-out-only items, no users outside the target set (`integrity.csv`).
- No temporal leakage: history ends 2017‑09‑30 and the Test period starts 2017‑10‑01, with zero shared user–item pairs.
- E6 reproduces the E5 metric files exactly (maximum absolute difference 6.1e‑18 across Recall, NDCG, user counts and coverage).
- The pandas and Spark implementations agree to 1e‑12 on every per-user column; 24 hand-computed unit tests pass.
- Retraining with seed 42 reproduces the E5 Validation metrics exactly.

## Runtime (local[2], 2 vCPU)

ALS fit on 819,984 Train ratings takes 20.0 s; ALS candidate generation dominates inference (73.6 s Validation, 125.8 s Test, of which `recommendForUserSubset` is 120.4 s). End-to-end CLI times: Popularity 75–79 s, ALS generate 131 s, ALS test-final with refit 250 s. Cloud runtime and cost are reported by E8.

## Limitations

Single 5% user cohort; local execution; K limited to 10 by the stored lists; bootstrap assumes independent users; category diversity uses partial metadata (75–80% of recommended slots); the implicit-ALS ablation is untuned and Validation-only.
