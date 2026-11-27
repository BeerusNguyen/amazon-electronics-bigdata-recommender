# E6 — Model Evaluation (T25–T28)

Tài liệu này mô tả cách E6 đánh giá ba phương án gợi ý Top‑10 của E5 (Popularity, ALS, ALS + Popularity fallback), cách kiểm chứng độ tin cậy của chính phép đánh giá, kết quả đo thật trên cohort Main của E4 và cách đọc từng biểu đồ.

Mọi con số trong tài liệu đều sinh từ code trong repo và có file nguồn đi kèm (cột "Nguồn"). Không có số nào được ước lượng bằng tay.

## 0. Tóm tắt kết quả

Bảng chính (Test, cohort Main 5% user, K = 10, relevant = rating ≥ 4, 75.009 target users, 55.347 users có ít nhất một mục tiêu relevant). Ngoặc vuông là khoảng tin cậy 95% bootstrap theo user.

| Phương án | Recall@10 | NDCG@10 | HitRate@10 | User coverage ≥1 | Catalog coverage | Personalization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random (sàn kiểm tra) | 0,000018 [0; 0,000054] | 0,000009 | 0,000018 | 100% | 97,4% | 1,000 |
| **Popularity** | **0,008577** [0,007873; 0,009297] | **0,005936** [0,005406; 0,006496] | **0,011148** | 100% | 0,007% (14 item) | 0,005 |
| ALS only | 0,000018 [0; 0,000054] | 0,000006 | 0,000018 | 38,6% | 5,0% | 0,997 |
| ALS + Popularity | 0,005552 [0,004988; 0,006147] | 0,003783 [0,003370; 0,004241] | 0,006685 | 100% | 5,0% | 0,623 |

Nguồn: `results/e6/model_comparison.csv` (mọi file kết quả nằm trong `results/e6/`, mục 13).

Năm kết luận chính, đều đã qua kiểm định có hiệu chỉnh Holm:

1. **Popularity thắng trên cả Validation lẫn Test.** Chênh lệch NDCG@10 Popularity − ALS+Popularity trên Test là +0,00215 (CI95 [0,00183; 0,00248], p_Holm ≤ 0,001). Trên cùng tập user, 248 user được Popularity phục vụ tốt hơn, chỉ 1 user ngược lại.
2. **ALS explicit của E5 không tốt hơn Random.** Validation: p_Holm = 0,43; Test: hai phương án trúng đúng 1 user mỗi bên, p_Holm = 1,0. ALS gần như không học được tín hiệu xếp hạng trên dữ liệu này.
3. **Fallback là thứ duy nhất làm ALS+Popularity có điểm.** 61,4% target user là user mới (không có lịch sử) nên ALS không phục vụ được; fallback nâng user coverage từ 38,6% lên 100%. Toàn bộ Recall của ALS+Popularity đến từ phần Popularity phục vụ user mới; ở phần user cũ, ALS chiếm chỗ của Popularity nên kéo điểm xuống.
4. **Nguyên nhân là mục tiêu explicit-rating, không phải pipeline.** ALS dự đoán rating trên cặp warm có RMSE 3,08, tệ hơn đoán bằng trung bình toàn cục (1,33). Tăng regParam không cứu được (RMSE 2,92–3,09). Ablation chẩn đoán với `implicitPrefs=True` làm Recall@10 của ALS tăng khoảng 34 lần trên Validation (0,000045 → 0,001537), nhưng vẫn dưới Popularity.
5. **Tuning của E5 nằm trong vùng nhiễu.** Bốn trial chênh nhau 1,5e‑5 NDCG, trong khi đổi seed với cùng config đã làm dao động 8,8e‑6. Config được chọn không khác biệt có ý nghĩa với các trial còn lại.

Giá trị tuyệt đối thấp (Recall ~1%) là đặc trưng của giao thức nghiêm ngặt chứ không phải lỗi: chia theo thời gian, 61% user là user mới, trung bình mỗi user chỉ có 1,4 mục tiêu relevant giữa ~200.000 item. Trần lý thuyết (mọi item ứng viên phải nằm trong catalog lịch sử) là Recall@10 = 0,854 trên Test, nên còn rất nhiều khoảng trống để cải thiện.

## 1. Phạm vi và bàn giao theo ticket

| Ticket | Yêu cầu | File bàn giao |
| --- | --- | --- |
| T25 | Recall@10 + unit test | `src/evaluation/recall_at_k.py`, `src/evaluation/protocol.py` (ground truth), `tests/test_e6_metrics.py` |
| T26 | NDCG@10 + unit test | `src/evaluation/ndcg_at_k.py`, `tests/test_e6_metrics.py` |
| T27 | User coverage, fallback effectiveness | `src/evaluation/coverage.py`, `results/e6/coverage.csv`, `results/e6/fallback.csv` |
| T28 | So sánh ALS vs Popularity công bằng | `src/evaluation/run_e6.py`, `results/e6/model_comparison.csv`, `docs/model_findings.md`, `results/e6/figures/` |
| Bổ sung | Metric phụ, kiểm định, beyond-accuracy, độ tin cậy, chẩn đoán | `per_user_metrics.py`, `significance.py`, `beyond_accuracy.py`, `als_diagnostics.py`, `e6_figures.py`, notebook, `tests/test_e6_spark_parity.py` |

Quan hệ với các Epic khác:

- E6 đọc nguyên trạng `data/processed/e4/` (split, mapping) và `outputs/e5/` (recommendations, metrics, model, tuning). E6 không sửa dữ liệu E4 hay output E5.
- E6 không đọc Test để chọn cấu hình. Config ALS đánh giá trên Test là config E5 đã khóa trong `outputs/e5/tuning/best_validation.json`.
- E7/E8 dùng lại được `run_e6 evaluate` trên GCP vì đầu vào là các đường dẫn Parquet; các hàm Spark tương đương đã có sẵn (mục 9.3).

## 2. Giao thức đánh giá (T18 đã khóa, E6 thực thi)

| Quy tắc | Áp dụng trong E6 |
| --- | --- |
| K | 10 (thêm K = 1, 3, 5 chỉ để phân tích độ nhạy, lấy tiền tố của cùng danh sách) |
| Relevant | rating held-out ≥ 4. Rating ≥ 5 chỉ dùng cho phân tích độ nhạy, không dùng để chọn mô hình |
| Ground truth | mỗi cặp (user, item) relevant trong held-out; loại item user đã đánh giá trong lịch sử; khử trùng cặp |
| Lịch sử | Validation: Train. Test: Train + Validation |
| Catalog ứng viên | item xuất hiện trong lịch sử của snapshot |
| Target users | mọi user distinct trong held-out (mẫu số của coverage) |
| Scored users | target user có ≥ 1 mục tiêu relevant (mẫu số của Recall/NDCG) |
| User không có gợi ý | vẫn nằm trong mẫu số, nhận 0 điểm, không bị loại im lặng |
| Hai phạm vi | `all`: giữ cả mục tiêu cold-item trong mẫu số (phản ánh giới hạn hệ thống). `warm_item`: chỉ mục tiêu có trong catalog (đo năng lực xếp hạng thuần) |
| Trung bình | macro: tính từng user rồi lấy trung bình |

Ví dụ tính tay (trùng README E4/E5, đã thành unit test): relevant = {A, C}, danh sách = [X, A, B].

- Recall@10 = 1/2 = 0,5
- DCG = 1/log2(3) = 0,6309; IDCG = 1 + 1/log2(3) = 1,6309; NDCG@10 = 0,3868528
- Precision@10 = 1/10; HitRate = 1; MRR = 1/2; AP@10 = (1/2) / min(2, 10) = 0,25

## 3. Cách chạy lại

Chạy từ thư mục gốc repo, môi trường như E5 (Python 3.10, JDK 17, PySpark 3.5.3) cộng `matplotlib`.

```powershell
# note: E5 phải chạy theo đúng thứ tự này để có đủ recommendations cho cả hai phase
python -m src.models.run_e5 run-all --phase validation
python -m src.models.run_e5 tune --phase validation
python -m src.models.run_e5 generate --phase validation      # sinh lại als-only/als cho config đã chọn
python -m src.models.run_e5 test-final --phase test
python -m src.models.run_e5 popularity --phase test          # test-final không lưu Popularity Test

# E6
python -m pytest -q tests/test_e6_metrics.py tests/test_e6_spark_parity.py
python -m src.evaluation.run_e6 evaluate --phase validation --metadata-dir <E3>/processed/metadata_clean
python -m src.evaluation.run_e6 evaluate --phase test --metadata-dir <E3>/processed/metadata_clean
python -m src.evaluation.run_e6 diagnose          # rating error, item support, bảng tuning E5
python -m src.evaluation.run_e6 seed-stability    # Spark, chỉ Validation
python -m src.evaluation.run_e6 ablation          # Spark, chỉ Validation
python -m src.evaluation.run_e6 compare --timings <file step,seconds>
python -m src.evaluation.run_e6 figures
python -m src.evaluation.run_e6 report            # chạy notebook và xuất báo cáo HTML
```

### 3.1 Xem kết quả: notebook và báo cáo HTML

Có ba cách xem, tất cả đọc cùng một bộ file trong `results/e6/`:

| Cách xem | Dành cho | Cần cài gì |
| --- | --- | --- |
| `results/e6/E6_evaluation_report.html` | đọc nhanh, gửi cô hoặc nhóm, đính kèm báo cáo | chỉ cần trình duyệt |
| `src/evaluation/notebooks/E6_evaluation.ipynb` | theo dõi và làm mới sau mỗi lần E5 thay đổi | Python + pandas + matplotlib + jupyter |
| `results/e6/*.csv` | trích số cho báo cáo T38 và slide | không cần gì |

Notebook là lớp trình bày mỏng: mọi phép tính vẫn nằm trong `src/evaluation/*.py` đã có unit test. Notebook gồm bảng kiểm tra độ tin cậy tự động (PASS/FAIL), bảng tô sáng giá trị tốt nhất, biểu đồ kèm hướng dẫn đọc và phần kết luận được sinh từ số liệu. Mặc định notebook chỉ đọc kết quả có sẵn (khoảng 15 giây). Bật `RUN_EVALUATE = True` ở ô cấu hình để chạy lại đánh giá.

Làm mới notebook và báo cáo HTML sau khi chạy lại pipeline chỉ cần một lệnh: `python -m src.evaluation.run_e6 report`. Lệnh này chạy lại toàn bộ notebook, lưu output vào notebook, rồi xuất `E6_evaluation_report.html` đã ẩn code và nhúng sẵn biểu đồ, nên chỉ cần gửi một file duy nhất (cần `pip install nbconvert ipykernel`).

Hai lưu ý môi trường đã gặp khi chạy thật:

- Máy 8 GB: đặt `SPARK_DRIVER_MEMORY=3g` cho `seed-stability`/`ablation`. Với 5g, JVM bị hệ điều hành kill do Python cũng cần bộ nhớ.
- JDK 21: Arrow 12 trong PySpark 3.5 lỗi khi chuyển Spark ↔ pandas. E6 tự tắt Arrow trong phiên Spark của mình nên chạy được cả JDK 17 và 21.

Thời gian chạy đo được (local[2], 2 vCPU): `evaluate` khoảng 2,5–4 phút mỗi phase; `seed-stability` 5 phút; `ablation` 7,6 phút; E5 tune 14,7 phút.

## 4. T25 — Recall@10

Định nghĩa:

```text
Recall@K(u) = |TopK(u) ∩ R(u)| / |R(u)|
Recall@K    = trung bình Recall@K(u) trên các scored users
```

Mẫu số là toàn bộ |R(u)|, kể cả khi |R(u)| > K. Vì vậy một danh sách hoàn hảo của user có 12 mục tiêu chỉ đạt 10/12. Đây là lựa chọn của T18 và được giữ nguyên để so sánh được với E5.

Cài đặt:

- `recall_at_k_user`: hàm tham chiếu cho một user, dùng trong unit test.
- `per_user_ranking_metrics` (pandas, vectorised) và `per_user_ranking_metrics_spark`: tính đồng thời mọi metric theo user. Cùng một bảng theo user được dùng cho trung bình, khoảng tin cậy và kiểm định cặp.
- Bảo vệ đầu vào: danh sách trùng item hoặc trùng rank thì dừng bằng `ValueError`; rank phải bắt đầu từ 1.

Unit test (giá trị kỳ vọng tính tay, ghi ngay trong comment của test):

| Test | Tình huống | Kỳ vọng |
| --- | --- | --- |
| `test_recall_readme_example` | ví dụ README | 0,5 |
| `test_recall_uses_full_relevant_count_even_above_k` | 12 relevant, 10 hit | 10/12 |
| `test_recall_truncates_at_k` | relevant ở rank 11 | 0 |
| `test_recall_no_relevant_user_is_excluded` | user không có relevant | None (không vào mẫu số) |
| `test_recall_macro_keeps_unserved_relevant_users_as_zero` | user có mục tiêu nhưng không có list | tính 0, mẫu số = 2 |
| `test_recall_rejects_duplicates` | danh sách trùng | ValueError |

## 5. T26 — NDCG@10

```text
DCG@K(u)  = Σ_{i=1..K} gain_i / log2(i + 1)
IDCG@K(u) = DCG của thứ tự lý tưởng, cắt tại K
NDCG@K(u) = DCG@K(u) / IDCG@K(u)
```

- **Binary (chính, theo T18):** gain = 1 nếu relevant.
- **Graded (bổ sung):** gain = 2^level − 1 với level = rating − 3, nên mục tiêu 5 sao có trọng số 3, mục tiêu 4 sao có trọng số 1. Chỉ báo cáo kèm, không dùng để chọn mô hình.
- Không nén vị trí sau khi bỏ item không relevant (một hit ở rank 3 sau hai item trượt có NDCG = 0,5).
- Cùng quy tắc mẫu số và ứng viên với Recall.

Unit test: ví dụ README (0,3868528), danh sách hoàn hảo (1), danh sách rỗng (0), IDCG cắt tại K, vị trí không bị nén (0,5), graded NDCG với [4★, 5★] = (1 + 3/log2 3)/(3 + 1/log2 3) = 0,796708, macro (0,1934264).

Kết quả: Binary và graded NDCG gần như trùng nhau (Test Popularity 0,005936 và 0,005961) vì 84% mục tiêu relevant trên Test là 5 sao (Validation 83%), nên thứ tự xếp hạng không đổi khi dùng graded gain.

## 6. Metric bổ sung và lý do thêm

| Metric | Ý nghĩa | Vì sao cần |
| --- | --- | --- |
| Precision@10 | hit / 10 | đo hiệu quả dùng 10 ô gợi ý, phạt danh sách dài vô ích |
| HitRate@10 | tỷ lệ user có ít nhất 1 hit | dễ giải thích cho người đọc báo cáo, ổn định khi user chỉ có 1 mục tiêu |
| MRR@10 | 1 / rank của hit đầu tiên | đo hit có ở đầu danh sách không |
| MAP@10 | trung bình precision tại các vị trí hit, chia min(|R|, K) | kết hợp thứ hạng và số hit |
| Phạm vi `warm_item` | chỉ mục tiêu có trong catalog | tách lỗi "không thể gợi ý" khỏi lỗi "xếp hạng kém" |
| Trần reachable | min(10, mục tiêu warm) / mục tiêu, trung bình theo user | đặt kết quả tuyệt đối vào đúng bối cảnh |

## 7. T27 — Coverage và hiệu quả fallback

Hai khái niệm được tách riêng, đúng Definition of Done của T27:

- **User coverage** (mẫu số = mọi target user, kể cả user không có mục tiêu relevant): tỷ lệ nhận ≥ 1 gợi ý và tỷ lệ nhận đủ 10 gợi ý.
- **Catalog coverage** (mẫu số = số item trong catalog lịch sử): tỷ lệ item từng xuất hiện trong ít nhất một danh sách. Nằm trong `results/e6/beyond_accuracy.csv`.

Trạng thái phục vụ của ALS trước fallback (Test, 75.009 target users):

| Trạng thái | Định nghĩa | Số user | Tỷ lệ |
| --- | --- | ---: | ---: |
| `full_list` | ALS trả đủ 10 item | 28.936 | 38,6% |
| `short_list` | ALS trả 1–9 item | 0 | 0% |
| `known_unserved` | có lịch sử nhưng ALS không trả gì | 0 | 0% |
| `unknown_user` | không có lịch sử | 46.073 | 61,4% |

Trước fallback user coverage = 38,6%, sau fallback = 100% (cả ≥ 1 và đủ 10). Fallback "cứu" 46.073 user khỏi danh sách rỗng bằng cách điền đủ 10 ô cho mỗi user mới (tính trên toàn bộ target users là 6,14 ô Popularity mỗi user).

Phát hiện khi kiểm chứng: ALS không bao giờ trả danh sách ngắn (over-fetch 50 ứng viên luôn đủ sau khi lọc seen), nên mọi danh sách ALS+Popularity là thuần ALS hoặc thuần Popularity. Không có danh sách "Mixed" nào, dù E5 README có mô tả nhãn này. Ngoài ra cột `recommendation_source` của E5 thực tế lưu nguồn theo từng dòng, không phải theo danh sách; E6 ghi nguồn của danh sách hybrid theo trạng thái ALS trong cột `hybrid_list_source` của `fallback.csv` (`full_list` → ALS, `short_list` → Mixed, còn lại → Popularity). Nhóm E9 cần biết điều này khi hiển thị nguồn gợi ý trong demo.

Chất lượng theo trạng thái (Test, ALS+Popularity): user `unknown_user` đạt Recall 0,009306 (do Popularity phục vụ); user `full_list` chỉ đạt 0,000045 (do ALS phục vụ). Nguồn: cột `hybrid_recall`, `hybrid_ndcg` của `results/e6/fallback.csv`.

## 8. T28 — So sánh công bằng

### 8.1 Điều kiện công bằng được kiểm tra bằng code

`results/e6/protocol.json` ghi lại các mẫu số mà mọi phương án dùng chung: cùng 75.009 target users, cùng lịch sử 926.131 cặp, cùng catalog 203.850 item, cùng K = 10, cùng ngưỡng relevant 4, cùng ground truth, cùng quy tắc mẫu số. Mọi bảng gợi ý phải qua `integrity.csv` (mục 9.1) trước khi được tính điểm.

Config ALS trên Test là config đã khóa từ Validation (rank 8, maxIter 5, regParam 0,05, seed 42). Không có lựa chọn nào dựa trên Test.

### 8.2 Bảng so sánh đầy đủ

`results/e6/model_comparison.csv` có một dòng cho mỗi (phase, phương án) với: Recall, NDCG (kèm CI95), Precision, HitRate, MRR, MAP, NDCG graded ở cả hai phạm vi `all` và `warm_item`; user coverage; catalog coverage, Gini, head share, novelty, personalization; thời gian train/inference/end-to-end.

Validation (55.265 scored users):

| Phương án | Recall@10 | NDCG@10 | HitRate@10 |
| --- | ---: | ---: | ---: |
| Random | 0,000018 | 0,000005 | 0,000018 |
| Popularity | 0,005992 [0,005379; 0,006639] | 0,002353 [0,002125; 0,002599] | 0,008305 |
| ALS only | 0,000045 [0,000004; 0,000106] | 0,000034 | 0,000090 |
| ALS + Popularity | 0,003747 [0,003261; 0,004250] | 0,001465 [0,001284; 0,001659] | 0,004886 |

Cả 5 kết luận kiểm định cặp (mục 8.3) đều giống nhau giữa Validation và Test: Popularity > ALS + Popularity > (ALS only ≈ Random). ALS only và Random đổi chỗ khi chỉ sắp theo trung bình (Validation ALS cao hơn, Test Random cao hơn), nhưng chênh lệch không có ý nghĩa ở cả hai phase, nên đó là hòa chứ không phải đảo thứ tự. Kết luận vì vậy không phụ thuộc vào một giai đoạn thời gian cụ thể.

### 8.3 Kiểm định thống kê

Một trung bình macro trên hàng chục nghìn user có phân phối rất lệch (hơn 99% user có điểm 0), nên chênh lệch nhỏ có thể chỉ là nhiễu. E6 dùng:

- **CI95 bootstrap** (2.000 lần lấy mẫu lại user) cho từng trung bình.
- **CI95 bootstrap cặp** cho chênh lệch trên cùng user.
- **Sign-flip randomisation test** (10.000 hoán vị) cho giả thuyết chênh lệch trung bình = 0. Đây là kiểm định cặp, không giả định phân phối chuẩn.
- **Hiệu chỉnh Holm** cho 10 kiểm định mỗi phase (5 cặp × Recall/NDCG). Giá trị nhỏ nhất kiểm định này báo được là 0,0001, sau Holm là 0,001, nên bảng ghi "≤ 0,001".

Kết quả Test (NDCG@10, A − B):

| So sánh | Chênh lệch | CI95 | User A tốt hơn / B tốt hơn | p_Holm |
| --- | ---: | --- | --- | ---: |
| Popularity − Random | +0,00593 | [0,00540; 0,00649] | 617 / 1 | ≤ 0,001 |
| ALS only − Random | −0,000003 | [−0,000027; 0,000019] | 1 / 1 | 1,0 |
| ALS only − Popularity | −0,00593 | [−0,00649; −0,00540] | 1 / 617 | ≤ 0,001 |
| ALS+Pop − Popularity | −0,00215 | [−0,00248; −0,00183] | 1 / 248 | ≤ 0,001 |
| ALS+Pop − ALS only | +0,00378 | [0,00337; 0,00423] | 369 / 0 | ≤ 0,001 |

Nguồn: `results/e6/significance.csv`.

### 8.4 Theo cohort cold-start (E4)

Recall@10 trên Test (`results/e6/cohorts.csv`):

| Cohort | Scored users | Popularity | ALS only | ALS + Popularity |
| --- | ---: | ---: | ---: | ---: |
| all | 55.347 | 0,008577 | 0,000018 | 0,005552 |
| warm user / warm item | 20.026 | 0,008619 | 0,000050 | 0,000050 |
| cold user / warm item | 28.674 | 0,010913 | 0 | 0,010913 |
| warm user / cold item | 4.522 | 0 | 0 | 0 |
| cold user / cold item | 5.605 | 0 | 0 | 0 |
| history 1–4 | 15.887 | 0,007894 | 0,000063 | 0,000063 |
| history ≥ 5 | 6.546 | 0,006571 | 0 | 0 |

Đọc bảng: các cohort cold-item bằng 0 với mọi phương án vì item không có trong catalog (giới hạn cấu trúc, không phải lỗi mô hình). Ngay cả user có ≥ 5 rating, nhóm mà ALS lẽ ra mạnh nhất, Popularity vẫn hơn.

### 8.5 Độ nhạy theo K và theo ngưỡng relevant

- K = 1, 3, 5, 10 (`metrics.csv`, cột `k`): thứ tự Popularity > ALS + Popularity > (ALS only ≈ Random) giữ nguyên ở mọi K. NDCG@1 của Popularity (0,0051) cao hơn NDCG@3 (0,0045) là bình thường vì NDCG không đơn điệu theo K.
- Relevant ≥ 5 thay vì ≥ 4 (`metrics.csv`, cột `relevance`): số scored users giảm (Test 55.347 → 47.616), điểm tăng nhẹ, thứ tự Popularity > ALS + Popularity > (ALS only ≈ Random) giữ nguyên. Kết luận không phụ thuộc vào ngưỡng.

### 8.6 Beyond-accuracy: danh sách trông như thế nào

| Chỉ số (Test) | Random | Popularity | ALS only | ALS + Pop |
| --- | ---: | ---: | ---: | ---: |
| Số item khác nhau được gợi ý | 198.581 | 14 | 10.232 | 10.242 |
| Catalog coverage | 97,4% | 0,007% | 5,0% | 5,0% |
| Gini phơi bày item | 0,29 | 1,00 | 0,98 | 0,99 |
| Tỷ lệ ô thuộc nhóm head | 0,6% | 100% | 0% | 61,4% |
| Novelty (bit) | 17,74 | 9,17 | 16,43 | 11,97 |
| Personalization | 1,000 | 0,005 | 0,997 | 0,623 |
| Category ILD (cấp 2) | 0,757 | 0,678 | 0,743 | 0,704 |

Popularity thắng về độ chính xác nhưng gần như gợi ý cùng một danh sách 14 item cho mọi người. Trong demo E9, mọi user sẽ thấy gần như cùng một danh sách. Đây là trade-off cần nêu rõ trong báo cáo. Category ILD tính trên các item có metadata E3 (75–80% số ô gợi ý), vì thư mục `metadata_clean` trên máy hiện chỉ có 3 file part (00000, 00002, 00003) với 567.061 ASIN.

### 8.7 Thời gian chạy (đo thật)

| Phase | Phương án | Train (fit) | Inference (stage timer E5) | End-to-end CLI |
| --- | --- | ---: | ---: | ---: |
| Validation | Popularity | gộp | gộp | 75,1 s |
| Validation | ALS only | 20,0 s | 73,6 s | 131,0 s |
| Validation | ALS + Popularity | 20,0 s | 83,8 s | 131,0 s |
| Test | Popularity | gộp | gộp | 78,6 s |
| Test | ALS only | không tách | 125,8 s | 249,9 s |
| Test | ALS + Popularity | không tách | 138,9 s | 249,9 s |

Inference ALS bị chi phối bởi `recommendForUserSubset` (Test: 120,4 s trên 125,8 s). Train ALS chỉ mất 20 s. Thời gian Popularity là cả lệnh CLI vì E5 không tách bước. Đo trên local[2], 2 vCPU; không dùng để kết luận về GCP (thuộc E8). Nguồn: `results/e6/runtime.csv`.

## 9. Kiểm chứng độ tin cậy của phép đánh giá

Mục tiêu: chứng minh các con số trên phản ánh mô hình chứ không phản ánh lỗi đo.

### 9.1 Tính toàn vẹn của từng bảng gợi ý

Mọi kiểm tra của mục 9 được ghi vào một file duy nhất `results/e6/reliability_checks.csv` (cột `phase`, `check`, `subject`, `value`, `status`, `detail`); notebook tóm tắt file này thành bảng PASS/FAIL ở đầu báo cáo.

Với cả bốn phương án và cả hai phase, mọi kiểm tra đều bằng 0 vi phạm (PASS): không trùng cặp user–item, không trùng rank, rank liên tục từ 1, không danh sách dài hơn 10, không gợi ý lại item đã thấy, không gợi ý item ngoài catalog lịch sử, không gợi ý item chỉ có trong held-out (rò rỉ tương lai), không user ngoài tập target, không khóa null. Nếu một bảng FAIL, `evaluate` ghi lại bằng chứng rồi dừng, không tính điểm trên danh sách lỗi.

### 9.2 Không rò rỉ thời gian

`temporal_integrity`: timestamp lớn nhất của lịch sử nhỏ hơn timestamp nhỏ nhất của held-out (Validation: 31/01/2017 < 01/02/2017; Test: 30/09/2017 < 01/10/2017) và 0 cặp user–item chung giữa lịch sử và held-out.

### 9.3 Bốn lớp đối chiếu độc lập

| Đối chiếu | Kết quả |
| --- | --- |
| Hàm tham chiếu từng user ↔ bảng vectorised pandas | khớp trên 60 user ngẫu nhiên, mọi metric (`test_vectorised_table_matches_reference_functions`) |
| Pandas ↔ Spark (ground truth + 14 cột metric) | sai lệch ≤ 1e‑12 (`test_e6_spark_parity.py`) |
| E6 ↔ hàm `ranking_metrics` của E5 trên toy data | khớp (`test_matches_e5_ranking_metrics`) |
| E6 ↔ file metric E5 trên dữ liệu thật (Recall, NDCG, số user, coverage, cả 3 phương án × 2 phase) | sai lệch lớn nhất 6,1e‑18 (`check` bắt đầu bằng `e5_cross_check`) |

### 9.4 Tái lập

Huấn luyện lại ALS với seed 42 trong E6 cho đúng Recall 4,478422e‑05 và NDCG 0,0014653337 của E5. Pipeline cho cùng một kết quả khi chạy lại với cùng seed.

### 9.5 Sàn và trần

- Random là sàn: một mô hình có ích phải thắng Random có ý nghĩa thống kê. Popularity thắng; ALS only thì không.
- Trần reachable: Recall@10 tối đa 0,829 (Validation) và 0,854 (Test) khi chỉ gợi ý item trong catalog. 14,5% mục tiêu Test là cold item, không phương án nào trong E5 chạm tới được.

### 9.6 Độ ổn định theo seed

Cùng config đã khóa, ba seed 42/7/2026 (`diagnostics/seed_stability.csv`): NDCG@10 của ALS+Popularity dao động 0,0014565–0,0014653 (biên độ 8,8e‑6); ALS only 2,5e‑5–3,4e‑5. Chênh lệch giữa bốn trial tuning là 1,5e‑5, cùng bậc với nhiễu seed. Nghĩa là T22 chưa chọn được config tốt hơn một cách đáng tin cậy.

### 9.7 Điều E6 không khẳng định

- Không suy rộng ra toàn bộ 20,99M rating: đây là cohort 5% user.
- Không kết luận "ALS luôn kém Popularity": chỉ kết luận cho ALS explicit với lưới tham số của E5.
- Ablation implicit chỉ chạy trên Validation, không tuning, không chạy Test, không thay thế mô hình của E5.

## 10. Chẩn đoán: vì sao ALS thua

### 10.1 Dự đoán rating sai nghiêm trọng

`diagnostics/als_rating_prediction.csv`, chỉ trên cặp warm user–warm item của held-out:

| Predictor | RMSE Validation | RMSE Test |
| --- | ---: | ---: |
| ALS (cắt về 1–5) | 3,080 | 3,074 |
| ALS (không cắt) | 5,220 | 5,249 |
| Trung bình toàn cục | 1,335 | 1,374 |
| Trung bình theo item | 1,388 | 1,420 |
| Trung bình theo user | 1,482 | 1,517 |

ALS dự đoán rating còn tệ hơn hằng số. Spark ALS không có bias term (global, user, item), trong khi phần lớn user chỉ có 1–4 rating. Vector ẩn không đủ dữ liệu để vừa tái tạo mức rating ~4,2 vừa xếp hạng. Điểm dự đoán trung bình của các item ALS xếp đầu là ~9,2–9,6 trên thang 1–5.

### 10.2 ALS đẩy item ít được đánh giá lên đầu

`diagnostics/als_item_support.csv`: item ALS gợi ý có trung vị 5 rating lịch sử (Popularity: 657 trên Validation, 741 trên Test), và 32% ô gợi ý là item có ≤ 3 rating. Đây là dấu hiệu kinh điển của explicit MF trên dữ liệu thưa: item ít rating có vector cực trị nên điểm dự đoán cao.

### 10.3 Ablation chẩn đoán (chỉ Validation)

`diagnostics/als_ablation.csv`, giữ rank 8, maxIter 5:

| Biến thể | RMSE warm | ALS only Recall | ALS only NDCG | ALS+Pop NDCG |
| --- | ---: | ---: | ---: | ---: |
| Explicit, regParam 0,05 (E5 đã khóa) | 3,08 | 0,000045 | 0,000034 | 0,001465 |
| Explicit, regParam 0,5 | 2,92 | 0,000056 | 0,000038 | 0,001469 |
| Explicit, regParam 1,0 | 3,09 | 0,000008 | 0,000004 | 0,001436 |
| Implicit, alpha 1 | (không áp dụng) | 0,001537 | 0,000787 | 0,002218 |
| Implicit, alpha 10 | (không áp dụng) | 0,001431 | 0,000713 | 0,002144 |
| Popularity (tham chiếu) | | | | 0,002353 |

Tăng regularisation không giúp; đổi sang implicit feedback giúp rất rõ. RMSE không có ý nghĩa cho mô hình implicit vì nó dự đoán mức ưa thích 0–1, không phải rating.

### 10.4 Khuyến nghị cho E5 (ngoài phạm vi E6, không thay đổi kết quả Test hiện tại)

1. Chuyển sang `implicitPrefs=True` và tune rộng hơn: rank 16–64, regParam 0,01–1, alpha 1–40, maxIter 10–15.
2. Tune trên ALS only, chỉ warm users, vì 61% user cold nhận cùng danh sách Popularity ở mọi trial. Chỉ số ALS+Popularity làm loãng tín hiệu.
3. Mỗi trial chạy ≥ 3 seed; chỉ chấp nhận config thắng nếu chênh lệch vượt nhiễu seed hoặc qua kiểm định cặp.
4. Xem xét hybrid theo phân đoạn: Popularity cho user ít lịch sử, ALS (implicit) cho user nhiều lịch sử, và chỉ giữ nếu thắng có ý nghĩa trên Validation.

Sau mọi thay đổi, chỉ cần chạy lại E5 rồi `run_e6 evaluate` để có so sánh mới với cùng giao thức.

## 11. Giải thích các biểu đồ

Tất cả nằm trong `results/e6/figures/`, vẽ từ CSV bởi `src/evaluation/e6_figures.py`. Màu theo phương án và không đổi giữa các biểu đồ: Popularity xanh dương, ALS only cam, ALS + Popularity xanh ngọc, Random xám (đường tham chiếu). Bảng màu đã chạy validator: phân biệt được với người mù màu ở mọi cặp. Mọi cột đều có nhãn giá trị và có CSV đi kèm.

| File | Đọc thế nào | Thông điệp chính |
| --- | --- | --- |
| `fig01_quality_ci.png` | Recall@10 (trái) và NDCG@10 (phải); mỗi nhóm là một phase; thanh lỗi là CI95 bootstrap | Popularity cao nhất ở cả hai phase; CI không chồng lên ALS+Popularity; ALS only và Random bằng 0 trên thang này |
| `fig02_significance.png` | Forest plot: chấm là chênh lệch NDCG cặp, đoạn là CI95; xám = không có ý nghĩa sau Holm | Mọi so sánh có ý nghĩa trừ ALS only − Random ở cả hai phase |
| `fig03_coverage_fallback.png` | Trái: thanh 100% chia target users theo trạng thái ALS. Phải: user coverage trước/sau fallback (Test) | 61% user là user mới; fallback đưa coverage từ 38,6% lên 100%; không có danh sách ngắn |
| `fig04_cohorts.png` | Recall@10 theo 7 cohort E4 (Test); n dưới nhãn là số scored users của cohort | Điểm của ALS+Popularity đến từ cohort cold user (Popularity phục vụ); cột cold item bằng 0 với mọi phương án |
| `fig05_k_sensitivity.png` | Recall@K và NDCG@K với K = 1, 3, 5, 10 (Test) | Thứ tự phương án ổn định theo K; ALS only và Random trùng nhau sát 0 |
| `fig06_beyond_accuracy.png` | Bốn ô nhỏ: catalog coverage, tỷ lệ ô từ head items, novelty, personalization (Test) | Popularity chính xác nhất nhưng chỉ dùng 14 item và không cá nhân hóa; ALS đa dạng nhưng không trúng |
| `fig07_als_item_support.png` | Độ phổ biến lịch sử của item được Popularity và ALS gợi ý (Test) | ALS chọn item trung vị 5 rating, 32% item ≤ 3 rating, giải thích vì sao hit gần bằng 0 |
| `fig08_als_ablation.png` | NDCG@10 của 5 biến thể ALS; đường dọc là Popularity (Validation) | regParam không cứu explicit ALS; implicit tăng mạnh nhưng chưa vượt Popularity |
| `fig09_tuning_vs_seed_noise.png` | Trục phóng to; chấm là 4 trial tuning, vùng xám là dải ba seed của config được chọn | Chênh lệch giữa trial cùng bậc nhiễu seed: lựa chọn của T22 chưa đáng tin cậy |
| `fig10_recall_ceiling.png` | Thang log; thanh xám là trần reachable | Kết quả cách trần hơn 2 bậc độ lớn: không gian cải thiện còn rất lớn |

Các biểu đồ chi tiết (fig03–fig07) vẽ cho Test vì Test là kết quả cuối; số liệu Validation tương ứng nằm trong cùng file CSV (cột `phase`) và trong notebook.

## 12. Hạn chế

- Một cohort (5% user, seed 42), local[2]. Kết quả về thời gian chỉ mang tính tương đối.
- Bootstrap theo user giả định các user độc lập.
- K tối đa là 10 vì E5 chỉ lưu Top‑10; muốn K = 20 cần E5 sinh danh sách dài hơn.
- Category ILD dựa trên metadata chưa đầy đủ (3 file part của `metadata_clean`, 567.061 ASIN); nên chạy lại khi có đủ bản E3.
- Thời gian train trên lịch sử Test không được E5 đo riêng; chỉ có tổng thời gian lệnh `test-final`.
- Ablation implicit không được tune và không chạy trên Test theo đúng nguyên tắc không chọn mô hình bằng Test.

## 13. Danh mục và vị trí file

Nguyên tắc sắp xếp: **code ở `src/evaluation/`, kết quả ở `results/e6/`, tài liệu ở `docs/`**. Trong `results/e6/`, mỗi chủ đề là một file, `phase` là một cột, nên tìm số liệu chỉ cần biết chủ đề.

```text
src/evaluation/
  protocol.py              ground truth, mẫu số, trần reachable, kiểm tra toàn vẹn (T18)
  per_user_metrics.py      bảng metric theo user, pandas + Spark (lõi của T25/T26)
  recall_at_k.py           T25
  ndcg_at_k.py             T26 (binary + graded)
  coverage.py              T27 user coverage và fallback
  beyond_accuracy.py       catalog coverage, Gini, head share, novelty, personalization, ILD
  significance.py          bootstrap, kiểm định cặp sign-flip, Holm
  als_diagnostics.py       chẩn đoán ALS cần Spark: rating error, seed, ablation
  run_e6.py                CLI duy nhất của E6
  e6_figures.py            10 biểu đồ
  notebooks/E6_evaluation.ipynb   lớp trình bày (đã chạy, có output)
tests/
  test_e6_metrics.py       24 test tính tay
  test_e6_spark_parity.py  pandas ↔ Spark
results/e6/
  E6_evaluation_report.html  báo cáo một file, không cần Python
  model_comparison.csv       T28: một dòng cho mỗi phase × phương án
  metrics.csv                T25/T26 và metric phụ, mọi K, ngưỡng, phạm vi, kèm CI95
  significance.csv           kiểm định cặp
  coverage.csv               T27: user coverage theo nhóm user
  fallback.csv               T27: trước/sau fallback theo trạng thái ALS, nguồn danh sách, chất lượng
  cohorts.csv                metric theo cohort cold-start
  beyond_accuracy.csv        độ đa dạng và cá nhân hóa
  reliability_checks.csv     mọi kiểm tra độ tin cậy, PASS/FAIL
  protocol.json              mẫu số, trần reachable, mốc thời gian theo phase
  runtime.csv                thời gian đo được
  diagnostics/               als_rating_prediction, als_item_support, als_tuning,
                             seed_stability, als_ablation
  figures/                   fig01 ... fig10
outputs/e6/per_user/{phase}_{method}.parquet   bảng theo user (gitignored)
docs/E6_README.md, docs/model_findings.md
```

Các file khác trong `src/evaluation/` không thuộc E6 và được giữ nguyên vì các Epic khác đang import: `split_data.py` (E4, dùng bởi `prepare_e4.py`), `ranking_metrics.py` (E5, dùng bởi `run_e5.py`), `analyze_processed_data.py` và `audit_e5_input.py` (script kiểm tra của E5).
