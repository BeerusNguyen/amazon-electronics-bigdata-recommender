# E5 — Popularity, Spark ALS và Cold-start Fallback

Tài liệu này mô tả cách chạy và tái lập Epic E5 của dự án **Amazon Electronics Big Data Recommender**.
E5 cung cấp ba phương án xếp hạng Top‑10:

1. **Popularity** làm baseline.
2. **Spark MLlib ALS explicit-feedback** làm mô hình cá nhân hóa.
3. **ALS + Popularity fallback** bổ sung gợi ý cho cold start, user chưa được phục vụ hoặc danh sách ALS ngắn.

E5 không thay đổi dữ liệu E4, không đọc lại ratings sạch của E3 để huấn luyện, không chạy Test
tự động trong quá trình tuning và không thêm metadata hay mô hình ML thứ ba.

---

## 1. Mục tiêu và phạm vi

E5 triển khai các task sau:

- **T20:** baseline Popularity và xếp hạng Top‑10.
- **T21:** huấn luyện Spark ALS explicit-feedback, lưu và nạp lại mô hình.
- **T23:** sinh candidate ALS, loại seen item và giữ thứ tự Top‑10.
- **T24:** fallback Popularity theo từng user và gắn provenance cho gợi ý.
- **T22:** tune ALS trên Validation, khóa config rồi mới đánh giá Test.

Các nhóm metric hỗ trợ E6 nằm trong `src/evaluation/ranking_metrics.py`.

---

## 2. Mapping với mã task và file source

| Thành phần | File |
| --- | --- |
| E5 CLI orchestration | `src/models/run_e5.py` |
| Popularity | `src/models/popularity.py` |
| ALS config, training và reload | `src/models/als.py` |
| Sinh candidate và fallback ranking | `src/models/candidates.py` |
| Popularity fallback cho Spark | `src/models/fallback.py` |
| Kiểm tra mapping E4 | `src/models/mappings.py` |
| Parameter grid và chọn trial | `src/models/tuning.py` |
| Recall/NDCG/coverage | `src/evaluation/ranking_metrics.py` |
| Cấu hình mặc định | `configs/e5.json` |
| Test fixture E5 | `tests/test_e5_models.py` |

CLI chính:

```text
python -m src.models.run_e5 ACTION [OPTIONS]
```

Các action:

| Action | Chức năng |
| --- | --- |
| `popularity` | Tính Popularity, sinh gợi ý baseline và metric |
| `train` | Huấn luyện ALS, lưu model và thông tin chạy vào console |
| `generate` | Nạp model ALS đã lưu, sinh ALS thuần và ALS + fallback |
| `run-all` | Chạy Popularity, train ALS, sinh ALS thuần và ALS + fallback trên Validation; in bảng so sánh |
| `tune` | Chạy grid trên Validation và chọn config |
| `test-final` | Refit Train + Validation bằng config đã khóa, rồi đánh giá Test |

---

## 3. Dữ liệu E4 bắt buộc

Chạy toàn bộ lệnh dưới đây từ thư mục gốc repository.

Mặc định, E5 đọc:

```text
data/processed/e4/
├── train.parquet
├── validation.parquet
├── test.parquet
├── split_manifest.json
├── split_statistics.csv
└── audit/
    ├── cold_start.csv
    ├── snapshot_manifest.json
    ├── validation/
    │   ├── user_id_map.parquet
    │   └── item_id_map.parquet
    └── test/
        ├── user_id_map.parquet
        └── item_id_map.parquet
```

### Giao thức snapshot

| Phase | History được phép dùng | Held-out |
| --- | --- | --- |
| `validation` | `train.parquet` | `validation.parquet` |
| `test` | `train.parquet` + `validation.parquet` | `test.parquet` |

Các mapping bắt buộc là mapping tương ứng với phase:

- Validation dùng `audit/validation/user_id_map.parquet` và `audit/validation/item_id_map.parquet`.
- Test dùng `audit/test/user_id_map.parquet` và `audit/test/item_id_map.parquet`.

Trước khi huấn luyện hoặc sinh gợi ý, E5 kiểm tra:

- Cột ID là string không rỗng.
- Cột index là int32, không null.
- `user_idx` và `item_idx` duy nhất.
- Index liên tục bắt đầu từ `0`.
- Mapping khớp chính xác danh sách user/item trong history.
- Không đổi ID hoặc mapping khi chạy giữa các trial.

Nếu thiếu file hoặc sai schema/mapping, chương trình dừng bằng `ValueError` hoặc `FileNotFoundError`,
không drop silent, không tiếp tục bằng mapping sai.

---

## 4. Quy tắc rating, seen và relevant

- Rating hợp lệ của E4 là số nguyên trong khoảng **1–5**.
- **ALS explicit-feedback** huấn luyện tất cả rating 1–5 trong history.
- **Relevant** trong held-out là `rating >= 4`.
- **Seen** gồm mọi sản phẩm user đã đánh giá trong history, kể cả rating 1–3.
- Catalog ứng viên là tập sản phẩm xuất hiện trong history của snapshot.
- Không lấy candidate hoặc catalog từ metadata E3.
- Không đưa item xuất hiện chỉ trong held-out vào catalog của snapshot.
- Cặp user–item trong held-out không làm thay đổi candidate catalog.

### Popularity

Popularity đếm số lần `rating >= 4` cho từng item trong history.

- Item có nhiều positive nhất xếp trước.
- Khi điểm bằng nhau, `product_id ASC`.
- Item có **0 positive** vẫn nằm trong catalog, được xếp sau các item có positive.
- Seen item bị loại cho từng user.

### ALS

ALS được cấu hình:

- `implicitPrefs=False`
- `ratingCol=rating`
- `coldStartStrategy=drop`
- `seed` cố định

Trong MLlib, `coldStartStrategy=drop` chỉ loại prediction không xác định trong generator.
Không được coi đó là fallback và không làm giảm mẫu số của metric Recall/NDCG.

---

## 5. Chuẩn bị môi trường

Môi trường đã xác nhận cho repo:

- Python 3.10
- JDK 17
- PySpark 3.5.3
- Pandas/PyArrow tương thích với môi trường E4

Kiểm tra:

```powershell
py -3.10 -c "import sys,pyspark,pandas,pyarrow; print(sys.version); print(pyspark.__version__, pandas.__version__, pyarrow.__version__)"
java -version
```

Nếu chưa có môi trường phù hợp, tạo từ thư mục gốc:

```powershell
py -3.10 -m venv .venv-e5
.\.venv-e5\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "pyspark==3.5.3" "pandas>=2.0,<3" "pyarrow>=15" "numpy>=1.24,<2" "pytest>=8"
```

Trên PowerShell, đặt các biến môi trường Spark:

```powershell
$env:SPARK_MASTER = "local[2]"
$env:SPARK_DRIVER_MEMORY = "3g"
$env:SPARK_LOCAL_IP = "127.0.0.1"
$python = (Get-Command python).Source
$env:PYSPARK_PYTHON = $python
$env:PYSPARK_DRIVER_PYTHON = $python
$env:SPARK_STORAGE_LEVEL = "MEMORY_AND_DISK"
```

Hãy kích hoạt virtual environment E5 trước khi chạy đoạn này. Dùng executable
đầy đủ của Python đang chạy lệnh để Spark không phải tự tìm `python3` trên Windows.

`SPARK_DRIVER_MEMORY` là cấu hình JVM, không phải peak memory đã đo.

Nếu Windows báo lỗi Hadoop native library hoặc Spark không ghi/đọc Parquet, chạy trong WSL2/Linux
bằng cùng Python 3.10 và JDK 17. Không cần đặt `HADOOP_HOME` trỏ tới thư mục không tồn tại.

---

## 6. Kiểm tra nhanh trước khi chạy mô hình

Chạy test fixture E5 và test E4:

```powershell
python -m pytest -q tests/test_e5_models.py tests/test_e4_audit.py
```

Các test chủ yếu kiểm tra:

- Đếm positive và giữ zero-positive catalog items.
- Lọc seen item và loại duplicate.
- Rank liên tục.
- Mixed ALS/Popularity fallback.
- Công thức manual Recall/NDCG.
- Mapping E4 hợp lệ và collision.
- Tuning grid và rule chọn config.

Kiểm tra compile:

```powershell
python -m compileall -q src tests
```

---

## 7. Cấu hình E5

Cấu hình mặc định tại `configs/e5.json`:

```json
{
  "top_k": 10,
  "candidate_pool": 100,
  "candidate_multiplier": 5,
  "relevance_threshold": 4,
  "als": {
    "rank": 16,
    "max_iter": 5,
    "reg_param": 0.1,
    "seed": 42,
    "implicit_prefs": false,
    "alpha": 1.0
  },
  "tuning_grid": {
    "rank": [8, 16],
    "max_iter": [5],
    "reg_param": [0.05, 0.1]
  }
}
```

Ý nghĩa:

- `top_k`: số gợi ý mục tiêu.
- `candidate_pool`: số Popularity item giới hạn dùng cho fallback.
- `candidate_multiplier`: số candidate ALS over-fetch theo công thức `Top-K × candidate_multiplier`.
- `relevance_threshold`: rating được xem là positive trong held-out.
- `tuning_grid`: các tổ hợp ALS được thử trên Validation.

Tạo cấu hình tùy chỉnh bằng file JSON mới và truyền qua `--config`.

Ví dụ:

```powershell
python -m src.models.run_e5 tune --config configs/e5.json --shuffle-partitions 8
```

---

## 8. Các tùy chọn CLI chung

```text
--phase {validation,test}
--data-dir ĐƯỜNG_DẪN_E4
--out ĐƯỜNG_DẪN_OUTPUT
--config ĐƯỜNG_DẪN_CONFIG
--shuffle-partitions SỐ_PARTITION
--run-id ID_TÙY_CHỌN
```

Mặc định:

```text
--phase validation
--data-dir data/processed/e4
--out outputs/e5
--config configs/e5.json
--shuffle-partitions 24
```

Ví dụ dùng pilot hoặc E4 ở vị trí khác:

```powershell
python -m src.models.run_e5 popularity `
  --phase validation `
  --data-dir data/subsets/e4-pilot `
  --out outputs/e5-pilot `
  --shuffle-partitions 8
```

Không ghi đè một thư mục model/recommendation đã chốt bằng config chưa kiểm chứng.
Chiến lược mặc định của implementation là ghi đè model khi action model được gọi;
nếu cần bảo toàn bằng chứng, truyền `--out` hoặc thư mục run riêng cho mỗi lần chạy.

### In Top‑10 recommendations ra console

Các action `popularity`, `generate`, `run-all` và `test-final` đều in bảng
Top‑10 cho từng user theo thứ tự `user_id`, `rank`, `product_id`, `score` và
`recommendation_source`.

Mỗi model được in riêng:

- `Popularity`
- `ALS`
- `ALS + Popularity`

Bảng này là bản xem nhanh của cùng dữ liệu recommendation được lưu trong
`outputs/e5/recommendations/`. Nó không thay thế hoặc ghi thêm dữ liệu.

---

## 9. Chạy Popularity baseline

Validation:

```powershell
python -m src.models.run_e5 popularity --phase validation
```

Test fallback benchmark có thể tính sau khi Test-final được chạy:

```powershell
python -m src.models.run_e5 popularity --phase test
```

Lưu ý: Popularity cho Test phải dùng **Train + Validation** làm history.
Test không được dùng trong tuning để chọn config.

Đầu ra:

```text
outputs/e5/popularity/validation/
outputs/e5/recommendations/popularity-validation/
outputs/e5/metrics/popularity-validation.json
```

Popularity table chứa:

- `product_id`
- `popularity_count`
- `score`

Recommendations chứa schema chuẩn:

```text
user_id, product_id, rank, score, item_source,
recommendation_source, phase
```

---

## 10. Huấn luyện ALS

Huấn luyện ALS trên history Validation:

```powershell
python -m src.models.run_e5 train --phase validation
```

Lưu ý: `train` không dùng held-out Validation để huấn luyện.
Hàm training chỉ nhận `train.parquet` làm history.

Model được lưu tại:

```text
outputs/e5/models/als-validation/
outputs/e5/models/als-validation.json
```

File JSON cạnh model là sidecar provenance, lưu `run_id`, phase, cấu hình ALS
và mapping paths đã dùng khi train. `generate` dùng thông tin sidecar để báo
cấu hình model thực tế thay vì lấy nhầm cấu hình ALS hiện tại.

ALS training dùng:

```text
userCol = user_idx
itemCol = item_idx
ratingCol = rating
implicitPrefs = false
coldStartStrategy = drop
seed = config ALS
```

Model reload được kiểm tra ở action `generate`.

---

## 11. Sinh recommendation bằng ALS và fallback

Sau khi `train --phase validation`, chạy:

```powershell
python -m src.models.run_e5 generate --phase validation
```

Action này:

1. Nạp model từ `outputs/e5/models/als-validation/`.
2. Lấy toàn bộ distinct user trong held-out Validation làm target.
3. Over-fetch candidate ALS theo `k × candidate_multiplier`.
4. Loại seen items bằng history.
5. Sắp xếp:
   - score ALS giảm dần;
   - `product_id ASC` nếu tie.
6. Tạo fallback từ Popularity.
7. Trả lại danh sách hợp nhất ALS trước, Popularity bám sau.
8. Lưu metric riêng cho ALS thuần và cho ALS + fallback.

Khi candidate ALS không đủ:

- User rank 1–n giữ thứ tự ALS.
- Popularity chỉ fill các vị trí còn thiếu.
- ALS duplicate và Popularity duplicate bị loại.
- Item đã seen không được bổ sung lại.
- Danh sách có thể ngắn hơn 10 nếu catalog unseen không còn đủ item.

Đầu ra:

```text
outputs/e5/recommendations/als-only-validation/
outputs/e5/metrics/als-only-validation.json
outputs/e5/recommendations/als-validation/
outputs/e5/metrics/als-validation.json
```

Nếu thiếu bảng Popularity khi chạy `generate`, chương trình tạo lại từ history
của snapshot rồi lưu vào `outputs/e5/popularity/validation/`.

---

## 11.1. Chạy toàn bộ ba phương án trên Validation

Để chạy Popularity baseline, train ALS một lần, rồi sinh và so sánh ALS thuần
cùng ALS + Popularity fallback, dùng:

```powershell
python -m src.models.run_e5 run-all `
  --phase validation `
  --data-dir data/processed/e4 `
  --out outputs/e5 `
  --config configs/e5.json `
  --shuffle-partitions 24
```

`run-all` chạy tuần tự:

1. Popularity trên Train, đánh giá trên Validation.
2. Huấn luyện ALS trên Train và lưu model.
3. Nạp model đã lưu, sinh ALS thuần và ALS + fallback, đánh giá cả hai trên cùng Validation users.
4. In kết luận dạng bảng với Recall@K, NDCG@K, coverage >=1, coverage đủ K và runtime các bước.
5. Ghi `outputs/e5/run-summary-validation.json`.

Lệnh này dùng config hiện tại, không tự tune. Kết quả là Validation; không chạy Test
và không gọi cấu hình mặc định là “best”. Muốn chọn tham số thì chạy `tune` riêng;
chỉ sau khi khóa lựa chọn Validation mới dùng lệnh `test-final`.

Các tệp metric và recommendation:

```text
outputs/e5/recommendations/popularity-validation/
outputs/e5/metrics/popularity-validation.json
outputs/e5/recommendations/als-only-validation/
outputs/e5/metrics/als-only-validation.json
outputs/e5/recommendations/als-validation/
outputs/e5/metrics/als-validation.json
outputs/e5/metrics/popularity-cohorts-validation.json
outputs/e5/metrics/als-only-cohorts-validation.json
outputs/e5/metrics/als-cohorts-validation.json
outputs/e5/run-summary-validation.json
```

Mỗi metric file lưu `run_id`, `phase`, model/model path, cấu hình ALS thực tế
(đọc từ sidecar cạnh model), mapping đã kiểm tra và chính sách fallback. Nếu
model cũ không có sidecar, cấu hình được ghi là chưa xác định thay vì suy đoán
từ file config hiện tại. Truyền `--run-id` để gắn một ID dễ nhận diện; mặc định
CLI tạo ID UTC duy nhất cho lần gọi.

Các file `*-cohorts-*.json` so sánh cùng phương án theo:

- `all_relevant_users`: mọi user có ít nhất một positive held-out.
- `warm_user_warm_item`: positive held-out của user và item đều có trong history.
- `cold_users`: user không có tương tác trong history.
- `history_1_4` và `history_ge_5`: số dòng tương tác của user trong history.

Recall/NDCG mỗi cohort dùng các relevant positives đúng với cohort đó; riêng
warm-user/warm-item chỉ tính positive thuộc item đã thấy trong history. Coverage
được tính trên những user có relevant positive trong cohort, không dùng phép
chia điểm tổng thể cho coverage ALS.

`runtime_seconds.generation_stages` đo các bước riêng bằng Spark action/materialization:
ALS candidate generation, seen filtering/ranking, Popularity fallback, final
assembly, metric computation và output writing. Thời gian từng bước là số đo
thực tế của lần chạy, không phải dự đoán; Spark startup/snapshot/model reload
vẫn nằm trong thời gian generation/evaluation tổng thể.

`run-all` cố ý chỉ nhận `--phase validation` để tránh chạy Test ngầm.
Đây là workload toàn bộ Validation; dùng E4 pilot và output path riêng trước
khi chạy dữ liệu/cohort lớn.

---

## 12. Ý nghĩa fallback

Fallback phân biệt user theo khả năng được phục vụ:

| Trạng thái | Định nghĩa |
| --- | --- |
| `Unknown user` | Không xuất hiện trong history |
| `Known / unserved` | Có history nhưng không nhận được candidate ALS |
| `Short list` | Nhận 1–9 gợi ý ALS |
| `Full list` | Nhận đủ 10 gợi ý ALS |

Nhãn provenance theo từng item:

- `item_source = ALS`
- `item_source = Popularity`

Nhãn provenance theo danh sách:

- `recommendation_source = ALS`
- `recommendation_source = Popularity`
- `recommendation_source = Mixed`

Nếu không có recommendation nào, user không được thêm vào bảng recommendation.
Người nhận 0 gợi ý có thể được kiểm tra bằng phép anti-join với target users;
CLI hiện chưa tạo riêng bảng user-level summary. E5 không pad bằng seen item
hoặc item ngoài catalog.

Không so sánh trực tiếp ALS prediction score với positive_count:

- ALS score là rating prediction.
- Popularity score là số positive rating.
- Hai thang đo khác nhau và không được sort chung bằng cách cộng/trừ.

---

## 13. Tune ALS trên Validation

Chạy grid mặc định:

```powershell
python -m src.models.run_e5 tune --phase validation
```

Lưu ý: `tune` chỉ nhận `--phase validation`.
Nếu truyền `--phase test`, chương trình dừng bằng `ValueError`.

Trong mỗi trial, tune giữ cố định:

- History = Train.
- Mapping Validation.
- Candidate catalog = item trong Train.
- Target user = user trong held-out Validation.
- `k = 10`.
- Toàn bộ user distinct trong held-out được sử dụng; chưa có `--user-limit`.
- Seen filter.
- Popularity fallback.

Grid mặc định tạo tối đa 4 trial:

| Trial | rank | maxIter | regParam |
| ---: | ---: | ---: | ---: |
| 1 | 8 | 5 | 0.05 |
| 2 | 8 | 5 | 0.10 |
| 3 | 16 | 5 | 0.05 |
| 4 | 16 | 5 | 0.10 |

Có thể giảm grid trong test:

```json
{
  "tuning_grid": {
    "rank": [8],
    "max_iter": [5],
    "reg_param": [0.1]
  }
}
```

Quy tắc chọn:

1. NDCG@10 cao nhất.
2. Nếu bằng nhau, Recall@10 cao nhất.
3. Nếu vẫn bằng nhau, giữ trial xuất hiện trước trong thứ tự input.

Quy tắc này được áp dụng trước khi xem kết quả Test.

Trial được ghi tại:

```text
outputs/e5/tuning/validation_results.json
```

Config trial được chọn và kết quả gắn:

```text
outputs/e5/tuning/best_validation.json
```

Sau mỗi trial có thể xem log:

```text
Validation trial 1/4: ...
```

Tune đồng thời huấn luyện lại config tốt nhất lên `Train` và lưu model tại:

```text
outputs/e5/models/als-validation/
```

Tune không tạo `configs/als_best.yaml`; implementation hiện lưu JSON provenance
tại `outputs/e5/tuning/best_validation.json`.

---

## 14. Chốt config và chạy Test-final

Chỉ chạy Test sau khi đã chọn config bằng Validation:

```powershell
python -m src.models.run_e5 test-final --phase test
```

Điều kiện bắt buộc:

```text
outputs/e5/tuning/best_validation.json
```

`test-final` thực hiện:

1. Đọc config đã chọn từ `best_validation.json`.
2. Dùng history **Train + Validation**.
3. Dùng mapping `audit/test/`.
4. Ghi nhận target user từ `test.parquet`.
5. Huấn luyện lại ALS.
6. Tính lại Popularity trên history Test.
7. Sinh ALS + fallback trên Test.
8. Tính metric một lần rồi ghi kết quả.

Sau bước này không đổi cấu hình dựa trên số liệu Test.

Đầu ra:

```text
outputs/e5/models/als-test-final/
outputs/e5/popularity/test/
outputs/e5/recommendations/als-test/
outputs/e5/metrics/als-test.json
```

Metric Test chứa thêm:

```text
selected_validation_parameters
```

để truy vết config Trial đã chọn trước khi xem Test.

---

## 15. Định nghĩa metric

### Recall@10

Với user `u`:

```text
Recall@10(u) = hits_in_top10(u) / relevant_items(u)
```

Macro average chỉ tính user có ít nhất một relevant item trong held-out.
User có relevant nhưng nhận 0 gợi ý được tính 0.
User không có relevant không vào mẫu số quality.

### NDCG@10

Tại vị trí i bắt đầu từ 1:

```text
DCG@10(u) = sum(rel_i / log2(i + 1))
IDCG@10(u) = sum(1 / log2(i + 1), i = 1..min(10, relevant_count))
NDCG@10(u) = DCG@10(u) / IDCG@10(u)
```

Không nén danh sách sau khi đối chiếu với relevant.
Không loại item non-relevant khỏi danh sách trước khi tính vị trí.

Ví dụ:

```text
relevant = {A, C}
recommendations = [X, A, B]

Recall = 1 / 2 = 0.5
DCG = 1 / log2(3)
IDCG = 1 + 1 / log2(3)
NDCG ≈ 0.3868528
```

### Coverage

Metric coverage tính trên toàn bộ target user:

- `user_coverage`: tỷ lệ user nhận ít nhất một gợi ý.
- `full_k_coverage`: tỷ lệ user nhận đủ 10 gợi ý.

User không có positive được giữ trong denominator coverage, dù không vào macro quality.

### Nhóm E4

Ranking metric nên được báo cáo riêng cho:

- `all`
- `warm_user/warm_item`
- `cold_user/warm_item`
- `warm_user/cold_item`
- `cold_user/cold_item`
- `history_0`
- `history_1-4`
- `history_5+`

Các group support không rời nhau hoàn toàn; một user có thể xuất hiện ở nhiều cohort pair.

Trong nhóm `all`, relevant cold-item vẫn tham gia Recall và NDCG dù item nằm ngoài candidate
catalog. Trong nhóm warm-item chỉ xét relevant warm-item của warm-user.

---

## 16. User, item và target user

Implementation hiện lấy target users từ distinct `user_id` của held-out:

- Validation: target là tất cả user xuất hiện trong `validation.parquet`.
- Test: target là tất cả user xuất hiện trong `test.parquet`.
- Pilot chưa giới hạn bằng tham số `--user-limit` trong CLI hiện tại.
- Nếu cần pilot, dùng một bộ E4 pilot do `src/splitting/run_e4.py` tạo ra.

Không dùng target user làm đồng nghĩa với toàn bộ catalog.
Không dùng rating held-out để chọn target user phục vụ tuning.

---

## 17. Popularity và candidate không cross toàn bộ user × catalog

`popularity_fallback` giới hạn:

- Popularity pool bởi `candidate_pool`.
- Cross join chỉ giữa target held-out và pool Popularity.
- Candidate ALS giới hạn theo `k × candidate_multiplier`.

Không materialize user × toàn bộ catalog.
Bộ nhớ vẫn phụ thuộc target user, pool và số partition, nên pilot nên chạy trước khi chạy cohort chính.

Các trade-off:

- Candidate pool nhỏ chạy nhanh nhưng có thể không đủ item unseen.
- Candidate pool lớn tăng khả năng fill 10 nhưng tăng shuffle và bộ nhớ.
- Candidate ALS over-fetch lớn cải thiện short-list nhưng không bảo đảm đủ 10.

---

## 18. Lưu và kiểm tra đầu ra

Các thư mục output thường nằm dưới `outputs/e5/`; thư mục `outputs/` đã bị `.gitignore`.

Cấu trúc mẫu:

```text
outputs/e5/
├── models/
│   ├── als-validation/
│   └── als-test-final/
├── popularity/
│   ├── validation/
│   └── test/
├── recommendations/
│   ├── popularity-validation/
│   ├── als-validation/
│   └── als-test/
├── metrics/
│   ├── popularity-validation.json
│   ├── als-validation.json
│   └── als-test.json
└── tuning/
    ├── validation_results.json
    └── best_validation.json
```

Kiểm tra recommendations bằng Pandas:

```python
import pandas as pd

recs = pd.read_parquet("outputs/e5/recommendations/als-validation")
print(recs.groupby("user_id").size().describe())
print(recs.groupby("recommendation_source").size())
print(recs.columns.tolist())
```

Kiểm tra duplicate và seen:

```python
import pandas as pd

recs = pd.read_parquet("outputs/e5/recommendations/als-validation")
train = pd.read_parquet("data/processed/e4/train.parquet")
seen = set(map(tuple, train[["user_id", "product_id"]].itertuples(index=False, name=None)))
pairs = set(map(tuple, recs[["user_id", "product_id"]].itertuples(index=False, name=None)))
print("duplicate_pairs =", recs.duplicated(["user_id", "product_id"]).sum())
print("seen_in_recommendations =", len(pairs & seen))
```

Kiểm tra ranks liên tục trong recommendation:

```python
import pandas as pd

recs = pd.read_parquet("outputs/e5/recommendations/als-validation")
bad = (
    recs.groupby("user_id")["rank"]
    .apply(lambda ranks: sorted(ranks.tolist()) != list(range(1, len(ranks) + 1)))
)
print("users_with_non_contiguous_ranks =", int(bad.sum()))
```

---

## 19. Chạy pilot trên 0,5% user

Tạo pilot E4 trước:

```powershell
python -m src.splitting.run_e4 prepare `
  --cohort-bps 50 `
  --out data/subsets/e4-pilot

python -m src.splitting.run_e4 audit --out data/subsets/e4-pilot
```

Sau đó chạy E5 trên pilot:

```powershell
$env:SPARK_MASTER = "local[2]"
$env:SPARK_DRIVER_MEMORY = "3g"
$env:SPARK_LOCAL_IP = "127.0.0.1"
$python = (Get-Command python).Source
$env:PYSPARK_PYTHON = $python
$env:PYSPARK_DRIVER_PYTHON = $python

$pilot = "data/subsets/e4-pilot"
if (-not (Test-Path "$pilot\train.parquet")) {
  if (Test-Path $pilot) {
    throw "Pilot folder exists but has no train.parquet. Use a new --out folder or inspect it before rerunning prepare."
  }
  if (-not (Test-Path "data/raw/Electronics.csv")) {
    throw "Missing data/raw/Electronics.csv; provide the E4 ratings CSV before preparing the pilot."
  }
  & $python -m src.splitting.run_e4 prepare --cohort-bps 50 --out $pilot
  if ($LASTEXITCODE -ne 0) { throw "E4 pilot preparation failed; stopping before E5." }
}
if (-not (Test-Path "$pilot\audit\validation\user_id_map.parquet")) {
  & $python -m src.splitting.run_e4 audit --out $pilot
  if ($LASTEXITCODE -ne 0) { throw "E4 pilot audit failed; stopping before E5." }
}

& $python -m src.models.run_e5 run-all `
  --phase validation `
  --data-dir $pilot `
  --out outputs/e5-pilot `
  --config configs/e5.json `
  --run-id "pilot-$((Get-Date).ToString('yyyyMMdd-HHmmss'))" `
  --shuffle-partitions 8
if ($LASTEXITCODE -ne 0) { throw "E5 pilot run failed." }
```

Không copy metric của pilot sang chạy chính.
Pilot chỉ dùng để kiểm tra pipeline, I/O, mapping, reload và tài nguyên local.

---

## 20. Pipeline Validation đầy đủ

Để chạy đủ ba phương án trên Validation trong một lệnh và in bảng so sánh cuối,
dùng `run-all`:

```powershell
python -m src.models.run_e5 run-all `
  --phase validation `
  --data-dir data/processed/e4 `
  --out outputs/e5 `
  --config configs/e5.json `
  --shuffle-partitions 24
```

Lệnh này train ALS theo config hiện tại, không chạy parameter tuning.
Nếu mục tiêu là chọn cấu hình, chạy `tune` riêng sau khi xác nhận kết quả trên pilot.

Hoặc chạy từng bước độc lập:

```powershell
$env:SPARK_MASTER = "local[2]"
$env:SPARK_DRIVER_MEMORY = "3g"
$env:SPARK_LOCAL_IP = "127.0.0.1"
$python = (Get-Command python).Source
$env:PYSPARK_PYTHON = $python
$env:PYSPARK_DRIVER_PYTHON = $python

& $python -m src.models.run_e5 popularity --phase validation
if ($LASTEXITCODE -ne 0) { throw "Popularity baseline failed." }
& $python -m src.models.run_e5 train --phase validation --shuffle-partitions 24
if ($LASTEXITCODE -ne 0) { throw "ALS training failed." }
& $python -m src.models.run_e5 generate --phase validation --shuffle-partitions 24
if ($LASTEXITCODE -ne 0) { throw "ALS recommendation generation failed." }
```

Kết quả kỳ vọng:

```text
Popularity metrics
ALS + fallback metrics
validation_results.json
best_validation.json
ALS model nạp lại thành công
Recommendations và coverage
```

Đừng xem metric pilot hoặc fixture là metric thực của cohort chính.

---

## 21. Pipeline Test-final đầy đủ

Chỉ chạy sau khi:

1. Validation tuning đã xong.
2. `best_validation.json` tồn tại.
3. Mapping Test đã được kiểm tra.
4. Đã ghi rõ config Validation được chọn.

```powershell
python -m src.models.run_e5 test-final --phase test --shuffle-partitions 24
```

Trong bước này:

- Train history = `train.parquet + validation.parquet`.
- Held-out = `test.parquet`.
- Mapping = `audit/test/`.
- Không tự tune lại.
- Không sửa config dựa trên metric Test.

---

## 22. Khôi phục hoặc nạp model bằng Python

Trong PowerShell, dùng `python -c`:

```powershell
python -c "from src.models.als import load_als; m=load_als('outputs/e5/models/als-validation'); print(m.rank)"
```

Nếu model không tải được:

- Kiểm tra thư mục có file Spark model.
- Kiểm tra cùng phiên bản PySpark khi lưu và nạp.
- Kiểm tra log lỗi JVM.
- Không gọi hành động save thành công nếu reload thất bại.

---

## 23. Troubleshooting

### `pyspark` hoặc package bị thiếu

```powershell
py -3.10 -m pip install "pyspark==3.5.3" "pandas>=2.0,<3" "pyarrow>=15"
```

Không cài đè PySpark 4.0.5 nếu mô hình và code đang kiểm chứng với 3.5.3.

### `ModuleNotFoundError`

Chạy từ thư mục gốc:

```powershell
Set-Location D:\msis406\amazon-electronics-bigdata-recommender
python -m src.models.run_e5 popularity --phase validation
```

Nếu đang dùng venv:

```powershell
.\.venv-e5\Scripts\Activate.ps1
python -m pip show pyspark pandas pyarrow
```

### `JAVA_HOME` hoặc JVM

```powershell
java -version
$env:JAVA_HOME = "C:\Đường-dẫn-JDK17"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
java -version
```

Chỉ đặt `JAVA_HOME` tới thư mục JDK thực tế.

### OutOfMemory hoặc JVM exited

Giảm tài nguyên:

```powershell
$env:SPARK_MASTER = "local[2]"
$env:SPARK_DRIVER_MEMORY = "2g"
$env:SPARK_STORAGE_LEVEL = "DISK_ONLY"
```

Chạy pilot:

```powershell
python -m src.models.run_e5 tune `
  --data-dir data/subsets/e4-pilot `
  --out outputs/e5-pilot
```

Giảm `rank`, `maxIter`, `candidate_pool`, hoặc số partition phù hợp.

### Winutils/Hadoop native lỗi

Không bịa success bằng fallback Pandas cho Spark model.
Đối chiếu:

- Lỗi ghi bảng Parquet không nhất thiết là lỗi save ALS model.
- Lỗi Spark model có thể làm hỏng JVM hoặc persistence.
- Chạy WSL2/Linux nếu native I/O không hoạt động trên Windows.

### `Missing required E4 data`

Kiểm tra:

```powershell
Test-Path data/processed/e4/train.parquet
Test-Path data/processed/e4/validation.parquet
Test-Path data/processed/e4/audit/validation/user_id_map.parquet
Test-Path data/processed/e4/audit/validation/item_id_map.parquet
```

Nếu file thiếu, chạy lại E4:

```powershell
python -m src.splitting.run_e4 prepare --cohort-bps 500
python -m src.splitting.run_e4 audit
```

Không chạy E5 bằng Parquet rating clean hoặc enriched thay thế.

### Model thiếu nhưng `generate` chạy

Chạy lại `train` trước:

```powershell
python -m src.models.run_e5 train --phase validation
python -m src.models.run_e5 generate --phase validation
```

Hoặc chạy tune:

```powershell
python -m src.models.run_e5 tune --phase validation
```

---

## 24. Kết quả nên xuất trình cho Final Report

Không ghi “đã train” hoặc “đã có best” trừ khi đã đo bằng data thực.

Nên lưu bảng:

| Method | Phase | Recall@10 | NDCG@10 | Coverage >=1 | Coverage >=10 | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | Validation | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| ALS | Validation | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| ALS + fallback | Validation | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

Đối chiếu cùng:

- Target user.
- History.
- Candidate catalog.
- Seen policy.
- `k=10`.
- Relevance threshold.
- Seed.
- Mapping.

Bổ sung:

- Count target user, evaluation user và relevant user.
- User/items trong catalog.
- Model size và file size.
- Runtime preparation, fit, save, reload và generate.
- Cấu hình Spark master, memory, shuffle partition.
- RAM thực tế chỉ khi có công cụ sampling đo process tree.
- Density:

```text
density = R / (U × I)
```

- Ước lượng:

```text
dense float32 matrix ≈ 4 × U × I bytes
ALS factor arrays ≈ 4 × k × (U + I) bytes
```

Hai công thức trên chưa tính overhead, shuffle, cache hoặc overhead JVM.

---

## 25. Hạn chế hiện tại

Các giới hạn cần ghi rõ:

- E5 chưa thực sự train trên toàn bộ cohort chính trong quá trình triển khai.
- Không chạy benchmark cluster hoặc GCP.
- Không triển khai API/UI/cloud.
- Không triển khai RMSE thay thế ranking metric.
- Không sử dụng metadata để cold-item hoặc content-based recommendation.
- ALS tie candidate ở ranh giới catalog không được coi hoàn toàn deterministic trên mọi runtime.
- Fixed seed chỉ giúp kiểm soát input/partition; Spark output có thể đổi giữa runtime.
- Candidate pool 100 có thể làm danh sách dưới 10.
- Fallback không gợi ý sản phẩm chưa từng xuất hiện trong history snapshot.
- Cấu hình mặc định chưa được gọi là best nếu chưa có validation metric thực.
- CLI hiện chưa có tham số giới hạn target user như `--user-limit`; dùng E4 pilot để kiểm soát scope.
- File mapping và model cần đủ provenance để tránh load nhầm phase.

---

## 26. Checklist bàn giao

- [ ] Đã xác nhận Python 3.10, JDK 17 và PySpark 3.5.3.
- [ ] Đã đọc ba split và bảng mapping Validation/Test.
- [ ] Đã chạy test fixture E5 và E4.
- [ ] Đã chạy baseline Popularity trên pilot.
- [ ] Đã chạy train ALS Validation.
- [ ] Đã reload model và kiểm tra generation.
- [ ] Đã chạy tune Validation.
- [ ] Đã lưu `validation_results.json` và `best_validation.json`.
- [ ] Đã kiểm tra seen/duplicate/rank/output schema.
- [ ] Đã tính được Recall/NDCG/coverage.
- [ ] Chưa đọc hoặc chọn config từ Test trước bước final.
- [ ] Đã chạy `test-final` sau khi khóa config.
- [ ] Đã ghi rõ pilot/full, runtime và thiết bị/tài nguyên.
- [ ] Đã phân biệt dữ liệu thực đo và tài liệu/ước tính.

## 27. Official API references

Khi báo cáo API thực dùng, tham khảo tài liệu chính thức:

- Spark MLlib ALS:
  `https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALS.html`
- Spark ALSModel:
  `https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALSModel.html`
- Spark DataFrame:
  `https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql.DataFrame.html`
- Pandas Parquet:
  `https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html`
