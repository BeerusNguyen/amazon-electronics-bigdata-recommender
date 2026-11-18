# E2 — Báo cáo Exploratory Data Analysis và phân tích 5V

## 1. Mục tiêu

E2 thực hiện exploratory data analysis (EDA) trên bộ dữ liệu Amazon Electronics để:

- kiểm tra cấu trúc và chất lượng dữ liệu ratings;
- thống kê user, item, rating, timestamp và mức độ trùng lặp;
- phân tích độ thưa, mức độ tập trung tương tác và ngưỡng hỗ trợ cho bài toán recommender;
- kiểm tra độ đầy đủ của metadata và khả năng nối metadata với ratings;
- tổng hợp bằng chứng cho năm chiều **5V: Volume, Variety, Velocity, Veracity, Value**.

EDA chỉ mô tả dữ liệu hiện có. E2 chưa huấn luyện hoặc đánh giá mô hình Popularity/ALS và chưa chứng minh chất lượng kinh doanh của hệ thống recommendation.

## 2. Dữ liệu đầu vào

| Nguồn | Nội dung |
|---|---|
| `data/raw/Electronics.csv` | Ratings gồm `user_id`, `item_id`, `rating`, `timestamp` Unix |
| `data/raw/meta_Electronics.json` hoặc `.gz` | Metadata sản phẩm: title, brand, price, category và các trường tùy chọn |

Khóa nối giữa hai nguồn là `ratings.item_id = metadata.asin`. File ratings cục bộ có kích thước **876,247,561 bytes**, SHA-256:

`1ce756643c0abed1b0d8a0738e67fc2510f3086d2bd52567942e18582dd01791`

## 3. Setup môi trường

Môi trường sử dụng:

- Python 3.9–3.12;
- Java 17;
- Jupyter Notebook hoặc VS Code với Python/Jupyter kernel;
- Apache Spark chạy local.

Có thể tạo môi trường và cài dependencies bằng PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
java -version
```

Notebook được chạy từ thư mục gốc repository:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\data_exploration.ipynb
```

Lần chạy được dùng để tạo các kết quả trong báo cáo sử dụng **PySpark 3.5.3**, `local[2]` và `shuffle_partitions=8`. Khi chạy lại, cần bảo đảm phiên bản PySpark tương thích với JDK và Spark runtime đang cài đặt.

## 4. Quy trình thực hiện

1. Đọc ratings bằng Spark với schema rõ ràng, không dùng Pandas để đọc toàn bộ CSV.
2. Kiểm tra dòng malformed, giá trị thiếu, rating ngoài miền hợp lệ và timestamp không hợp lệ.
3. Tính tổng số dòng, số user/item, số cặp user-item duy nhất, duplicate và sparsity.
4. Phân tích phân bố rating, hoạt động theo tháng/năm và số lượng tương tác trên mỗi user/item.
5. Tính concentration của nhóm user/item có nhiều tương tác và các ngưỡng lọc hỗ trợ.
6. Đọc metadata JSON Lines, đo mức độ đầy đủ của các trường và phân tích cấu trúc metadata.
7. Nối item trong ratings với metadata để đo coverage.
8. Sinh các bảng CSV, JSON, Markdown và hình ảnh trong `outputs/reports/Electronics/`.
9. Kiểm tra sanity checks để bảo đảm các tổng và histogram nhất quán.

## 5. Kết quả chạy

### 5.1. Quy mô và chất lượng ratings

| Chỉ số | Kết quả |
|---|---:|
| Tổng dòng đã đọc | 20,994,353 |
| Dòng hợp lệ | 20,994,353 |
| Dòng không hợp lệ | 0 |
| Số user | 9,838,676 |
| Số item | 756,489 |
| Cặp user-item duy nhất | 20,553,480 |
| Duplicate dư chính xác | 426,262 |
| Dư cặp user-item lặp lại | 440,873 |
| Sparsity | 99.99972385% |
| Rating trung bình | 4.0737 |
| Khoảng thời gian UTC | 1997-12-04 đến 2018-10-05 |

Phân bố rating:

- 1 sao: 11.51%;
- 2 sao: 5.43%;
- 3 sao: 7.29%;
- 4 sao: 15.75%;
- 5 sao: 60.03%.

Do đó, rating 4–5 sao chiếm khoảng **75.78%**. Đây là phân bố rating, không phải phân tích sentiment.

### 5.2. Hành vi tương tác và độ thưa

- 6,330,952 user chỉ có một item duy nhất, tương đương khoảng **64.35%** số user.
- Nếu giữ user có ít nhất 5 tương tác, còn 788,140 user và 7,358,908 cặp duy nhất, tương ứng **8.01% user** và **35.80% cặp**.
- Top 1% item, tương đương 7,565 item, chiếm khoảng **43.08%** các cặp tương tác duy nhất.
- Top 1% user chiếm khoảng **11.65%** các cặp tương tác duy nhất.

Các số liệu trên cho thấy dữ liệu có long-tail mạnh và cold-start đáng kể. Những ngưỡng này là thống kê mô tả; E2 chưa áp dụng chúng để tạo tập train/test.

### 5.3. Metadata và coverage

Metadata có 756,077 item. Trong số đó:

- 756,063 item có title;
- 750,731 item có brand;
- 300,600 item có price;
- 751,940 item có category chính;
- 644,664 item có description;
- 623,386 item có feature.

Trong 756,489 item xuất hiện ở ratings, có 756,061 item match metadata, đạt tỷ lệ **99.9434%**. Metadata phù hợp để enrich kết quả recommendation, nhưng không phải mọi trường đều đầy đủ; đặc biệt price chỉ hiện diện ở khoảng 39.76% metadata item.

### 5.4. Thời gian xử lý

Lần chạy full-data được ghi nhận khoảng **1,501 giây**. Thời gian này bao gồm Spark startup, đọc dữ liệu, EDA, kiểm tra, hash cục bộ và tạo biểu đồ; không bao gồm toàn bộ thời gian ghi report và shutdown Spark.

Biểu đồ và bảng hoạt động theo tháng/năm cho thấy đây là dữ liệu lịch sử. Timestamp phản ánh thời điểm event trong dataset, không phản ánh tốc độ ingest, throughput hoặc SLA realtime.

## 6. Phân tích 5V

| V | Phân tích và bằng chứng | Giới hạn |
|---|---|---|
| **Volume** | 20.99 triệu dòng, 9.84 triệu user, 756 nghìn item và file ratings khoảng 876 MB. | Quy mô file chưa tự chứng minh cần cluster; cần benchmark CSV/Parquet và workload thực tế. |
| **Variety** | Ratings có ID chuỗi, rating số và Unix timestamp; metadata JSON bổ sung title, brand, price, category, feature, image và quan hệ sản phẩm. | Dữ liệu hiện tại chủ yếu là CSV + JSONL; chưa có review text đầy đủ để phân tích sentiment. |
| **Velocity** | Có lịch sử event từ 1997 đến 2018 và các báo cáo monthly/yearly activity. | Không có tốc độ ingest, dữ liệu streaming hay yêu cầu realtime nên chưa kết luận được velocity hệ thống. |
| **Veracity** | 0 dòng invalid theo schema/rule đã kiểm tra; duplicate dư 426,262; các sanity checks đều đạt. | Schema hợp lệ không đảm bảo review không spam, không đảm bảo dữ liệu đại diện và duplicate chưa phải bằng chứng dữ liệu sai. |
| **Value** | Dữ liệu phù hợp để thử Popularity và ALS Top-10; metadata có coverage item 99.9434%; có thể phân tích coverage và cold-start. | E2 chưa có Recall@10, NDCG@10, coverage mô hình hay chỉ số kinh doanh nên chưa chứng minh giá trị recommendation. |

## 7. Các output đã tạo

Các kết quả chính nằm trong `outputs/reports/Electronics/`:

- `eda_summary.md`: tóm tắt EDA và method notes;
- `ratings_profile.json`: schema, quality, cardinality, duplicate, sparsity, rating và time statistics;
- `rating_distribution.csv`, `monthly_activity.csv`, `yearly_activity.csv`;
- `user_interaction_hist.csv`, `item_interaction_hist.csv`;
- `top20_users.csv`, `top20_items.csv`;
- `eda_extended/data_5v.md`: bảng bằng chứng 5V;
- `eda_extended/eda_insights.md`: các insight và giới hạn;
- `eda_extended/quality_rates.csv`, `concentration.csv`, `support_thresholds.csv`;
- `eda_extended/figures/`: dashboard, long-tail concentration và monthly rating trends;
- `metadata_deep_eda/`: completeness, category/brand concentration, price và các phân tích metadata chi tiết.

## 8. Kết luận

E2 đã hoàn thành EDA full-data và tạo được bằng chứng định lượng cho 5V. Dataset có volume lớn, cực kỳ sparse, phân bố tương tác lệch về một số item và số user một-tương-tác cao. Chất lượng schema tốt theo các rule đã kiểm tra, trong khi metadata có coverage nối rất cao nhưng completeness giữa các trường không đồng đều.

Về hướng triển khai tiếp theo, cần dùng split theo thời gian và học mọi quy tắc lọc từ tập train; sau đó so sánh Popularity với ALS bằng Recall@10, NDCG@10, coverage và đánh giá cold-start. Các kết luận về velocity, tính đại diện, spam và giá trị kinh doanh cần dữ liệu hoặc thí nghiệm bổ sung, không thể suy ra chỉ từ EDA này.
