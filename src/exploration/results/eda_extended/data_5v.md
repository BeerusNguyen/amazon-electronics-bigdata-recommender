# Phân tích 5V dựa trên bằng chứng

| V | Bằng chứng | Giới hạn |
|---|---|---|
| Volume | 20,994,353 dòng đọc được; 20,994,353 hợp lệ; 876,247,561 bytes; 20,553,480 cặp duy nhất. | Quy mô file không tự chứng minh cần cluster; cần benchmark riêng. |
| Variety | CSV gồm ID chuỗi, rating số và Unix timestamp. | Một định dạng; không có JSON/metadata/text trong phạm vi hiện tại. |
| Velocity | Lịch sử 1997-12-04T00:00:00+00:00 → 2018-10-05T00:00:00+00:00; xem monthly_rating_detail.csv. | Chỉ event frequency lịch sử, không có thông tin tốc độ ingest hoặc SLA realtime. |
| Veracity | 0 dòng invalid; 426,262 trùng dư; kiểm tra quality_rates.csv. | Valid schema không đồng nghĩa nội dung đúng hoặc mẫu không thiên lệch. |
| Value | Bài toán Top-10; 64.35% user có một tương tác, top1% item chiếm 43.08% cặp. | Cần đánh giá ALS/Popularity bằng ranking metrics, coverage và xử lý cold-start; chưa chứng minh cải thiện kinh doanh. |

Input version: `workspace-data`. CSV SHA256: `1ce756643c0abed1b0d8a0738e67fc2510f3086d2bd52567942e18582dd01791`.
