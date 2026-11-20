# EDA mở rộng — Amazon Electronics

- Tập có 20,994,353 rating hợp lệ, 9,838,676 user và 756,489 item; độ thưa 99.99972385%.
- 75.78% rating nằm ở mức 4–5 sao. Cần xem phân bố rating khi chọn baseline và đánh giá; đây không phải phân tích sentiment.
- 64.35% user chỉ có một item duy nhất; cần báo cáo độ phủ dự đoán và trường hợp cold-start khi chia theo thời gian.
- Top 1% item (làm tròn lên 7,565 item) chiếm 43.08% các cặp tương tác duy nhất; cần đối chiếu ALS với Popularity và báo cáo coverage.
- Nếu chỉ xét user có ít nhất 5 item, còn 8.01% user và 35.80% cặp. Đây là số liệu mô tả, chưa áp dụng lọc.
- Tháng có nhiều rating nhất trong dữ liệu quan sát là 2016-01 (427,413); không suy ra nguyên nhân hoặc tăng trưởng ngoài phạm vi dataset.
- Có 0 dòng vi phạm quy tắc hợp lệ và 426,262 dòng trùng dư sau chuẩn hóa; số 0 không chứng minh không có spam/review giả.

## Giới hạn

- Metadata sản phẩm được phân tích riêng ở phần metadata deep EDA; dataset không có nội dung review để phân tích sentiment.
- Dữ liệu lịch sử; tần suất event không phải tốc độ ingest hoặc realtime.
- User/item ID là định danh trong dataset, không chứng minh đại diện toàn bộ khách hàng/sản phẩm Amazon.
- Năm/tháng biên có thể chưa đủ kỳ; không so trực tiếp tổng kỳ thiếu với kỳ đầy đủ để kết luận tăng trưởng.
- Đây là EDA toàn tập; quy tắc lọc, mapping và lựa chọn tham số mô hình phải được học từ train.
- Không dựng ma trận user-item dạng dense; dùng biểu diễn tương tác thưa cho recommender.
- Cell mới chưa chạy trên CSV đầy đủ tại môi trường biên tập; output mới sẽ sinh khi bạn chạy notebook.