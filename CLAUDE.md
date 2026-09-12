# LUẬT VÀ LỆNH ƯU TIÊN CHO CLAUDE AI (PROJECT: HOCMAYCOBAN)

## 1. TÔN CHỈ HOẠT ĐỘNG (Rules)

- Luôn đọc file này trước khi thực hiện bất kỳ yêu cầu nào.
- Code phải viết bằng Python 3.10+, sạch sẽ, có Type Hints và Comment bằng Tiếng Việt.
- BẮT BUỘC thực hiện công việc TỪNG BƯỚC MỘT theo chỉ định của người dùng, không tự ý làm trước các bước sau.

## 2. QUY TRÌNH PHÁT TRIỂN (Làm từng bước)

- **Bước 1**: Viết code huấn luyện mô hình KNN và Lưu Model (`colab/train_knn.py`).
- **Bước 2**: Viết code API Server FastAPI (`app/main.py`).
- **Bước 3**: Viết code kiểm thử tự động Pytest (`tests/test_api.py`).

## 3. THÔNG TIN KỸ THUẬT

- Thuật toán sử dụng: K-Nearest Neighbors (KNN).
- Thư viện chính: scikit-learn, numpy, fastapi, uvicorn, pytest.
- Thư mục lưu Model: `colab/knn_model.pkl` và `colab/scaler.pkl`.
