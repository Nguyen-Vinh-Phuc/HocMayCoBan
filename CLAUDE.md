# HƯỚNG DẪN DỰ ÁN: DỰ ĐOÁN GIÁ XE MÁY CŨ VIỆT NAM (KNN REGRESSOR)

## 1. Thông tin dự án
- Bài toán: Hồi quy (Regression) dự đoán giá xe máy cũ (Đơn vị: Triệu VNĐ).
- Thư mục `colab/`: Chứa script sinh dữ liệu thực tế thị trường VN, huấn luyện KNN Regressor, xuất ảnh báo cáo.
- Thư mục `app/`: Triển khai FastAPI nhận input (năm đăng ký, số km, dung tích cc) và trả về giá dự đoán.
- Thư mục `tests/`: Kiểm thử tự động API bằng Pytest.

## 2. Quy chuẩn Code
- Dùng Python 3.10+, code sạch, comment tiếng Việt.
- Sử dụng StandardScaler chuẩn hóa đặc trưng trước khi đưa vào KNN.