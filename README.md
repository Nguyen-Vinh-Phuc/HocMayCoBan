# Dự Đoán Giá Xe Máy Cũ Việt Nam

## 📌 Giới Thiệu

Dự án huấn luyện mô hình **KNN Regressor** để dự đoán giá xe máy cũ theo năm đăng ký,
số km đã đi và dung tích xi-lanh. Giá trả về được tính theo triệu VNĐ.

## 📁 Cấu Trúc Dự Án

- `app/`: API FastAPI nhận dữ liệu xe máy và trả về giá dự đoán.
- `colab/train_motorbike.py`: Sinh dữ liệu mẫu, huấn luyện và lưu model/scaler.
- `tests/`: Kiểm thử API bằng Pytest.

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Khởi tạo môi trường

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# Cài đặt thư viện:
pip install -r requirements.txt
```

### 2. Huấn luyện model

```bash
python colab/train_motorbike.py
```

Lệnh này tạo model, scaler và hai biểu đồ trong `colab/`.

### 3. Chạy API

```bash
uvicorn app.main:app --reload
```

Mở `http://127.0.0.1:8000/docs` để thử API. Endpoint `POST /predict` nhận:

```json
{ "year_bought": 2020, "mileage_km": 25000, "engine_cc": 125 }
```

### 4. Chạy kiểm thử

```bash
pytest -q
```

### Chạy bằng Docker

```bash
docker compose up --build
```
