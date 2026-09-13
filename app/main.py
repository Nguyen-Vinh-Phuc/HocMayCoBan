from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Motorbike Price Prediction API (Vietnam)",
    description="API Dự đoán giá bán xe máy cũ tại Việt Nam sử dụng thuật toán KNN Regressor",
    version="1.0.0"
)

class MotorbikeInput(BaseModel):
    year_bought: int = Field(
        ..., ge=1990, le=2026, description="Năm đăng ký xe"
    )
    mileage_km: float = Field(..., ge=0, description="Số km đã đi")
    engine_cc: int = Field(..., ge=50, le=2000, description="Dung tích xi-lanh")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "colab" / "knn_model.pkl"
SCALER_PATH = PROJECT_ROOT / "colab" / "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except (FileNotFoundError, OSError, ValueError) as e:
    print(f"Lỗi load model: {e}")
    model, scaler = None, None

@app.get("/")
def home():
    return {"status": "online", "message": "API Dự Đoán Giá Xe Máy Cũ Việt Nam đang hoạt động!"}

@app.post("/predict")
def predict_price(data: MotorbikeInput):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Mô hình chưa sẵn sàng. Hãy chạy colab/train_motorbike.py trước.",
        )

    features = pd.DataFrame(
        [[data.year_bought, data.mileage_km, data.engine_cc]],
        columns=["year_bought", "mileage_km", "engine_cc"],
    )
    features_scaled = scaler.transform(features)
    predicted_price = model.predict(features_scaled)[0]

    return {
        "input_specs": {
            "nam_dang_ky": data.year_bought,
            "so_km_da_di": data.mileage_km,
            "dung_tich_cc": data.engine_cc
        },
        "predicted_price_million_vnd": round(float(predicted_price), 2),
        "message": f"Giá dự đoán cho chiếc xe này là khoảng {round(float(predicted_price), 1)} triệu VNĐ"
    }