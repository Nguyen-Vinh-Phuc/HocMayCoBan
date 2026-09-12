from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title="KNN Model Service API",
    description="API dự đoán cho mô hình K-Nearest Neighbors",
    version="1.0.0"
)

# Khai báo cấu trúc dữ liệu đầu vào (Ví dụ: Iris Dataset - 4 đặc trưng)
class PredictionInput(BaseModel):
    features: list[float] = Field(
        ..., 
        example=[5.1, 3.5, 1.4, 0.2],
        description="Danh sách các đặc trưng (Sepal Length, Sepal Width, Petal Length, Petal Width)"
    )


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "colab" / "knn_model.pkl"
    scaler_path = Path(__file__).resolve().parent.parent / "colab" / "scaler.pkl"

    if model_path.exists() and scaler_path.exists():
        return joblib.load(model_path), joblib.load(scaler_path)

    iris = load_iris()
    scaler = StandardScaler()
    features = scaler.fit_transform(iris.data)
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    model.fit(features, iris.target)
    return model, scaler


model, scaler = load_model()
class_names = ["Setosa", "Versicolor", "Virginica"]

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "KNN Machine Learning API đang hoạt động bình thường!"
    }

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        if len(data.features) != 4:
            raise HTTPException(status_code=400, detail="Đầu vào phải chứa đúng 4 đặc trưng!")

        input_array = np.array(data.features, dtype=float).reshape(1, -1)
        pred_class = int(model.predict(scaler.transform(input_array))[0])

        return {
            "input_features": data.features,
            "predicted_class_id": pred_class,
            "predicted_class_name": class_names[pred_class]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))