from typing import Any

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_predict_endpoint() -> None:
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    result: dict[str, Any] = response.json()

    assert response.status_code == 200
    assert result["input_features"] == payload["features"]
    assert result["predicted_class_id"] == 0
    assert result["predicted_class_name"] == "Setosa"


def test_predict_rejects_wrong_feature_count() -> None:
    response = client.post("/predict", json={"features": [5.1, 3.5]})

    assert response.status_code == 400


def test_predict_rejects_missing_features() -> None:
    # Pydantic phải từ chối request không có trường features.
    response = client.post("/predict", json={})

    assert response.status_code == 422