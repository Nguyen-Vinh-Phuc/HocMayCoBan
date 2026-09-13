from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_predict_endpoint() -> None:
    payload = {"year_bought": 2020, "mileage_km": 25000, "engine_cc": 125}
    response = client.post("/predict", json=payload)
    result = response.json()

    assert response.status_code == 200
    assert result["input_specs"] == {
        "nam_dang_ky": 2020,
        "so_km_da_di": 25000,
        "dung_tich_cc": 125,
    }
    assert isinstance(result["predicted_price_million_vnd"], float)
    assert result["predicted_price_million_vnd"] > 0


def test_predict_rejects_wrong_feature_count() -> None:
    response = client.post(
        "/predict",
        json={"year_bought": 2020, "mileage_km": -1, "engine_cc": 125},
    )

    assert response.status_code == 422


def test_predict_rejects_missing_features() -> None:
    # Pydantic phải từ chối request không có trường features.
    response = client.post("/predict", json={})

    assert response.status_code == 422