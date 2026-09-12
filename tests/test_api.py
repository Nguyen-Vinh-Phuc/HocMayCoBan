from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict_endpoint():
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_class_name"] == "Setosa"


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [5.1, 3.5]})
    assert response.status_code == 400