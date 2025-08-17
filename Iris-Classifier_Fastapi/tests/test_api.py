import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_single():
    payload = {"measurements":[5.1,3.5,1.4,0.2]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "species" in data and "probabilities" in data

def test_predict_batch():
    payload = {"batch":[[5.1,3.5,1.4,0.2],[6.1,2.8,4.7,1.2]]}
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 2