# Iris Classifier FastAPI Service

Predict Iris species from sepal/petal measurements via a REST API.

## Tech
- **Model**: Logistic Regression (scikit-learn) in a Pipeline with StandardScaler
- **API**: FastAPI + Uvicorn
- **Packaging**: requirements.txt / Docker
- **Testing**: pytest + httpx TestClient

## Setup (Local, no Docker)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/docs
```

## Example Requests

### cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json"   -d '{"measurements":[5.1,3.5,1.4,0.2]}'
```

### HTTPie
```bash
http POST :8000/predict measurements:='[6.1,2.8,4.7,1.2]'
```

### Python client
```python
import requests
url = "http://127.0.0.1:8000/predict"
payload = {"measurements":[6.3,3.3,6.0,2.5]}
print(requests.post(url, json=payload).json())
```

### Batch
```bash
curl -X POST "http://127.0.0.1:8000/predict_batch" -H "Content-Type: application/json"   -d '{"batch":[[5.1,3.5,1.4,0.2],[6.1,2.8,4.7,1.2],[6.3,3.3,6.0,2.5]]}'
```

## Docker
```bash
docker build -t iris-fastapi .
docker run -p 8000:8000 iris-fastapi
```

## Testing
```bash
pytest -q
```

## Math (Logistic Regression, Multinomial)
We model class probability with softmax over linear scores:
\[ p(y=k \mid x) = \frac{\exp(w_k^T x + b_k)}{\sum_j \exp(w_j^T x + b_j)} \]
The model is trained by minimizing cross-entropy loss with L2 regularization using LBFGS.
A **StandardScaler** ensures features are zero-mean/unit-variance.

## Production Tips
- Pin versions in `requirements.txt`; build reproducible Docker images.
- Validate/parse requests with **Pydantic**; return 4xx on user errors.
- Add request logging, structured logs (JSON), and correlation IDs.
- Health endpoints (`GET /`) and readiness checks for orchestration.
- Use CI to run tests, mypy, ruff/flake8.
- Containerize; scale via replicas; use a process manager or uvicorn workers.
- Monitor with Prometheus + Grafana; add tracing (OpenTelemetry).
- Consider model versioning (MLflow) and feature stores.
- Rate limiting and auth (API keys/JWT) if public.
```