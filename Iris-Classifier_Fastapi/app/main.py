
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
import joblib
from pathlib import Path
from typing import List

SPECIES = ["setosa", "versicolor", "virginica"]

class IrisRequest(BaseModel):
    # 4 measurements in cm: sepal length, sepal width, petal length, petal width
    measurements: conlist(float, min_items=4, max_items=4) = Field(
        ..., description="Order: sepal_length, sepal_width, petal_length, petal_width (cm)"
    )

class IrisBatchRequest(BaseModel):
    batch: List[conlist(float, min_items=4, max_items=4)]

class IrisResponse(BaseModel):
    species: str
    species_id: int
    probabilities: List[float] = Field(description="Class probabilities in order: setosa, versicolor, virginica")

# Create the FastAPI app
app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements using a trained Logistic Regression model.",
    version="1.0.0",
)

# Load model on startup
model_path = Path(__file__).resolve().parents[1] / "models" / "iris_model.joblib"
if not model_path.exists():
    raise RuntimeError(f"Model file not found at {model_path}. Please run `python train_model.py` first.")

model = joblib.load(model_path)

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "Iris API is running"}

@app.post("/predict", response_model=IrisResponse, tags=["inference"])
def predict(req: IrisRequest):
    try:
        import numpy as np
        X = np.array(req.measurements).reshape(1, -1)
        proba = model.predict_proba(X)[0].tolist()
        pred_id = int(model.predict(X)[0])
        return {
            "species": SPECIES[pred_id],
            "species_id": pred_id,
            "probabilities": proba
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", tags=["inference"])
def predict_batch(req: IrisBatchRequest):
    """Return predictions for a batch of rows."""
    try:
        import numpy as np
        X = np.array(req.batch)
        proba = model.predict_proba(X).tolist()
        preds = model.predict(X).tolist()
        return [
            {"species": SPECIES[int(pid)], "species_id": int(pid), "probabilities": p}
            for pid, p in zip(preds, proba)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
