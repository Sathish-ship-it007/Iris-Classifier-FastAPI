
import os
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_and_save():
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: Standardize -> Logistic Regression (multinomial)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, multi_class="multinomial", solver="lbfgs"))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, preds))

    outdir = Path("models")
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "iris_model.joblib")
    print("Model saved to", outdir / "iris_model.joblib")

if __name__ == "__main__":
    train_and_save()
