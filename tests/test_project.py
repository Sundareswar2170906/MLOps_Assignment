
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from joblib import dump
from sklearn.linear_model import LogisticRegression

from src.data import load_and_clean_data
from src.train import build_pipeline
import src.api as api


def test_dataset_has_binary_target():
    df = load_and_clean_data()

    assert not df.empty
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})


def test_pipeline_fits_and_predicts():
    # Synthetic data for a quick, deterministic unit test.
    rows = []

    for label in [0, 1]:
        for i in range(12):
            rows.append({
                "age": 35 + i + label * 10,
                "sex": i % 2,
                "cp": 1 + (i % 4),
                "trestbps": 110 + i,
                "chol": 180 + i * 3,
                "fbs": i % 2,
                "restecg": i % 3,
                "thalach": 170 - i,
                "exang": i % 2,
                "oldpeak": float(i % 4),
                "slope": 1 + (i % 3),
                "ca": float(i % 4),
                "thal": float(3 + (i % 3)),
            })

    X = pd.DataFrame(rows)
    y = [0] * 12 + [1] * 12

    pipeline = build_pipeline(
        LogisticRegression(max_iter=2000)
    )
    pipeline.fit(X, y)

    assert len(pipeline.predict(X)) == len(X)
    assert pipeline.predict_proba(X).shape == (len(X), 2)


def test_model_can_be_saved_and_loaded(tmp_path):
    df = load_and_clean_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = build_pipeline(
        LogisticRegression(max_iter=2000)
    )
    pipeline.fit(X, y)

    path = tmp_path / "model.joblib"
    dump(pipeline, path)

    assert path.exists()


def test_api_health():
    api.model = object()

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_api_prediction():
    df = load_and_clean_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    api.model = build_pipeline(
        LogisticRegression(max_iter=2000)
    ).fit(X, y)

    api.metadata = {"selected_model": "test_model"}

    payload = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }

    with TestClient(api.app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability_class_1"] <= 1
