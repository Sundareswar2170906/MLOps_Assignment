
from pathlib import Path
import json
import logging
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from joblib import load
from prometheus_client import Counter, Histogram, make_asgi_app

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Educational demonstration, not a clinical diagnostic tool.",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_http_requests",
    "Total HTTP requests, excluding the metrics endpoint",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTION_COUNT = Counter(
    "api_predictions",
    "Number of successful predictions",
)

PREDICTION_ERRORS = Counter(
    "api_prediction_errors",
    "Number of prediction errors",
)

model = None
metadata = {}

# Prometheus collects metrics from this endpoint.
app.mount("/metrics", make_asgi_app())


class PatientFeatures(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-10, le=20)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=0, le=7)


@app.on_event("startup")
def load_model():
    global model, metadata

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run python -m src.train first."
        )

    model = load(MODEL_PATH)

    if METADATA_PATH.exists():
        metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

    logger.info(
        "Model loaded: %s",
        metadata.get("selected_model", "unknown"),
    )


@app.middleware("http")
async def monitor_requests(request, call_next):
    path = request.url.path

    # Avoid recording Prometheus scrapes as normal API requests.
    if path == "/metrics" or path.startswith("/metrics/"):
        return await call_next(request)

    started = perf_counter()
    status = "500"

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        elapsed = perf_counter() - started

        REQUEST_COUNT.labels(
            method=request.method,
            path=path,
            status=status,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            path=path,
        ).observe(elapsed)

        logger.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            path,
            status,
            elapsed * 1000,
        )


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "metrics": "/metrics",
        "disclaimer": "Educational demonstration, not medical advice.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(patient: PatientFeatures):
    try:
        row = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]

        PREDICTION_COUNT.inc()

        logger.info("Prediction completed: class=%s", prediction)

        return {
            "prediction": prediction,
            "predicted_class": (
                "heart_disease_present"
                if prediction == 1
                else "heart_disease_absent"
            ),
            "probability_class_1": round(
                float(probabilities[1]), 4
            ),
            "model_name": metadata.get(
                "selected_model", "unknown"
            ),
            "disclaimer": (
                "Educational use only. Not a medical diagnosis."
            ),
        }

    except Exception:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        )
