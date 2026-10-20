
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import create_eda_plots, load_and_clean_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SCREENSHOTS_DIR = ROOT / "screenshots"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal",
]


def build_pipeline(classifier):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def main():
    for directory in [MODEL_DIR, REPORT_DIR, SCREENSHOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data()
    create_eda_plots(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease-classification")

    results = []

    for name, classifier in models.items():
        pipeline = build_pipeline(classifier)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        cv_metrics = {
            f"cv_{metric}": float(scores[f"test_{metric}"].mean())
            for metric in scoring
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)

            for key, value in cv_metrics.items():
                mlflow.log_metric(key, value)

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            test_metrics = {
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }

            for key, value in test_metrics.items():
                mlflow.log_metric(key, float(value))

            # Save and track evaluation plots.
            ConfusionMatrixDisplay.from_predictions(
                y_test, predictions
            )
            plt.tight_layout()
            cm_path = SCREENSHOTS_DIR / f"{name}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close("all")
            mlflow.log_artifact(str(cm_path))

            RocCurveDisplay.from_predictions(y_test, probabilities)
            plt.tight_layout()
            roc_path = SCREENSHOTS_DIR / f"{name}_roc_curve.png"
            plt.savefig(roc_path, dpi=150)
            plt.close("all")
            mlflow.log_artifact(str(roc_path))

            report_path = REPORT_DIR / f"{name}_classification_report.txt"
            report_path.write_text(
                classification_report(
                    y_test, predictions, zero_division=0
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path))

            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                skops_trusted_types=["numpy.dtype","sklearn.tree._tree.Tree"]
            )

            results.append({
                "model": name,
                **cv_metrics,
                **test_metrics,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # Selection is based on CV ROC-AUC, not test-set performance.
    best_name = max(
        results,
        key=lambda row: row["cv_roc_auc"],
    )["model"]

    best_pipeline = build_pipeline(models[best_name])
    best_pipeline.fit(X_train, y_train)

    dump(best_pipeline, MODEL_DIR / "heart_disease_pipeline.joblib")

    metadata = {
        "selected_model": best_name,
        "features": list(X.columns),
        "target_definition": "num == 0: 0; num > 0: 1",
        "random_state": 42,
        "test_size": 0.20,
    }

    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nModel comparison:")
    print(results_df.round(3).to_string(index=False))
    print("\nSelected model:", best_name)
    print("Saved model:", MODEL_DIR / "heart_disease_pipeline.joblib")
    print("MLflow tracking directory:", ROOT / "mlruns")


if __name__ == "__main__":
    main()
