
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCREENSHOTS_DIR = ROOT / "screenshots"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num",
]


def load_and_clean_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "heart_disease_raw.csv"
    clean_path = DATA_DIR / "heart_disease_cleaned.csv"

    if not raw_path.exists():
        df = pd.read_csv(
            DATA_URL,
            header=None,
            names=COLUMNS,
            na_values="?",
        )
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path, na_values="?")

    df = df.drop_duplicates().copy()

    # Do not impute missing values here. The model pipeline
    # learns imputation values from training folds only.
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    df.to_csv(clean_path, index=False)

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1")

    return df


def create_eda_plots(df):
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Feature histograms
    df.hist(figsize=(15, 12), bins=20, edgecolor="black")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "feature_histograms.png", dpi=150)
    plt.close("all")

    # Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Heart Disease Feature Correlations")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Class balance
    counts = df["target"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["No disease (0)", "Disease (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
    )
    plt.ylabel("Number of records")
    plt.title("Target Class Balance")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "class_balance.png", dpi=150)
    plt.close()

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        missing.to_csv(SCREENSHOTS_DIR / "missing_values.csv")


if __name__ == "__main__":
    dataset = load_and_clean_data()

    print("Dataset shape:", dataset.shape)
    print("\nMissing values:")
    print(dataset.isnull().sum())
    print("\nTarget distribution:")
    print(dataset["target"].value_counts())

    create_eda_plots(dataset)
    print("\nStage 1 complete. Data and plots saved locally.")
