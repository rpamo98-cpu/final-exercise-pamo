"""
Wine Quality Prediction System
Automatically downloads the UCI red-wine CSV if it is missing,
then trains/evaluates the model and saves the artifacts.

Run from this folder:
    python train_model.py
"""
from pathlib import Path
import json
import urllib.request

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "wine_quality.csv"
MODEL_FILE = BASE_DIR / "wine_model.pkl"
METRICS_FILE = BASE_DIR / "evaluation_results.json"
REPORT_FILE = BASE_DIR / "evaluation_report.txt"


def download_dataset():
    """Download the UCI CSV only when it is not already in the project folder."""
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:
        print(f"Using existing dataset: {DATA_FILE.name}")
        return

    print("Dataset not found. Downloading the UCI red-wine CSV...")
    temporary_file = DATA_FILE.with_suffix(".csv.download")
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; WineQualityStudentProject/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()

        if not data or b"fixed acidity" not in data[:500]:
            raise RuntimeError(
                "The downloaded file does not look like the expected wine-quality CSV."
            )

        temporary_file.write_bytes(data)
        temporary_file.replace(DATA_FILE)
        print(f"Downloaded dataset successfully: {DATA_FILE.name}")
    except Exception as exc:
        if temporary_file.exists():
            temporary_file.unlink()
        raise RuntimeError(
            "Could not download the UCI dataset. Check your internet connection "
            "and run `python train_model.py` again."
        ) from exc


def load_data():
    download_dataset()
    df = pd.read_csv(DATA_FILE, sep=";")
    if "quality" not in df.columns:
        raise ValueError("Dataset must contain a 'quality' target column.")
    return df


def main():
    df = load_data()
    print("Dataset:", DATA_FILE.name)
    print("Original dataset shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))

    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])  # Keep target out of model inputs.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=[0, 1], zero_division=0
    )
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Needs improvement", "Good quality"],
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "accuracy": float(accuracy),
        "classes": {
            "Needs improvement": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1_score": float(f1[0]),
                "support": int(support[0]),
            },
            "Good quality": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1_score": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": matrix.tolist(),
        "test_size": int(len(y_test)),
        "random_state": 42,
        "target_rule": "quality >= 6 = GOOD QUALITY; otherwise NEEDS IMPROVEMENT",
    }

    joblib.dump(model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REPORT_FILE.write_text(
        "WINE QUALITY MODEL EVALUATION\n"
        "=============================\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"{report}\n"
        "Confusion matrix (rows=true, columns=predicted; class order 0,1):\n"
        f"{matrix}\n",
        encoding="utf-8",
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(report)
    print("\nSaved files:")
    print(f"- {MODEL_FILE.name}")
    print(f"- {METRICS_FILE.name}")
    print(f"- {REPORT_FILE.name}")
    print("Input features:", list(X.columns))


if __name__ == "__main__":
    main()
