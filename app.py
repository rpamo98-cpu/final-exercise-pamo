from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "wine_model.pkl"
DATA_FILE = BASE_DIR / "wine_quality.csv"
METRICS_FILE = BASE_DIR / "evaluation_results.json"

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")


def prepare_project():
    """On Streamlit Community Cloud, create the CSV/model if they are not present."""
    if not (MODEL_FILE.exists() and DATA_FILE.exists()):
        with st.spinner("First-time setup: downloading the dataset and training the model..."):
            try:
                from train_model import main as train_model_main
                train_model_main()
            except Exception as exc:
                st.error(
                    "Automatic setup failed. Check that the app repository contains "
                    "`train_model.py` and `requirements.txt`, and that the deployment "
                    f"has internet access. Details: {exc}"
                )
                st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE, sep=";")


prepare_project()
model = load_model()
data = load_dataset()

st.title("🍷 Wine Quality Prediction System")
st.caption("Classroom prototype for preliminary red-wine quality classification")

a, b, c = st.columns(3)
a.metric("Dataset records", f"{len(data):,}")
b.metric("Average quality score", f"{data['quality'].mean():.2f}")
c.metric("Good-quality records", f"{int((data['quality'] >= 6).sum()):,}")

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Model evaluation")
if METRICS_FILE.exists():
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    st.metric("Test accuracy", f"{metrics['accuracy']:.3f}")
    metric_rows = []
    for class_name, values in metrics["classes"].items():
        metric_rows.append({
            "Class": class_name,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1-score": values["f1_score"],
            "Support": values["support"],
        })
    st.dataframe(pd.DataFrame(metric_rows).set_index("Class"), use_container_width=True)
else:
    st.info("Evaluation file not found.")

st.subheader("Predict uploaded wine samples")
st.write("Upload a CSV containing the 11 physicochemical predictor columns. Do not include the quality target.")
features = list(model.feature_names_in_)
template = data[features].head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv",
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [col for col in features if col not in new_data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data[features])
            results = new_data.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                for value in predictions
            ]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted good-quality samples", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                "wine_quality_predictions.csv",
                "text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")

st.divider()
st.caption(
    "Limitation: predictions are preliminary screening estimates from a public red-wine dataset. "
    "They are not a substitute for laboratory testing, sensory evaluation, or professional quality control."
)
