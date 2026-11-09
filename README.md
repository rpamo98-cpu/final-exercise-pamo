# Wine Quality Prediction System

A classroom decision-support prototype that classifies red-wine samples as **GOOD QUALITY** or **NEEDS IMPROVEMENT** using physicochemical measurements and a Random Forest classifier.

## Prediction rule
- `quality >= 6` → `GOOD QUALITY` (class 1)
- `quality < 6` → `NEEDS IMPROVEMENT` (class 0)

The original `quality` column is the target only and is excluded from model inputs to prevent target leakage.

## Dataset
UCI Machine Learning Repository — [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), red-wine CSV. The training script automatically downloads the CSV from the UCI repository the first time you run `python train_model.py`. The downloaded file is saved as `wine_quality.csv` in this project folder. If it is already there, the script reuses it.

## Files
- `train_model.py` — downloads data, preprocesses, trains, evaluates, and saves the model and metrics.
- `app.py` — Streamlit dashboard and CSV prediction interface.
- `requirements.txt` — Python dependencies.
- `SAD_mini_analysis.md` — system analysis and design mini-analysis.
- `reflection_answers.md` — brief answers to the laboratory reflection questions.
- `SUBMISSION_NOTES.md` — steps to complete evaluation evidence, screenshots, and GitHub publication.

## Setup and run
Use Python 3.10 or later.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal (normally `http://localhost:8501`). In the dashboard, download the sample input CSV, upload it, and inspect/download the predictions.

## Model and evaluation
A stratified 80/20 train-test split is used (`random_state=42`). The preprocessing/model pipeline uses median imputation, standard scaling, and `RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)`. Running `train_model.py` generates:
- `wine_model.pkl`
- `evaluation_results.json`
- `evaluation_report.txt`

Use the generated values in the submission. Metrics are intentionally calculated by the script on the actual downloaded dataset rather than invented in this README.

## Limitations and responsible use
This model is trained on a particular public red-wine dataset. It may not generalize to other wine types, producers, regions, or testing conditions. Predictions are preliminary screening results, not final quality decisions and not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.

## GitHub publication
Create a repository named `wine-quality-prediction` on GitHub, then run:

```bash
git init
git add train_model.py app.py requirements.txt README.md SAD_mini_analysis.md reflection_answers.md SUBMISSION_NOTES.md
git commit -m "Build wine quality prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/wine-quality-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your own GitHub username. Never put passwords or access tokens in source code.


## Deploying to Streamlit Community Cloud

This version is designed for Community Cloud. On the first app launch, `app.py`
automatically runs `train_model.py` if the CSV or trained model is missing.
The dataset is downloaded from UCI and the model/evaluation files are generated
inside the app's runtime. The first launch may take a little longer.

Push the whole project folder to a GitHub repository, then create/deploy the app
from Streamlit Community Cloud using `app.py` as the main file.
