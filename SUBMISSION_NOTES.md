# Submission Notes — Complete Before Uploading

## Included in this package
- [x] SAD mini-analysis
- [x] `train_model.py`
- [x] `app.py`
- [x] `requirements.txt`
- [x] `README.md`
- [x] Brief reflection answers
- [x] Script to calculate accuracy, precision, recall, and F1-score from the real dataset

## Must be completed on your computer
1. Install the requirements and run `python train_model.py`.
2. Open `evaluation_report.txt` and copy the actual accuracy, precision, recall, and F1-score into your submission/report. The values are not prefilled because they must come from executing the model against the dataset.
3. Run `streamlit run app.py`.
4. Capture a genuine screenshot of the dashboard. Then upload a sample CSV from the dashboard and capture a second screenshot showing prediction results.
5. Create your own GitHub repository and push the project. Replace the placeholder URL with your actual repository URL.

## Suggested screenshot evidence
- `01_dashboard.png`: dashboard title, dataset summary, preview, and model evaluation.
- `02_predictions.png`: uploaded sample rows and displayed prediction labels.

## GitHub URL
`https://github.com/YOUR-USERNAME/wine-quality-prediction`

Replace the placeholder after publishing. Do not submit the placeholder as if it were a real repository.

## Final checks
- Confirm `wine_model.pkl`, `wine_quality.csv`, `evaluation_results.json`, and `evaluation_report.txt` were created by training.
- Confirm the app accepts a valid sample CSV and reports missing required columns for an invalid CSV.
- Confirm the result CSV downloads successfully.
- Keep screenshots genuine and captured from your running Streamlit app.
