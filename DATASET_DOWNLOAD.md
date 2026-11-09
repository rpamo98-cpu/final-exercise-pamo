# Automatic dataset download

You do not need to download `wine_quality.csv` manually.

1. Open this project folder in VS Code.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Run `python train_model.py`.

If `wine_quality.csv` is missing, `train_model.py` downloads it from the
UCI Machine Learning Repository and saves it in this folder. Later runs
reuse the saved CSV. An internet connection is needed for the first run.

After training, the script creates `wine_model.pkl`,
`evaluation_results.json`, and `evaluation_report.txt`.
