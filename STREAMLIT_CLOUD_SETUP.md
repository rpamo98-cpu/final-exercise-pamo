# Streamlit Community Cloud setup

The website version must be connected to a GitHub repository. Upload/push
the complete project folder to GitHub, including at least:

- `app.py`
- `train_model.py`
- `requirements.txt`

In Streamlit Community Cloud, deploy the repository and choose `app.py`
as the Main file path.

This cloud-ready `app.py` checks for `wine_quality.csv` and `wine_model.pkl`.
If either is missing, it automatically runs the training script, which
downloads the UCI CSV and trains the model. The first app startup can take
a few minutes. After deployment, use **Manage app → Reboot app** if you
need to trigger a fresh startup.

Do not upload a local `.venv` folder to GitHub.
