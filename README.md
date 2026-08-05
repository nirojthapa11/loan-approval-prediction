# Loan Approval Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A machine learning project that predicts whether a loan application will be
approved or rejected, based on applicant financial and demographic details.
Built as a 6-day, end-to-end certification project: data cleaning, feature
engineering, EDA, multi-model comparison, and deployment via a Streamlit
app, a Flask REST API, and Docker.

> **Status:** ✅  Completed. Decision Tree model selected (Test F1 = 1.0000 — see [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) for full results and an honest discussion of why this dataset supports unusually high scores).

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Setup & Quick Start](#setup--quick-start-windows--cmdexe)
- [Roadmap](#roadmap)
- [License](#license)

## Problem Statement

Loan approval decisions depend on a mix of numerical factors (income, loan
amount, credit history/CIBIL score, asset values) and categorical factors
(education level, employment type, number of dependents). This project
builds a classification model to predict loan approval status and exposes
it through both a Streamlit web application and a Flask REST API.

## Dataset

**Loan Approval Prediction Dataset** (Kaggle, Archit Sharma)
Contains applicant details such as:
- `no_of_dependents`, `education`, `self_employed`
- `income_annum`, `loan_amount`, `loan_term`, `cibil_score`
- `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
- `loan_status` (target: Approved / Rejected)

Download link: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

Place the downloaded CSV in `data/loan_approval_dataset.csv`.

## Documentation

This README is the entry point. Everything else lives in its own file,
so each topic is easy to find and doesn't bloat this page:

| Guide | What's in it |
|---|---|
| [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | Full technical walkthrough — folder structure, dataset column reference, notebook-by-notebook explanation, model rationale, retraining instructions, troubleshooting table, and a full **Git/GitHub reference** (section 11) — install, commit workflow, this project's branching strategy, undoing commits, tagging releases |
| [`DOCKER_GUIDE.md`](DOCKER_GUIDE.md) | Installing Docker, building the image, running via `docker run` or `docker compose`, a troubleshooting table |
| [`FLASK_API_GUIDE.md`](FLASK_API_GUIDE.md) | Every API endpoint, tested request/response examples, error handling, how the API relates to the Streamlit app |
| [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) | Results for all 7 models, best hyperparameters, and why the top scores are unusually high (explained honestly, not glossed over) |
| [`POSTMAN_COLLECTION.json`](POSTMAN_COLLECTION.json) | Import directly into Postman — 6 ready-to-run requests against the API |

## Project Structure

```
loan-approval-prediction/
├── app.py                      # Streamlit application (Predict + About tabs)
├── requirements.txt            # Python dependencies
├── README.md                   # You are here
├── PROJECT_GUIDE.md             # Full technical walkthrough + Git reference
├── DOCKER_GUIDE.md              # Docker build/run/troubleshooting guide
├── FLASK_API_GUIDE.md           # API endpoint reference, tested examples
├── MODEL_COMPARISON.md          # Full model results and selection rationale
├── POSTMAN_COLLECTION.json      # Importable Postman collection
├── LICENSE
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
│
├── data/                        # Dataset (gitignored — download separately)
├── notebooks/                    # 4 notebooks: loading, cleaning, EDA, modeling
├── models/                       # trained_model.pkl, scaler.pkl, feature_columns.pkl
├── images/                       # EDA/model plots, screenshots
├── reports/                      # Timestamped prediction reports (gitignored CSVs)
│
├── src/                          # Reusable pipeline modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
└── api/                          # Flask REST API (standalone, shares src/predict.py)
    ├── app.py
    ├── routes.py
    ├── inference.py
    ├── model_loader.py
    └── requirements.txt
```

## Setup & Quick Start (Windows / cmd.exe)

```cmd
:: 1. Clone the repository
git clone https://github.com/nirojthapa11/loan-approval-prediction.git
cd loan-approval-prediction

:: 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate.bat

:: 3. Install dependencies
pip install -r requirements.txt

:: 4. Download the dataset (see Dataset section above) into data/

:: 5. Either retrain from scratch...
python src\train.py

:: ...or use the already-committed models/*.pkl and skip straight to running an interface:

:: 6a. Run the Streamlit app
streamlit run app.py

:: 6b. OR run the Flask API
python api\app.py

:: 6c. OR run both via Docker
docker compose up
```

Full detail on each of these steps is in the guides linked above —
this is deliberately just enough to get something running.

## Screenshots

Auto-generated by the notebooks (already in `images/`):
- EDA: `univariate_histograms.png`, `bivariate_boxplots.png`, `correlation_heatmap.png`, `pairplot.png`
- Model evaluation: `model_comparison.png`, `confusion_matrices.png`, `roc_curves.png`

Added to `images/`:
- [x] `github_repo_created.png` — repo homepage right after creation
- [x] `git_log_final.png` — output of `git log --oneline --graph --all` showing full commit history across all 6 days
- [x] `streamlit_predict_approved.png` / `streamlit_predict_rejected.png` — the Predict tab showing both outcomes
- [x] `streamlit_about_tab.png` — the About & Model Info tab
- [x] `postman_predict_response.png` — a successful `POST /predict` response in Postman
- [x] `docker_build_success.png` / `docker_ps_running.png` — Docker build and running containers

## Roadmap

- [x]  Project setup, dataset loading, initial exploration
- [x]  Data cleaning & feature engineering
- [x]  Exploratory data analysis & feature selection
- [x]  Model training, comparison, tuning, evaluation
- [x]  Streamlit application
- [x]  Flask API, Docker, final documentation

## License

MIT — see [LICENSE](LICENSE).
