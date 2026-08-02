# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be
approved or rejected, based on applicant financial and demographic details.

> **Status:** 🚧 In development 

## Problem Statement

Loan approval decisions depend on a mix of numerical factors (income, loan
amount, credit history/CIBIL score, asset values) and categorical factors
(education level, employment type, number of dependents). This project
builds a classification model to predict loan approval status and exposes
it through a Streamlit web application.

## Dataset

**Loan Approval Prediction Dataset** (Kaggle)
Contains applicant details such as:
- `no_of_dependents`, `education`, `self_employed`
- `income_annum`, `loan_amount`, `loan_term`, `cibil_score`
- `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
- `loan_status` (target: Approved / Rejected)

Download link: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

Place the downloaded CSV in `data/loan_approval_dataset.csv`.

## Project Structure

```
loan-approval-prediction/
├── app.py                  # Streamlit application 
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── data/                    # Dataset (not committed — see .gitignore)
├── notebooks/                # Jupyter notebooks (EDA, experimentation)
├── models/                   # Saved trained model, scaler, encoder
├── images/                   # Plots and screenshots
├── src/                      # Reusable Python modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
└── api/                      # Flask REST API (bonus)
```

## Setup (Windows / cmd.exe)

```cmd
:: 1. Clone the repository
git clone https://github.com/nirojthapa11/loan-approval-prediction.git
cd loan-approval-prediction

:: 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate.bat

:: 3. Install dependencies
pip install -r requirements.txt

:: 4. Launch Jupyter to explore the notebooks
jupyter notebook
```

## Roadmap

- [x] Day 1 — Project setup, dataset loading, initial exploration
- [x] Day 2 — Data cleaning & feature engineering
- [x] Day 3 — Exploratory data analysis & feature selection
- [x] Day 4 — Model training, comparison, tuning, evaluation
- [x] Day 5 — Streamlit application
- [ ] Day 6 — Documentation, Docker, final polish

## License

See [LICENSE](LICENSE).
