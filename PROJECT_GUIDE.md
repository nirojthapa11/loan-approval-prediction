# Project Guide

A complete walkthrough of the Loan Approval Prediction project: what every
piece does, how to run it, and how to troubleshoot common problems.

---

## 1. Project Overview

**Goal:** Predict whether a loan application will be **Approved** or
**Rejected**, given applicant details (income, credit history, assets,
dependents, education, employment type).

**Type:** Binary classification.

**Pipeline:**
```
Raw CSV → Cleaning → Feature Engineering → EDA → Feature Selection →
Model Training (multiple algorithms) → Comparison → Tuning → Evaluation →
Save Best Model → Streamlit App (+ Flask API, Docker as bonus layers)
```

---

## 2. Folder Structure (full project, by the end of Day 6)

```
loan-approval-prediction/
│
├── app.py                       # Streamlit application entry point (Predict + About tabs)
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview (this is the "front door")
├── PROJECT_GUIDE.md              # This file (includes Git/GitHub reference in section 11)
├── DOCKER_GUIDE.md               # Docker reference
├── FLASK_API_GUIDE.md            # API reference
├── MODEL_COMPARISON.md           # Model results and selection rationale
├── POSTMAN_COLLECTION.json       # Importable Postman collection
├── LICENSE
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   └── loan_approval_dataset.csv     # Raw dataset (gitignored — download separately)
│
├── notebooks/
│   ├── 01_data_loading_exploration.ipynb
│   ├── 02_data_cleaning_feature_engineering.ipynb
│   ├── 03_eda_feature_selection.ipynb
│   └── 04_model_training_evaluation.ipynb
│
├── models/
│   ├── trained_model.pkl         # Best-performing model, serialized
│   ├── scaler.pkl                # Fitted StandardScaler
│   └── feature_columns.pkl       # Exact feature order expected at inference time
│
├── images/                       # Saved plots + app/GitHub screenshots
├── reports/                      # Timestamped prediction reports (runtime output)
│
├── src/
│   ├── preprocessing.py          # Cleaning functions (missing values, outliers, encoding)
│   ├── feature_engineering.py    # Derived feature creation
│   ├── feature_selection.py      # Correlation analysis, multicollinearity checks
│   ├── train.py                  # Model training + hyperparameter tuning
│   ├── evaluate.py               # Metrics, confusion matrix, ROC-AUC
│   ├── predict.py                # Load model, run inference on new input
│   └── utils.py                  # Shared helpers (e.g. load/save artifacts)
│
└── api/                           # Flask REST API (standalone, shares src/predict.py)
    ├── app.py                     # Flask entry point
    ├── routes.py                  # GET /, GET /health, POST /predict
    ├── inference.py               # Input validation + prediction wrapper
    ├── model_loader.py            # Cached artifact loading
    └── requirements.txt           # Minimal deps for API-only deployment
```

---

## 3. Installation

```cmd
git clone https://github.com/<username>/loan-approval-prediction.git
cd loan-approval-prediction
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Why a virtual environment?** It isolates this project's dependencies from
your system Python and other projects, so version conflicts don't happen.
`venv\Scripts\activate.bat` puts you "inside" it — you'll see `(venv)` at
the start of your command prompt when it's active.

---

## 4. Dataset

Source: Kaggle — `architsharma01/loan-approval-prediction-dataset`.

Place the CSV at `data/loan_approval_dataset.csv`. Key columns:

| Column | Type | Meaning |
|---|---|---|
| `no_of_dependents` | numeric | Number of dependents |
| `education` | categorical | Graduate / Not Graduate |
| `self_employed` | categorical | Yes / No |
| `income_annum` | numeric | Annual income |
| `loan_amount` | numeric | Requested loan amount |
| `loan_term` | numeric | Loan term (years) |
| `cibil_score` | numeric | Credit score (300-900) |
| `residential_assets_value` | numeric | Value of residential assets |
| `commercial_assets_value` | numeric | Value of commercial assets |
| `luxury_assets_value` | numeric | Value of luxury assets |
| `bank_asset_value` | numeric | Value of bank assets |
| `loan_status` | categorical (target) | Approved / Rejected |

---

## 5. Notebook Walkthrough

- **01 — Loading & Exploration:** confirms the CSV reads correctly; checks
  shape, dtypes, missing values, duplicates, target balance.
- **02 — Cleaning & Feature Engineering:** handles missing values, removes
  duplicates, treats outliers (e.g. via IQR capping), encodes categorical
  columns, scales numeric columns, and derives new features (e.g.
  total asset value, debt-to-income-style ratios).
- **03 — EDA & Feature Selection:** univariate → bivariate → multivariate
  analysis (histograms, boxplots, correlation heatmap, pairplot), then
  selects the final feature set for modeling based on correlation with
  the target and multicollinearity checks.
- **04 — Modeling:** trains and compares Logistic Regression, Decision
  Tree, Random Forest, KNN, Naive Bayes, SVM, and XGBoost; runs
  cross-validation and hyperparameter tuning (GridSearchCV/RandomizedSearchCV);
  evaluates with accuracy, precision, recall, F1, ROC-AUC, and confusion
  matrices; saves the best model.

### A note on `encoder.pkl` (or the lack of one)

`education` and `self_employed` are binary categorical columns encoded
with a **fixed, deterministic mapping** (`Graduate`→1, `Not Graduate`→0,
etc. — see `src/preprocessing.encode_categorical_columns`), not a fitted
`LabelEncoder`/`OneHotEncoder` object. A fixed mapping doesn't need to be
serialized and reloaded the way a *fitted* transformer does — the same
two lines of mapping code run identically at training and inference
time. This is why `models/` has `scaler.pkl` and `feature_columns.pkl`
but no `encoder.pkl`: there's nothing stateful to save. If a future
version of this project encoded a higher-cardinality categorical column
(e.g. a `state` or `city` field) with something like `OneHotEncoder`,
*that* would need to be pickled and reloaded, since its output shape
depends on which categories were seen during training.

---

## 6. Model Explanation

Each algorithm is included for a reason:
- **Logistic Regression** — simple, interpretable baseline.
- **Decision Tree** — captures non-linear splits, easy to visualize.
- **Random Forest** — ensemble of trees, usually strong out of the box.
- **KNN** — distance-based, sensitive to feature scaling (hence scaling matters).
- **Naive Bayes** — fast probabilistic baseline, assumes feature independence.
- **SVM** — effective with clear margins between classes.
- **XGBoost** — gradient boosting, typically the top performer on tabular data.

The **best model** is selected by comparing cross-validated F1 score (a
balanced metric for classification, especially if the target classes are
imbalanced) and confirmed with ROC-AUC on a held-out test set. See
`MODEL_COMPARISON.md` for the actual numbers once Day 4 is complete.

---

## 7. How Streamlit Works Here

`app.py` loads `models/trained_model.pkl`, `models/scaler.pkl`, and
`models/feature_columns.pkl`, presents input widgets (sliders/dropdowns) for
each raw applicant detail in the sidebar, applies the same feature
engineering and encoding used in training (`src/feature_engineering.py`,
`src/preprocessing.py`), scales with the same fitted scaler, and calls
`.predict()` / `.predict_proba()` to show the approval decision and its
confidence. Model info, project background, and example applicant profiles
are in a separate "About & Model Info" tab in the main content area.

**Run it:**
```cmd
streamlit run app.py
```
This starts a local server (default `http://localhost:8501`) and opens
it in your browser automatically.

---

## 7b. How the Flask API Works Here

`api/app.py` exposes the same prediction logic as a REST API, for
integration into other systems rather than a browser UI. It reuses
`src/predict.py` directly — the API and the Streamlit app can never
silently diverge in behavior, since both call the same function.

**Run it:**
```cmd
python api\app.py
```
See `FLASK_API_GUIDE.md` for the full endpoint reference and tested
example requests, and `POSTMAN_COLLECTION.json` for a ready-to-import
Postman collection.

---

## 8. Retraining the Model

If you update the dataset or preprocessing logic:
```cmd
python src/train.py
```
This re-runs the full training + tuning pipeline and overwrites the
`.pkl` files in `models/`. Restart the Streamlit app or Flask API
afterward to pick up the new model.

---

## 9. Common Errors & Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | Package not installed, or venv not activated | Run `venv\Scripts\activate.bat` then `pip install -r requirements.txt` |
| `FileNotFoundError: data/loan_approval_dataset.csv` | Dataset not downloaded/placed | Download from Kaggle, place in `data/` |
| Streamlit shows a blank/error page | Model files missing or mismatched preprocessing | Run `python src/train.py` to regenerate `models/*.pkl` |
| `ValueError: X has N features, but model expects M` | Input at inference time wasn't encoded/scaled the same way as training data | Ensure `predict.py` uses the same `scaler.pkl` and `feature_columns.pkl` order |
| Jupyter kernel keeps dying/restarting | Usually a memory issue or a corrupted environment | Restart kernel, or recreate the venv |
| `git push` rejected (`non-fast-forward`) | Remote has commits you don't have locally | `git pull` first, resolve any conflicts, then push |
| Flask API returns HTML instead of JSON on unknown routes | Error handler registered on the blueprint instead of the app | Already fixed in `api/app.py` — handlers are registered via `app.errorhandler`, not `blueprint.errorhandler` |
| Docker build/run issues | See `DOCKER_GUIDE.md` — has a dedicated troubleshooting table |

---

## 10. Future Improvements

- Add SHAP values to explain individual predictions in the Streamlit app.
- Add a monitoring/logging layer to the Flask API for tracking prediction drift.
- Expand hyperparameter tuning with Bayesian optimization (e.g. Optuna) instead of grid search.
- Add automated tests (`pytest`) for `src/preprocessing.py` and `src/predict.py`.
- Set up a CI pipeline (GitHub Actions) to run tests and linting on every push.

---

## 11. Version Control (Git) Reference

### Setup (one-time)
```cmd
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Connecting a local project to GitHub
```cmd
git init
git add .
git commit -m "Day 1: Initial project setup"
git branch -M main
git remote add origin https://github.com/<username>/loan-approval-prediction.git
git push -u origin main
```

### The core day-to-day cycle
```cmd
git status                          :: see what changed
git add .                           :: stage changes
git commit -m "Day N: description"
git push                            :: send commits to GitHub
```

| Command | What it does |
|---|---|
| `git status` | Shows staged/unstaged/untracked files |
| `git add .` / `git add <file>` | Stages changes |
| `git commit -m "message"` | Creates a snapshot with a message |
| `git push` / `git pull` | Sync commits with GitHub |
| `git log --oneline --graph --all` | Compact visual history across all branches |
| `git clone <url>` | Downloads a full copy of a repo (all history included) |

### Branching strategy used in this project
- `main` — always working, deployable code.
- One short-lived branch per day (`feature/data-cleaning`, `feature/eda`, `feature/model-training`, etc.), merged back into `main` with `--no-ff` once that day's work is verified — this keeps an explicit, visible merge commit per day in the history rather than a flattened linear log.

```cmd
git checkout -b feature/data-cleaning
:: ... do the day's work, commit ...
git push -u origin feature/data-cleaning
git checkout main
git merge feature/data-cleaning --no-ff
git push
git branch -d feature/data-cleaning
git push origin --delete feature/data-cleaning
```

**Resolving a merge conflict:** Git marks the conflicting section
directly in the file (`<<<<<<<`, `=======`, `>>>>>>>` markers) — edit
to keep the correct content, delete the markers, then `git add <file>`
and `git commit` to finalize.

### Undoing things

| Situation | Command | Effect |
|---|---|---|
| Unstage a file (keep changes) | `git restore --staged <file>` | Moves file back to unstaged |
| Discard uncommitted changes | `git restore <file>` | **Destructive** — reverts to last commit |
| Undo last commit, keep changes staged | `git reset --soft HEAD~1` | Commit removed, changes remain staged |
| Undo an already-pushed commit | `git revert <commit-hash>` | Creates a new commit that undoes the old one (safe for shared repos) |
| Recover after a hard reset | `git reflog` | Shows where HEAD has been — almost nothing is truly lost immediately |

### Marking a release
```cmd
git tag -a v1.0 -m "Final submission for certification"
git push origin v1.0
```

### Best practices
- Commit often, in small logical chunks — one commit, one coherent change.
- Write commit messages in the imperative mood: "Add feature engineering," not "Added."
- Never commit secrets (`.env` files, API keys) — that's what `.gitignore` is for.
- Pull before you push if working across multiple machines.
