"""
train.py

Model training, cross-validation, and hyperparameter tuning for the
Loan Approval Prediction project.

Trains and tunes seven classifiers, compares them on cross-validated
F1 score, and saves the best-performing model + fitted scaler to disk.

Run directly: `python src/train.py` (from the project root, with the
venv activated).
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils import save_artifact, print_section

DATA_PATH = "data/loan_approval_selected.csv"
TARGET = "loan_status"
RANDOM_STATE = 42

# Model + hyperparameter grid definitions. Grids are intentionally small —
# wide enough to demonstrate real tuning, narrow enough to run in a few
# minutes on a laptop. Widen these if you have time/compute to spare.
MODEL_GRID = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "params": {
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_split": [2, 5, 10],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [6, 10, None],
            "min_samples_split": [2, 5],
        },
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
        },
    },
    "Naive Bayes": {
        "estimator": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9, 1e-8, 1e-7],
        },
    },
    "SVM": {
        "estimator": SVC(probability=True, random_state=RANDOM_STATE),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        },
    },
}


def load_and_split(data_path: str = DATA_PATH, test_size: float = 0.2):
    """
    Load the feature-selected dataset and perform a stratified
    train/test split.

    Stratification matters here because loan_status is imbalanced
    (62% Approved / 38% Rejected, per Day 1/3 findings) — an unstratified
    split risks a test set with a meaningfully different class balance.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on the TRAINING split only.

    Fitting on the full dataset
    before splitting would leak test-set statistics into training.
    Fitting here, after the split, keeps the test set genuinely unseen.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def tune_model(name: str, estimator, params: dict, X_train, y_train, cv_folds: int = 5):
    """
    Run GridSearchCV for a single model, optimizing F1 score (chosen
    over accuracy given the target class imbalance).

    Returns
    -------
    dict with keys: name, best_estimator, best_params, best_cv_f1
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return {
        "name": name,
        "best_estimator": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_cv_f1": grid.best_score_,
    }


def train_all_models(X_train, y_train):
    """
    Run tune_model for every model in MODEL_GRID.

    Returns
    -------
    list of dict, one per model (see tune_model's return shape).
    """
    results = []
    for name, config in MODEL_GRID.items():
        print_section(f"Tuning: {name}")
        result = tune_model(
            name, config["estimator"], config["params"], X_train, y_train
        )
        print(f"Best CV F1: {result['best_cv_f1']:.4f}")
        print(f"Best params: {result['best_params']}")
        results.append(result)
    return results


def select_best_model(results: list) -> dict:
    """Return the result dict with the highest cross-validated F1 score."""
    return max(results, key=lambda r: r["best_cv_f1"])

if __name__ == "__main__":
    print_section("Loading data and splitting (stratified, 80/20)")
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print_section("Fitting scaler on TRAINING data only")
    scaler = fit_scaler(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    results = train_all_models(X_train_scaled, y_train)

    best = select_best_model(results)
    print_section(f"Best model: {best['name']} (CV F1 = {best['best_cv_f1']:.4f})")

    save_artifact(best["best_estimator"], "models/trained_model.pkl")
    save_artifact(scaler, "models/scaler.pkl")
    save_artifact(list(X_train.columns), "models/feature_columns.pkl")

