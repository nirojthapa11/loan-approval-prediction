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