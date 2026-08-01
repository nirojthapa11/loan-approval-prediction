"""
evaluate.py

Model evaluation utilities: metrics, confusion matrix, and ROC curve
plotting for the Loan Approval Prediction project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Compute the full evaluation metric suite for a fitted model on the
    held-out test set.

    Parameters
    ----------
    name : str
        Model name, for labeling.
    model : fitted sklearn-compatible estimator
    X_test, y_test : test split

    Returns
    -------
    dict with name, accuracy, precision, recall, f1, roc_auc, y_pred, y_proba
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def print_evaluation_report(name: str, y_test, y_pred) -> None:
    """Print the sklearn classification report for a single model."""
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    

def plot_confusion_matrix(name: str, y_test, y_pred, ax=None):
    """Plot a labeled confusion matrix heatmap for a single model."""
    cm = confusion_matrix(y_test, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return ax     
