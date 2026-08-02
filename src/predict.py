"""
predict.py

Inference pipeline for a single loan application. Takes raw applicant
inputs (the same shape a person would naturally provide — individual
asset values, not pre-computed ratios), runs them through the SAME
encoding, feature engineering, and scaling steps used in training, and
returns a prediction with confidence.

This is used by both app.py (Streamlit) and, if built, the Flask API —
keeping inference logic in one place avoids the two ever drifting apart.
"""

import pandas as pd

from feature_engineering import (
    add_total_assets_value, add_loan_to_income_ratio, add_asset_to_loan_ratio
)
from utils import load_artifact

MODEL_PATH = "models/trained_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def load_inference_artifacts(
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    feature_columns_path: str = FEATURE_COLUMNS_PATH,
):
    """
        Load the trained model, fitted scaler, and expected feature column
        order. Raises a clear error if any file is missing, rather than a
        cryptic pickle traceback.
    
        Returns
        -------
        tuple: (model, scaler, feature_columns)
        """
    try:
        model = load_artifact(model_path)
        scaler = load_artifact(scaler_path)
        feature_columns = load_artifact(feature_columns_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing model artifact: {e}. Run `python src/train.py` from the "
            f"project root first to generate models/*.pkl."
        )
    return model, scaler, feature_columns


def build_feature_row(raw_input: dict) -> pd.DataFrame:
    """
    Convert raw applicant input into the exact engineered feature row
    the model expects — mirrors preprocessing.py's encoding and
    feature_engineering.py's derived features, applied to a single row.
    
    Parameters
    ----------
    raw_input : dict
        Expected keys:
        no_of_dependents, education ("Graduate"/"Not Graduate"),
        self_employed ("Yes"/"No"), income_annum, loan_amount, loan_term,
        cibil_score, residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value
    
    Returns
    -------
    pd.DataFrame
        Single-row dataframe with raw + engineered columns (not yet
        scaled or column-ordered — see preprocess_input for that).
    """
    df = pd.DataFrame([raw_input])

    # Same deterministic mappings as preprocessing.encode_categorical_columns
    df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
    df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})

    # Same feature engineering as feature_engineering.engineer_features
    # (cibil_band is intentionally skipped -- it was dropped in Day 3's
    # feature selection as redundant with the raw cibil_score)
    df = add_total_assets_value(df)
    df = add_loan_to_income_ratio(df)
    df = add_asset_to_loan_ratio(df)

    return df


def preprocess_input(raw_input: dict, scaler, feature_columns: list) -> pd.DataFrame:
    """
    Full preprocessing: build engineered features, select+order columns
    to match training, then apply the SAME fitted scaler used in training.

    Parameters
    ----------
    raw_input : dict
    scaler : fitted StandardScaler (from models/scaler.pkl)
    feature_columns : list of str (from models/feature_columns.pkl) —
        defines both which columns to keep and their exact order.

    Returns
    -------
    pd.DataFrame
        Single-row, scaled, correctly-ordered feature row ready for
        model.predict().
    """
    df = build_feature_row(raw_input)
    df = df[feature_columns]  # enforce exact column set + order used in training
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=feature_columns)


def predict_loan_status(raw_input: dict, model=None, scaler=None, feature_columns=None) -> dict:
    """
    End-to-end prediction for a single application.

    If model/scaler/feature_columns aren't passed in, they're loaded
    from disk (convenient for one-off calls; pass them in explicitly
    when calling this repeatedly, e.g. from a Streamlit app, to avoid
    reloading from disk on every prediction).

    Returns
    -------
    dict with keys: prediction ("Approved"/"Rejected"), approved_probability,
    rejected_probability
    """
    if model is None or scaler is None or feature_columns is None:
        model, scaler, feature_columns = load_inference_artifacts()

    # Apply preprocessing
    processed_input = preprocess_input(raw_input, scaler, feature_columns)

    # Predict class
    prediction = model.predict(processed_input)[0]
    
    # Predict probabilities
    probabilities = model.predict_proba(processed_input)[0]

    # Convert numeric prediction to readable label
    label = "Approved" if prediction == 1 else "Rejected"
    
    return {
        "prediction": label,
        "approved_probability": float(probabilities[1]),
        "rejected_probability": float(probabilities[0]),
    }
    

if __name__ == "__main__":
    # Quick manual smoke test
    sample = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 5000000,
        "loan_amount": 15000000,
        "loan_term": 10,
        "cibil_score": 750,
        "residential_assets_value": 3000000,
        "commercial_assets_value": 2000000,
        "luxury_assets_value": 5000000,
        "bank_asset_value": 1000000,
    }
    result = predict_loan_status(sample)
    print(result)
