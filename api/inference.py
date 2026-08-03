"""
predict.py (api/)

Input validation and prediction wrapper for the Flask API. Delegates
the actual prediction logic to src/predict.py -- the same code path
the Streamlit app uses -- so the API and the app can never silently
diverge in behavior.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from predict import predict_loan_status  # noqa: E402
from model_loader import get_model_artifacts

REQUIRED_FIELDS = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
]


def validate_input(data: dict) -> list:
    """
    Check the incoming JSON body for missing fields and obviously
    invalid values. Returns a list of error messages -- empty list
    means the input is valid.

    Parameters
    ----------
    data : dict
        Parsed JSON body from the request.

    Returns
    -------
    list of str
        Human-readable validation errors, if any.
    """
    errors = []

    if not isinstance(data, dict):
        return ["Request body must be a JSON object."]

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        errors.append(f"Missing required field(s): {', '.join(missing)}")
        return errors  # no point checking values if fields are missing

    if data.get("education") not in ("Graduate", "Not Graduate"):
        errors.append("education must be 'Graduate' or 'Not Graduate'")

    if data.get("self_employed") not in ("Yes", "No"):
        errors.append("self_employed must be 'Yes' or 'No'")

    numeric_fields = [f for f in REQUIRED_FIELDS if f not in ("education", "self_employed")]
    for field in numeric_fields:
        value = data.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{field} must be a number")
        elif value < 0:
            errors.append(f"{field} must not be negative")

    if not errors:
        cibil_score = data.get("cibil_score")
        if not (300 <= cibil_score <= 900):
            errors.append("cibil_score must be between 300 and 900")

    return errors


def run_prediction(data: dict) -> dict:
    """
    Validate input, then run the shared prediction pipeline.

    Returns
    -------
    dict
        Either {"error": [...]} or the prediction result dict from
        src.predict.predict_loan_status.
    """
    errors = validate_input(data)
    if errors:
        return {"error": errors}

    model, scaler, feature_columns = get_model_artifacts()
    raw_input = {field: data[field] for field in REQUIRED_FIELDS}
    result = predict_loan_status(raw_input, model, scaler, feature_columns)
    return result
