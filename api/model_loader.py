"""
model_loader.py

Loads the trained model, scaler, and feature column order for the
Flask API. Reuses the same artifacts produced by src/train.py  --
the API and the Streamlit app both consume the exact same .pkl files,
so predictions are guaranteed consistent between the two interfaces.
"""

import os
import sys

# Add the project's src/ folder to the path so we can reuse the same
# preprocessing/feature engineering/prediction logic as the Streamlit
# app, rather than duplicating it here.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from predict import load_inference_artifacts  # noqa: E402

_model = None
_scaler = None
_feature_columns = None


def get_model_artifacts():
    """
    Load model/scaler/feature_columns once per process and cache them
    in module-level variables. Avoids re-reading .pkl files from disk
    on every API request.

    Returns
    -------
    tuple: (model, scaler, feature_columns)
    """
    global _model, _scaler, _feature_columns

    if _model is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "trained_model.pkl")
        scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
        feature_columns_path = os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")
        _model, _scaler, _feature_columns = load_inference_artifacts(
            model_path, scaler_path, feature_columns_path
        )

    return _model, _scaler, _feature_columns
