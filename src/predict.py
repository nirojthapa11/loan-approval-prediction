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


def preprocess_input(df, scaler, feature_columns):

    df = df[feature_columns]  # enforce exact column set + order used in training
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=feature_columns)


def predict_loan_status(raw_input, model, scaler, feature_columns):
    # Convert raw user input into a dataframe
    input_df = build_feature_row(raw_input)

    # Apply preprocessing
    processed_input = preprocess_input(
        input_df,
        scaler,
        feature_columns,
    )

    # Generate prediction
    prediction = model.predict(processed_input)[0]

    # Convert numeric prediction to readable label
    label = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": label
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
