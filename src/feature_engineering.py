"""
feature_engineering.py

Derived feature creation for the Loan Approval Prediction project.
All functions operate on the already-cleaned dataframe (see
preprocessing.clean_data).
"""

import pandas as pd


def add_total_assets_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum the four individual asset columns into one total_assets_value
    feature. A single combined figure is often more predictive on its
    own, and reduces multicollinearity risk from the four correlated
    source columns.
    """
    df = df.copy()
    asset_cols = [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]
    if all(c in df.columns for c in asset_cols):
        df["total_assets_value"] = df[asset_cols].sum(axis=1)
    return df


def add_loan_to_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    loan_amount / income_annum — how large the requested loan is
    relative to the applicant's annual income. A common, intuitive
    signal in real-world credit risk models.
    """
    df = df.copy()
    if "loan_amount" in df.columns and "income_annum" in df.columns:
        df["loan_to_income_ratio"] = df["loan_amount"] / df["income_annum"].replace(0, pd.NA)
        df["loan_to_income_ratio"] = df["loan_to_income_ratio"].fillna(0)
    return df


def add_asset_to_loan_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    total_assets_value / loan_amount — how well the applicant's total
    assets cover the requested loan. Requires total_assets_value to
    already exist (run add_total_assets_value first).
    """
    df = df.copy()
    if "total_assets_value" in df.columns and "loan_amount" in df.columns:
        df["asset_to_loan_ratio"] = df["total_assets_value"] / df["loan_amount"].replace(0, pd.NA)
        df["asset_to_loan_ratio"] = df["asset_to_loan_ratio"].fillna(0)
    return df


def add_cibil_score_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket cibil_score into standard credit-risk bands. Kept as a
    separate categorical feature alongside the raw numeric score,
    since tree-based models can exploit the raw value while the band
    can help simpler models (e.g. Logistic Regression) capture
    non-linear risk thresholds.

    Bands: Poor (<650), Fair (650-749), Good (750-799), Excellent (800+)
    """
    df = df.copy()
    if "cibil_score" in df.columns:
        bins = [0, 649, 749, 799, 900]
        labels = ["Poor", "Fair", "Good", "Excellent"]
        df["cibil_band"] = pd.cut(df["cibil_score"], bins=bins, labels=labels)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full Day 2 feature engineering pipeline end to end.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of preprocessing.clean_data).

    Returns
    -------
    pd.DataFrame
        Dataframe with total_assets_value, loan_to_income_ratio,
        asset_to_loan_ratio, and cibil_band added.
    """
    df = add_total_assets_value(df)
    df = add_loan_to_income_ratio(df)
    df = add_asset_to_loan_ratio(df)
    df = add_cibil_score_band(df)
    return df
