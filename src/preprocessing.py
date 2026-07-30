"""
preprocessing.py

Data cleaning functions for the Loan Approval Prediction project.

Design note: scaling is intentionally NOT performed here. Fitting a scaler
on the full dataset before a train/test split leaks test-set statistics
into training. Scaling is fit only on the training split, inside
`train.py` (Day 4). This module handles everything that's safe to do
before that split: dropping identifiers, fixing invalid values, removing
duplicates, and deterministic categorical encoding.
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw loan approval CSV and normalize whitespace.

    The Kaggle source file has leading whitespace in both column names
    and string values (e.g. " Approved" instead of "Approved").

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def drop_identifier_columns(df: pd.DataFrame, columns=("loan_id",)) -> pd.DataFrame:
    """
    Drop columns that are pure row identifiers with no predictive value.

    Parameters
    ----------
    df : pd.DataFrame
    columns : tuple of str
        Column names to drop, if present.

    Returns
    -------
    pd.DataFrame
    """
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing)


def fix_negative_asset_values(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Clip negative values in asset/financial columns to 0.

    A negative asset value (e.g. residential_assets_value = -100000) is
    not physically meaningful and is treated as a data entry artifact
    rather than dropped, since it's a single-column issue on an
    otherwise valid row.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to check. Defaults to the four asset value columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if columns is None:
        columns = [
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ]
    for col in columns:
        if col in df.columns:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                df[col] = df[col].clip(lower=0)
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, if any."""
    return df.drop_duplicates().reset_index(drop=True)


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply fixed, deterministic mappings to categorical columns.

    This is safe to do before the train/test split because the mapping
    is fixed (not fitted from the data's distribution), unlike scaling.

    Mappings:
        education:      Graduate -> 1, Not Graduate -> 0
        self_employed:  Yes -> 1, No -> 0
        loan_status:    Approved -> 1, Rejected -> 0   (target)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if "education" in df.columns:
        df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})

    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})

    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    return df


def clean_data(path: str) -> pd.DataFrame:
    """
    Run the full Day 2 cleaning pipeline end to end.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe: identifiers dropped, negative asset values
        fixed, duplicates removed, categorical columns encoded.
    """
    df = load_raw_data(path)
    df = drop_identifier_columns(df)
    df = fix_negative_asset_values(df)
    df = remove_duplicate_rows(df)
    df = encode_categorical_columns(df)
    return df
