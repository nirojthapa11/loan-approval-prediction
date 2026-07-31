"""
feature_selection.py

Feature selection helpers for the Loan Approval Prediction project.
Used in Day 3 (EDA notebook) and referenced again in Day 4 (train.py)
to keep the final feature list consistent between analysis and modeling.
"""

import pandas as pd


def get_correlation_with_target(df: pd.DataFrame, target: str = "loan_status") -> pd.Series:
    """
    Compute each numeric feature's correlation with the target, sorted
    by absolute strength (strongest predictors first).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned + engineered dataframe (numeric/encoded columns only
        contribute; non-numeric columns are ignored automatically).
    target : str
        Name of the target column.

    Returns
    -------
    pd.Series
        Correlation of each feature with the target, sorted by
        absolute value descending. The target itself is excluded.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def find_highly_correlated_pairs(df: pd.DataFrame, threshold: float = 0.85) -> list:
    """
    Identify pairs of numeric features whose correlation with EACH OTHER
    (not the target) exceeds the given threshold — candidates for
    multicollinearity-driven removal.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        Absolute correlation above which a pair is flagged. 0.85 is a
        common default in practice.

    Returns
    -------
    list of tuple
        (feature_a, feature_b, correlation) for each flagged pair.
    """
    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr().abs()
    pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr_matrix.iloc[i, j]
            if value > threshold:
                pairs.append((columns[i], columns[j], round(value, 3)))
    return sorted(pairs, key=lambda x: -x[2])


def select_features(df: pd.DataFrame, drop_columns: list, target: str = "loan_status"):
    """
    Apply the final feature selection decision: drop the specified
    columns (e.g. redundant asset columns once total_assets_value
    exists, or cibil_band once the raw cibil_score is kept).

    Parameters
    ----------
    df : pd.DataFrame
    drop_columns : list of str
        Columns to remove, decided from EDA (correlation analysis).
    target : str
        Target column name, kept regardless of drop_columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with only the selected features + target.
    """
    columns_to_drop = [c for c in drop_columns if c in df.columns and c != target]
    return df.drop(columns=columns_to_drop)
