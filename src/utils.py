"""
utils.py

Shared helper functions used across preprocessing, training, and
inference scripts.
"""

import joblib


def save_artifact(obj, path: str) -> None:
    """
    Serialize any Python object (model, scaler, encoder) to disk with joblib.

    Parameters
    ----------
    obj : object
        The object to save.
    path : str
        Destination file path, e.g. 'models/trained_model.pkl'.
    """
    joblib.dump(obj, path)
    print(f"Saved: {path}")


def load_artifact(path: str):
    """
    Load a previously saved artifact (model, scaler, encoder).

    Parameters
    ----------
    path : str
        Path to the .pkl file.

    Returns
    -------
    object
        The deserialized object.
    """
    return joblib.load(path)


def print_section(title: str) -> None:
    """Print a consistent section header, used to keep notebook/script output readable."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
