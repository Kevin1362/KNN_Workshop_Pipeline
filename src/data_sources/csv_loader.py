from __future__ import annotations
import pandas as pd

def load_csv(path: str, target_col: str):
    """Load a CSV file and return (X_df, y_series)."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found. Available columns: {df.columns.tolist()}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y
