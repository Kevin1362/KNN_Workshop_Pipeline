from __future__ import annotations
import pandas as pd
import numpy as np

from src.data_sources.csv_loader import load_csv
from src.pipeline import run_knn_pipeline

def main():
    X_df, y = load_csv("data/iris.csv", target_col="label")

    # Convert to numpy for our from-scratch model
    X = X_df.to_numpy(dtype=float)
    y = y.to_numpy()

    result = run_knn_pipeline(X, y, k=5, distance="euclidean", test_size=0.2, seed=42, standardize=True)

    print("Loaded from data/iris.csv")
    print("Accuracy:", round(result.accuracy, 3))
    print("Confusion matrix:\n", result.confusion_matrix)

if __name__ == "__main__":
    main()
