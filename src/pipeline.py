from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import train_test_split_np, standardize_fit, standardize_transform
from .knn import KNNClassifier
from .evaluation import evaluate_classification

@dataclass
class PipelineResult:
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str
    k: int

def run_knn_pipeline(X: np.ndarray, y: np.ndarray, *, k: int = 5, distance: str = "euclidean",
                     test_size: float = 0.2, seed: int = 42, standardize: bool = True) -> PipelineResult:
    # 1) Split
    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=test_size, seed=seed)

    # 2) Preprocess
    if standardize:
        mean, std = standardize_fit(X_train)
        X_train = standardize_transform(X_train, mean, std)
        X_test = standardize_transform(X_test, mean, std)

    # 3) Train
    model = KNNClassifier(k=k, distance=distance).fit(X_train, y_train)

    # 4) Predict
    y_pred = model.predict(X_test)

    # 5) Evaluate
    acc, cm, report = evaluate_classification(y_test, y_pred)

    return PipelineResult(accuracy=acc, confusion_matrix=cm, classification_report=report, k=k)
