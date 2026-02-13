from __future__ import annotations
import numpy as np
from collections import Counter

class KNNClassifier:
    """KNN classifier implemented from scratch (NumPy).

    Parameters
    ----------
    k : int
        Number of neighbors.
    distance : str
        'euclidean' or 'manhattan'
    """

    def __init__(self, k: int = 5, distance: str = "euclidean"):
        if k <= 0:
            raise ValueError("k must be > 0")
        if distance not in {"euclidean", "manhattan"}:
            raise ValueError("distance must be 'euclidean' or 'manhattan'")
        self.k = int(k)
        self.distance = distance
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X_train = X
        self.y_train = y
        return self

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.distance == "euclidean":
            # sqrt(sum((x - Xi)^2))
            return np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        # manhattan: sum(|x - Xi|)
        return np.abs(self.X_train - x).sum(axis=1)

    def predict_one(self, x: np.ndarray):
        if self.y_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        d = self._compute_distances(x)
        k = min(self.k, len(d))
        nn_idx = np.argpartition(d, k - 1)[:k]
        nn_labels = self.y_train[nn_idx].tolist()
        counts = Counter(nn_labels)
        # deterministic tie-breaker: sort by (-count, label)
        return sorted(counts.items(), key=lambda t: (-t[1], str(t[0])))[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(row) for row in X])
