from __future__ import annotations
import numpy as np

def train_test_split_np(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Simple train/test split using NumPy only."""
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize_fit(X: np.ndarray):
    """Return (mean, std) for standardization."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std
