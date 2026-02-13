from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from src.pipeline import run_knn_pipeline

def main():
    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target

    # Try a few k values (simple model selection demo)
    ks = [1, 3, 5, 7, 9, 11]
    accs = []
    for k in ks:
        result = run_knn_pipeline(X, y, k=k, distance="euclidean", test_size=0.2, seed=42, standardize=True)
        accs.append(result.accuracy)

    best_k = ks[int(np.argmax(accs))]
    best_acc = max(accs)

    print("KNN (from scratch) — Iris dataset")
    print(f"Best k = {best_k} with accuracy = {best_acc:.3f}")
    print("\nConfusion matrix (best k):")
    result = run_knn_pipeline(X, y, k=best_k, distance="euclidean", test_size=0.2, seed=42, standardize=True)
    print(result.confusion_matrix)
    print("\nClassification report:")
    print(result.classification_report)

    # Plot accuracy vs k
    plt.figure()
    plt.plot(ks, accs, marker="o")
    plt.xlabel("k (neighbors)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k on Iris (test split)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
