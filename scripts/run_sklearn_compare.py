from __future__ import annotations
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)  # Euclidean
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    print("KNN (scikit-learn) — Iris dataset")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
