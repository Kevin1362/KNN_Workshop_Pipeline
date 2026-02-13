from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_classification(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels)
    return acc, cm, report
