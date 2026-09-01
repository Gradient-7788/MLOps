import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
)

class ModelTrainer:
    MODEL_TYPES = {
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression
    }

    def __init__(self, model_type: str = "GradientBoosting", params: Dict[str, Any] = None):
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {model_type}. Allowed: {list(self.MODEL_TYPES.keys())}")
        self.model_type = model_type
        self.params = params or {}
        self.model = self.MODEL_TYPES[model_type](**self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else y_pred
        
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "roc_auc": float(roc_auc_score(y, y_prob)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "log_loss": float(log_loss(y, y_prob)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
