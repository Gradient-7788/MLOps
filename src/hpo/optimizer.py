import optuna
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

class HyperparameterOptimizer:
    def __init__(self, n_trials: int = 15, timeout: int = 60):
        self.n_trials = n_trials
        self.timeout = timeout

    def optimize(self, X: np.ndarray, y: np.ndarray, model_type: str = "GradientBoosting") -> Tuple[Dict[str, Any], float]:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def objective(trial: optuna.Trial) -> float:
            if model_type == "GradientBoosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "random_state": 42
                }
                model_cls = GradientBoostingClassifier
            elif model_type == "RandomForest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 4, 12),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "random_state": 42
                }
                model_cls = RandomForestClassifier
            else:
                params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True), "max_iter": 500, "random_state": 42}
                model_cls = None

            scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]

                if model_cls:
                    clf = model_cls(**params)
                else:
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(**params)

                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_va)[:, 1]
                scores.append(roc_auc_score(y_va, probs))

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_score = study.best_value
        return best_params, best_score
