import time
import pandas as pd
from typing import Dict, Any
from src.data.data_loader import generate_synthetic_loan_data, save_dataset
from src.data_validation.validator import DataValidator
from src.feature_pipeline.pipeline import FeaturePipeline
from src.hpo.optimizer import HyperparameterOptimizer
from src.training.trainer import ModelTrainer
from src.experiment_tracking.tracker import ExperimentTracker
from src.model_registry.registry import ModelRegistry
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class AutoRetrainer:
    def __init__(self):
        self.validator = DataValidator()
        self.tracker = ExperimentTracker(experiment_name="auto_retraining_runs")
        self.registry = ModelRegistry()

    def run_pipeline(
        self,
        n_samples: int = 3000,
        inject_drift: bool = False,
        model_type: str = "GradientBoosting",
        run_hpo: bool = True,
        df: pd.DataFrame = None,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """model_type: GradientBoosting | RandomForest | LogisticRegression | auto
        (auto tunes all three with Optuna and keeps the family with the best CV ROC-AUC).
        df: train on this dataframe (e.g. a recent production window) instead of generating one."""
        logs = []
        if df is None:
            logs.append(f"[1/7 Data Ingestion] Generating dataset (n_samples={n_samples}, inject_drift={inject_drift})...")
            df = generate_synthetic_loan_data(n_samples=n_samples, random_state=int(time.time()) % 10000, inject_drift=inject_drift)
        else:
            logs.append(f"[1/7 Data Ingestion] Using supplied dataset ({len(df)} rows)...")
        save_dataset(df, "latest_train.csv")

        logs.append("[2/7 Data Validation] Running schema and range assertions...")
        val_report = self.validator.validate(df, is_training=True)
        if not val_report["is_valid"]:
            raise ValueError(f"Data validation failed: {val_report['summary']}")

        # Stratified hold-out split BEFORE fitting any transformation (no validation leakage)
        df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["is_default"], random_state=42)
        y_train, y_val = df_train["is_default"].values, df_val["is_default"].values

        logs.append("[3/7 Feature Pipeline] Fitting transformations on the training split only...")
        pipeline = FeaturePipeline()
        X_train, feature_names = pipeline.fit_transform(df_train)
        X_val = pipeline.transform(df_val)

        params, cv_scores = {}, {}
        families = ["GradientBoosting", "RandomForest", "LogisticRegression"] if model_type == "auto" else [model_type]
        if run_hpo:
            logs.append(f"[4/7 Hyperparameter Optimization] Optuna TPE, stratified 3-fold CV, families={families}...")
            optimizer = HyperparameterOptimizer(n_trials=n_trials, timeout=60)
            best = {}
            for fam in families:
                fam_params, fam_score = optimizer.optimize(X_train, y_train, model_type=fam)
                best[fam] = (fam_params, fam_score)
                cv_scores[fam] = round(fam_score, 4)
                logs.append(f"    {fam}: best CV ROC-AUC {fam_score:.4f} params={fam_params}")
            model_type = max(best, key=lambda f: best[f][1])
            params = best[model_type][0]
            if model_type == "LogisticRegression":
                params = {**params, "max_iter": 500}
        elif model_type == "auto":
            model_type = "GradientBoosting"

        logs.append(f"[5/7 Training & Evaluation] Training {model_type} model...")
        trainer = ModelTrainer(model_type=model_type, params=params)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_val, y_val)
        logs.append(f"    Validation Metrics -> ROC-AUC: {metrics['roc_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

        logs.append("[6/7 Experiment Tracking] Logging run parameters, metrics, and metadata...")
        run_id = self.tracker.log_run(
            model_type=model_type,
            params=params,
            metrics=metrics,
            dataset_info={"n_samples": len(df), "num_features": X_train.shape[1], "drift_injected": inject_drift,
                          "cv_roc_auc_by_family": cv_scores},
            tags={"trigger": "automated_retraining"}
        )

        logs.append("[7/7 Model Registry & Quality Gate] Champion vs challenger on the SAME validation split...")
        promoted = False
        try:
            prod_model, prod_pipe, prod_meta = self.registry.get_production_model()
            champ_probs = prod_model.predict_proba(prod_pipe.transform(df_val))[:, 1]
            champ_auc = float(roc_auc_score(y_val, champ_probs))
            logs.append(f"    Champion {prod_meta['model_id']} ROC-AUC on this validation split: {champ_auc:.4f} | challenger: {metrics['roc_auc']:.4f}")
            registered_record = self.registry.register_model(
                model_obj=trainer.model, feature_pipeline=pipeline, model_type=model_type,
                metrics=metrics, params=params, run_id=run_id)
            if metrics["roc_auc"] >= champ_auc:
                logs.append("    -> Quality gate PASSED: challenger promoted to Production.")
                self.registry.transition_stage(registered_record["model_id"], "Production")
                promoted = True
            else:
                logs.append("    -> Challenger did not beat the champion: registered as Staging.")
        except RuntimeError:
            logs.append("    -> No active production model. Promoting initial model to Production.")
            registered_record = self.registry.register_model(
                model_obj=trainer.model, feature_pipeline=pipeline, model_type=model_type,
                metrics=metrics, params=params, run_id=run_id)
            promoted = True

        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "model_id": registered_record["model_id"],
            "model_type": model_type,
            "promoted_to_production": promoted,
            "metrics": metrics,
            "cv_roc_auc_by_family": cv_scores,
            "logs": logs
        }
