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
        run_hpo: bool = True
    ) -> Dict[str, Any]:
        logs = []
        logs.append(f"[1/7 Data Ingestion] Generating dataset (n_samples={n_samples}, inject_drift={inject_drift})...")
        df = generate_synthetic_loan_data(n_samples=n_samples, random_state=int(time.time()) % 10000, inject_drift=inject_drift)
        save_dataset(df, "latest_train.csv")

        logs.append("[2/7 Data Validation] Running schema and range assertions...")
        val_report = self.validator.validate(df, is_training=True)
        if not val_report["is_valid"]:
            raise ValueError(f"Data validation failed: {val_report['summary']}")

        logs.append("[3/7 Feature Pipeline] Fitting feature transformations & domain engineering...")
        pipeline = FeaturePipeline()
        X, feature_names = pipeline.fit_transform(df)
        y = df["is_default"].values

        # Split train/val
        split_idx = int(len(df) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        params = {}
        if run_hpo:
            logs.append("[4/7 Hyperparameter Optimization] Running Optuna search...")
            optimizer = HyperparameterOptimizer(n_trials=10, timeout=30)
            best_params, best_hpo_score = optimizer.optimize(X_train, y_train, model_type=model_type)
            params = best_params
            logs.append(f"    Optuna found best params: {best_params} (CV AUC: {best_hpo_score:.4f})")

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
            dataset_info={"n_samples": len(df), "num_features": X.shape[1], "drift_injected": inject_drift},
            tags={"trigger": "automated_retraining"}
        )

        logs.append("[7/7 Model Registry & Quality Gate] Evaluating against current production model...")
        promoted = False
        try:
            prod_model, prod_pipe, prod_meta = self.registry.get_production_model()
            prod_auc = prod_meta["metrics"].get("roc_auc", 0.0)
            logs.append(f"    Current Production Model ({prod_meta['model_id']}) ROC-AUC: {prod_auc:.4f}")
            
            # If candidate model ROC-AUC is equal or better, or if forced retraining
            if metrics["roc_auc"] >= prod_auc - 0.02: # Allow small margin or better
                logs.append("    -> Quality gate PASSED! Registering & promoting candidate model to Production stage.")
                registered_record = self.registry.register_model(
                    model_obj=trainer.model,
                    feature_pipeline=pipeline,
                    model_type=model_type,
                    metrics=metrics,
                    params=params,
                    run_id=run_id
                )
                self.registry.transition_stage(registered_record["model_id"], "Production")
                promoted = True
            else:
                logs.append("    -> Candidate model did not beat existing Production model threshold. Registering as Staging.")
                registered_record = self.registry.register_model(
                    model_obj=trainer.model,
                    feature_pipeline=pipeline,
                    model_type=model_type,
                    metrics=metrics,
                    params=params,
                    run_id=run_id
                )
        except RuntimeError:
            # First model in registry
            logs.append("    -> No active production model. Promoting initial model to Production.")
            registered_record = self.registry.register_model(
                model_obj=trainer.model,
                feature_pipeline=pipeline,
                model_type=model_type,
                metrics=metrics,
                params=params,
                run_id=run_id
            )
            promoted = True

        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "model_id": registered_record["model_id"],
            "promoted_to_production": promoted,
            "metrics": metrics,
            "logs": logs
        }
