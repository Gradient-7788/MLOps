import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import generate_synthetic_loan_data
from src.data_validation.validator import DataValidator
from src.feature_pipeline.pipeline import FeaturePipeline
from src.training.trainer import ModelTrainer
from src.hpo.optimizer import HyperparameterOptimizer
from src.experiment_tracking.tracker import ExperimentTracker
from src.model_registry.registry import ModelRegistry
from src.drift_detection.drift_detector import DriftDetector
from src.auto_retraining.retrainer import AutoRetrainer

def test_data_loader():
    df = generate_synthetic_loan_data(n_samples=100, inject_drift=False)
    assert len(df) == 100
    assert "is_default" in df.columns
    assert "credit_score" in df.columns

def test_data_validator():
    df = generate_synthetic_loan_data(n_samples=100, inject_drift=False)
    validator = DataValidator()
    report = validator.validate(df, is_training=True)
    assert report["is_valid"] is True
    assert len(report["missing_columns"]) == 0

def test_feature_pipeline():
    df = generate_synthetic_loan_data(n_samples=100, inject_drift=False)
    pipeline = FeaturePipeline()
    X, feature_names = pipeline.fit_transform(df)
    assert X.shape[0] == 100
    assert len(feature_names) > 0

    # Test transform on new batch
    X_new = pipeline.transform(df)
    assert X_new.shape == X.shape

def test_model_trainer_and_hpo():
    df = generate_synthetic_loan_data(n_samples=150, inject_drift=False)
    pipeline = FeaturePipeline()
    X, _ = pipeline.fit_transform(df)
    y = df["is_default"].values

    optimizer = HyperparameterOptimizer(n_trials=3, timeout=10)
    best_params, best_score = optimizer.optimize(X, y, model_type="GradientBoosting")
    assert isinstance(best_params, dict)

    trainer = ModelTrainer(model_type="GradientBoosting", params=best_params)
    trainer.train(X, y)
    metrics = trainer.evaluate(X, y)
    assert metrics["roc_auc"] >= 0.5
    assert "accuracy" in metrics

def test_experiment_tracker():
    tracker = ExperimentTracker(experiment_name="test_experiment")
    run_id = tracker.log_run(
        model_type="RandomForest",
        params={"n_estimators": 50},
        metrics={"roc_auc": 0.85},
        dataset_info={"n_samples": 100}
    )
    assert run_id.startswith("run_")
    run_data = tracker.get_run(run_id)
    assert run_data["metrics"]["roc_auc"] == 0.85

def test_model_registry():
    df = generate_synthetic_loan_data(n_samples=100, inject_drift=False)
    pipeline = FeaturePipeline()
    X, _ = pipeline.fit_transform(df)
    y = df["is_default"].values
    
    trainer = ModelTrainer(model_type="LogisticRegression")
    trainer.train(X, y)
    metrics = trainer.evaluate(X, y)

    registry = ModelRegistry()
    record = registry.register_model(
        model_obj=trainer.model,
        feature_pipeline=pipeline,
        model_type="LogisticRegression",
        metrics=metrics,
        params={},
        run_id="test_run_123"
    )
    assert record["model_id"].startswith("model_v")
    
    prod_model, prod_pipe, prod_meta = registry.get_production_model()
    assert prod_model is not None
    assert prod_meta["model_id"] == record["model_id"]

def test_drift_detector():
    df_ref = generate_synthetic_loan_data(n_samples=300, inject_drift=False)
    df_curr = generate_synthetic_loan_data(n_samples=300, inject_drift=True)
    
    detector = DriftDetector()
    report = detector.detect_drift(df_ref, df_curr)
    assert "drift_detected" in report
    assert "feature_reports" in report
    assert report["max_psi"] > 0.0

def test_auto_retrainer():
    retrainer = AutoRetrainer()
    result = retrainer.run_pipeline(n_samples=150, run_hpo=False)
    assert result["status"] == "SUCCESS"
    assert "run_id" in result
    assert len(result["logs"]) > 0
