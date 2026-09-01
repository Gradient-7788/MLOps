import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pathlib import Path

from src.data.data_loader import generate_synthetic_loan_data, save_dataset, load_dataset
from src.data_validation.validator import DataValidator
from src.feature_pipeline.pipeline import FeaturePipeline
from src.training.trainer import ModelTrainer
from src.hpo.optimizer import HyperparameterOptimizer
from src.experiment_tracking.tracker import ExperimentTracker
from src.model_registry.registry import ModelRegistry
from src.monitoring.monitor import ModelMonitor
from src.drift_detection.drift_detector import DriftDetector
from src.auto_retraining.retrainer import AutoRetrainer

app = FastAPI(
    title="End-to-End Production MLOps Platform",
    description="Automated Model Training, HPO, Experiment Tracking, Registry, Deployment, Monitoring, Drift Detection & Retraining",
    version="1.0.0"
)

# Setup Templates
templates_dir = Path(__file__).resolve().parent / "templates"
templates_dir.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize Core Services
registry = ModelRegistry()
monitor = ModelMonitor()
drift_detector = DriftDetector()
retrainer = AutoRetrainer()
tracker = ExperimentTracker()
validator = DataValidator()

# Check if an initial model exists; if not, train baseline model
def ensure_baseline_model():
    try:
        registry.get_production_model()
    except Exception:
        print("[MLOps Init] No baseline model found. Training initial baseline model...")
        retrainer.run_pipeline(n_samples=2000, inject_drift=False, model_type="GradientBoosting", run_hpo=True)

ensure_baseline_model()

# Pydantic Schemas
class LoanPredictionInput(BaseModel):
    credit_score: float = Field(..., example=720.0, ge=300, le=850)
    annual_income: float = Field(..., example=75000.0, ge=0)
    debt_to_income: float = Field(..., example=18.5, ge=0, le=100)
    employment_length_years: float = Field(..., example=6.0, ge=0, le=50)
    loan_amount: float = Field(..., example=15000.0, ge=500, le=200000)
    revolving_utilization: float = Field(..., example=0.25, ge=0.0, le=2.0)
    home_ownership: str = Field(..., example="MORTGAGE")
    loan_intent: str = Field(..., example="PERSONAL")

class RetrainRequest(BaseModel):
    n_samples: int = Field(default=2500, ge=500, le=20000)
    inject_drift: bool = Field(default=False)
    model_type: str = Field(default="GradientBoosting")
    run_hpo: bool = Field(default=True)

class PromoteRequest(BaseModel):
    model_id: str
    target_stage: str # "Production", "Staging", "Archived"

# API Endpoints
@app.get("/", response_class=HTMLResponse)
def get_dashboard(request: Request):
    html_path = templates_dir / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/predict")
def predict_single(input_data: LoanPredictionInput):
    start_time = time.time()
    try:
        model, pipeline, meta = registry.get_production_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")

    df_input = pd.DataFrame([input_data.model_dump()])
    
    # Feature transformation
    X_trans = pipeline.transform(df_input)
    
    # Inference
    prediction = int(model.predict(X_trans)[0])
    probability = float(model.predict_proba(X_trans)[0][1])
    
    latency_ms = (time.time() - start_time) * 1000.0
    
    # Log prediction to monitoring store
    monitor.log_prediction(
        features=input_data.model_dump(),
        prediction=prediction,
        probability=probability,
        latency_ms=latency_ms,
        model_version=meta["model_id"]
    )
    
    return {
        "is_default_prediction": prediction,
        "default_probability": round(probability, 4),
        "risk_level": "HIGH RISK" if probability > 0.45 else ("MEDIUM RISK" if probability > 0.20 else "LOW RISK"),
        "model_version": meta["model_id"],
        "latency_ms": round(latency_ms, 2)
    }

@app.post("/pipeline/train")
def train_pipeline(req: RetrainRequest):
    result = retrainer.run_pipeline(
        n_samples=req.n_samples,
        inject_drift=req.inject_drift,
        model_type=req.model_type,
        run_hpo=req.run_hpo
    )
    return result

@app.post("/pipeline/simulate-drift")
def simulate_drift(n_samples: int = 1500, inject_drift: bool = True):
    # Load baseline dataset
    try:
        df_ref = load_dataset("latest_train.csv")
    except Exception:
        df_ref = generate_synthetic_loan_data(n_samples=2000, inject_drift=False)
        save_dataset(df_ref, "latest_train.csv")

    # Generate current production batch with optional drift
    df_curr = generate_synthetic_loan_data(n_samples=n_samples, inject_drift=inject_drift)
    save_dataset(df_curr, "latest_prod_batch.csv")

    drift_report = drift_detector.detect_drift(df_ref, df_curr)
    
    # If severe drift detected, send warning in response
    return {
        "drift_report": drift_report,
        "n_samples": n_samples,
        "drift_injected": inject_drift
    }

@app.post("/pipeline/auto-retrain")
def trigger_auto_retrain():
    # Check if drift exists in production batch
    try:
        df_ref = load_dataset("latest_train.csv")
        df_curr = load_dataset("latest_prod_batch.csv")
        drift_report = drift_detector.detect_drift(df_ref, df_curr)
    except Exception:
        drift_report = {"drift_detected": True, "max_psi": 0.35}

    # Trigger automatic retraining
    retrain_result = retrainer.run_pipeline(
        n_samples=3000,
        inject_drift=False, # Train on clean adapted dataset
        model_type="GradientBoosting",
        run_hpo=True
    )
    
    return {
        "drift_cause": drift_report.get("summary", "Manual trigger"),
        "retraining_result": retrain_result
    }

@app.get("/registry/models")
def list_models():
    return {"models": registry.list_models()}

@app.post("/registry/promote")
def promote_model(req: PromoteRequest):
    updated = registry.transition_stage(req.model_id, req.target_stage)
    return {"status": "SUCCESS", "model": updated}

@app.get("/experiments")
def list_experiments():
    return {"runs": tracker.list_runs()}

@app.get("/monitoring/stats")
def monitoring_stats():
    return monitor.get_summary_metrics()

@app.get("/monitoring/logs")
def monitoring_logs(limit: int = 50):
    return {"logs": monitor.get_recent_predictions(limit=limit)}
