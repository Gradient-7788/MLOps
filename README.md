# Production MLOps Platform

### Automated Model Training • Hyperparameter Optimization • Model Registry • Deployment • Monitoring • Drift Detection • Auto-Retraining

A production-oriented **end-to-end MLOps platform** that demonstrates how a machine learning model can move from data ingestion and validation to optimized training, experiment tracking, model registration, real-time inference, production monitoring, statistical drift detection, and automated retraining.

The platform is built around a synthetic **loan-default prediction** problem and is designed to reproduce the core engineering components required to operate ML systems beyond the experimentation stage.

---

## 🚀 Overview

Building a high-performing ML model is only one part of a production ML system.

A deployed model must also be:

* reproducible
* versioned
* monitored
* evaluated continuously
* robust to changing data distributions
* automatically retrained when its assumptions become stale
* accessible through a production inference API

This project implements these capabilities as a modular MLOps platform.

```text
                         ┌─────────────────────┐
                         │    Data Generation  │
                         │   / Data Ingestion   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Validation   │
                         │ Schema + Quality     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         │   Transformations    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Hyperparameter Optimization  │
                    │           Optuna              │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   Model Training    │
                         │ GB / RF / Logistic  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Experiment Tracking │
                         │ Metrics + Parameters │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Model Registry   │
                         │ Version + Lifecycle  │
                         └──────────┬──────────┘
                                    │
                              Production
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    FastAPI Serving  │
                         │   Real-time /predict│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Production Monitor  │
                         │ Latency + Predictions│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Drift Detection   │
                         │ PSI + KS + Wasserstein│
                         └──────────┬──────────┘
                                    │
                             Significant Drift
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Auto-Retraining   │
                         │ HPO → Train → Register│
                         └─────────────────────┘
```

---

## ✨ Key Capabilities

### 1. End-to-End ML Pipeline

The platform integrates the complete ML lifecycle rather than treating training and deployment as separate scripts.

**Pipeline:**

```text
Data
 ↓
Validation
 ↓
Feature Engineering
 ↓
Hyperparameter Optimization
 ↓
Training
 ↓
Evaluation
 ↓
Experiment Tracking
 ↓
Model Registration
 ↓
Production Deployment
 ↓
Monitoring
 ↓
Drift Detection
 ↓
Automatic Retraining
```

---

### 2. Multiple Model Architectures

The training framework supports multiple classical ML models behind a common training interface:

* **Gradient Boosting Classifier**
* **Random Forest Classifier**
* **Logistic Regression**

The trainer exposes a unified training/evaluation interface, allowing models to be swapped without changing the surrounding MLOps pipeline.

Evaluation includes:

* Accuracy
* ROC-AUC
* Precision
* Recall
* F1 score
* Log loss
* Confusion matrix components

This makes model comparison reproducible and suitable for automated experiment tracking.

---

### 3. Automated Hyperparameter Optimization

The platform integrates **Optuna** for automated hyperparameter search.

Instead of relying on manually selected parameters, the optimizer performs cross-validated optimization using **Stratified K-Fold validation** and maximizes ROC-AUC.

Example search spaces include:

**Gradient Boosting**

```text
n_estimators
learning_rate
max_depth
subsample
```

**Random Forest**

```text
n_estimators
max_depth
min_samples_split
```

The optimizer returns the best parameter configuration together with its validation score, which can then be passed directly into the training pipeline.

---

### 4. Experiment Tracking

Each training run can be persisted in a dedicated experiment store.

Tracked information includes:

* model type
* hyperparameters
* evaluation metrics
* training metadata
* model identifiers
* experiment results

This provides reproducibility and makes it possible to compare different training runs rather than treating every model as an isolated artifact.

---

### 5. Model Registry & Lifecycle Management

Models are persisted in a dedicated registry with explicit lifecycle stages.

Supported stages include:

```text
Staging
Production
Archived
```

The registry allows the system to:

* register trained models
* maintain model versions
* retrieve the production model
* transition models between lifecycle stages
* separate experimental models from production models

This creates a controlled path from experimentation to deployment.

---

### 6. Production Inference API

The trained production model is exposed through **FastAPI**.

The primary inference endpoint is:

```http
POST /predict
```

A prediction request contains borrower information such as:

```json
{
  "credit_score": 720,
  "annual_income": 75000,
  "debt_to_income": 18.5,
  "employment_length_years": 6,
  "loan_amount": 15000,
  "revolving_utilization": 0.25,
  "home_ownership": "MORTGAGE",
  "loan_intent": "PERSONAL"
}
```

The API returns:

```json
{
  "is_default_prediction": 0,
  "default_probability": 0.0834,
  "risk_level": "LOW RISK",
  "model_version": "...",
  "latency_ms": 4.21
}
```

The inference layer also records prediction metadata for downstream monitoring.

---

## 📊 Production Monitoring

Every inference request records operational information such as:

* prediction
* predicted probability
* model version
* input features
* inference latency

The platform exposes monitoring endpoints for retrieving:

```text
Prediction logs
Aggregate statistics
Recent inference activity
```

This makes the model observable after deployment rather than treating deployment as the end of the ML lifecycle.

---

# 🔬 Statistical Drift Detection

One of the central components of the platform is automated **data drift detection**.

Production data is compared against a reference training distribution using multiple statistical measures.

### Population Stability Index — PSI

PSI is calculated using percentile-based buckets derived from the reference distribution.

The implementation smooths zero-frequency buckets to avoid numerical instability and calculates the resulting distribution shift.

Conceptually:

```text
Reference Distribution
        │
        ├── Bucketization
        │
        ▼
Current Production Distribution
        │
        ▼
PSI Score
```

### Kolmogorov–Smirnov Test

The system additionally applies the two-sample KS test to determine whether reference and production distributions differ statistically.

### Wasserstein Distance

The Wasserstein distance is also calculated to quantify the magnitude of distributional displacement.

Therefore each monitored feature produces:

```text
PSI
KS statistic
KS p-value
Wasserstein distance
Drift status
Severity
```

A feature is flagged when either the PSI threshold or KS significance threshold is exceeded.

The implementation also classifies drift severity based on PSI magnitude:

```text
LOW
MEDIUM
HIGH
```

This multi-metric approach is stronger than relying on a single drift statistic.

---

# 🔄 Automated Retraining

Drift detection is connected to the retraining pipeline.

The system can:

1. Compare reference and production data
2. Detect statistically significant distribution shift
3. Trigger model retraining
4. Run hyperparameter optimization
5. Train a new model
6. Evaluate the candidate
7. Track the experiment
8. Register the resulting model
9. Make the new model available for lifecycle promotion

This creates the foundation for a closed-loop ML system:

```text
        Production
            │
            ▼
       Monitoring
            │
            ▼
     Drift Detection
            │
      ┌─────┴─────┐
      │           │
   Stable      Drifted
      │           │
      │           ▼
      │      Retraining
      │           │
      │           ▼
      │      HPO + Training
      │           │
      │           ▼
      │      Model Registry
      │           │
      └───────────┘
```

---

# 🏗️ Project Architecture

```text
MLOps/
│
├── data_store/
│   └── datasets and production batches
│
├── experiments_store/
│   └── experiment runs and metrics
│
├── models_store/
│   └── registered model artifacts
│
├── monitoring_store/
│   └── prediction and monitoring logs
│
├── src/
│   │
│   ├── api/
│   │   ├── app.py
│   │   └── templates/
│   │
│   ├── data/
│   │   └── data loading / generation
│   │
│   ├── data_validation/
│   │   └── validation logic
│   │
│   ├── feature_pipeline/
│   │   └── feature transformations
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── hpo/
│   │   └── optimizer.py
│   │
│   ├── experiment_tracking/
│   │   └── experiment tracking
│   │
│   ├── model_registry/
│   │   └── model lifecycle management
│   │
│   ├── monitoring/
│   │   └── prediction monitoring
│   │
│   ├── drift_detection/
│   │   └── drift_detector.py
│   │
│   ├── auto_retraining/
│   │   └── retraining orchestration
│   │
│   └── config.py
│
├── tests/
│   └── test_pipeline.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The repository separates **data, models, experiments, monitoring artifacts, application logic, and tests**, providing a clear boundary between ML artifacts and application code.

---

# 🧰 Technology Stack

| Component                   | Technology                           |
| --------------------------- | ------------------------------------ |
| Language                    | Python                               |
| API                         | FastAPI                              |
| Server                      | Uvicorn                              |
| ML                          | scikit-learn                         |
| Numerical Computing         | NumPy                                |
| Data Processing             | Pandas                               |
| Statistical Testing         | SciPy                                |
| Hyperparameter Optimization | Optuna                               |
| Validation                  | Pydantic                             |
| Containerization            | Docker                               |
| Multi-container Deployment  | Docker Compose                       |
| Testing                     | Pytest                               |
| Model Lifecycle             | Custom Model Registry                |
| Experiment Tracking         | Custom Experiment Tracker            |
| Monitoring                  | Custom Monitoring Layer              |
| Drift Detection             | PSI + KS Test + Wasserstein Distance |

The current dependency stack is intentionally lightweight, using FastAPI, Pandas, NumPy, scikit-learn, SciPy, Optuna and Pydantic rather than requiring a large external MLOps platform.

---

# 🐳 Docker Deployment

The entire application can be containerized using Docker.

The repository includes:

```text
Dockerfile
docker-compose.yml
```

Docker Compose exposes the application on:

```text
http://localhost:8000
```

Persistent directories are mounted for:

```text
data_store
models_store
experiments_store
monitoring_store
```

This allows trained models, experiments, datasets and monitoring information to survive container restarts.

---

# ⚙️ Installation

## Option 1 — Local Python Environment

Clone the repository:

```bash
git clone https://github.com/Gradient-7788/MLOps.git
cd MLOps
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Option 2 — Docker

Build and launch the platform:

```bash
docker compose up --build
```

The API will then be available at:

```text
http://localhost:8000
```

---

# ▶️ Running the API

Start the FastAPI application with:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

FastAPI's interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

# 🔌 API Reference

## Health Check

```http
GET /health
```

Returns the current service health status.

---

## Model Prediction

```http
POST /predict
```

Runs inference using the currently registered production model.

The endpoint performs:

```text
Request Validation
       ↓
Feature Transformation
       ↓
Production Model Loading
       ↓
Prediction
       ↓
Probability Estimation
       ↓
Monitoring Log
       ↓
Response
```

---

## Train a Model

```http
POST /pipeline/train
```

Supports:

* configurable sample size
* model selection
* optional drift injection
* optional hyperparameter optimization

Example:

```json
{
  "n_samples": 2500,
  "inject_drift": false,
  "model_type": "GradientBoosting",
  "run_hpo": true
}
```

---

## Simulate Production Drift

```http
POST /pipeline/simulate-drift
```

Generates a production batch and compares it against the reference training distribution.

This endpoint is useful for demonstrating the complete monitoring → drift detection workflow.

---

## Trigger Auto-Retraining

```http
POST /pipeline/auto-retrain
```

Checks production data for drift and executes the retraining pipeline.

---

## Model Registry

List registered models:

```http
GET /registry/models
```

Promote a model:

```http
POST /registry/promote
```

Supported lifecycle stages:

```text
Staging
Production
Archived
```

---

## Experiment Tracking

```http
GET /experiments
```

Returns recorded experiment runs.

---

## Monitoring

Retrieve aggregate monitoring statistics:

```http
GET /monitoring/stats
```

Retrieve recent prediction logs:

```http
GET /monitoring/logs
```

---

# 🧪 Testing

The project contains a dedicated test suite under:

```text
tests/
```

Run tests with:

```bash
pytest
```

The test layer is intended to verify the core pipeline behavior independently of the API deployment layer.

---

# 🔁 Example Production Workflow

A typical production lifecycle looks like this:

### Step 1 — Train

```text
Training Data
      ↓
Validation
      ↓
Feature Pipeline
      ↓
HPO
      ↓
Model Training
```

### Step 2 — Evaluate

```text
Accuracy
ROC-AUC
Precision
Recall
F1
Log Loss
Confusion Matrix
```

### Step 3 — Track

```text
Parameters
Metrics
Model Configuration
Run Metadata
```

### Step 4 — Register

```text
Candidate Model
      ↓
Staging
      ↓
Production
```

### Step 5 — Serve

```text
Client
  ↓
FastAPI
  ↓
Feature Pipeline
  ↓
Production Model
  ↓
Prediction
```

### Step 6 — Monitor

```text
Predictions
Latency
Probabilities
Model Version
```

### Step 7 — Detect Drift

```text
Reference Data
      vs
Production Data
      ↓
PSI
KS Test
Wasserstein Distance
```

### Step 8 — Retrain

```text
Drift Detected
      ↓
New Dataset
      ↓
HPO
      ↓
Training
      ↓
Evaluation
      ↓
Experiment Tracking
      ↓
Model Registry
```

---

# 💡 Engineering Decisions

## Why separate model artifacts from experiments?

A production model and an experiment are fundamentally different objects.

Experiments answer:

> "What happened during this training run?"

The model registry answers:

> "Which model should the production system use?"

Keeping these concerns separate allows experiments to accumulate without accidentally changing the production model.

---

## Why use multiple drift metrics?

No single statistical test completely captures distribution shift.

The platform combines:

* PSI for stability-oriented distribution monitoring
* KS testing for statistical significance
* Wasserstein distance for distributional displacement

This provides complementary evidence before triggering retraining.

---

## Why use HPO inside the retraining loop?

Simply retraining with the previous hyperparameters assumes that the optimal configuration remains unchanged.

The retraining pipeline instead allows the model to search for a better configuration on the new training distribution.

```text
Distribution Shift
       ↓
New Training Data
       ↓
Hyperparameter Search
       ↓
Candidate Models
       ↓
Best Configuration
       ↓
Production Candidate
```

---

# 🎯 What This Project Demonstrates

This project is designed to demonstrate practical ML engineering skills beyond model development.

### Machine Learning

* Binary classification
* Ensemble learning
* Model evaluation
* Cross-validation
* Hyperparameter optimization

### MLOps

* Experiment tracking
* Model versioning
* Model registry
* Model lifecycle management
* Production inference
* Monitoring
* Data drift detection
* Automated retraining

### Software Engineering

* Modular Python architecture
* Typed interfaces
* REST API design
* Input validation
* Automated testing
* Persistent artifact stores
* Containerized deployment

### Statistical / ML Engineering

* Population Stability Index
* Kolmogorov–Smirnov testing
* Wasserstein distance
* Distribution-shift analysis
* Probability-based model monitoring

---

# 📈 Why This Matters

A conventional ML project often ends here:

```text
Dataset → Model → Accuracy
```

This project goes substantially further:

```text
Dataset
   ↓
Validation
   ↓
Features
   ↓
Optimization
   ↓
Training
   ↓
Evaluation
   ↓
Experiment Tracking
   ↓
Model Registry
   ↓
Production API
   ↓
Monitoring
   ↓
Drift Detection
   ↓
Automatic Retraining
   ↺
```

The objective is to demonstrate that an ML system is not merely a model — it is a **software system with a continuously evolving lifecycle**.

---

# 🔮 Future Improvements

Potential extensions include:

* CI/CD with GitHub Actions
* Cloud object storage for model/data artifacts
* PostgreSQL-backed metadata storage
* MLflow integration
* Kubernetes deployment
* Prometheus/Grafana observability
* Authentication and API authorization
* Canary deployments
* Champion/challenger model evaluation
* Automated model promotion based on evaluation thresholds
* Feature-level drift dashboards
* Model performance monitoring with delayed ground truth
* Scheduled retraining
* Data versioning with DVC
* Infrastructure-as-code with Terraform

---

# 📌 Project Structure at a Glance

```text
                    ┌──────────────────────────┐
                    │       FastAPI API        │
                    └────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
     Prediction             Training Pipeline       Monitoring
          │                      │                      │
          ▼                      ▼                      ▼
      Production            HPO + Training         Prediction Logs
       Model                      │                      │
          │                      ▼                      │
          │               Experiment Tracker             │
          │                      │                      │
          │                      ▼                      │
          │                Model Registry               │
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
                         Drift Detection
                                 │
                         ┌───────┴───────┐
                         │               │
                       Stable         Drifted
                         │               │
                         │               ▼
                         │        Auto-Retraining
                         │               │
                         └───────────────┘
```

---

# 👨‍💻 Author

**Yesh Lohchab**

Machine Learning • AI • MLOps • Quantitative Engineering

---

## ⭐ Project Highlights

> **End-to-end production ML platform implementing automated training, Optuna-based HPO, experiment tracking, model registry, FastAPI inference, production monitoring, PSI/KS/Wasserstein drift detection, and automated retraining in a Dockerized architecture.**

---

## 📄 License

This project is intended as an educational and engineering portfolio project demonstrating production-oriented machine learning system design.
