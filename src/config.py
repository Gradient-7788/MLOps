import os
from pathlib import Path

# MLOPS_HOME lets tests and deployments keep artifacts outside the source tree
BASE_DIR = Path(os.environ.get("MLOPS_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = BASE_DIR / "data_store"
MODELS_DIR = BASE_DIR / "models_store"
EXPERIMENTS_DIR = BASE_DIR / "experiments_store"
MONITORING_DIR = BASE_DIR / "monitoring_store"

for d in [DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR, MONITORING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset columns
NUMERICAL_FEATURES = [
    "credit_score",
    "annual_income",
    "debt_to_income",
    "employment_length_years",
    "loan_amount",
    "revolving_utilization"
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_intent"
]

TARGET_COL = "is_default"

# Validation thresholds
VALIDATION_RULES = {
    "credit_score": {"min": 300, "max": 850},
    "annual_income": {"min": 0, "max": 1_000_000},
    "debt_to_income": {"min": 0.0, "max": 100.0},
    "employment_length_years": {"min": 0, "max": 50},
    "loan_amount": {"min": 500, "max": 200_000},
    "revolving_utilization": {"min": 0.0, "max": 2.0}
}

# Drift Thresholds
DRIFT_PSI_THRESHOLD = 0.20
DRIFT_KS_PVALUE_THRESHOLD = 0.05
DRIFT_KS_STAT_THRESHOLD = 0.10   # KS effect size: with large samples p-values alone flag trivial shifts
DRIFT_CHI2_PVALUE_THRESHOLD = 0.01
