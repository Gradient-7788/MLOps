import pandas as pd
import numpy as np
from typing import Tuple
from src.config import DATA_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

def generate_synthetic_loan_data(n_samples: int = 2500, random_state: int = 42, inject_drift: bool = False) -> pd.DataFrame:
    np.random.seed(random_state)
    
    if not inject_drift:
        credit_score = np.random.normal(700, 50, n_samples).clip(300, 850)
        annual_income = np.random.lognormal(10.8, 0.5, n_samples).clip(15000, 500000)
        debt_to_income = np.random.gamma(2.5, 5, n_samples).clip(1.0, 60.0)
        employment_length = np.random.exponential(5, n_samples).clip(0, 40)
        loan_amount = np.random.uniform(2000, 35000, n_samples)
        revolving_util = np.random.beta(2, 5, n_samples).clip(0.01, 0.99)
        
        home_ownership = np.random.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.45, 0.45, 0.10])
        loan_intent = np.random.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"], size=n_samples)
    else:
        # Inject significant covariate & concept drift (e.g., economic recession simulation)
        credit_score = np.random.normal(610, 75, n_samples).clip(300, 850) # Shifted down
        annual_income = np.random.lognormal(10.2, 0.6, n_samples).clip(10000, 350000) # Lower income
        debt_to_income = np.random.gamma(4.5, 7, n_samples).clip(5.0, 85.0) # Higher DTI ratio
        employment_length = np.random.exponential(3, n_samples).clip(0, 30) # Shorter tenure
        loan_amount = np.random.uniform(5000, 50000, n_samples) # Higher loan requests
        revolving_util = np.random.beta(4, 2, n_samples).clip(0.05, 1.2) # High utilization
        
        home_ownership = np.random.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.60, 0.35, 0.05])
        loan_intent = np.random.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"], size=n_samples, p=[0.4, 0.2, 0.3, 0.05, 0.05])

    # Calculate probabilistic target variable is_default
    logit = (
        -0.015 * (credit_score - 650)
        - 0.000015 * (annual_income - 50000)
        + 0.05 * (debt_to_income - 15)
        - 0.08 * employment_length
        + 0.00004 * (loan_amount - 10000)
        + 1.5 * (revolving_util - 0.3)
    )
    if inject_drift:
        logit += 0.8 # Shift baseline default probability up
        
    prob = 1 / (1 + np.exp(-logit))
    is_default = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "credit_score": np.round(credit_score, 1),
        "annual_income": np.round(annual_income, 2),
        "debt_to_income": np.round(debt_to_income, 2),
        "employment_length_years": np.round(employment_length, 1),
        "loan_amount": np.round(loan_amount, 2),
        "revolving_utilization": np.round(revolving_util, 3),
        "home_ownership": home_ownership,
        "loan_intent": loan_intent,
        TARGET_COL: is_default
    })
    
    return df

def save_dataset(df: pd.DataFrame, filename: str) -> str:
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return str(path)

def load_dataset(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset {filename} not found at {path}")
    return pd.read_csv(path)
