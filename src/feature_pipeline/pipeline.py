import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR

class FeaturePipeline:
    def __init__(self):
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names: list = []

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Domain feature engineering: ratio of loan amount to annual income
        df["loan_to_income_ratio"] = np.round(df["loan_amount"] / (df["annual_income"] + 1e-5), 4)
        
        # Risk index combining credit score and DTI
        df["credit_dti_risk_index"] = np.round((850 - df["credit_score"]) * df["debt_to_income"] / 100.0, 4)
        
        # Log income transformation to smooth skewed distributions
        df["log_annual_income"] = np.log1p(df["annual_income"])
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, list]:
        df_eng = self._engineer_features(df)
        
        num_cols = NUMERICAL_FEATURES + ["loan_to_income_ratio", "credit_dti_risk_index", "log_annual_income"]
        cat_cols = CATEGORICAL_FEATURES
        
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_cols),
                ("cat", cat_transformer, cat_cols)
            ]
        )
        
        X_transformed = self.pipeline.fit_transform(df_eng)
        
        cat_encoded_cols = list(self.pipeline.named_transformers_["cat"].get_feature_names_out(cat_cols))
        self.feature_names = num_cols + cat_encoded_cols
        
        return X_transformed, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("FeaturePipeline is not fitted yet.")
        df_eng = self._engineer_features(df)
        return self.pipeline.transform(df_eng)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, filepath: str) -> "FeaturePipeline":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.pipeline = data["pipeline"]
        obj.feature_names = data["feature_names"]
        return obj
