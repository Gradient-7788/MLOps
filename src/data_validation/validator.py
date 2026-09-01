import pandas as pd
from typing import Dict, Any, List
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL, VALIDATION_RULES

class DataValidator:
    def __init__(self, rules: Dict[str, Dict[str, float]] = None):
        self.rules = rules or VALIDATION_RULES
        self.required_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    def validate(self, df: pd.DataFrame, is_training: bool = True) -> Dict[str, Any]:
        report = {
            "is_valid": True,
            "total_rows": len(df),
            "missing_columns": [],
            "null_counts": {},
            "range_violations": {},
            "summary": "Data validation passed successfully."
        }
        
        expected_cols = self.required_columns + ([TARGET_COL] if is_training else [])
        for col in expected_cols:
            if col not in df.columns:
                report["missing_columns"].append(col)
                report["is_valid"] = False

        if report["missing_columns"]:
            report["summary"] = f"Missing required columns: {report['missing_columns']}"
            return report

        # Check null values
        null_counts = df[expected_cols].isnull().sum().to_dict()
        report["null_counts"] = {k: int(v) for k, v in null_counts.items() if v > 0}
        if any(v > 0 for v in null_counts.values()):
            report["is_valid"] = False
            report["summary"] = "Dataset contains missing values."

        # Check numerical ranges
        for col, bound in self.rules.items():
            if col in df.columns:
                min_val, max_val = bound["min"], bound["max"]
                out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_bounds) > 0:
                    report["range_violations"][col] = {
                        "count": len(out_of_bounds),
                        "expected_range": [min_val, max_val],
                        "actual_min": float(df[col].min()),
                        "actual_max": float(df[col].max())
                    }

        if report["range_violations"]:
            # If range violations exist, flag in summary
            report["summary"] = f"Detected out-of-range values in columns: {list(report['range_violations'].keys())}"
            # Note: We can allow soft validation or strict validation
            report["is_valid"] = True # Soft warning so pipeline handles or clips outliers

        return report
