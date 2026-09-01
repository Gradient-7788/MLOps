import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import ks_2samp, wasserstein_distance
from src.config import NUMERICAL_FEATURES, DRIFT_PSI_THRESHOLD, DRIFT_KS_PVALUE_THRESHOLD

class DriftDetector:
    def __init__(self, psi_threshold: float = DRIFT_PSI_THRESHOLD, ks_threshold: float = DRIFT_KS_PVALUE_THRESHOLD):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    @staticmethod
    def calculate_psi(reference: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        # Prevent division by zero / single constant series
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(reference, percentiles)
        buckets[0] = -np.inf
        buckets[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=buckets)
        curr_counts, _ = np.histogram(current, bins=buckets)

        ref_percents = ref_counts / len(reference)
        curr_percents = curr_counts / len(current)

        # Smooth zero counts
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)

        psi_val = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        return float(psi_val)

    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        feature_reports = {}
        overall_drift_detected = False
        max_psi = 0.0

        for col in NUMERICAL_FEATURES:
            if col in reference_df.columns and col in current_df.columns:
                ref_vals = reference_df[col].dropna().values
                curr_vals = current_df[col].dropna().values

                psi = self.calculate_psi(ref_vals, curr_vals)
                ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
                w_dist = wasserstein_distance(ref_vals, curr_vals)

                feature_drift = (psi > self.psi_threshold) or (p_value < self.ks_threshold)
                if feature_drift:
                    overall_drift_detected = True

                if psi > max_psi:
                    max_psi = psi

                feature_reports[col] = {
                    "psi": round(psi, 4),
                    "ks_p_value": round(float(p_value), 6),
                    "ks_statistic": round(float(ks_stat), 4),
                    "wasserstein_distance": round(float(w_dist), 4),
                    "drift_detected": bool(feature_drift),
                    "severity": "HIGH" if psi > 0.25 else ("MEDIUM" if psi > 0.10 else "LOW")
                }

        return {
            "drift_detected": overall_drift_detected,
            "max_psi": round(max_psi, 4),
            "psi_threshold": self.psi_threshold,
            "feature_reports": feature_reports,
            "summary": "Drift alert triggered! Significant covariate shift detected." if overall_drift_detected else "Data distribution is stable. No drift detected."
        }
