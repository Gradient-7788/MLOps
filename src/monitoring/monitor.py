import json
import time
from pathlib import Path
from typing import Dict, Any, List
from src.config import MONITORING_DIR

class ModelMonitor:
    def __init__(self):
        self.logs_file = MONITORING_DIR / "prediction_logs.jsonl"

    def log_prediction(self, features: Dict[str, Any], prediction: int, probability: float, latency_ms: float, model_version: str):
        record = {
            "timestamp": time.time(),
            "date_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "features": features,
            "prediction": int(prediction),
            "probability": float(probability),
            "latency_ms": float(latency_ms),
            "model_version": model_version
        }
        with open(self.logs_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_recent_predictions(self, limit: int = 500) -> List[Dict[str, Any]]:
        if not self.logs_file.exists():
            return []
        
        logs = []
        with open(self.logs_file, "r") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    logs.append(json.loads(line.strip()))
                except Exception:
                    continue
        return logs

    def get_summary_metrics(self) -> Dict[str, Any]:
        logs = self.get_recent_predictions(1000)
        if not logs:
            return {
                "total_predictions": 0,
                "avg_latency_ms": 0.0,
                "default_rate": 0.0,
                "avg_confidence": 0.0
            }

        latencies = [l["latency_ms"] for l in logs]
        preds = [l["prediction"] for l in logs]
        probs = [l["probability"] for l in logs]

        return {
            "total_predictions": len(logs),
            "avg_latency_ms": round(float(sum(latencies) / len(latencies)), 2),
            "p95_latency_ms": round(float(sorted(latencies)[int(len(latencies) * 0.95)]), 2) if len(latencies) > 5 else round(float(max(latencies)), 2),
            "default_rate": round(float(sum(preds) / len(preds)), 4),
            "avg_confidence": round(float(sum(probs) / len(probs)), 4)
        }
