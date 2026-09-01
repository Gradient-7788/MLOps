import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List
from src.config import EXPERIMENTS_DIR

class ExperimentTracker:
    def __init__(self, experiment_name: str = "loan_default_prediction"):
        self.experiment_name = experiment_name
        self.exp_dir = EXPERIMENTS_DIR / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_info: Dict[str, Any],
        artifacts: Dict[str, str] = None,
        tags: Dict[str, str] = None
    ) -> str:
        run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        run_data = {
            "run_id": run_id,
            "experiment_name": self.experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type,
            "params": params,
            "metrics": metrics,
            "dataset_info": dataset_info,
            "artifacts": artifacts or {},
            "tags": tags or {}
        }
        
        filepath = self.exp_dir / f"{run_id}.json"
        with open(filepath, "w") as f:
            json.dump(run_data, f, indent=2)
            
        return run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = []
        for file in sorted(self.exp_dir.glob("run_*.json"), reverse=True):
            try:
                with open(file, "r") as f:
                    runs.append(json.load(f))
            except Exception:
                continue
        return runs

    def get_run(self, run_id: str) -> Dict[str, Any]:
        filepath = self.exp_dir / f"{run_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Run {run_id} not found.")
        with open(filepath, "r") as f:
            return json.load(f)
