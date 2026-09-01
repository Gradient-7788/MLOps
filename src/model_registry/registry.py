import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from src.config import MODELS_DIR

class ModelRegistry:
    def __init__(self):
        self.registry_dir = MODELS_DIR / "registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "models_meta.json"
        self._init_metadata()

    def _init_metadata(self):
        if not self.metadata_file.exists():
            with open(self.metadata_file, "w") as f:
                json.dump({"models": []}, f, indent=2)

    def _read_metadata(self) -> Dict[str, Any]:
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def _write_metadata(self, meta: Dict[str, Any]):
        with open(self.metadata_file, "w") as f:
            json.dump(meta, f, indent=2)

    def register_model(
        self,
        model_obj: Any,
        feature_pipeline: Any,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        run_id: str
    ) -> Dict[str, Any]:
        meta = self._read_metadata()
        version = len(meta["models"]) + 1
        model_id = f"model_v{version}"
        
        model_file = self.registry_dir / f"{model_id}.pkl"
        pipeline_file = self.registry_dir / f"{model_id}_pipeline.pkl"

        with open(model_file, "wb") as f:
            pickle.dump(model_obj, f)
            
        feature_pipeline.save(str(pipeline_file))

        record = {
            "model_id": model_id,
            "version": version,
            "stage": "Staging" if version > 1 else "Production", # Auto set v1 to Production
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type,
            "metrics": metrics,
            "params": params,
            "run_id": run_id,
            "model_path": str(model_file),
            "pipeline_path": str(pipeline_file)
        }

        # If this is set to Production, archive old production models
        if record["stage"] == "Production":
            for m in meta["models"]:
                if m["stage"] == "Production":
                    m["stage"] = "Archived"

        meta["models"].append(record)
        self._write_metadata(meta)
        return record

    def transition_stage(self, model_id: str, new_stage: str) -> Dict[str, Any]:
        if new_stage not in ["None", "Staging", "Production", "Archived"]:
            raise ValueError(f"Invalid stage: {new_stage}")

        meta = self._read_metadata()
        target_model = None

        if new_stage == "Production":
            # Demote existing Production models to Archived
            for m in meta["models"]:
                if m["stage"] == "Production":
                    m["stage"] = "Archived"

        for m in meta["models"]:
            if m["model_id"] == model_id:
                m["stage"] = new_stage
                target_model = m
                break

        if not target_model:
            raise FileNotFoundError(f"Model ID {model_id} not found in registry.")

        self._write_metadata(meta)
        return target_model

    def get_production_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        meta = self._read_metadata()
        prod_meta = None
        for m in meta["models"]:
            if m["stage"] == "Production":
                prod_meta = m
                break

        if not prod_meta:
            # Fallback to latest model if no model marked as Production
            if meta["models"]:
                prod_meta = meta["models"][-1]
            else:
                raise RuntimeError("No models registered in Model Registry.")

        with open(prod_meta["model_path"], "rb") as f:
            model_obj = pickle.load(f)

        from src.feature_pipeline.pipeline import FeaturePipeline
        pipeline_obj = FeaturePipeline.load(prod_meta["pipeline_path"])

        return model_obj, pipeline_obj, prod_meta

    def list_models(self) -> List[Dict[str, Any]]:
        meta = self._read_metadata()
        return meta["models"]
