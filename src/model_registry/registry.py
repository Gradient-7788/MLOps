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
        self._cache: Dict[str, Tuple[Any, Any]] = {}
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
            "stage": "Staging" if version > 1 else "Production",  # first model bootstraps Production
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type,
            "metrics": metrics,
            "params": params,
            "run_id": run_id,
            # stored relative to the registry dir so the registry is portable across machines
            "model_path": model_file.name,
            "pipeline_path": pipeline_file.name
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

        return self._load(prod_meta)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() and p.exists() else self.registry_dir / p.name

    def _load(self, meta: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
        """Unpickle a registered model + feature pipeline, cached by model_id so serving does not
        hit the disk on every request."""
        mid = meta["model_id"]
        if mid not in self._cache:
            with open(self._resolve(meta["model_path"]), "rb") as f:
                model_obj = pickle.load(f)
            from src.feature_pipeline.pipeline import FeaturePipeline
            pipeline_obj = FeaturePipeline.load(str(self._resolve(meta["pipeline_path"])))
            self._cache[mid] = (model_obj, pipeline_obj)
        model_obj, pipeline_obj = self._cache[mid]
        return model_obj, pipeline_obj, meta

    def list_models(self) -> List[Dict[str, Any]]:
        meta = self._read_metadata()
        return meta["models"]
