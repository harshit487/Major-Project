"""
Stage 7: Singleton Model Management & Inference Service.

Responsibilities:
- Loads the trained Keras model (`efficientnetb0_ham10000_binary.keras`) once
  at application startup.
- Verifies that the model file exists on disk before attempting inference.
- Runs preprocessing + forward pass to compute Benign and Malignant probabilities
  that sum to 1.0.
- Invokes Grad-CAM generation when requested.
- Stores recent prediction metadata in memory so PDF reports can be generated
  by `prediction_id` without re-uploading.
"""

from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
from PIL import Image

from backend.config import (
    BINARY_CLASSES,
    CLASS_MAPPING_PATH,
    DATASET_AUDIT_PATH,
    EVALUATION_METRICS_PATH,
    GRADCAM_DIR,
    HAM10000_DIAGNOSIS_DETAILS,
    INPUT_SHAPE,
    MEDICAL_DISCLAIMER,
    MODEL_ARCHITECTURE,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHTS_PATH,
    PROJECT_METADATA,
    get_active_decision_threshold,
)
from backend.services.gradcam import generate_and_save_gradcam_artifacts
from backend.services.preprocessing import preprocess_image_for_inference


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested but the trained model file is missing or not loaded."""


class ModelService:
    """
    Thread-safe singleton managing the trained EfficientNetB0 binary model,
    evaluation metadata, and inference pipeline.
    """

    def __init__(self, model_path: Path = MODEL_WEIGHTS_PATH) -> None:
        self._lock = threading.Lock()
        self.model_path: Path = Path(model_path)
        self.model = None
        self.model_loaded: bool = False
        self.load_error: Optional[str] = None
        self.loaded_at: Optional[str] = None
        self._predictions_cache: Dict[str, Dict[str, Any]] = {}

    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the trained Keras model from `self.model_path` if it exists.
        Never fabricates a model or substitutes fake weights.
        """
        with self._lock:
            if self.model_loaded and self.model is not None and not force_reload:
                return True

            if not self.model_path.exists() or not self.model_path.is_file():
                self.model = None
                self.model_loaded = False
                self.load_error = (
                    f"Trained model file not found at '{self.model_path.name}'. "
                    "Please train the model using `python scripts/train_model.py`."
                )
                return False

            try:
                import tensorflow as tf

                self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
                self.model_loaded = True
                self.load_error = None
                self.loaded_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                return True
            except Exception as exc:
                self.model = None
                self.model_loaded = False
                self.load_error = f"Failed to load model weights: {exc}"
                return False

    def ensure_ready(self) -> None:
        """
        Confirm that the model file exists on disk and is loaded in memory
        before performing inference.
        """
        if not self.model_path.exists():
            self.model_loaded = False
            self.model = None
            raise ModelNotLoadedError(
                f"Model weights file '{self.model_path.name}' does not exist before inference. "
                "Train the model first via `python scripts/train_model.py`."
            )
        if not self.model_loaded or self.model is None:
            loaded = self.load_model(force_reload=True)
            if not loaded:
                raise ModelNotLoadedError(
                    self.load_error or "Trained model could not be loaded for inference."
                )

    def get_health_status(self) -> Dict[str, Any]:
        """Return health status for `GET /api/health`."""
        if not self.model_loaded and self.model_path.exists():
            self.load_model()
        return {
            "status": "healthy",
            "api_online": True,
            "model_loaded": bool(self.model_loaded and self.model is not None and self.model_path.exists()),
            "model_architecture": MODEL_ARCHITECTURE,
            "model_file_exists": self.model_path.exists(),
            "status_message": (
                "Trained EfficientNetB0 model loaded and ready for inference."
                if (self.model_loaded and self.model_path.exists())
                else (self.load_error or "Model weights not yet trained.")
            ),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return comprehensive model and evaluation metadata for `GET /api/model-info`."""
        if not self.model_loaded and self.model_path.exists():
            self.load_model()

        threshold = get_active_decision_threshold()

        model_config = None
        if MODEL_CONFIG_PATH.exists():
            try:
                with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                    model_config = json.load(f)
            except Exception:
                model_config = None

        evaluation_metrics = None
        if EVALUATION_METRICS_PATH.exists():
            try:
                with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
                    evaluation_metrics = json.load(f)
            except Exception:
                evaluation_metrics = None

        dataset_audit = None
        if DATASET_AUDIT_PATH.exists():
            try:
                with open(DATASET_AUDIT_PATH, "r", encoding="utf-8") as f:
                    dataset_audit = json.load(f)
            except Exception:
                dataset_audit = None

        return {
            "model_architecture": MODEL_ARCHITECTURE,
            "model_loaded": bool(self.model_loaded and self.model_path.exists()),
            "loaded_at": self.loaded_at,
            "input_dimensions": {
                "height": INPUT_SHAPE[0],
                "width": INPUT_SHAPE[1],
                "channels": INPUT_SHAPE[2],
                "shape": list(INPUT_SHAPE),
            },
            "supported_classes": list(BINARY_CLASSES.values()),
            "class_encoding": {str(k): v for k, v in BINARY_CLASSES.items()},
            "positive_class": "Malignant",
            "decision_threshold": threshold,
            "original_to_binary_mapping": HAM10000_DIAGNOSIS_DETAILS,
            "model_config": model_config,
            "evaluation": evaluation_metrics,
            "dataset_audit": dataset_audit,
            "project_metadata": PROJECT_METADATA,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }

    def predict_image(
        self,
        image_source: Union[str, Path, bytes, Image.Image],
        prediction_id: str,
        include_gradcam: bool = True,
        gradcam_output_dir: Path = GRADCAM_DIR,
    ) -> Dict[str, Any]:
        """
        Perform real model inference (and optional Grad-CAM explainability) on
        a validated dermoscopic image.
        """
        self.ensure_ready()

        _, resized_rgb_uint8, preprocessed_batch = preprocess_image_for_inference(image_source)

        raw_pred = self.model.predict(preprocessed_batch, verbose=0)
        malignant_prob = float(np.clip(raw_pred[0][0], 0.0, 1.0))
        benign_prob = float(1.0 - malignant_prob)

        # Guarantee exact 6-decimal rounding that sums to 1.0
        malignant_prob_rounded = round(malignant_prob, 6)
        benign_prob_rounded = round(1.0 - malignant_prob_rounded, 6)

        threshold = get_active_decision_threshold()
        is_malignant = malignant_prob_rounded >= threshold
        predicted_class = "Malignant" if is_malignant else "Benign"
        predicted_label = 1 if is_malignant else 0
        confidence_score = round(
            malignant_prob_rounded if is_malignant else benign_prob_rounded,
            6,
        )

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        visualizations: Dict[str, Any] = {}
        internal_paths: Dict[str, Path] = {}
        if include_gradcam:
            gradcam_res = generate_and_save_gradcam_artifacts(
                model=self.model,
                resized_rgb_uint8=resized_rgb_uint8,
                preprocessed_batch=preprocessed_batch,
                prediction_id=prediction_id,
                output_dir=gradcam_output_dir,
                threshold=threshold,
            )
            visualizations = {
                "original_image_url": gradcam_res["original_url"],
                "gradcam_heatmap_url": gradcam_res["heatmap_url"],
                "gradcam_overlay_url": gradcam_res["overlay_url"],
                "gradcam_comparison_url": gradcam_res["comparison_url"],
                "target_layer": gradcam_res["target_layer"],
                "explanation": gradcam_res["explanation"],
            }
            internal_paths = {
                "original_path": gradcam_res["original_path"],
                "heatmap_path": gradcam_res["heatmap_path"],
                "overlay_path": gradcam_res["overlay_path"],
                "comparison_path": gradcam_res["comparison_path"],
            }

        response_payload: Dict[str, Any] = {
            "prediction_id": prediction_id,
            "prediction_status": "success",
            "timestamp": timestamp,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "benign_probability": benign_prob_rounded,
            "malignant_probability": malignant_prob_rounded,
            "confidence_score": confidence_score,
            "calibration_note": (
                "Uncalibrated softmax/sigmoid model probability estimate. "
                "Does not represent true clinical post-test disease probability."
            ),
            "decision_threshold": threshold,
            "model_architecture": MODEL_ARCHITECTURE,
            "input_dimensions": list(INPUT_SHAPE),
            "visualizations": visualizations,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }

        # Store in cache (with internal file paths kept server-side only)
        self._predictions_cache[prediction_id] = {
            **response_payload,
            "_internal_paths": internal_paths,
        }
        return response_payload

    def get_cached_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a previously computed prediction by ID for PDF report generation."""
        return self._predictions_cache.get(prediction_id)


# Global singleton instance initialized on module import / Flask startup
model_service = ModelService()
