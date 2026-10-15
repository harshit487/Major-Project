"""
Prediction and Explainable AI (Grad-CAM) Routes
(`POST /api/predict`, `POST /api/gradcam`, `GET /api/static/gradcam/<filename>`).
"""

from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from backend.config import GRADCAM_DIR, get_active_decision_threshold
from backend.services.gradcam import generate_and_save_gradcam_artifacts
from backend.services.model_service import ModelNotLoadedError, model_service
from backend.services.preprocessing import preprocess_image_for_inference
from backend.utils.validation import ImageValidationError, validate_uploaded_image

prediction_bp = Blueprint("prediction", __name__)


def _extract_uploaded_file():
    """Accept either 'image' or 'file' form field from multipart/form-data."""
    if "image" in request.files:
        return request.files["image"]
    if "file" in request.files:
        return request.files["file"]
    return None


@prediction_bp.route("/api/predict", methods=["POST"])
def predict_lesion():
    """
    Validate an uploaded dermoscopic image, perform binary inference
    (Benign vs. Malignant), generate Grad-CAM visualizations, and return JSON.
    """
    try:
        uploaded = _extract_uploaded_file()
        validated = validate_uploaded_image(uploaded)

        include_gradcam = request.form.get("include_gradcam", "true").lower() != "false"
        result = model_service.predict_image(
            image_source=validated.pil_image,
            prediction_id=validated.unique_id,
            include_gradcam=include_gradcam,
        )
        return jsonify(result), 200

    except ImageValidationError as exc:
        return (
            jsonify(
                {
                    "prediction_status": "error",
                    "error": exc.message,
                }
            ),
            exc.status_code,
        )
    except ModelNotLoadedError as exc:
        return (
            jsonify(
                {
                    "prediction_status": "error",
                    "error": str(exc),
                }
            ),
            503,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "prediction_status": "error",
                    "error": f"Inference pipeline error: {exc}",
                }
            ),
            500,
        )


@prediction_bp.route("/api/gradcam", methods=["POST"])
def generate_gradcam():
    """
    Generate and return Grad-CAM visualizations (heatmap, overlay, and
    side-by-side comparison) for a valid uploaded dermoscopic image.
    """
    try:
        uploaded = _extract_uploaded_file()
        validated = validate_uploaded_image(uploaded)

        model_service.ensure_ready()
        _, resized_rgb_uint8, preprocessed_batch = preprocess_image_for_inference(
            validated.pil_image
        )
        threshold = get_active_decision_threshold()

        gradcam_artifacts = generate_and_save_gradcam_artifacts(
            model=model_service.model,
            resized_rgb_uint8=resized_rgb_uint8,
            preprocessed_batch=preprocessed_batch,
            prediction_id=validated.unique_id,
            output_dir=GRADCAM_DIR,
            threshold=threshold,
        )

        # If caller explicitly requests raw image file response (?format=image)
        if request.args.get("format") == "image":
            overlay_path = gradcam_artifacts["overlay_path"]
            return send_from_directory(
                str(overlay_path.parent),
                overlay_path.name,
                mimetype="image/png",
            )

        return (
            jsonify(
                {
                    "status": "success",
                    "prediction_id": gradcam_artifacts["prediction_id"],
                    "target_layer": gradcam_artifacts["target_layer"],
                    "original_image_url": gradcam_artifacts["original_url"],
                    "gradcam_heatmap_url": gradcam_artifacts["heatmap_url"],
                    "gradcam_overlay_url": gradcam_artifacts["overlay_url"],
                    "gradcam_comparison_url": gradcam_artifacts["comparison_url"],
                    "explanation": gradcam_artifacts["explanation"],
                }
            ),
            200,
        )

    except ImageValidationError as exc:
        return jsonify({"status": "error", "error": exc.message}), exc.status_code
    except ModelNotLoadedError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"status": "error", "error": f"Grad-CAM error: {exc}"}), 500


@prediction_bp.route("/api/static/gradcam/<path:filename>", methods=["GET"])
def serve_gradcam_file(filename: str):
    """Serve generated Grad-CAM PNG images without exposing internal server paths."""
    safe_name = secure_filename(filename)
    if not safe_name:
        abort(404)
    target = GRADCAM_DIR / safe_name
    if not target.exists() or not target.is_file():
        return jsonify({"error": "Requested visualization not found."}), 404
    return send_from_directory(str(GRADCAM_DIR), safe_name, mimetype="image/png")
