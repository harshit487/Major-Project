"""
PDF Report Generation Route (`POST /api/report`).
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request, send_file

from backend.config import GRADCAM_DIR
from backend.services.model_service import ModelNotLoadedError, model_service
from backend.services.report_service import generate_prediction_pdf_report
from backend.utils.validation import ImageValidationError, validate_uploaded_image

reports_bp = Blueprint("reports", __name__)


@reports_bp.route("/api/report", methods=["POST"])
def create_report():
    """
    Generate and return a downloadable PDF prediction report.

    Supports two invocation modes:
    1. JSON payload containing `prediction_id` (from a prior `/api/predict` call)
       and/or explicit prediction fields.
    2. Multipart form upload containing an image (`image` or `file`), which
       executes inference + Grad-CAM and returns the generated PDF directly.
    """
    try:
        prediction_data = None
        orig_path = None
        heat_path = None
        over_path = None

        # Mode A: Multipart image upload
        if "image" in request.files or "file" in request.files:
            uploaded = request.files.get("image") or request.files.get("file")
            validated = validate_uploaded_image(uploaded)
            prediction_data = model_service.predict_image(
                image_source=validated.pil_image,
                prediction_id=validated.unique_id,
                include_gradcam=True,
            )
            cached = model_service.get_cached_prediction(validated.unique_id)
            if cached and "_internal_paths" in cached:
                orig_path = cached["_internal_paths"].get("original_path")
                heat_path = cached["_internal_paths"].get("heatmap_path")
                over_path = cached["_internal_paths"].get("overlay_path")

        # Mode B: JSON request with prediction_id or prediction fields
        elif request.is_json:
            body = request.get_json(silent=True) or {}
            pred_id = body.get("prediction_id")
            if pred_id:
                cached = model_service.get_cached_prediction(str(pred_id))
                if cached:
                    prediction_data = cached
                    internal = cached.get("_internal_paths", {})
                    orig_path = internal.get("original_path")
                    heat_path = internal.get("heatmap_path")
                    over_path = internal.get("overlay_path")
                else:
                    # Check if Grad-CAM files exist on disk for this prediction_id
                    safe_id = "".join(c for c in str(pred_id) if c.isalnum() or c in ("_", "-"))
                    cand_orig = GRADCAM_DIR / f"{safe_id}_original.png"
                    cand_heat = GRADCAM_DIR / f"{safe_id}_heatmap.png"
                    cand_over = GRADCAM_DIR / f"{safe_id}_overlay.png"
                    if cand_orig.exists():
                        orig_path = cand_orig
                    if cand_heat.exists():
                        heat_path = cand_heat
                    if cand_over.exists():
                        over_path = cand_over

            if prediction_data is None:
                if "predicted_class" not in body:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "error": (
                                    "Prediction ID not found in active session and no "
                                    "'predicted_class' details were provided."
                                ),
                            }
                        ),
                        400,
                    )
                prediction_data = body
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "Provide a JSON payload with 'prediction_id' or upload an image file.",
                    }
                ),
                400,
            )

        pdf_path = generate_prediction_pdf_report(
            prediction_data=prediction_data,
            original_image_path=orig_path,
            heatmap_image_path=heat_path,
            overlay_image_path=over_path,
        )

        return send_file(
            str(pdf_path),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=pdf_path.name,
        )

    except ImageValidationError as exc:
        return jsonify({"status": "error", "error": exc.message}), exc.status_code
    except ModelNotLoadedError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"status": "error", "error": f"PDF report generation failed: {exc}"}), 500
