"""
Health and Model Metadata API Routes (`GET /api/health`, `GET /api/model-info`,
`GET /api/figures/<filename>`).
"""

from __future__ import annotations

from flask import Blueprint, abort, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from backend.config import FIGURES_DIR
from backend.services.model_service import model_service

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def get_health():
    """Return API status and whether the trained model loaded successfully."""
    status = model_service.get_health_status()
    return jsonify(status), 200


@health_bp.route("/api/model-info", methods=["GET"])
def get_model_info():
    """
    Return model architecture, input dimensions, supported classes,
    decision threshold, and available test-set evaluation metrics.
    """
    info = model_service.get_model_info()
    return jsonify(info), 200


@health_bp.route("/api/figures/<path:filename>", methods=["GET"])
def serve_evaluation_figure(filename: str):
    """Serve generated evaluation plots from `results/figures/` safely."""
    safe_name = secure_filename(filename)
    if not safe_name:
        abort(404)
    target = FIGURES_DIR / safe_name
    if not target.exists() or not target.is_file():
        return jsonify({"error": f"Figure '{safe_name}' not yet generated."}), 404
    return send_from_directory(str(FIGURES_DIR), safe_name)
