"""
Stage 7: Flask Application Entry Point (`backend/app.py`).

Loads the trained EfficientNetB0 model once at application startup, registers
all API blueprints (`/api/health`, `/api/model-info`, `/api/predict`,
`/api/gradcam`, `/api/report`), configures CORS for the React frontend, and
handles upload size/validation errors gracefully without exposing internal paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# Ensure project root is on sys.path when running `python backend/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CORS_ORIGINS, MAX_CONTENT_LENGTH
from backend.routes.health import health_bp
from backend.routes.prediction import prediction_bp
from backend.routes.reports import reports_bp
from backend.services.model_service import model_service


def create_app(load_model_on_start: bool = True) -> Flask:
    """Create and configure the Flask backend application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # Configure CORS for local React Vite development server
    CORS(
        app,
        resources={r"/api/*": {"origins": CORS_ORIGINS}},
        expose_headers=["Content-Disposition"],
    )

    # Register modular route blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(reports_bp)

    # Global error handlers
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_err):
        max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
        return (
            jsonify(
                {
                    "prediction_status": "error",
                    "error": f"Uploaded file exceeds the maximum allowed size of {max_mb:.1f} MB.",
                }
            ),
            413,
        )

    @app.errorhandler(404)
    def handle_not_found(_err):
        return jsonify({"status": "error", "error": "Requested API endpoint was not found."}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(_err):
        return jsonify({"status": "error", "error": "HTTP method not allowed for this endpoint."}), 405

    if load_model_on_start:
        loaded = model_service.load_model()
        if loaded:
            print(f"[Startup] Loaded trained model from: {model_service.model_path.name}")
        else:
            print(f"[Startup] Notice: {model_service.load_error}")

    return app


app = create_app(load_model_on_start=os.getenv("SKIP_INITIAL_MODEL_LOAD", "0") != "1")


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
