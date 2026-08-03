"""
routes.py

Flask route definitions for the Loan Approval Prediction API.

Endpoints:
    GET  /         -> API status
    GET  /health   -> Health check
    POST /predict  -> Run a prediction on a single application
"""

from flask import Blueprint, jsonify, request

from inference import run_prediction

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/", methods=["GET"])
def index():
    """Basic status endpoint -- confirms the API is up and reachable."""
    return jsonify({
        "status": "ok",
        "service": "Loan Approval Prediction API",
        "endpoints": {
            "GET /": "This status message",
            "GET /health": "Health check",
            "POST /predict": "Run a prediction (see FLASK_API_GUIDE.md for the request body)",
        },
    })


@api_blueprint.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint. Attempts to load model artifacts to confirm
    the API is not just running, but actually able to serve predictions.
    """
    try:
        from model_loader import get_model_artifacts
        get_model_artifacts()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "model_loaded": False, "error": str(e)}), 503


@api_blueprint.route("/predict", methods=["POST"])
def predict():
    """
    Run a prediction on a single loan application.

    Expects a JSON body with all required fields (see
    inference.REQUIRED_FIELDS / FLASK_API_GUIDE.md). Returns a 400 with
    validation errors if the input is malformed, or a 200 with the
    prediction result if successful.
    """
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": ["Request body must be valid JSON with Content-Type: application/json"]
        }), 400

    result = run_prediction(data)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result), 200
