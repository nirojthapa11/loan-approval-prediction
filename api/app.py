"""
app.py (api/)

Flask application entry point for the Loan Approval Prediction API.

Run with: python api/app.py (from the project root, with the venv activated)

"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
from routes import api_blueprint


def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_blueprint)

    # Registered at the APP level (not the blueprint level) specifically
    # so they also catch 404s for URLs that don't match any route at all --
    # Blueprint.errorhandler only fires for errors raised inside that
    # blueprint's own view functions, not unmatched routing.
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found. See GET / for available endpoints."}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error."}), 500

    return app


app = create_app()

if __name__ == "__main__":
    # host="0.0.0.0" so the API is reachable from outside a Docker container;
    # debug=False for anything beyond local testing.
    app.run(host="0.0.0.0", port=5000, debug=False)
