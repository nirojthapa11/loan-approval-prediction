# Loan Approval Prediction -- Dockerfile
#
# Builds an image capable of running EITHER the Streamlit app or the
# Flask API (selected via the CMD you override at `docker run` time, or
# via docker-compose.yml, which runs both as separate services from
# this same image).

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by some scientific Python packages
# (e.g. compiling scikit-learn/xgboost dependencies on some platforms).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer from the app code)
# so Docker can cache this step and skip reinstalling on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Streamlit's default port (8501) and the Flask API's default port (5000)
EXPOSE 8501
EXPOSE 5000

# Default command runs the Streamlit app. docker-compose.yml overrides
# this for the API service to run `python api/app.py` instead.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
