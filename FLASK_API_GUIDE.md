# Flask API Guide

Reference for the Loan Approval Prediction REST API — a standalone
alternative to the Streamlit app, useful for integrating predictions
into other systems.

## Overview

The API lives in `api/` and reuses the exact same trained model,
scaler, and prediction logic as the Streamlit app (`src/predict.py`) —
so results from the API and the app are always identical for the same
input.

## Running the API

```cmd
cd C:\Projects\loan-approval-prediction
venv\Scripts\activate.bat
python api\app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

Leave this terminal running while you test the endpoints from another
terminal, Postman, or your browser.

## Endpoints

### `GET /`
Status message and a list of available endpoints.

**Example:**
```cmd
curl http://localhost:5000/
```

**Response:**
```json
{
  "status": "ok",
  "service": "Loan Approval Prediction API",
  "endpoints": {
    "GET /": "This status message",
    "GET /health": "Health check",
    "POST /predict": "Run a prediction (see FLASK_API_GUIDE.md for the request body)"
  }
}
```

### `GET /health`
Confirms the API is running AND able to load the model (not just that
the server process is alive).

**Example:**
```cmd
curl http://localhost:5000/health
```

**Response (healthy):**
```json
{"status": "healthy", "model_loaded": true}
```

**Response (unhealthy, e.g. missing model files) — HTTP 503:**
```json
{"status": "unhealthy", "model_loaded": false, "error": "..."}
```

### `POST /predict`
Runs a prediction on a single loan application.

**Required JSON fields:**

| Field | Type | Notes |
|---|---|---|
| `no_of_dependents` | number | |
| `education` | string | `"Graduate"` or `"Not Graduate"` |
| `self_employed` | string | `"Yes"` or `"No"` |
| `income_annum` | number | |
| `loan_amount` | number | |
| `loan_term` | number | years |
| `cibil_score` | number | 300–900 |
| `residential_assets_value` | number | |
| `commercial_assets_value` | number | |
| `luxury_assets_value` | number | |
| `bank_asset_value` | number | |

**Example request (curl) — tested, genuinely returns Approved:**
```cmd
curl -X POST http://localhost:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"no_of_dependents\": 2, \"education\": \"Graduate\", \"self_employed\": \"No\", \"income_annum\": 8000000, \"loan_amount\": 12000000, \"loan_term\": 10, \"cibil_score\": 780, \"residential_assets_value\": 5000000, \"commercial_assets_value\": 4000000, \"luxury_assets_value\": 6000000, \"bank_asset_value\": 3000000}"
```

*(Note the `^` line-continuation and escaped quotes — that's cmd.exe syntax. If you're using PowerShell or Git Bash instead, single-line with normal quoting works fine, or just use Postman — see below.)*

**Response — HTTP 200:**
```json
{"prediction": "Approved", "approved_probability": 1.0, "rejected_probability": 0.0}
```

**Example request — tested, genuinely returns Rejected:**
```json
{
  "no_of_dependents": 3, "education": "Not Graduate", "self_employed": "Yes",
  "income_annum": 2000000, "loan_amount": 15000000, "loan_term": 5,
  "cibil_score": 420, "residential_assets_value": 500000,
  "commercial_assets_value": 0, "luxury_assets_value": 500000,
  "bank_asset_value": 200000
}
```
**Response — HTTP 200:**
```json
{"prediction": "Rejected", "approved_probability": 0.0, "rejected_probability": 1.0}
```

## Error Handling

**Missing fields — HTTP 400:**
```json
{"error": ["Missing required field(s): no_of_dependents, education, ..."]}
```

**Invalid value (e.g. CIBIL score out of range) — HTTP 400:**
```json
{"error": ["cibil_score must be between 300 and 900"]}
```

**Malformed / non-JSON body — HTTP 400:**
```json
{"error": ["Request body must be valid JSON with Content-Type: application/json"]}
```

**Unknown route — HTTP 404:**
```json
{"error": "Endpoint not found. See GET / for available endpoints."}
```

## Testing with Postman

1. Open Postman → **File > Import** → select `POSTMAN_COLLECTION.json` from the project root.
2. The collection includes: status check, health check, a valid "likely approved" request, a valid "likely rejected" request, and two intentionally invalid requests (missing fields, out-of-range CIBIL score) — all pre-filled and ready to send.
3. Confirm the `base_url` variable (top of the collection) points to `http://localhost:5000` — adjust if you're running the API elsewhere (e.g. inside Docker, see `DOCKER_GUIDE.md`).

## Testing with curl (Git Bash / WSL / macOS / Linux syntax)

If you're using Git Bash instead of cmd.exe, single-quote the whole `-d` body — no line continuation or escaping needed:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"no_of_dependents": 2, "education": "Graduate", "self_employed": "No", "income_annum": 8000000, "loan_amount": 12000000, "loan_term": 10, "cibil_score": 780, "residential_assets_value": 5000000, "commercial_assets_value": 4000000, "luxury_assets_value": 6000000, "bank_asset_value": 3000000}'
```

## How the Streamlit App and the API Relate

Both are **standalone** — you can run either one independently. Neither
calls the other. What they share is the underlying code path:

```
                    src/predict.py
                   (shared logic)
                    /            \
             app.py               api/inference.py
        (Streamlit UI,               (Flask API,
         direct function call)        JSON validation + same call)
```

This means: if you retrain the model (`python src/train.py`), **both**
interfaces immediately use the new model — there's only one place the
actual prediction logic lives.
