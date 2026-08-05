# Docker Guide

Reference for containerizing and running the Loan Approval Prediction
project — both the Streamlit app and the Flask API.

> **Honesty note:** the Dockerfile/docker-compose.yml below were written
> carefully and their YAML syntax has been validated, but they have
> **not** been run against an actual Docker daemon while building this
> project (no Docker available in the environment used to build it).
> Test them yourself following the steps below, and if anything doesn't
> behave as expected, that's useful, real information for your
> troubleshooting section — not a sign you did something wrong.

## 1. Installing Docker

**Windows:**
1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Run the installer (requires WSL 2 — the installer will prompt you to enable it if it isn't already).
3. Restart your machine if prompted.
4. Launch Docker Desktop, wait for it to say "Docker Desktop is running."
5. Verify from cmd:
   ```cmd
   docker --version
   docker compose version
   ```

## 2. Building the Image

From the project root (where `Dockerfile` lives):

```cmd
cd C:\Projects\loan-approval-prediction
docker build -t loan-approval-prediction .
```

**What this does:** reads the `Dockerfile`, installs Python + dependencies
from `requirements.txt`, copies the project into the image, and tags the
result as `loan-approval-prediction`.

**Expected output:** a series of numbered build steps, ending with
`Successfully tagged loan-approval-prediction:latest` (exact wording
varies by Docker version).

## 3. Running the Streamlit App Container

```cmd
docker run -p 8501:8501 loan-approval-prediction
```

Open `http://localhost:8501` in your browser — should show the same
Streamlit app you've been running locally.

`-p 8501:8501` maps port 8501 inside the container to port 8501 on your
machine. If 8501 is already in use locally, map to a different host
port: `-p 8600:8501`, then visit `http://localhost:8600`.

## 4. Running the Flask API Container

```cmd
docker run -p 5000:5000 loan-approval-prediction python api/app.py
```

Test it the same way as `FLASK_API_GUIDE.md` describes, just against
this container instead of a local `python api/app.py` process.

## 5. Using Docker Compose (Both Services at Once)

```cmd
docker compose up
```

This builds (if needed) and starts **both** the Streamlit app
(`localhost:8501`) and the Flask API (`localhost:5000`) together, using
the service definitions in `docker-compose.yml`.

Run it in the background instead:
```cmd
docker compose up -d
```

**Stop everything:**
```cmd
docker compose down
```

**View logs:**
```cmd
docker compose logs -f
```
(`-f` follows the log output live; Ctrl+C to stop watching, without
stopping the containers.)

**Check running containers:**
```cmd
docker ps
```

## 6. Retraining Without Rebuilding the Image

`docker-compose.yml` mounts `./models` from your host machine into the
container. This means if you retrain locally (`python src/train.py`,
outside Docker) and it overwrites `models/*.pkl`, restarting the
container picks up the new model without needing `docker build` again:

```cmd
docker compose restart
```

## 7. Common Errors & Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `docker: command not found` | Docker Desktop not installed or not on PATH | Reinstall Docker Desktop, restart terminal |
| `Cannot connect to the Docker daemon` | Docker Desktop isn't running | Launch Docker Desktop from the Start menu, wait for it to fully start |
| `port is already allocated` | Something else is using 8501 or 5000 | Change the host-side port mapping, e.g. `-p 8600:8501` |
| Build fails installing `xgboost` or `scikit-learn` | Missing build tools in a slimmer base image | The Dockerfile already installs `build-essential` for this reason — if it still fails, try `python:3.11` (non-slim) as the base image instead |
| App runs but shows "Model artifacts not found" | `models/*.pkl` wasn't copied into the image, or the volume mount path is wrong | Confirm `models/trained_model.pkl` exists on your host before building; check the volume line in `docker-compose.yml` matches your actual folder name |
| Container exits immediately | Check `docker compose logs` for the actual Python traceback | Fix the underlying error, rebuild with `docker compose up --build` |

## 8. Cleaning Up

```cmd
docker compose down          :: stop and remove containers
docker rmi loan-approval-prediction   :: remove the built image
docker system prune          :: remove unused Docker data (use with care)
```
