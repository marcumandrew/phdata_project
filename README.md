# Housing Price API — README

Predict single-home prices for the Seattle area via a FastAPI service backed by a trained ML model. The repo includes:

* Two training pipelines (the original **KNN** baseline and an improved **XGBoost** model)
* A production-style API with a *single-row* prediction endpoint and a “minimal fields” bonus endpoint
* Model versioning + hot-reload without downtime
* Tools and examples to test correctness and load
* Visual diagnostics (parity plots, error histograms) and an **interactive Folium map** of large validation errors

---

## Contents

```
app/
  main.py                 # FastAPI app (pred endpoints, admin/reload, model hot-load)
  create_model.py         # XGBoost training pipeline + metrics + Folium map + artifacts
  original_create_model.py# Original script for reference
  deploy.py               # Post-train deploy helper (bumps model version, calls /admin/reload)

data/
  kc_house_data.csv
  zipcode_demographics.csv

model/
  model.pkl                   # latest trained model
  model_features.json         # ordered list of features expected at inference
  feature_medians.json        # numeric medians for inference-time imputation
  model_meta.json             # { "version": "vX" } — incremented on deploy
  ... (plots, metrics, maps written by training scripts)

.vscode/
  launch.json                 # VS Code run config: Gunicorn + UvicornWorker (multi-process)

Dockerfile                    # Container image definition (for local dev or prod)
requirements.txt              # Python deps
```

---

## Quick Start (VS Code Dev Container)

This project is set up to run entirely inside a VS Code **Dev Container**.

1. **Prereqs**

   * VS Code + “Dev Containers” extension
   * Docker Desktop running

2. **Open in container**

   * `File → Open Folder…` the repo root
   * VS Code will prompt “Reopen in Container” — accept
   * Container build installs `requirements.txt`, copies `app/` and runs in `/app`

3. **Train a model (XGBoost)**

   ```bash
   python app/create_model.py
   ```

   Artifacts appear in `model/` and plots/maps are written there too:

   * `model.pkl`, `model_features.json`, `feature_medians.json`, `model_meta.json`
   * `parity_*.png`, `hist_spe_*.png`, `val_bad_preds_map.html` (interactive map)

4. **Run the API (multi-process)**

   * In VS Code: **Run and Debug → “FastAPI (Gunicorn)”**
   * Or from terminal:

     ```bash
     python -m gunicorn app.main:app -k uvicorn.workers.UvicornWorker \
       -w 12 -b 0.0.0.0:8000 --timeout 60 --keep-alive 5 \
       --graceful-timeout 30 --max-requests 5000 --max-requests-jitter 500
     ```
   * Open docs at: `http://localhost:8000/docs`

5. **(Optional) Deploy step after training**

   * Bump the model version and hot-reload the API:

     ```bash
     python app/deploy.py
     ```
   * This increments `model/model_meta.json` (e.g., v3 → v4) and calls `POST /admin/reload`.

---

## API Overview

### Endpoints

* `GET /live` — liveness probe
* `GET /ready` — readiness probe; returns current `model_version`
* `POST /predict` — **single-row** prediction with the full schema (matches `kc_house_data.csv` fields; demographics are joined in the backend)
* `POST /predict/minimal` — **single-row** “bonus” endpoint with the minimum viable fields (`sqft_living, bedrooms, bathrooms, zipcode`)
* `POST /admin/reload` — hot-reload model artifacts without restarting the process

### Request/Response examples

**Full schema**

```json
POST /predict
{
  "row": {
    "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800,
    "sqft_lot": 5000, "floors": 1, "waterfront": 0, "view": 0,
    "condition": 3, "grade": 7, "sqft_above": 1800, "sqft_basement": 0,
    "yr_built": 1992, "yr_renovated": 0, "zipcode": "98052",
    "lat": 47.61, "long": -122.23, "sqft_living15": 1600, "sqft_lot15": 3000
  }
}
```

**Minimal schema**

```json
POST /predict/minimal
{ "row": { "sqft_living": 1800, "bedrooms": 3, "bathrooms": 2, "zipcode": "98052" } }
```

**Response**

```json
{
  "prediction": 532188.12,
  "model_version": "v4",
  "latency_ms": 7
}
```

### Under the Hood

1. Builds a single-row DataFrame from the request
2. Joins demographics by `zipcode`
3. Adds missing trained features and imputes numeric columns with medians
4. Ensures correct feature order
5. Returns prediction and metadata

> Imputation allows the API to handle missing optional fields (like `sqft_basement`) gracefully.

---

## Training Pipelines

### 1) `app/create_model.py` — XGBoost model

* Train/validation/test split (90/10 outer, 15% of train for validation)
* Metrics: R², RMSE, MAE, MAPE, ME, bias stats
* Outputs plots: parity plots, signed % error histograms
* Generates `val_bad_preds_map.html` (Folium)
* Saves model artifacts in `model/`

### 2) `app/original_create_model.py` — reference version

* Original script before refactoring
* Shows early design logic for model creation and evaluation

---

## Model Versioning & Reload

* **deploy.py** automatically increments the version in `model_meta.json` and triggers the `/admin/reload` endpoint.
* Reload swaps the model atomically without downtime; Gunicorn workers continue handling requests.

---

## Scaling Notes

* Gunicorn runs multiple Uvicorn workers (12 by default via `launch.json`).
* Each worker handles ~5000 requests before recycling (`--max-requests 5000`).
* `--max-requests-jitter 500` adds randomness to avoid all workers restarting simultaneously.

---

## Testing

### Simple curl test

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"row":{"sqft_living":1800,"bedrooms":3,"bathrooms":2,"zipcode":"98052"}}'
```

### Python test

```python
import requests
base = "http://localhost:8000"
print(requests.get(base + "/ready").json())
payload = {"row": {"sqft_living": 1800, "bedrooms": 3, "bathrooms": 2, "zipcode": "98052"}}
print(requests.post(base + "/predict/minimal", json=payload).json())
```



---

**Author:** Andrew Marcum
**Environment:** VS Code Dev Container (Dockerized FastAPI + XGBoost)
