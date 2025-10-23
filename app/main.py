import json
import os
import time
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Annotated, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from threading import Lock  # <-- thread-safety

# ---------- Paths / Config ----------
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "model"))
DEMOGRAPHICS_CSV = Path(os.getenv("DEMOGRAPHICS_CSV", "data/zipcode_demographics.csv"))

# Global (hot) artifacts
_model = None
_model_features: List[str] = []
_feature_medians: Dict[str, float] = {}
_model_meta: Dict[str, Any] = {}
_demographics_df: Optional[pd.DataFrame] = None

# Protects hot artifact swaps & snapshots
_reload_lock = Lock()

# Minimal feature set for the bonus endpoint
MIN_REQUIRED = ["sqft_living", "bedrooms", "bathrooms", "zipcode"]

# ---------- Tags for Swagger ----------
tags_metadata = [
    {"name": "health", "description": "Liveness and readiness checks."},
    {"name": "predict", "description": "Prediction endpoints for housing prices."},
    {"name": "admin", "description": "Operational endpoints (reload artifacts, etc.)."}
]

# ---------- Schemas ----------
class Features(BaseModel):
    bedrooms: Optional[float] = None
    bathrooms: Optional[float] = None
    sqft_living: Optional[float] = None
    sqft_lot: Optional[float] = None
    floors: Optional[float] = None
    waterfront: Optional[float] = None
    view: Optional[float] = None
    condition: Optional[float] = None
    grade: Optional[float] = None
    sqft_above: Optional[float] = None
    sqft_basement: Optional[float] = None
    yr_built: Optional[float] = None
    yr_renovated: Optional[float] = None
    zipcode: str
    lat: Optional[float] = None
    long: Optional[float] = None
    sqft_living15: Optional[float] = None
    sqft_lot15: Optional[float] = None

class BasicFeatures(BaseModel):
    sqft_living: float
    bedrooms: float
    bathrooms: float
    zipcode: str

class PredictRequest(BaseModel):
    row: Features

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "row": {
                        "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800,
                        "sqft_lot": 5000, "floors": 1, "waterfront": 0, "view": 0,
                        "condition": 3, "grade": 7, "sqft_above": 1800,
                        "sqft_basement": 0, "yr_built": 1992, "yr_renovated": 0,
                        "zipcode": "98052", "lat": 47.61, "long": -112.23,
                        "sqft_living15": 1600, "sqft_lot15": 3000
                    }
                }
            ]
        }
    }

class PredictBasicRequest(BaseModel):
    row: BasicFeatures

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"row": {"sqft_living": 1800, "bedrooms": 3, "bathrooms": 2, "zipcode": "98052"}}
            ]
        }
    }

class PredictResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: int

# ---------- Helpers ----------
def df_from_row(row: BaseModel) -> pd.DataFrame:
    df = pd.DataFrame([row.model_dump()])  # single-row DataFrame
    df["zipcode"] = df["zipcode"].astype(str)
    return df

def _load_artifacts_from_disk() -> Tuple[Any, List[str], Dict[str, float], Dict[str, Any], pd.DataFrame]:
    """Load artifacts from disk WITHOUT holding the lock."""
    try:
        with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(ARTIFACT_DIR / "model_features.json", "r") as f:
            model_features = json.load(f)
        with open(ARTIFACT_DIR / "feature_medians.json", "r") as f:
            feature_medians = json.load(f)
        with open(ARTIFACT_DIR / "model_meta.json", "r") as f:
            model_meta = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}")

    try:
        demo_df = pd.read_csv(DEMOGRAPHICS_CSV, dtype={"zipcode": str})
    except Exception as e:
        raise RuntimeError(f"Failed to load demographics CSV: {e}")

    return model, model_features, feature_medians, model_meta, demo_df

def _swap_artifacts(
    model: Any,
    model_features: List[str],
    feature_medians: Dict[str, float],
    model_meta: Dict[str, Any],
    demo_df: pd.DataFrame,
) -> None:
    """Atomically replace global references under lock."""
    global _model, _model_features, _feature_medians, _model_meta, _demographics_df
    with _reload_lock:
        _model = model
        _model_features = model_features
        _feature_medians = feature_medians
        _model_meta = model_meta
        _demographics_df = demo_df

def load_artifacts() -> None:
    """Public loader used at startup and by /admin/reload."""
    model, feats, meds, meta, demo = _load_artifacts_from_disk()
    _swap_artifacts(model, feats, meds, meta, demo)

def ensure_ready():
    with _reload_lock:
        ready = (_model is not None) and (_demographics_df is not None) and bool(_model_features)
    if not ready:
        raise HTTPException(status_code=503, detail="Model or dependencies not loaded")

def _snapshot_artifacts() -> Tuple[Any, List[str], Dict[str, float], Dict[str, Any], pd.DataFrame]:
    """Take a consistent snapshot of the current artifacts for use in a request."""
    with _reload_lock:
        if _model is None or _demographics_df is None or not _model_features:
            raise HTTPException(status_code=503, detail="Model or dependencies not loaded")
        return _model, _model_features[:], dict(_feature_medians), dict(_model_meta), _demographics_df

# ---------- Preprocessing ----------
def merge_demographics(df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(demo_df, how="left", on="zipcode")
    if "zipcode" in merged.columns:
        merged = merged.drop(columns=["zipcode"])
    return merged

def apply_imputation_and_order(df: pd.DataFrame, feature_medians: Dict[str, float], model_features: List[str]) -> pd.DataFrame:
    for col in model_features:
        if col not in df.columns:
            df[col] = None
    for col, med in feature_medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)
    df = df.fillna(0)
    df = df[model_features]
    return df

# ---------- App (Swagger/OpenAPI setup) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    # cleanup here if needed

app = FastAPI(
    title="Housing Price API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "displayRequestDuration": True,
        "tryItOutEnabled": True
    },
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# (Optional) enable CORS if you plan to call from browsers/other origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ---------- Routes ----------
@app.get("/", tags=["health"], summary="Landing")
def root():
    return {
        "message": "KC Housing Price API",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }

@app.get("/live", tags=["health"], summary="Liveness probe")
def liveness():
    return {"status": "ok"}

@app.get("/ready", tags=["health"], summary="Readiness probe")
def readiness():
    ensure_ready()
    # Read under lock to avoid torn reads
    with _reload_lock:
        version = _model_meta.get("version")
    return {"status": "ready", "model_version": version}

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["predict"],
    summary="Predict price",
)
def predict(
    req: PredictRequest = Body(
        ...,
        description="predict returns price."
    )
):
    # Take a consistent snapshot quickly, then release lock for heavy work
    model, model_features, feature_medians, model_meta, demo_df = _snapshot_artifacts()

    t0 = time.time()
    base_df = df_from_row(req.row)
    merged = merge_demographics(base_df, demo_df)
    final_df = apply_imputation_and_order(merged, feature_medians, model_features)

    try:
        prediction = model.predict(final_df).tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    latency = int((time.time() - t0) * 1000)
    return PredictResponse(
        prediction=float(prediction[0]),
        model_version=model_meta.get("version", "unknown"),
        latency_ms=latency
    )

@app.post(
    "/predict/minimal",
    response_model=PredictResponse,
    tags=["predict"],
    summary="Predict price basic",
)
def predict_minimal(
    req: PredictBasicRequest = Body(
        ...,
        description="Basic prediction with subset of features"
    )
):
    model, model_features, feature_medians, model_meta, demo_df = _snapshot_artifacts()

    t0 = time.time()
    df = pd.DataFrame([req.row.model_dump()])
    df["zipcode"] = df["zipcode"].astype(str)

    merged = merge_demographics(df, demo_df)
    final_df = apply_imputation_and_order(merged, feature_medians, model_features)

    try:
        prediction = model.predict(final_df).tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    latency = int((time.time() - t0) * 1000)
    return PredictResponse(
        prediction=float(prediction[0]),
        model_version=model_meta.get("version", "unknown"),
        latency_ms=latency
    )

@app.post("/admin/reload", tags=["admin"], summary="Reload model artifacts")
def admin_reload():
    """
    Safe hot-reload:
    - Load from disk without holding the lock.
    - Acquire lock only to atomically swap references.
    - Keep in-flight predictions consistent via snapshotting.
    """
    model, feats, meds, meta, demo = _load_artifacts_from_disk()
    _swap_artifacts(model, feats, meds, meta, demo)
    return {"status": "reloaded", "model_version": meta.get("version")}
