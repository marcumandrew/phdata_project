# app/train_model_knn.py
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, pipeline, neighbors, preprocessing
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

# ---------------- Config ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_PATH = DATA_DIR / "kc_house_data.csv"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"  # load all cols; join on zipcode

MODEL_VERSION = "knn_v1"

# === Limited feature set from the original assignment ===
# (We will also load lat/long just for mapping, but exclude them from the model inputs.)
SALES_COLUMN_SELECTION: List[str] = [
    "price",
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement",
    "zipcode",
    # map-only (NOT used by model)
    "lat", "long"
]

# ---------------- Data loading ----------------
def load_data(
    sales_path: Path,
    demographics_path: Path,
    sales_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    # Load limited sales columns (+ lat/long for mapping)
    data = pd.read_csv(sales_path, usecols=sales_cols, dtype={"zipcode": str})

    # Load ALL demographics; keep zipcode as str for join
    demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})

    # Keep lat/long aside for mapping; we'll merge them back after splits
    latlong = data[["lat", "long"]].copy() if {"lat", "long"}.issubset(data.columns) else None

    # Merge with demographics on zipcode (left), then drop zipcode (as in original)
    merged = data.merge(demographics, how="left", on="zipcode").drop(columns=["zipcode"])

    # Target and features
    y = merged.pop("price")
    X = merged

    # Put lat/long back as columns for mapping convenience, but we will exclude them from modeling
    if latlong is not None:
        X["lat"] = latlong["lat"]
        X["long"] = latlong["long"]

    print(f"✅ Loaded {X.shape[0]:,} rows and {X.shape[1]:,} columns after merge.")
    return X, y

# ---------------- Metrics & plots ----------------
def signed_pe(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return (y_pred - y_true) / y_true

def bias_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    spe = signed_pe(y_true, y_pred)
    me  = float(np.mean(y_pred - y_true))
    mpe = float(np.mean(spe))
    mdpe = float(np.median(spe))
    over_rate = float(np.mean(y_pred > y_true))
    return {"ME": me, "MPE": mpe, "MdPE": mdpe, "over_rate": over_rate}

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))

def plot_signed_pe_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    spe_pct = signed_pe(y_true, y_pred) * 100.0
    spe_pct = np.clip(spe_pct, -200, 200)
    plt.figure(figsize=(6, 4))
    plt.hist(spe_pct, bins=40, alpha=0.9)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Signed % Error ( (pred-actual)/actual × 100 )")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def eval_split(name: str, model, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> dict:
    yhat = model.predict(X[feature_cols])
    r2 = metrics.r2_score(y, yhat)
    rmse = root_mean_squared_error(y, yhat)
    mae = metrics.mean_absolute_error(y, yhat)
    mape_val = mape(y, yhat)
    bias = bias_stats(y.values, yhat)

    print(f"\n📊 {name} Evaluation")
    print(f"  R²   : {r2:.3f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  MAPE : {mape_val*100:.2f}%")
    print(f"  ME   : {bias['ME']:,.0f} $  "
          f"(MPE: {bias['MPE']*100:.2f}%, MdPE: {bias['MdPE']*100:.2f}%, "
          f"Over-rate: {bias['over_rate']*100:.1f}%)")

    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape_val, "bias": bias, "yhat": yhat}

# -------- Folium map of >10% error on Validation --------
def map_bad_preds_val_folium(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    yhat_val: np.ndarray,
    out_path: Path,
    threshold: float = 0.10
) -> None:
    if not {"lat", "long"}.issubset(X_val.columns):
        print("⚠️ Skipping folium map: lat/long not present.")
        return

    df = X_val.copy()
    df["actual"] = y_val.values
    df["pred"] = yhat_val
    df["ape"] = np.abs(df["actual"] - df["pred"]) / df["actual"]

    center_lat = float(df["lat"].mean()) if df["lat"].notna().any() else 47.6062
    center_lon = float(df["long"].mean()) if df["long"].notna().any() else -122.3321

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")

    vmax = max(0.5, float(df["ape"].quantile(0.99)))
    cmap = cm.LinearColormap(colors=["#fee08b", "#fdae61", "#f46d43", "#d73027"],
                             vmin=threshold, vmax=vmax)
    cmap.caption = "Absolute Percentage Error (Validation)"

    all_cluster = MarkerCluster(name="All validation points", disableClusteringAtZoom=14)
    bad_cluster = MarkerCluster(name=f"> {int(threshold*100)}% error", disableClusteringAtZoom=14)

    for _, r in df.iterrows():
        lat, lon = r["lat"], r["long"]
        if pd.isna(lat) or pd.isna(lon):
            continue

        # popup_html = (
        #     f"<b>Actual:</b> ${r['actual']:,.0f}<br>"
        #     f"<b>Pred:</b>   ${r['pred']:,.0f}<br>"
        #     f"<b>APE:</b>    {r['ape']*100:.1f}%"
        # )


        popup_html = (
            f"<b>Actual:</b> ${r['actual']:,.0f}<br>"
            f"<b>Pred:</b>   ${r['pred']:,.0f}<br>"
            f"<b>APE:</b>    {r['ape']*100:.1f}%<br>"
            f"<b>Sqft:</b>   {r.get('sqft_living', 'n/a')}<br>"
            f"<b>Bed/Bath:</b> {r.get('bedrooms', 'n/a')}/{r.get('bathrooms', 'n/a')}<br>"
            f"<b>Grade:</b>  {r.get('grade', 'n/a')}<br>"
            f"<b>Condition:</b>  {r.get('condition', 'n/a')}<br>"
            f"<b>Year Built:</b>  {r.get('yr_built', 'n/a')}<br>"
            f"<b>Year Renovated:</b>  {r.get('yr_renovated', 'n/a')}<br>"
            f"<b>View:</b>  {r.get('view', 'n/a')}<br>"
            f"<b>Floors:</b>  {r.get('floors', 'n/a')}<br>"
            f"<b>Sqft Above:</b>  {r.get('sqft_above', 'n/a')}<br>"
            f"<b>Sqft Basement:</b>  {r.get('sqft_basement', 'n/a')}<br>"
            f"<b>Sqft_Living15:</b>  {r.get('sqft_living15', 'n/a')}"
        )

        popup = folium.Popup(popup_html, max_width=250)

        if r["ape"] > threshold:
            color = cmap(min(r["ape"], vmax))
            radius = float(np.clip(r["ape"] * 30, 6, 18))
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=popup,
            ).add_to(bad_cluster)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#888888",
                fill=True,
                fill_color="#888888",
                fill_opacity=0.5,
                popup=popup,
            ).add_to(all_cluster)

    all_cluster.add_to(m)
    bad_cluster.add_to(m)
    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))
    print(f"🗺️  Wrote interactive map → {out_path.resolve()}")

# ---------------- Main ----------------
def main() -> None:
    X_all, y_all = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # Determine model feature columns (EXCLUDE lat/long if present)
    model_feature_cols = [c for c in X_all.columns if c not in ("lat", "long")]

    # Train/Test split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_all, y_all, test_size=0.10, random_state=42
    )

    # Further split training into train/validation for evaluation parity with your XGB script
    X_tr, X_val, y_tr, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    # Build KNN pipeline (as in original: RobustScaler + KNeighborsRegressor)
    # You can tune n_neighbors, weights, p, etc. if desired.
    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(with_centering=True, with_scaling=True),
        neighbors.KNeighborsRegressor()
    )

    # Fit on the training split (X_tr) using only model features
    model.fit(X_tr[model_feature_cols], y_tr)

    # ---- Evaluate on train/val/test ----
    train_metrics = eval_split("Train", model, X_tr, y_tr, model_feature_cols)
    val_metrics   = eval_split("Validation", model, X_val, y_val, model_feature_cols)
    test_metrics  = eval_split("Test (held-out)", model, X_test, y_test, model_feature_cols)

    # ---- Bias histograms (signed % error) ----
    plot_signed_pe_hist(y_tr.values,  train_metrics["yhat"], "Signed % Error: Train (KNN)", OUT_DIR / "knn_hist_spe_train.png")
    plot_signed_pe_hist(y_val.values, val_metrics["yhat"], "Signed % Error: Validation (KNN)", OUT_DIR / "knn_hist_spe_val.png")
    plot_signed_pe_hist(y_test.values, test_metrics["yhat"], "Signed % Error: Test (KNN)", OUT_DIR / "knn_hist_spe_test.png")

    # ---- Parity plots ----
    parity_plot(y_tr.values,  train_metrics["yhat"], "Parity: Train (KNN)", OUT_DIR / "knn_parity_train.png")
    parity_plot(y_val.values, val_metrics["yhat"], "Parity: Validation (KNN)", OUT_DIR / "knn_parity_val.png")
    parity_plot(y_test.values, test_metrics["yhat"], "Parity: Test (KNN)", OUT_DIR / "knn_parity_test.png")

    # ---- Folium map of >10% error points on Validation ----
    map_bad_preds_val_folium(X_val, y_val, val_metrics["yhat"], OUT_DIR / "knn_val_bad_preds_map.html", threshold=0.10)

    # ---- Save artifacts expected by your API ----
    # Keep the exact feature order used by the model at train time
    feature_names = model_feature_cols.copy()

    # Medians for numeric imputation at inference time (API uses these)
    feature_medians = X_train[feature_names].median(numeric_only=True).to_dict()

    with open(OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    (OUT_DIR / "model_features.json").write_text(json.dumps(feature_names))
    (OUT_DIR / "feature_medians.json").write_text(json.dumps(feature_medians))
    (OUT_DIR / "model_meta.json").write_text(json.dumps({"version": MODEL_VERSION}))

    print(f"\n📦 KNN artifacts & plots written to: {OUT_DIR.resolve()}")
    print(" - model.pkl")
    print(" - model_features.json")
    print(" - feature_medians.json")
    print(" - model_meta.json")
    print(" - knn_parity_train.png")
    print(" - knn_parity_val.png")
    print(" - knn_parity_test.png")
    print(" - knn_hist_spe_train.png")
    print(" - knn_hist_spe_val.png")
    print(" - knn_hist_spe_test.png")
    print(" - knn_val_bad_preds_map.html")

if __name__ == "__main__":
    main()
