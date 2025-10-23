# app/train_model.py
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
import os
import requests  # NEW: to call your API

# ---------------- Config ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_PATH = DATA_DIR / "kc_house_data.csv"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"

MODEL_VERSION = "v1"

# include lat/long so we can map points
SALES_COLUMN_SELECTION: List[str] = [
    'price', 'sqft_living', 'bedrooms', 'sqft_lot', 'floors',
    'yr_built', 'yr_renovated', 'sqft_above', 'sqft_basement',
    'waterfront', 'view', 'condition', 'grade',
    'sqft_living15', 'bathrooms', 'sqft_lot15',
    'zipcode', 'lat', 'long'
]

DEMOGRAPHIC_COLUMN_SELECTION: List[str] = [
    'ppltn_qty','urbn_ppltn_qty','sbrbn_ppltn_qty','farm_ppltn_qty','non_farm_qty',
    'medn_hshld_incm_amt','medn_incm_per_prsn_amt','hous_val_amt',
    'edctn_less_than_9_qty','edctn_9_12_qty','edctn_high_schl_qty','edctn_some_clg_qty',
    'edctn_assoc_dgre_qty','edctn_bchlr_dgre_qty','edctn_prfsnl_qty',
    'per_urbn','per_sbrbn','per_farm','per_non_farm',
    'per_less_than_9','per_9_to_12','per_hsd','per_some_clg','per_assoc','per_bchlr','per_prfsnl',
    'zipcode'
]

# ---------------- Data loading ----------------
def load_data(
    sales_path: Path,
    demographics_path: Path,
    sales_cols: List[str],
    demo_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(sales_path, usecols=sales_cols, dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path, usecols=demo_cols, dtype={'zipcode': str})
    merged = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    y = merged.pop("price")
    X = merged
    print(f"✅ Loaded {X.shape[0]:,} rows and {X.shape[1]:,} features after merge.")
    return X, y

# ---------------- Metrics & plots ----------------
def signed_pe(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed percentage error: (pred - actual) / actual."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return (y_pred - y_true) / y_true

def bias_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    spe = signed_pe(y_true, y_pred)
    me  = float(np.mean(y_pred - y_true))                # Mean Error ($)
    mpe = float(np.mean(spe))                            # Mean % Error (signed)
    mdpe = float(np.median(spe))                         # Median % Error (signed)
    over_rate = float(np.mean(y_pred > y_true))          # fraction over-predicted
    return {"ME": me, "MPE": mpe, "MdPE": mdpe, "over_rate": over_rate}

def plot_signed_pe_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    spe_pct = signed_pe(y_true, y_pred) * 100.0
    spe_pct = np.clip(spe_pct, -200, 200)  # cap for readability
    plt.figure(figsize=(6, 4))
    plt.hist(spe_pct, bins=40, alpha=0.9)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Signed % Error ( (pred-actual)/actual × 100 )")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))

def eval_split(name: str, model: XGBRegressor, X: pd.DataFrame, y: pd.Series) -> dict:
    yhat = model.predict(X)
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

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape_val,
        "bias": bias,
        "yhat": yhat
    }


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

# -------- NEW: interactive folium map --------
def map_bad_preds_val_folium(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    yhat_val: np.ndarray,
    out_path: Path,
    threshold: float = 0.10
) -> None:
    # Need coordinates
    if not {"lat", "long"}.issubset(X_val.columns):
        print("⚠️ Skipping folium map: lat/long not present.")
        return

    df = X_val.copy()
    df["actual"] = y_val.values
    df["pred"] = yhat_val
    df["ape"] = np.abs(df["actual"] - df["pred"]) / df["actual"]

    # Basic center (Seattle area) — fallback to mean of points
    center_lat = float(df["lat"].mean()) if df["lat"].notna().any() else 47.6062
    center_lon = float(df["long"].mean()) if df["long"].notna().any() else -122.3321

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")

    # Colormap for bad points (10% → yellow, 50% → red)
    vmax = max(0.5, float(df["ape"].quantile(0.99)))  # cap for robust coloring
    cmap = cm.LinearColormap(colors=["#fee08b", "#fdae61", "#f46d43", "#d73027"],
                             vmin=threshold, vmax=vmax)
    cmap.caption = "Absolute Percentage Error (Validation)"

    # Plot all points (light gray) with clustering
    all_cluster = MarkerCluster(name="All validation points", disableClusteringAtZoom=14)
    bad_cluster = MarkerCluster(name=f"> {int(threshold*100)}% error", disableClusteringAtZoom=14)

    for _, r in df.iterrows():
        lat, lon = r["lat"], r["long"]
        if pd.isna(lat) or pd.isna(lon):
            continue


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
            # color by error magnitude and size scaled by error
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

# ---------------- Diagnostics ----------------
def save_feature_importance(model: XGBRegressor, feature_names: List[str]) -> None:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    idx_to_name = {f"f{i}": name for i, name in enumerate(feature_names)}
    rows = [(idx_to_name.get(k, k), v) for k, v in score.items()]
    missing = set(feature_names) - {idx_to_name.get(k, k) for k in score.keys()}
    rows.extend([(m, 0.0) for m in missing])

    importances = pd.DataFrame(rows, columns=["feature", "gain_importance"]).sort_values(
        "gain_importance", ascending=False
    )
    importances.to_csv(OUT_DIR / "xgb_feature_importance.csv", index=False)
    print("\n🔎 Top features by XGBoost (gain):")
    print(importances.head(15).to_string(index=False))

def show_feature_target_correlation(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    numeric = x_train.select_dtypes(include="number")
    corr = numeric.corrwith(y_train).sort_values(ascending=False)
    corr.to_csv(OUT_DIR / "feature_target_correlation.csv", header=["correlation"])
    print("\n🎯 Correlation with target (price):")
    print(corr.head(15).to_string())

# ---------------- Main ----------------
def main() -> None:
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION, DEMOGRAPHIC_COLUMN_SELECTION)

    # Train/Test split
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    # Diagnostics
    show_feature_target_correlation(x_train, y_train)

    # Model
    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )

    # Early stopping split from training portion → yields train (x_tr), validation (x_val)
    x_tr, x_val, y_tr, y_val = model_selection.train_test_split(
        x_train, y_train, test_size=0.15, random_state=42
    )

    model.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val)],
        verbose=True
    )

    # ---- Evaluate on train/val/test with MAE & MAPE ----
    train_metrics = eval_split("Train", model, x_tr, y_tr)
    val_metrics   = eval_split("Validation", model, x_val, y_val)
    test_metrics  = eval_split("Test (held-out)", model, x_test, y_test)

    # ---- Bias histograms (signed % error) ----
    plot_signed_pe_hist(y_tr.values,  train_metrics["yhat"], "Signed % Error: Train", OUT_DIR / "hist_spe_train.png")
    plot_signed_pe_hist(y_val.values, val_metrics["yhat"], "Signed % Error: Validation", OUT_DIR / "hist_spe_val.png")
    plot_signed_pe_hist(y_test.values, test_metrics["yhat"], "Signed % Error: Test", OUT_DIR / "hist_spe_test.png")

    # Optionally dump numeric metrics to JSON for dashboards
    (OUT_DIR / "metrics.json").write_text(json.dumps({
        "train": {k: (v if k != "yhat" else None) for k,v in train_metrics.items()},
        "val":   {k: (v if k != "yhat" else None) for k,v in val_metrics.items()},
        "test":  {k: (v if k != "yhat" else None) for k,v in test_metrics.items()},
    }, indent=2))


    # ---- Save parity plots ----
    parity_plot(y_tr.values, train_metrics["yhat"], "Parity: Train", OUT_DIR / "parity_train.png")
    parity_plot(y_val.values, val_metrics["yhat"], "Parity: Validation", OUT_DIR / "parity_val.png")
    parity_plot(y_test.values, test_metrics["yhat"], "Parity: Test", OUT_DIR / "parity_test.png")

    # ---- Folium map of >10% error points on Validation ----
    map_bad_preds_val_folium(x_val, y_val, val_metrics["yhat"], OUT_DIR / "val_bad_preds_map.html", threshold=0.10)

    # ---- Save artifacts expected by API ----
    feature_names = list(x_train.columns)
    feature_medians = x_train.median(numeric_only=True).to_dict()

    with open(OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    (OUT_DIR / "model_features.json").write_text(json.dumps(feature_names))
    (OUT_DIR / "feature_medians.json").write_text(json.dumps(feature_medians))
    (OUT_DIR / "model_meta.json").write_text(json.dumps({"version": MODEL_VERSION}))

    # Diagnostics
    save_feature_importance(model, feature_names)

    print(f"\n📦 Artifacts & plots written to: {OUT_DIR.resolve()}")
    print(" - model.pkl")
    print(" - model_features.json")
    print(" - feature_medians.json")
    print(" - model_meta.json")
    print(" - xgb_feature_importance.csv")
    print(" - feature_target_correlation.csv")
    print(" - parity_train.png")
    print(" - parity_val.png")
    print(" - parity_test.png")
    print(" - val_bad_preds_map.html")

   

if __name__ == "__main__":
    main()
