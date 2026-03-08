from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
try:
    from streamlit_geolocation import streamlit_geolocation
except Exception:
    streamlit_geolocation = None

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "pipeline"
if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))

from pipeline.config import DATA_RAW, MODELS
from pipeline.coord_dms import format_lat_dms, format_lon_dms, parse_coordinate
from pipeline.generate_planet_features import build_feature_row, load_kernels
from pipeline.seismic_features import compute_history_features
from pipeline.train_location_models import location_features


st.set_page_config(page_title="Earthquake Risk Demo", layout="wide")
st.title("Earthquake Risk Demo")
st.caption("This interface is a research prototype and does not provide exact earthquake predictions.")


@st.cache_resource
def load_models():
    clf = joblib.load(MODELS / "location_classifier.joblib")
    calibrator = joblib.load(MODELS / "location_classifier_calibrator.joblib")
    ensemble = joblib.load(MODELS / "location_classifier_ensemble.joblib")
    reg = joblib.load(MODELS / "magnitude_regressor.joblib")
    reg_q10 = joblib.load(MODELS / "magnitude_regressor_q10.joblib")
    reg_q90 = joblib.load(MODELS / "magnitude_regressor_q90.joblib")
    feature_cols = json.loads((MODELS / "location_models_meta.json").read_text(encoding="utf-8"))["feature_columns"]
    return clf, calibrator, ensemble, reg, reg_q10, reg_q90, feature_cols


@st.cache_data
def load_eq_catalog():
    eq = pd.read_parquet(DATA_RAW / "earthquakes_m4_2000_2026.parquet")
    eq["time_utc"] = pd.to_datetime(eq["time_utc"], utc=True).dt.floor("min")
    return eq


def build_location_predictions(lat: float, lon: float, start_utc: pd.Timestamp, end_utc: pd.Timestamp, top_k: int):
    clf, calibrator, ensemble, reg, reg_q10, reg_q90, feature_cols = load_models()
    eq = load_eq_catalog()

    minutes = pd.date_range(start=start_utc, end=end_utc, freq="min", tz="UTC")
    if len(minutes) == 0:
        return pd.DataFrame()

    rows = []
    loc = location_features(lat, lon)

    load_kernels()
    try:
        for ts in minutes:
            feat = build_feature_row(ts)
            feat.update(loc)
            rows.append(feat)
    finally:
        import spiceypy as spice

        spice.kclear()

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    hist_input = pd.DataFrame({"time_utc": df["time_utc"], "latitude": lat, "longitude": lon})
    hist = compute_history_features(hist_input, eq, radius_km=300.0)
    df = pd.concat([df.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)

    X = df[feature_cols]
    p_cal = calibrator.predict_proba(X)[:, 1]
    ens_probs = np.vstack([m.predict_proba(X)[:, 1] for m in ensemble])
    p_std = ens_probs.std(axis=0)

    df["p_eq_m4_plus"] = p_cal
    df["p_eq_m4_plus_low95"] = np.clip(p_cal - 1.96 * p_std, 0.0, 1.0)
    df["p_eq_m4_plus_high95"] = np.clip(p_cal + 1.96 * p_std, 0.0, 1.0)
    df["pred_magnitude_if_event"] = reg.predict(X)
    df["pred_magnitude_q10"] = reg_q10.predict(X)
    df["pred_magnitude_q90"] = reg_q90.predict(X)
    df["latitude_dms"] = format_lat_dms(lat)
    df["longitude_dms"] = format_lon_dms(lon)

    return df.sort_values("p_eq_m4_plus", ascending=False)


def pick_top_with_min_gap(df: pd.DataFrame, top_k: int, min_gap_minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    if min_gap_minutes <= 0:
        return df.head(top_k)

    selected = []
    min_gap = pd.Timedelta(minutes=int(min_gap_minutes))
    for row in df.itertuples(index=False):
        t = pd.Timestamp(row.time_utc)
        if all(abs(t - pd.Timestamp(r["time_utc"])) >= min_gap for r in selected):
            selected.append(dict(zip(df.columns, row)))
        if len(selected) >= top_k:
            break
    return pd.DataFrame(selected)


def build_grid_predictions(
    ts_utc: pd.Timestamp,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step: float,
):
    clf, calibrator, ensemble, reg, reg_q10, reg_q90, feature_cols = load_models()
    eq = load_eq_catalog()

    lats = np.arange(lat_min, lat_max + 1e-9, step)
    lons = np.arange(lon_min, lon_max + 1e-9, step)

    load_kernels()
    try:
        base = build_feature_row(ts_utc)
    finally:
        import spiceypy as spice

        spice.kclear()

    rows = []
    for lat in lats:
        for lon in lons:
            r = dict(base)
            r.update(location_features(float(lat), float(lon)))
            rows.append(r)

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    hist_input = df[["time_utc", "latitude", "longitude"]].copy()
    hist = compute_history_features(hist_input, eq, radius_km=300.0)
    df = pd.concat([df.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)

    X = df[feature_cols]
    p_cal = calibrator.predict_proba(X)[:, 1]
    ens_probs = np.vstack([m.predict_proba(X)[:, 1] for m in ensemble])
    p_std = ens_probs.std(axis=0)

    df["p_eq_m4_plus"] = p_cal
    df["p_eq_m4_plus_low95"] = np.clip(p_cal - 1.96 * p_std, 0.0, 1.0)
    df["p_eq_m4_plus_high95"] = np.clip(p_cal + 1.96 * p_std, 0.0, 1.0)
    df["pred_magnitude_if_event"] = reg.predict(X)
    df["pred_magnitude_q10"] = reg_q10.predict(X)
    df["pred_magnitude_q90"] = reg_q90.predict(X)
    df["latitude_dms"] = df["latitude"].apply(format_lat_dms)
    df["longitude_dms"] = df["longitude"].apply(format_lon_dms)
    return df


tab1, tab2 = st.tabs(["Location Forecast", "Grid Heatmap"])

with tab1:
    st.subheader("Location Forecast")
    if "selected_lat" not in st.session_state:
        st.session_state["selected_lat"] = 39.93
    if "selected_lon" not in st.session_state:
        st.session_state["selected_lon"] = 32.85

    st.markdown("Allow location access to center the map on your current position. You can click the map to refine the point.")
    if streamlit_geolocation is not None:
        browser_loc = streamlit_geolocation()
        if browser_loc and browser_loc.get("latitude") is not None and browser_loc.get("longitude") is not None:
            st.session_state["selected_lat"] = float(browser_loc["latitude"])
            st.session_state["selected_lon"] = float(browser_loc["longitude"])
    else:
        st.info("Install 'streamlit-geolocation' to auto-center the map on your current location.")

    m = folium.Map(
        location=[st.session_state["selected_lat"], st.session_state["selected_lon"]],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.Marker(
        [st.session_state["selected_lat"], st.session_state["selected_lon"]],
        tooltip="Selected Location",
    ).add_to(m)
    folium.LatLngPopup().add_to(m)
    map_data = st_folium(m, height=420, width=None, key="location_picker")
    if map_data and map_data.get("last_clicked"):
        st.session_state["selected_lat"] = float(map_data["last_clicked"]["lat"])
        st.session_state["selected_lon"] = float(map_data["last_clicked"]["lng"])

    lat = float(st.session_state["selected_lat"])
    lon = float(st.session_state["selected_lon"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latitude (decimal)", f"{lat:.6f}")
    c2.metric("Longitude (decimal)", f"{lon:.6f}")
    c3.metric("Latitude (DMS)", format_lat_dms(lat))
    c4.metric("Longitude (DMS)", format_lon_dms(lon))

    c5, c6 = st.columns(2)
    top_k = c5.number_input("Top-K", min_value=5, max_value=200, value=20, step=5)
    min_gap_min = c6.number_input("Minimum gap between shown rows (minutes)", min_value=0, max_value=1440, value=360, step=30)

    d1, d2 = st.columns(2)
    start_date = d1.date_input("Start date (UTC)", datetime(2026, 4, 1).date())
    end_date = d2.date_input("End date (UTC)", datetime(2026, 4, 3).date())
    day_span = (end_date - start_date).days + 1
    if day_span > 31:
        st.warning(f"The selected range is {day_span} days. Computation may take longer.")

    if st.button("Run Forecast", type="primary"):
        try:
            start_ts = pd.Timestamp(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc))
            end_ts = pd.Timestamp(datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=23, minutes=59))
            with st.spinner("Computing forecast, please wait..."):
                ranked = build_location_predictions(lat, lon, start_ts, end_ts, int(top_k))
                out = pick_top_with_min_gap(ranked, int(top_k), int(min_gap_min))
                if out.empty:
                    st.warning("No rows left after applying the minimum-gap filter. Reduce the gap and try again.")
                    st.stop()

                out["uncertainty_width"] = out["p_eq_m4_plus_high95"] - out["p_eq_m4_plus_low95"]
                out["risk_level"] = pd.cut(
                    out["p_eq_m4_plus"],
                    bins=[-0.001, 0.33, 0.66, 1.001],
                    labels=["Low", "Medium", "High"],
                )
                out["confidence_level"] = pd.cut(
                    out["uncertainty_width"],
                    bins=[-0.001, 0.20, 0.40, 10.0],
                    labels=["High", "Medium", "Low"],
                )

            st.success("Forecast completed.")
            st.warning(
                "This output is probabilistic research output, not a deterministic earthquake prediction. "
                "Use it only as an exploratory signal."
            )
            st.dataframe(
                out[
                    [
                        "time_utc",
                        "latitude_dms",
                        "longitude_dms",
                        "p_eq_m4_plus",
                        "p_eq_m4_plus_low95",
                        "p_eq_m4_plus_high95",
                        "risk_level",
                        "confidence_level",
                        "pred_magnitude_if_event",
                        "pred_magnitude_q10",
                        "pred_magnitude_q90",
                    ]
                ],
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Error: {exc}")


with tab2:
    st.subheader("Grid Heatmap")
    g1, g2, g3 = st.columns(3)
    grid_time = g1.text_input("Grid timestamp (UTC)", "2026-04-01T00:00:00Z")
    step = g2.number_input("Step (degrees)", min_value=0.5, max_value=10.0, value=5.0, step=0.5)
    grid_topk = g3.number_input("Top-K", min_value=5, max_value=500, value=20, step=5)

    b1, b2, b3, b4 = st.columns(4)
    lat_min_in = b1.text_input("Lat Min (DMS)", "30.00.00N")
    lat_max_in = b2.text_input("Lat Max (DMS)", "50.00.00N")
    lon_min_in = b3.text_input("Lon Min (DMS)", "20.00.00E")
    lon_max_in = b4.text_input("Lon Max (DMS)", "45.00.00E")

    if st.button("Generate Grid"):
        try:
            ts = pd.Timestamp(grid_time)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")

            lat_min = parse_coordinate(lat_min_in, is_lat=True)
            lat_max = parse_coordinate(lat_max_in, is_lat=True)
            lon_min = parse_coordinate(lon_min_in, is_lat=False)
            lon_max = parse_coordinate(lon_max_in, is_lat=False)

            with st.spinner("Computing grid, please wait..."):
                grid = build_grid_predictions(ts, lat_min, lat_max, lon_min, lon_max, float(step))
                top = grid.sort_values("p_eq_m4_plus", ascending=False).head(int(grid_topk))

            fig, ax = plt.subplots(figsize=(10, 5))
            sc = ax.scatter(grid["longitude"], grid["latitude"], c=grid["p_eq_m4_plus"], cmap="inferno", s=55, alpha=0.9)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Grid Risk Heatmap")
            plt.colorbar(sc, ax=ax, label="Calibrated P(M>=4)")
            st.pyplot(fig, clear_figure=True, use_container_width=True)

            st.dataframe(
                top[
                    [
                        "latitude_dms",
                        "longitude_dms",
                        "p_eq_m4_plus",
                        "p_eq_m4_plus_low95",
                        "p_eq_m4_plus_high95",
                        "pred_magnitude_if_event",
                    ]
                ],
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Error: {exc}")
