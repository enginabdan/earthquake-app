import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from coord_dms import format_lat_dms, format_lon_dms, parse_coordinate
from config import DATA_RAW, MODELS
from generate_planet_features import build_feature_row, load_kernels
from seismic_features import compute_history_features
from train_location_models import location_features


def parse_args():
    p = argparse.ArgumentParser(description="Predict risk on a lat/lon grid for one timestamp")
    p.add_argument("--time", type=str, required=True, help="UTC, e.g. 2026-04-01T00:00:00Z")
    p.add_argument("--lat-min", type=str, default="60.00.00S", help="Decimal or DMS")
    p.add_argument("--lat-max", type=str, default="60.00.00N", help="Decimal or DMS")
    p.add_argument("--lon-min", type=str, default="180.00.00W", help="Decimal or DMS")
    p.add_argument("--lon-max", type=str, default="180.00.00E", help="Decimal or DMS")
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def load_feature_cols() -> list[str]:
    obj = json.loads((MODELS / "location_models_meta.json").read_text(encoding="utf-8"))
    return obj["feature_columns"]


def main():
    args = parse_args()
    lat_min = parse_coordinate(args.lat_min, is_lat=True)
    lat_max = parse_coordinate(args.lat_max, is_lat=True)
    lon_min = parse_coordinate(args.lon_min, is_lat=False)
    lon_max = parse_coordinate(args.lon_max, is_lat=False)

    ts = pd.Timestamp(args.time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    lats = np.arange(lat_min, lat_max + 1e-9, args.step)
    lons = np.arange(lon_min, lon_max + 1e-9, args.step)

    clf = joblib.load(MODELS / "location_classifier.joblib")
    calibrator = joblib.load(MODELS / "location_classifier_calibrator.joblib")
    ensemble = joblib.load(MODELS / "location_classifier_ensemble.joblib")
    reg = joblib.load(MODELS / "magnitude_regressor.joblib")
    reg_q10 = joblib.load(MODELS / "magnitude_regressor_q10.joblib")
    reg_q90 = joblib.load(MODELS / "magnitude_regressor_q90.joblib")

    base_planet = None
    load_kernels()
    try:
        base_planet = build_feature_row(ts)
    finally:
        import spiceypy as spice

        spice.kclear()

    rows = []
    for lat in lats:
        for lon in lons:
            r = dict(base_planet)
            r.update(location_features(float(lat), float(lon)))
            rows.append(r)

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)

    hist_input = df[["time_utc", "latitude", "longitude"]].copy()
    eq = pd.read_parquet(DATA_RAW / "earthquakes_m4_2000_2026.parquet")
    eq["time_utc"] = pd.to_datetime(eq["time_utc"], utc=True).dt.floor("min")
    hist = compute_history_features(hist_input, eq, radius_km=300.0)
    df = pd.concat([df.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)

    feature_cols = load_feature_cols()
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

    out = Path(args.out) if args.out else MODELS / "grid_risk_map.csv"
    df.to_csv(out, index=False)

    top = df.sort_values("p_eq_m4_plus", ascending=False).head(args.top_k)
    top_out = out.with_name(out.stem + "_topk.csv")
    top.to_csv(top_out, index=False)

    print(f"saved full grid -> {out}")
    print(f"saved top-{args.top_k} -> {top_out}")
    print(
        top[
            ["latitude_dms", "longitude_dms", "p_eq_m4_plus", "pred_magnitude_if_event"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
