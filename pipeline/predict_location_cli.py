import argparse
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
    p = argparse.ArgumentParser(description="Predict top earthquake-risk minutes for a location")
    p.add_argument("--lat", type=str, required=True, help="Decimal or DMS, e.g. 39.93 or 39.55.48N")
    p.add_argument("--lon", type=str, required=True, help="Decimal or DMS, e.g. 32.85 or 32.51.00E")
    p.add_argument("--start", type=str, required=True, help="UTC, e.g. 2026-04-01T00:00:00Z")
    p.add_argument("--end", type=str, required=True, help="UTC, e.g. 2026-04-03T00:00:00Z")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def json_load_feature_columns(path: Path) -> list:
    import json

    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["feature_columns"]


def main():
    args = parse_args()
    lat = parse_coordinate(args.lat, is_lat=True)
    lon = parse_coordinate(args.lon, is_lat=False)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    minutes = pd.date_range(start=start, end=end, freq="min", tz="UTC")

    if len(minutes) > 60 * 24 * 31:
        raise ValueError("Max range is 31 days for one run.")

    clf = joblib.load(MODELS / "location_classifier.joblib")
    calibrator = joblib.load(MODELS / "location_classifier_calibrator.joblib")
    ensemble = joblib.load(MODELS / "location_classifier_ensemble.joblib")
    reg = joblib.load(MODELS / "magnitude_regressor.joblib")
    reg_q10 = joblib.load(MODELS / "magnitude_regressor_q10.joblib")
    reg_q90 = joblib.load(MODELS / "magnitude_regressor_q90.joblib")

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
    eq = pd.read_parquet(DATA_RAW / "earthquakes_m4_2000_2026.parquet")
    eq["time_utc"] = pd.to_datetime(eq["time_utc"], utc=True).dt.floor("min")
    hist = compute_history_features(hist_input, eq, radius_km=300.0)
    df = pd.concat([df.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)

    feature_cols = json_load_feature_columns(MODELS / "location_models_meta.json")
    X = df[feature_cols]

    p_raw = clf.predict_proba(X)[:, 1]
    p_cal = calibrator.predict_proba(X)[:, 1]
    ens_probs = np.vstack([m.predict_proba(X)[:, 1] for m in ensemble])
    p_std = ens_probs.std(axis=0)
    p_low = np.clip(p_cal - 1.96 * p_std, 0.0, 1.0)
    p_high = np.clip(p_cal + 1.96 * p_std, 0.0, 1.0)

    m_point = reg.predict(X)
    m_lo = reg_q10.predict(X)
    m_hi = reg_q90.predict(X)

    df["p_eq_m4_plus_raw"] = p_raw
    df["p_eq_m4_plus"] = p_cal
    df["p_eq_m4_plus_low95"] = p_low
    df["p_eq_m4_plus_high95"] = p_high
    df["pred_magnitude_if_event"] = m_point
    df["pred_magnitude_q10"] = m_lo
    df["pred_magnitude_q90"] = m_hi
    df["latitude_dms"] = format_lat_dms(lat)
    df["longitude_dms"] = format_lon_dms(lon)

    out_df = df.sort_values("p_eq_m4_plus", ascending=False).head(args.top_k)

    out = Path(args.out) if args.out else MODELS / "location_predictions_topk.csv"
    out_df.to_csv(out, index=False)

    print(f"saved top-{args.top_k} -> {out}")
    print(
        out_df[
            [
                "time_utc",
                "latitude_dms",
                "longitude_dms",
                "p_eq_m4_plus",
                "p_eq_m4_plus_low95",
                "p_eq_m4_plus_high95",
                "pred_magnitude_if_event",
                "pred_magnitude_q10",
                "pred_magnitude_q90",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
