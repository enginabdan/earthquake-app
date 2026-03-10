import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline

from config import DATA_RAW, MODELS
from generate_planet_features import build_feature_row, load_kernels
from seismic_features import compute_history_features


CUTOFF = "2018-01-01T00:00:00Z"
ENSEMBLE_SEEDS = [11, 22, 33, 44, 55]


def location_features(lat: float, lon: float) -> dict:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "abs_latitude": float(abs(lat)),
        "sin_lat": float(np.sin(lat_r)),
        "cos_lat": float(np.cos(lat_r)),
        "sin_lon": float(np.sin(lon_r)),
        "cos_lon": float(np.cos(lon_r)),
    }


def build_rows(eq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    load_kernels()
    try:
        for r in eq.itertuples(index=False):
            feat = build_feature_row(r.time_utc)
            feat.update(location_features(r.latitude, r.longitude))
            feat["target_eq_m4_plus"] = 1
            feat["magnitude"] = float(r.magnitude)
            rows.append(feat)
    finally:
        import spiceypy as spice

        spice.kclear()
    return pd.DataFrame(rows)


def build_negative_rows(eq: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    min_t = eq["time_utc"].min().floor("min")
    max_t = eq["time_utc"].max().ceil("min")
    total_min = int((max_t - min_t).total_seconds() // 60)

    idx = rng.integers(0, len(eq), size=n)
    sampled_locs = eq.iloc[idx][["latitude", "longitude"]].reset_index(drop=True)

    times = [min_t + pd.Timedelta(minutes=int(rng.integers(0, total_min))) for _ in range(n)]
    neg = pd.DataFrame({"time_utc": pd.to_datetime(times, utc=True)})
    neg = pd.concat([neg, sampled_locs], axis=1)

    rows = []
    load_kernels()
    try:
        for r in neg.itertuples(index=False):
            feat = build_feature_row(r.time_utc)
            feat.update(location_features(r.latitude, r.longitude))
            feat["target_eq_m4_plus"] = 0
            feat["magnitude"] = np.nan
            rows.append(feat)
    finally:
        import spiceypy as spice

        spice.kclear()

    return pd.DataFrame(rows)


def hgb_clf(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    random_state=seed,
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=350,
                ),
            ),
        ]
    )


def hgb_reg(seed: int, loss: str = "squared_error", quantile: Optional[float] = None) -> Pipeline:
    kwargs = {
        "random_state": seed,
        "learning_rate": 0.05,
        "max_depth": 4,
        "max_iter": 300,
        "loss": loss,
    }
    if loss == "quantile" and quantile is not None:
        kwargs["quantile"] = quantile

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(**kwargs)),
        ]
    )


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    eq = pd.read_parquet(DATA_RAW / "earthquakes_m4_2000_2026.parquet")
    eq["time_utc"] = pd.to_datetime(eq["time_utc"], utc=True).dt.floor("min")
    eq = eq.dropna(subset=["latitude", "longitude", "magnitude"]).drop_duplicates(subset=["event_id"])

    pos_df = build_rows(eq[["time_utc", "latitude", "longitude", "magnitude"]])
    neg_df = build_negative_rows(eq[["time_utc", "latitude", "longitude", "magnitude"]], n=len(pos_df))
    ds = pd.concat([pos_df, neg_df], ignore_index=True)

    hist = compute_history_features(ds[["time_utc", "latitude", "longitude"]], eq)
    ds = pd.concat([ds.reset_index(drop=True), hist], axis=1)

    ds = ds.sort_values("time_utc").reset_index(drop=True)
    cutoff = pd.Timestamp(CUTOFF)
    train_df = ds[ds["time_utc"] < cutoff].copy()
    test_df = ds[ds["time_utc"] >= cutoff].copy()

    feature_cols = [c for c in ds.columns if c not in ["time_utc", "target_eq_m4_plus", "magnitude"]]

    # Time-aware train/calibration split inside train window
    n_train = len(train_df)
    n_cal = max(200, int(n_train * 0.2))
    core_df = train_df.iloc[: n_train - n_cal]
    cal_df = train_df.iloc[n_train - n_cal :]

    X_core = core_df[feature_cols]
    y_core = core_df["target_eq_m4_plus"].astype(int)
    X_cal = cal_df[feature_cols]
    y_cal = cal_df["target_eq_m4_plus"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["target_eq_m4_plus"].astype(int)

    clf = hgb_clf(seed=42)
    clf.fit(X_core, y_core)
    p_raw = clf.predict_proba(X_test)[:, 1]

    calibrator = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    calibrator.fit(X_cal, y_cal)
    p_cal = calibrator.predict_proba(X_test)[:, 1]

    auc_raw = float(roc_auc_score(y_test, p_raw))
    auc_cal = float(roc_auc_score(y_test, p_cal))
    brier_raw = float(brier_score_loss(y_test, p_raw))
    brier_cal = float(brier_score_loss(y_test, p_cal))

    pi = permutation_importance(clf, X_test, y_test, n_repeats=8, random_state=42, scoring="roc_auc")
    imp_df = pd.DataFrame({"feature": feature_cols, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
    imp_df.sort_values("importance_mean", ascending=False).to_csv(
        MODELS / "location_classifier_feature_importance.csv", index=False
    )

    ensemble = []
    ens_pred = []
    for s in ENSEMBLE_SEEDS:
        rng = np.random.default_rng(s)
        idx = rng.integers(0, len(X_core), size=len(X_core))
        X_boot = X_core.iloc[idx]
        y_boot = y_core.iloc[idx]
        m = hgb_clf(seed=s)
        m.fit(X_boot, y_boot)
        ensemble.append(m)
        ens_pred.append(m.predict_proba(X_test)[:, 1])
    ens_pred = np.vstack(ens_pred)
    ens_std_mean = float(ens_pred.std(axis=0).mean())

    pos_only = ds[ds["target_eq_m4_plus"] == 1].copy()
    pos_train = pos_only[pos_only["time_utc"] < cutoff]
    pos_test = pos_only[pos_only["time_utc"] >= cutoff]

    Xr_train = pos_train[feature_cols]
    yr_train = pos_train["magnitude"].astype(float)
    Xr_test = pos_test[feature_cols]
    yr_test = pos_test["magnitude"].astype(float)

    reg_point = hgb_reg(seed=42, loss="squared_error")
    reg_q10 = hgb_reg(seed=42, loss="quantile", quantile=0.1)
    reg_q90 = hgb_reg(seed=42, loss="quantile", quantile=0.9)

    reg_point.fit(Xr_train, yr_train)
    reg_q10.fit(Xr_train, yr_train)
    reg_q90.fit(Xr_train, yr_train)

    yhat = reg_point.predict(Xr_test)
    ylo = reg_q10.predict(Xr_test)
    yhi = reg_q90.predict(Xr_test)

    mae = float(mean_absolute_error(yr_test, yhat))
    in_band = float(((yr_test >= ylo) & (yr_test <= yhi)).mean())

    joblib.dump(clf, MODELS / "location_classifier.joblib")
    joblib.dump(calibrator, MODELS / "location_classifier_calibrator.joblib")
    joblib.dump(ensemble, MODELS / "location_classifier_ensemble.joblib")

    joblib.dump(reg_point, MODELS / "magnitude_regressor.joblib")
    joblib.dump(reg_q10, MODELS / "magnitude_regressor_q10.joblib")
    joblib.dump(reg_q90, MODELS / "magnitude_regressor_q90.joblib")

    meta = {
        "cutoff": CUTOFF,
        "n_rows": int(len(ds)),
        "n_positive": int(ds["target_eq_m4_plus"].sum()),
        "n_negative": int((ds["target_eq_m4_plus"] == 0).sum()),
        "classifier_roc_auc_raw": auc_raw,
        "classifier_roc_auc_calibrated": auc_cal,
        "classifier_brier_raw": brier_raw,
        "classifier_brier_calibrated": brier_cal,
        "ensemble_n_models": len(ENSEMBLE_SEEDS),
        "ensemble_mean_std": ens_std_mean,
        "regressor_mae": mae,
        "regressor_q10_q90_coverage": in_band,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_columns": feature_cols,
    }
    (MODELS / "location_models_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
