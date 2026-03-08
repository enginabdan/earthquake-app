import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def compute_history_features(
    samples: pd.DataFrame,
    catalog: pd.DataFrame,
    radius_km: float = 300.0,
) -> pd.DataFrame:
    """Causal features: only events strictly before sample time are used."""
    cat = catalog[["time_utc", "latitude", "longitude", "magnitude"]].copy()
    cat["time_utc"] = pd.to_datetime(cat["time_utc"], utc=True)
    cat = cat.sort_values("time_utc")

    out = []
    for r in samples.itertuples(index=False):
        t = pd.Timestamp(r.time_utc)
        prior = cat[cat["time_utc"] < t]

        if prior.empty:
            out.append(
                {
                    "hist_cnt_r300_1d": 0,
                    "hist_cnt_r300_7d": 0,
                    "hist_cnt_r300_30d": 0,
                    "hist_cnt_r300_365d": 0,
                    "hist_maxmag_r300_365d": 0.0,
                    "hist_meanmag_r300_365d": 0.0,
                    "hist_days_since_last_r300": 9999.0,
                }
            )
            continue

        dkm = haversine_km(r.latitude, r.longitude, prior["latitude"].values, prior["longitude"].values)
        local = prior[dkm <= radius_km].copy()

        if local.empty:
            out.append(
                {
                    "hist_cnt_r300_1d": 0,
                    "hist_cnt_r300_7d": 0,
                    "hist_cnt_r300_30d": 0,
                    "hist_cnt_r300_365d": 0,
                    "hist_maxmag_r300_365d": 0.0,
                    "hist_meanmag_r300_365d": 0.0,
                    "hist_days_since_last_r300": 9999.0,
                }
            )
            continue

        dt_days = (t - local["time_utc"]).dt.total_seconds() / 86400.0

        w1 = dt_days <= 1
        w7 = dt_days <= 7
        w30 = dt_days <= 30
        w365 = dt_days <= 365

        local365 = local[w365]
        if local365.empty:
            maxmag = 0.0
            meanmag = 0.0
        else:
            maxmag = float(local365["magnitude"].max())
            meanmag = float(local365["magnitude"].mean())

        out.append(
            {
                "hist_cnt_r300_1d": int(w1.sum()),
                "hist_cnt_r300_7d": int(w7.sum()),
                "hist_cnt_r300_30d": int(w30.sum()),
                "hist_cnt_r300_365d": int(w365.sum()),
                "hist_maxmag_r300_365d": maxmag,
                "hist_meanmag_r300_365d": meanmag,
                "hist_days_since_last_r300": float(dt_days.min()),
            }
        )

    return pd.DataFrame(out)
