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
    """Causal features using a bounded 365-day lookback window per sample."""
    cat = catalog[["time_utc", "latitude", "longitude", "magnitude"]].copy()
    cat["time_utc"] = pd.to_datetime(cat["time_utc"], utc=True).dt.tz_localize(None)
    cat = cat.sort_values("time_utc", kind="mergesort").reset_index(drop=True)

    cat_t = cat["time_utc"].to_numpy(dtype="datetime64[ns]")
    cat_lat = cat["latitude"].to_numpy(dtype=np.float64)
    cat_lon = cat["longitude"].to_numpy(dtype=np.float64)
    cat_mag = cat["magnitude"].to_numpy(dtype=np.float64)

    s = samples[["time_utc", "latitude", "longitude"]].copy()
    s["time_utc"] = pd.to_datetime(s["time_utc"], utc=True).dt.tz_localize(None)

    out = []
    lookback = np.timedelta64(365, "D")

    for r in s.itertuples(index=False):
        t = np.datetime64(r.time_utc.to_datetime64(), "ns")
        right = np.searchsorted(cat_t, t, side="left")
        if right <= 0:
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

        left = np.searchsorted(cat_t, t - lookback, side="left")
        if left >= right:
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

        sl_lat = cat_lat[left:right]
        sl_lon = cat_lon[left:right]
        sl_t = cat_t[left:right]
        sl_mag = cat_mag[left:right]

        dkm = haversine_km(float(r.latitude), float(r.longitude), sl_lat, sl_lon)
        m_local = dkm <= radius_km

        if not np.any(m_local):
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

        local_t = sl_t[m_local]
        local_mag = sl_mag[m_local]
        dt_days = (t - local_t) / np.timedelta64(1, "D")

        w1 = dt_days <= 1.0
        w7 = dt_days <= 7.0
        w30 = dt_days <= 30.0
        w365 = dt_days <= 365.0

        if np.any(w365):
            mag365 = local_mag[w365]
            maxmag = float(np.max(mag365))
            meanmag = float(np.mean(mag365))
        else:
            maxmag = 0.0
            meanmag = 0.0

        out.append(
            {
                "hist_cnt_r300_1d": int(np.sum(w1)),
                "hist_cnt_r300_7d": int(np.sum(w7)),
                "hist_cnt_r300_30d": int(np.sum(w30)),
                "hist_cnt_r300_365d": int(np.sum(w365)),
                "hist_maxmag_r300_365d": maxmag,
                "hist_meanmag_r300_365d": meanmag,
                "hist_days_since_last_r300": float(np.min(dt_days)) if dt_days.size else 9999.0,
            }
        )

    return pd.DataFrame(out)
