from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import spiceypy as spice

from config import DATA_PROCESSED, DATA_RAW, KERNEL_SPK, KERNEL_TLS, OBS, PLANETS, REF_FRAME


def load_kernels() -> None:
    spice.furnsh(str(KERNEL_TLS))
    spice.furnsh(str(KERNEL_SPK))


def vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return np.zeros_like(v)
    return v / n


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    u1, u2 = unit(v1), unit(v2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def planet_vector_at(ts_utc: pd.Timestamp, target: str) -> np.ndarray:
    et = spice.str2et(ts_utc.strftime("%Y-%m-%dT%H:%M:%S"))
    vec, _lt = spice.spkpos(target, et, REF_FRAME, "NONE", OBS)
    return np.array(vec, dtype=float)


def build_feature_row(ts: pd.Timestamp) -> dict:
    row: dict[str, float | str] = {"time_utc": ts}
    vectors: dict[str, np.ndarray] = {}

    for p in PLANETS:
        v = planet_vector_at(ts, p)
        pname = p.split()[0].lower()
        vectors[pname] = v
        row[f"dist_{pname}"] = vec_norm(v)
        row[f"x_{pname}"] = float(v[0])
        row[f"y_{pname}"] = float(v[1])
        row[f"z_{pname}"] = float(v[2])

    base = vectors["jupiter"]
    for pname, v in vectors.items():
        if pname == "jupiter":
            continue
        row[f"angle_{pname}_jupiter_deg"] = angle_deg(v, base)

    return row


def sample_negative_times(eq_times: pd.Series, n: int, seed: int = 42) -> pd.Series:
    min_t = eq_times.min().floor("min")
    max_t = eq_times.max().ceil("min")
    total_min = int((max_t - min_t).total_seconds() // 60)
    rng = np.random.default_rng(seed)

    sampled = []
    blocked = set((eq_times.dt.floor("min")).astype(str).tolist())
    while len(sampled) < n:
        m = int(rng.integers(0, total_min))
        ts = min_t + timedelta(minutes=m)
        if ts.strftime("%Y-%m-%d %H:%M:00+00:00") in blocked:
            continue
        sampled.append(ts)
    return pd.Series(pd.to_datetime(sampled, utc=True), name="time_utc")


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    eq = pd.read_parquet(DATA_RAW / "earthquakes_m4_2000_2026.parquet")
    eq["time_utc"] = pd.to_datetime(eq["time_utc"], utc=True).dt.floor("min")
    eq = eq.drop_duplicates(subset=["time_utc"]).sort_values("time_utc")

    neg = sample_negative_times(eq["time_utc"], n=len(eq))
    all_times = pd.concat([eq["time_utc"], neg], ignore_index=True).drop_duplicates().sort_values()

    load_kernels()
    rows = [build_feature_row(ts) for ts in all_times]
    spice.kclear()

    df = pd.DataFrame(rows)
    df.to_parquet(DATA_PROCESSED / "planet_features_eq_and_controls.parquet", index=False)
    print(f"saved {len(df)} rows")


if __name__ == "__main__":
    main()
