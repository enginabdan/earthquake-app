import calendar
from datetime import datetime

import pandas as pd
import requests

from coord_dms import format_lat_dms, format_lon_dms
from config import DATA_RAW, END_DATE, MIN_MAGNITUDE, START_DATE, USGS_QUERY_URL


def month_edges(start: str, end: str):
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    y, m = s.year, s.month
    while True:
        last_day = calendar.monthrange(y, m)[1]
        left = datetime(y, m, 1)
        right = datetime(y, m, last_day, 23, 59, 59)
        if right > e:
            right = e
        if left >= e:
            break
        yield left, right
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def fetch_window(start_dt: datetime, end_dt: datetime) -> list[dict]:
    params = {
        "format": "geojson",
        "starttime": start_dt.isoformat(),
        "endtime": end_dt.isoformat(),
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time-asc",
        "limit": 20000,
    }
    resp = requests.get(USGS_QUERY_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for f in data.get("features", []):
        p = f.get("properties", {})
        g = f.get("geometry", {})
        c = (g.get("coordinates") or [None, None, None])
        rows.append(
            {
                "event_id": f.get("id"),
                "time_utc": pd.to_datetime(p.get("time"), unit="ms", utc=True),
                "magnitude": p.get("mag"),
                "place": p.get("place"),
                "longitude": c[0],
                "latitude": c[1],
                "depth_km": c[2],
            }
        )
    return rows


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for left, right in month_edges(START_DATE, END_DATE):
        all_rows.extend(fetch_window(left, right))

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["event_id"]).sort_values("time_utc")
    df["latitude_dms"] = df["latitude"].apply(lambda x: format_lat_dms(x) if pd.notna(x) else "")
    df["longitude_dms"] = df["longitude"].apply(lambda x: format_lon_dms(x) if pd.notna(x) else "")
    out = DATA_RAW / "earthquakes_m4_2000_2026.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
