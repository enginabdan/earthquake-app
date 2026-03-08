import pandas as pd

from config import DATA_PROCESSED, DATA_RAW


def main():
    eq = pd.read_parquet(DATA_RAW / "earthquakes_m4_2000_2026.parquet")
    eq["time_utc"] = pd.to_datetime(eq["time_utc"], utc=True).dt.floor("min")
    eq = eq.drop_duplicates(subset=["time_utc"])
    eq["target_eq_m4_plus"] = 1

    feats = pd.read_parquet(DATA_PROCESSED / "planet_features_eq_and_controls.parquet")
    feats["time_utc"] = pd.to_datetime(feats["time_utc"], utc=True)

    ds = feats.merge(eq[["time_utc", "target_eq_m4_plus", "latitude", "longitude", "magnitude"]], on="time_utc", how="left")
    ds["target_eq_m4_plus"] = ds["target_eq_m4_plus"].fillna(0).astype(int)

    out = DATA_PROCESSED / "model_dataset.parquet"
    ds.to_parquet(out, index=False)
    print(ds["target_eq_m4_plus"].value_counts(dropna=False).to_string())
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
