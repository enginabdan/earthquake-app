import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from config import DATA_PROCESSED, MODELS


CUTS = [
    ("2008-01-01T00:00:00Z", "2012-01-01T00:00:00Z"),
    ("2012-01-01T00:00:00Z", "2016-01-01T00:00:00Z"),
    ("2016-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    ("2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROCESSED / "model_dataset.parquet")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.sort_values("time_utc").reset_index(drop=True)

    drop_cols = ["time_utc", "target_eq_m4_plus", "latitude", "longitude", "magnitude"]
    folds = []

    for train_end_s, test_end_s in CUTS:
        train_end = pd.Timestamp(train_end_s)
        test_end = pd.Timestamp(test_end_s)

        train_df = df[df["time_utc"] < train_end]
        test_df = df[(df["time_utc"] >= train_end) & (df["time_utc"] < test_end)]

        X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
        y_train = train_df["target_eq_m4_plus"].astype(int)
        X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        y_test = test_df["target_eq_m4_plus"].astype(int)

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=42,
                        learning_rate=0.05,
                        max_depth=4,
                        max_iter=300,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        fold = {
            "train_end": train_end_s,
            "test_end": test_end_s,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "positive_rate_test": float(y_test.mean()) if len(y_test) else 0.0,
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        }
        folds.append(fold)

    summary = {
        "folds": folds,
        "mean_roc_auc": float(pd.DataFrame(folds)["roc_auc"].mean()),
        "mean_pr_auc": float(pd.DataFrame(folds)["pr_auc"].mean()),
    }

    out = MODELS / "rolling_cv_metrics.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
