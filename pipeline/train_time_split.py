import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from config import DATA_PROCESSED, MODELS


CUTOFF = "2018-01-01T00:00:00Z"


def permutation_p_value(model, X_test, y_test, observed_auc: float, n_perm: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    p = model.predict_proba(X_test)[:, 1]
    null_aucs = [roc_auc_score(rng.permutation(y_test), p) for _ in range(n_perm)]
    null_aucs = np.array(null_aucs)
    return float((1 + np.sum(null_aucs >= observed_auc)) / (1 + n_perm))


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROCESSED / "model_dataset.parquet")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)

    cutoff = pd.Timestamp(CUTOFF)
    train_df = df[df["time_utc"] < cutoff].copy()
    test_df = df[df["time_utc"] >= cutoff].copy()

    drop_cols = ["time_utc", "target_eq_m4_plus", "latitude", "longitude", "magnitude"]
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

    auc = float(roc_auc_score(y_test, proba))
    ap = float(average_precision_score(y_test, proba))
    pval = permutation_p_value(model, X_test, y_test, auc)

    metrics = {
        "cutoff": CUTOFF,
        "roc_auc": auc,
        "pr_auc": ap,
        "permutation_p_value": pval,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }

    out = MODELS / "time_split_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
