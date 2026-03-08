import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import DATA_PROCESSED, MODELS


def permutation_p_value(model, X_test, y_test, observed_auc: float, n_perm: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    null_aucs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_test)
        p = model.predict_proba(X_test)[:, 1]
        null_aucs.append(roc_auc_score(y_perm, p))
    null_aucs = np.array(null_aucs)
    p_val = (1 + np.sum(null_aucs >= observed_auc)) / (1 + n_perm)
    return float(p_val)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROCESSED / "model_dataset.parquet")

    y = df["target_eq_m4_plus"].astype(int)
    drop_cols = ["time_utc", "target_eq_m4_plus", "latitude", "longitude", "magnitude"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
        "roc_auc": auc,
        "pr_auc": ap,
        "permutation_p_value": pval,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_test": float(y_test.mean()),
    }

    out = MODELS / "baseline_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
