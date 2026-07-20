"""
Anomaly detection on spending, three approaches compared (brief step 4).

The injected anomalies span four mechanisms — amount spikes, out-of-habit
category, duplicate charges and frequency bursts — so a single amount-based rule
cannot catch them all. We engineer features that expose every mechanism, then
compare three detectors against the ground-truth `is_anomaly` label:

    features per transaction:
        robust_z        — personal amount deviation (from behavior.py)
        log_amount      — raw scale
        same_day_count  — #tx for that user+merchant on that day (bursts)
        is_duplicate    — same user+amount+label within 15 min (double billing)
        cat_rarity      — 1 − share of the user's tx in this category (habit)

    detectors:
        z-score rule    — |robust_z| > 3.5 OR duplicate OR burst (transparent baseline)
        Isolation Forest — unsupervised, multivariate
        DBSCAN          — density clustering; points labelled noise = anomalies

Evaluation uses PR-AUC / precision / recall (anomalies are ~3%), which is the
honest metric under imbalance.

Run:
    python src/anomaly.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

from behavior import add_robust_zscore
from generate import RANDOM_STATE, load_raw

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

FEATURES = ["robust_z", "log_amount", "same_day_count", "is_duplicate",
            "cat_rarity"]
Z_THRESHOLD = 3.5


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the multivariate feature set that exposes all four anomaly types."""
    df = add_robust_zscore(df)
    df["log_amount"] = np.log1p(df["amount"])

    # same-day merchant burst count
    df["day"] = df["date"].dt.floor("D")
    df["same_day_count"] = (df.groupby(["user", "label", "day"])["amount"]
                            .transform("size"))

    # duplicate: identical user+amount+label within 15 minutes
    df = df.sort_values(["user", "label", "amount", "date"])
    dt = df.groupby(["user", "label", "amount"])["date"].diff()
    df["is_duplicate"] = (dt.notna() & (dt <= pd.Timedelta(minutes=15))).astype(int)

    # category habit rarity per user: 1 − (share of this user's tx in the cat)
    user_cat = df.groupby("user")["category"].transform(
        lambda s: s.map(s.value_counts(normalize=True)))
    df["cat_rarity"] = 1 - user_cat

    return df.sort_values(["user", "date"]).reset_index(drop=True)


def _evaluate(y_true, score, name, thr=None) -> dict:
    ap = average_precision_score(y_true, score)
    if thr is None:
        # pick threshold maximising F1 on the PR curve
        prec, rec, thresholds = precision_recall_curve(y_true, score)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        thr = thresholds[np.argmax(f1[:-1])] if len(thresholds) else 0.5
    pred = (score >= thr).astype(int)
    return {"method": name, "PR_AUC": ap,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "threshold": float(thr)}


def run(save: bool = True) -> pd.DataFrame:
    df = engineer_features(load_raw())
    y = df["is_anomaly"].astype(int).to_numpy()
    X = StandardScaler().fit_transform(df[FEATURES])

    rows = []

    # --- 1. transparent rule -------------------------------------------
    rule_score = ((df.abs_z > Z_THRESHOLD) | (df.is_duplicate == 1)
                  | (df.same_day_count >= 4)).astype(float).to_numpy()
    rows.append(_evaluate(y, rule_score, "Rule (z|dup|burst)", thr=0.5))

    # --- 2. Isolation Forest -------------------------------------------
    iso = IsolationForest(n_estimators=300, contamination=float(y.mean()),
                          random_state=RANDOM_STATE)
    iso.fit(X)
    iso_score = -iso.score_samples(X)
    rows.append(_evaluate(y, iso_score, "IsolationForest"))

    # --- 3. DBSCAN (noise = anomaly) -----------------------------------
    db = DBSCAN(eps=1.5, min_samples=8).fit(X)
    db_score = (db.labels_ == -1).astype(float)
    rows.append(_evaluate(y, db_score, "DBSCAN (noise)", thr=0.5))

    table = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    print("=== Anomaly detection comparison (vs injected labels) ===")
    print(table.round(3).to_string(index=False))

    # per-type recall of the best detector (IsolationForest at its F1 threshold)
    best_thr = table.loc[table.method == "IsolationForest", "threshold"].iloc[0]
    df["iso_flag"] = (iso_score >= best_thr).astype(int)
    by_type = (df[df.is_anomaly]
               .groupby("anomaly_type")
               .agg(n=("iso_flag", "size"), caught=("iso_flag", "sum")))
    by_type["recall"] = (by_type.caught / by_type.n).round(2)
    print("\nIsolationForest recall by anomaly type:")
    print(by_type.to_string())

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORT_DIR / "anomaly_comparison.csv", index=False)
        by_type.to_csv(REPORT_DIR / "anomaly_recall_by_type.csv")

        plt.figure(figsize=(8, 6))
        for score, name in [(rule_score, "Rule"), (iso_score, "IsolationForest"),
                            (db_score, "DBSCAN")]:
            prec, rec, _ = precision_recall_curve(y, score)
            ap = average_precision_score(y, score)
            plt.plot(rec, prec, label=f"{name} (PR-AUC={ap:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Détection d'anomalies de dépenses — comparaison")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "anomaly_pr_curves.png", dpi=130)
        plt.close()
        print(f"[anomaly] saved comparison + PR curves to {REPORT_DIR}")

    return table


if __name__ == "__main__":
    run()
