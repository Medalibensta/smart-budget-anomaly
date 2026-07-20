"""
Dynamic budget recommender + alert generation (brief step 5).

Turns the analysis into something a user would actually see:

    * per-user, per-category monthly budget = a robust central tendency of that
      user's *normal* months (median of monthly totals, anomalies excluded),
      with a tolerance band (median + k·MAD) as the "you're overspending" line;
    * month-to-date projection: given spend so far this month, linearly project
      the end-of-month total and flag categories on track to breach budget;
    * a human-readable alert feed combining budget breaches and the anomaly
      flags from anomaly.py.

This is deliberately simple and transparent — a recommender a user can trust and
override — rather than an opaque optimiser.

Run:
    python src/recommend.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anomaly import engineer_features
from generate import load_raw

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

MAD_SCALE = 1.4826
TOLERANCE_K = 1.5


def category_budgets(df: pd.DataFrame) -> pd.DataFrame:
    """Robust monthly budget + tolerance per (user, category)."""
    normal = df[~df.is_anomaly].copy()
    normal["month"] = normal.date.dt.to_period("M")
    monthly = (normal.groupby(["user", "category", "month"]).amount.sum()
               .reset_index())
    g = monthly.groupby(["user", "category"]).amount
    budget = g.median().rename("budget")
    mad = g.apply(lambda s: (s - s.median()).abs().median()).rename("mad")
    out = pd.concat([budget, mad], axis=1).reset_index()
    out["tolerance"] = out.budget + TOLERANCE_K * MAD_SCALE * out.mad
    return out


def month_to_date_alerts(df: pd.DataFrame, budgets: pd.DataFrame,
                         as_of: pd.Timestamp) -> pd.DataFrame:
    """Project current-month spend to month-end and flag likely breaches."""
    month = as_of.to_period("M")
    days_in_month = as_of.days_in_month
    frac = as_of.day / days_in_month

    cur = df[df.date.dt.to_period("M") == month].copy()
    spent = (cur.groupby(["user", "category"]).amount.sum()
             .rename("spent_so_far").reset_index())
    merged = spent.merge(budgets, on=["user", "category"], how="left")
    merged["projected"] = merged.spent_so_far / max(frac, 1e-6)
    merged["breach"] = merged.projected > merged.tolerance
    merged["overshoot_pct"] = ((merged.projected - merged.budget)
                               / merged.budget.replace(0, np.nan) * 100)
    return merged.sort_values("overshoot_pct", ascending=False)


def build_alert_feed(df_feat: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Human-readable anomaly alerts from the engineered/flagged transactions."""
    anom = df_feat[df_feat.is_anomaly].copy()
    msg = {
        "amount_spike": "Montant inhabituel",
        "wrong_category": "Achat hors habitude",
        "duplicate": "Débit en double probable",
        "frequency_burst": "Rafale de débits identiques",
    }
    anom["alert"] = anom.anomaly_type.map(msg).fillna("Dépense atypique")
    cols = ["date", "user", "label", "category", "amount", "alert"]
    return anom.sort_values("amount", ascending=False)[cols].head(n)


def run(save: bool = True) -> pd.DataFrame:
    df = load_raw()
    budgets = category_budgets(df)
    as_of = df.date.max().normalize()
    alerts = month_to_date_alerts(df, budgets, as_of)

    feat = engineer_features(df)
    feed = build_alert_feed(feat)

    print(f"=== Budget projection as of {as_of.date()} (breaches) ===")
    breaches = alerts[alerts.breach].head(10)
    print(breaches[["user", "category", "budget", "projected",
                    "overshoot_pct"]].round(1).to_string(index=False))
    print("\n=== Example alert feed ===")
    print(feed.to_string(index=False))

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        budgets.to_csv(REPORT_DIR / "category_budgets.csv", index=False)
        alerts.to_csv(REPORT_DIR / "budget_projection.csv", index=False)
        feed.to_csv(REPORT_DIR / "alert_feed.csv", index=False)

        # Budget vs actual for one user, latest month.
        u = "user_0"
        sub = alerts[alerts.user == u].set_index("category")
        sub = sub.sort_values("budget", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        y = np.arange(len(sub))
        ax.barh(y, sub.budget, color="#4c72b0", alpha=0.7, label="budget")
        ax.barh(y + 0.0, sub.projected, height=0.4, color="#c44e52",
                alpha=0.8, label="projeté fin de mois")
        ax.errorbar(sub.tolerance, y, fmt="k|", markersize=14,
                    label="seuil tolérance")
        ax.set_yticks(y); ax.set_yticklabels(sub.index)
        ax.set_xlabel("EUR / mois"); ax.set_title(f"Budget vs projection — {u}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "budget_vs_projection.png", dpi=130)
        plt.close()
        print(f"[recommend] saved budgets + projection + alert feed to {REPORT_DIR}")

    return alerts


if __name__ == "__main__":
    run()
