"""
Behavioural modelling of "normal" spending (brief step 3).

Anomaly is personal: 300 EUR is routine for one user's rent and alarming for
another's groceries. So we model a per-user, per-category baseline and score
each transaction by how far it deviates from that user's own history:

    * a robust rolling baseline per (user, category): expanding median and MAD
      up to (but excluding) each transaction — causal, no future leakage;
    * a robust z-score  z = 0.6745·(amount − median) / MAD;
    * monthly spend trajectories per category for visualisation.

Using median/MAD instead of mean/std keeps the baseline from being inflated by
the very spikes we want to catch.

Run:
    python src/behavior.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate import load_raw

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED = DATA_DIR / "processed" / "transactions_scored.csv"

MAD_SCALE = 0.6745
MIN_HISTORY = 5  # need a few prior points before scoring


def add_robust_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a causal per-(user, category) robust z-score to each row."""
    df = df.sort_values(["user", "category", "date"]).copy()

    def _score(group: pd.DataFrame) -> pd.Series:
        amt = group["amount"]
        # Expanding stats SHIFTED by 1 so a point is scored on its past only.
        med = amt.expanding(min_periods=MIN_HISTORY).median().shift(1)
        mad = (amt - med.ffill()).abs().expanding(
            min_periods=MIN_HISTORY).median().shift(1)
        z = MAD_SCALE * (amt - med) / mad.replace(0, np.nan)
        return z

    df["robust_z"] = (df.groupby(["user", "category"], group_keys=False)
                      .apply(_score))
    df["robust_z"] = df["robust_z"].fillna(0.0)
    df["abs_z"] = df["robust_z"].abs()
    return df.sort_values(["user", "date"]).reset_index(drop=True)


def monthly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Total spend per user, category and month."""
    m = df.assign(month=df.date.dt.to_period("M").dt.to_timestamp())
    return (m.groupby(["user", "category", "month"]).amount.sum()
            .reset_index())


def run(save: bool = True) -> pd.DataFrame:
    df = add_robust_zscore(load_raw())
    prof = monthly_profile(df)

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED, index=False)

        # Monthly spend trajectory for one illustrative user.
        u = "user_0"
        sub = prof[prof.user == u]
        piv = sub.pivot(index="month", columns="category", values="amount").fillna(0)
        ax = piv.plot(figsize=(11, 5.5), marker="o", colormap="tab10")
        ax.set(xlabel="Mois", ylabel="Dépense mensuelle (EUR)",
               title=f"Trajectoire de dépenses par catégorie — {u}")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "spend_trajectory.png", dpi=130)
        plt.close()

        # Distribution of robust z by anomaly label.
        plt.figure(figsize=(9, 5))
        for flag, name, color in [(False, "normal", "#4c72b0"),
                                  (True, "anomalie", "#c44e52")]:
            vals = df[df.is_anomaly == flag]["abs_z"].clip(upper=15)
            plt.hist(vals, bins=40, alpha=0.6, label=name, color=color,
                     density=True)
        plt.axvline(3.5, color="k", ls="--", alpha=0.6, label="seuil |z|=3.5")
        plt.xlabel("|z| robuste"); plt.ylabel("densité")
        plt.title("Séparation des anomalies par le z-score robuste")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "zscore_separation.png", dpi=130)
        plt.close()
        print(f"[behavior] scored {len(df):,} tx -> {PROCESSED.name}")
        print(f"[behavior] median |z| normal={df[~df.is_anomaly].abs_z.median():.2f}"
              f"  anomaly={df[df.is_anomaly].abs_z.median():.2f}")

    return df


if __name__ == "__main__":
    run()
