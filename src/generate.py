"""
Synthetic personal-finance transaction generator.

No open dataset combines realistic free-text merchant labels, per-user spending
habits, seasonality AND labelled anomalies — so we generate one with explicit,
documented rules. Every transaction carries GROUND TRUTH (`category`,
`is_anomaly`, `anomaly_type`) so both the NLP categoriser and the anomaly
detectors can be evaluated honestly.

Design (the realism that makes the problem non-trivial):
    * several users, each with an income level and a personal spending profile;
    * RECURRING expenses (rent, subscriptions, utilities, salary) on fixed
      monthly days and stable amounts;
    * VARIABLE expenses (groceries, restaurants, transport, leisure, health,
      shopping) with weekly rhythm (more on weekends) and seasonal effects
      (heating in winter, travel in summer);
    * free-text merchant LABELS drawn from realistic pools per category, with
      city/branch noise ("CARREFOUR CITY PARIS 12", "SNCF INTERNET"), so
      categorisation is a genuine text problem;
    * injected ANOMALIES of four kinds: amount spikes, out-of-habit category,
      duplicate charges (double billing), and frequency bursts.

Run:
    python src/generate.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_CSV = DATA_DIR / "raw" / "transactions.csv"

N_USERS = 8
MONTHS = 18
ANOMALY_RATE = 0.012  # ~1.2% of variable transactions perturbed

# category -> (merchant label pool, typical amount mean/std for a mid income)
CATEGORIES: dict[str, dict] = {
    "Groceries": {
        "merchants": ["CARREFOUR", "LIDL", "MONOPRIX", "AUCHAN", "INTERMARCHE",
                      "FRANPRIX", "LECLERC", "CASINO SUPERMARCHE"],
        "mean": 45, "std": 20, "freq_per_week": 2.5},
    "Restaurants": {
        "merchants": ["MCDONALDS", "UBER EATS", "DELIVEROO", "LE BISTROT",
                      "SUSHI SHOP", "STARBUCKS", "BOULANGERIE PAUL", "KFC"],
        "mean": 22, "std": 14, "freq_per_week": 1.8},
    "Transport": {
        "merchants": ["SNCF INTERNET", "RATP NAVIGO", "UBER", "TOTAL ENERGIES",
                      "ESSO STATION", "BLABLACAR", "VELIB", "PARKING INDIGO"],
        "mean": 30, "std": 25, "freq_per_week": 1.5},
    "Leisure": {
        "merchants": ["NETFLIX.COM", "SPOTIFY", "FNAC", "CINEMA UGC",
                      "STEAM GAMES", "AMAZON PRIME", "DECATHLON", "CULTURA"],
        "mean": 28, "std": 22, "freq_per_week": 1.0},
    "Health": {
        "merchants": ["PHARMACIE", "DOCTOLIB", "LABORATOIRE BIO", "OPTIC 2000",
                      "DENTISTE", "MUTUELLE"],
        "mean": 35, "std": 30, "freq_per_week": 0.4},
    "Shopping": {
        "merchants": ["ZARA", "AMAZON.FR", "H&M", "SHEIN", "IKEA", "ZALANDO",
                      "APPLE STORE", "SEPHORA"],
        "mean": 55, "std": 45, "freq_per_week": 0.7},
}

# Recurring monthly expenses: (label, category, day-of-month, amount base)
RECURRING = [
    ("LOYER APPARTEMENT", "Housing", 3, 780),
    ("EDF ELECTRICITE", "Utilities", 8, 65),
    ("ORANGE MOBILE", "Utilities", 12, 25),
    ("ASSURANCE MAAF", "Insurance", 15, 40),
    ("ABONNEMENT SALLE SPORT", "Leisure", 5, 30),
]


def _season_factor(month: int, category: str) -> float:
    """Multiplicative seasonal effect on amount/frequency."""
    if category == "Utilities" and month in (11, 12, 1, 2):
        return 1.5   # heating
    if category == "Transport" and month in (7, 8):
        return 1.4   # summer travel
    if category == "Shopping" and month in (11, 12):
        return 1.6   # holidays
    if category == "Restaurants" and month in (6, 7, 8):
        return 1.2
    return 1.0


def _make_label(fake: Faker, base: str) -> str:
    """Add realistic branch/city noise to a merchant name."""
    r = fake.random.random()
    if r < 0.35:
        return f"{base} {fake.city().upper()}"
    if r < 0.5:
        return f"{base} {fake.random_int(1, 20):02d}"
    if r < 0.6:
        return f"CB {base}"
    return base


def generate(save: bool = True) -> pd.DataFrame:
    fake = Faker("fr_FR")
    Faker.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    start = pd.Timestamp("2024-01-01")
    rows: list[dict] = []

    for uid in range(N_USERS):
        income_factor = rng.uniform(0.7, 1.6)  # personal spending scale
        for m in range(MONTHS):
            month_start = start + pd.DateOffset(months=m)
            month = month_start.month

            # --- recurring ---------------------------------------------
            for label, cat, day, base in RECURRING:
                amt = base * income_factor * _season_factor(month, cat)
                amt *= rng.uniform(0.98, 1.02)
                date = month_start + pd.Timedelta(days=day - 1)
                rows.append(dict(
                    user=f"user_{uid}", date=date, label=label,
                    amount=round(amt, 2), category=cat, recurring=True,
                    is_anomaly=False, anomaly_type="none"))

            # --- variable ----------------------------------------------
            for cat, spec in CATEGORIES.items():
                sf = _season_factor(month, cat)
                n = rng.poisson(spec["freq_per_week"] * 4.3 * sf)
                for _ in range(n):
                    dom = rng.integers(0, 28)
                    date = month_start + pd.Timedelta(days=int(dom))
                    # weekend uplift for leisure/restaurants
                    weekend = date.dayofweek >= 5
                    up = 1.25 if (weekend and cat in
                                  ("Restaurants", "Leisure", "Shopping")) else 1.0
                    amt = max(1.0, rng.normal(spec["mean"], spec["std"]))
                    amt *= income_factor * sf * up
                    base_merchant = rng.choice(spec["merchants"])
                    rows.append(dict(
                        user=f"user_{uid}", date=date,
                        label=_make_label(fake, base_merchant),
                        amount=round(amt, 2), category=cat, recurring=False,
                        is_anomaly=False, anomaly_type="none"))

    df = pd.DataFrame(rows)

    # ---- inject anomalies on a copy of variable transactions ----------
    df = _inject_anomalies(df, rng, fake)

    df = df.sort_values(["user", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    if save:
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
        print(f"[generate] {len(df):,} transactions, "
              f"{df.is_anomaly.sum()} anomalies "
              f"({df.is_anomaly.mean()*100:.2f}%) -> "
              f"{RAW_CSV.relative_to(DATA_DIR.parent)}")
    return df


def _inject_anomalies(df: pd.DataFrame, rng, fake) -> pd.DataFrame:
    """Perturb a small fraction of variable transactions into labelled anomalies."""
    var_idx = df.index[~df.recurring].to_numpy()
    n_anom = int(len(var_idx) * ANOMALY_RATE)
    chosen = rng.choice(var_idx, size=n_anom, replace=False)

    extra_rows = []
    for i in chosen:
        kind = rng.choice(["amount_spike", "wrong_category",
                           "duplicate", "frequency_burst"])
        if kind == "amount_spike":
            df.loc[i, "amount"] = round(df.loc[i, "amount"] *
                                        rng.uniform(6, 15), 2)
            df.loc[i, "anomaly_type"] = kind
            df.loc[i, "is_anomaly"] = True
        elif kind == "wrong_category":
            # A large purchase in a category the user rarely touches.
            df.loc[i, "category"] = "Shopping"
            df.loc[i, "label"] = _make_label(fake, "APPLE STORE")
            df.loc[i, "amount"] = round(rng.uniform(600, 2000), 2)
            df.loc[i, "anomaly_type"] = kind
            df.loc[i, "is_anomaly"] = True
        elif kind == "duplicate":
            dup = df.loc[i].copy()
            dup["date"] = dup["date"] + pd.Timedelta(minutes=2)
            dup["is_anomaly"] = True
            dup["anomaly_type"] = "duplicate"
            df.loc[i, "is_anomaly"] = True
            df.loc[i, "anomaly_type"] = "duplicate"
            extra_rows.append(dup)
        else:  # frequency_burst — several same-day charges at one merchant
            for _ in range(rng.integers(4, 8)):
                burst = df.loc[i].copy()
                burst["date"] = burst["date"] + pd.Timedelta(
                    hours=int(rng.integers(1, 10)))
                burst["is_anomaly"] = True
                burst["anomaly_type"] = "frequency_burst"
                extra_rows.append(burst)
            df.loc[i, "is_anomaly"] = True
            df.loc[i, "anomaly_type"] = "frequency_burst"

    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    return df


def load_raw() -> pd.DataFrame:
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV, parse_dates=["date"])
    return generate()


if __name__ == "__main__":
    frame = generate()
    print(frame.category.value_counts())
    print("\nanomaly types:\n", frame[frame.is_anomaly].anomaly_type.value_counts())
    print("\nsample labels:\n", frame.label.sample(10, random_state=1).tolist())
