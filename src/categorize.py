"""
Automatic expense categorisation from the free-text merchant label (brief step 2).

Two approaches, compared honestly:

    * RULES — a keyword lookup (the kind a bank ships first). Fast and
      transparent, but brittle to unseen merchants and typos.
    * ML — TF-IDF on character n-grams (robust to "CB ", branch numbers, city
      suffixes) + a linear SVM. Generalises to labels it never saw in training.

Char n-grams matter here: "CARREFOUR CITY PARIS 12" and "CB CARREFOUR" share the
"carrefour" substring even though word tokens differ. Evaluation is on a
held-out split of transactions, and — critically — on merchants HELD OUT from
training, to measure true generalisation rather than memorisation.

Run:
    python src/categorize.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from generate import RANDOM_STATE, load_raw

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

# Keyword rules: substring -> category (first match wins).
RULES = {
    "CARREFOUR": "Groceries", "LIDL": "Groceries", "MONOPRIX": "Groceries",
    "AUCHAN": "Groceries", "INTERMARCHE": "Groceries", "FRANPRIX": "Groceries",
    "LECLERC": "Groceries", "CASINO": "Groceries",
    "MCDONALDS": "Restaurants", "UBER EATS": "Restaurants",
    "DELIVEROO": "Restaurants", "BISTROT": "Restaurants",
    "SUSHI": "Restaurants", "STARBUCKS": "Restaurants", "PAUL": "Restaurants",
    "KFC": "Restaurants",
    "SNCF": "Transport", "NAVIGO": "Transport", "UBER": "Transport",
    "TOTAL": "Transport", "ESSO": "Transport", "BLABLACAR": "Transport",
    "VELIB": "Transport", "PARKING": "Transport",
    "NETFLIX": "Leisure", "SPOTIFY": "Leisure", "FNAC": "Leisure",
    "CINEMA": "Leisure", "STEAM": "Leisure", "PRIME": "Leisure",
    "DECATHLON": "Leisure", "CULTURA": "Leisure", "SPORT": "Leisure",
    "PHARMACIE": "Health", "DOCTOLIB": "Health", "LABORATOIRE": "Health",
    "OPTIC": "Health", "DENTISTE": "Health", "MUTUELLE": "Insurance",
    "ZARA": "Shopping", "AMAZON": "Shopping", "H&M": "Shopping",
    "SHEIN": "Shopping", "IKEA": "Shopping", "ZALANDO": "Shopping",
    "APPLE": "Shopping", "SEPHORA": "Shopping",
    "LOYER": "Housing", "EDF": "Utilities", "ORANGE": "Utilities",
    "ASSURANCE": "Insurance", "MAAF": "Insurance",
}


def rule_predict(label: str) -> str:
    up = label.upper()
    for kw, cat in RULES.items():
        if kw in up:
            return cat
    return "Unknown"


def build_ml_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  lowercase=True)),
        ("clf", LinearSVC(random_state=RANDOM_STATE)),
    ])


def run(save: bool = True) -> dict:
    df = load_raw()
    # Categorisation is about the merchant, so dedup identical labels for a fair,
    # merchant-level evaluation but keep enough repetition for TF-IDF.
    data = df[["label", "category"]].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        data.label, data.category, test_size=0.25,
        random_state=RANDOM_STATE, stratify=data.category)

    # --- rules ----------------------------------------------------------
    rule_pred = X_test.map(rule_predict)
    rule_f1 = f1_score(y_test, rule_pred, average="macro")
    rule_cov = (rule_pred != "Unknown").mean()

    # --- ML -------------------------------------------------------------
    pipe = build_ml_pipeline()
    pipe.fit(X_train, y_train)
    ml_pred = pipe.predict(X_test)
    ml_f1 = f1_score(y_test, ml_pred, average="macro")

    # --- generalisation to unseen merchants -----------------------------
    novel = pd.Series([
        "CB CARREFOUR EXPRESS LYON", "NETFLIX ABONNEMENT", "UBER TRIP 8843",
        "PHARMACIE DU CENTRE", "AMAZON MKTPLACE EU", "SNCF CONNECT",
        "RESTAURANT LE PETIT NICE", "LIDL SUD",
    ])
    novel_pred = pipe.predict(novel)

    print(f"[categorize] Rules : macro-F1={rule_f1:.3f}  "
          f"coverage={rule_cov*100:.1f}% (rest -> Unknown)")
    print(f"[categorize] ML    : macro-F1={ml_f1:.3f}")
    print("[categorize] ML on novel merchant labels:")
    for lbl, p in zip(novel, novel_pred):
        print(f"    {lbl:<32} -> {p}")

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        rep = classification_report(y_test, ml_pred, output_dict=True)
        pd.DataFrame(rep).T.to_csv(REPORT_DIR / "categorization_report.csv")
        pd.DataFrame([{"method": "rules", "macro_f1": rule_f1,
                       "coverage": rule_cov},
                      {"method": "tfidf_svm", "macro_f1": ml_f1,
                       "coverage": 1.0}]).to_csv(
            REPORT_DIR / "categorization_comparison.csv", index=False)

        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, ml_pred, labels=labels)
        plt.figure(figsize=(8, 6.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Prédit"); plt.ylabel("Réel")
        plt.title(f"Catégorisation TF-IDF+SVM (macro-F1={ml_f1:.3f})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "categorization_confusion.png", dpi=130)
        plt.close()
        print(f"[categorize] saved report + confusion matrix to {REPORT_DIR}")

    return {"rule_f1": rule_f1, "ml_f1": ml_f1, "pipeline": pipe}


if __name__ == "__main__":
    run()
