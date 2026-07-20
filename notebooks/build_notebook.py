"""
Build and execute notebooks/05_smart_budget.ipynb.

Light aggregates run live; heavy artefacts are displayed from the CSVs/figures
produced by src/, keeping the notebook fast and single-sourced.

Run:
    python notebooks/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "05_smart_budget.ipynb"


def md(t): return nbf.v4.new_markdown_cell(t.strip())
def code(t): return nbf.v4.new_code_cell(t.strip())


cells = [
    md("""
# Assistant budgétaire intelligent & détection d'anomalies de dépenses

**Angle décisionnel** — aider un particulier à (1) **catégoriser
automatiquement** ses dépenses à partir du libellé bancaire, (2) **comprendre**
son comportement de consommation normal, et (3) être **alerté** en cas de
dépense inhabituelle — le tout de façon transparente et personnalisée.

**Données** — aucun jeu ouvert ne combine libellés réalistes, habitudes
par utilisateur, saisonnalité **et** anomalies étiquetées. On **génère** donc
des transactions synthétiques réalistes (Faker + règles), avec une **vérité
terrain** (`category`, `is_anomaly`, `anomaly_type`) pour évaluer honnêtement
chaque brique.

**Plan** : données → catégorisation NLP → comportement normal → détection
d'anomalies → recommandations budgétaires → limites.
"""),
    code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from generate import load_raw

df = load_raw()
print(f"{len(df):,} transactions | {df.user.nunique()} utilisateurs | "
      f"{df.date.min().date()} → {df.date.max().date()}")
print(f"anomalies injectées : {df.is_anomaly.sum()} "
      f"({df.is_anomaly.mean()*100:.2f}%)")
df[["date", "user", "label", "amount", "category",
    "is_anomaly", "anomaly_type"]].head(8)
"""),
    md("""
## 1. Des données volontairement réalistes

Les libellés portent du bruit de succursale/ville (« CB CARREFOUR », « UBER
EATS 08 »), les habitudes varient par utilisateur, et quatre familles
d'anomalies sont injectées : **pic de montant**, **achat hors habitude**,
**débit en double** et **rafale de débits**. C'est ce qui rend le problème
non trivial.
"""),
    code("""
print("Catégories :")
display(df.category.value_counts())
print("Types d'anomalies :")
display(df[df.is_anomaly].anomaly_type.value_counts())
"""),
    md("""
## 2. Catégorisation automatique (NLP)

Deux approches comparées : **règles** par mots-clés (transparentes mais fragiles)
et **TF-IDF sur n-grammes de caractères + SVM linéaire** (robuste au bruit de
libellé). Les n-grammes de caractères permettent de reconnaître « carrefour »
dans « CB CARREFOUR EXPRESS LYON ».
"""),
    code("""
comp = pd.read_csv(ROOT / "reports" / "categorization_comparison.csv")
display(comp.round(3))
display(Image(ROOT / "reports" / "figures" /
              "categorization_confusion.png", width=560))
"""),
    md("""
**Lecture** — le modèle TF-IDF généralise à des libellés **jamais vus** en
entraînement (ex. « SNCF CONNECT » → Transport). Le score très élevé reflète
un vocabulaire de marchands partagé ; le vrai test de robustesse est la
généralisation aux libellés inédits (voir `src/categorize.py`). « Amazon » est
volontairement ambigu (Prime/Shopping) — une ambiguïté réaliste.
"""),
    md("""
## 3. Comportement « normal » personnalisé

Une anomalie est **relative à l'utilisateur** : 300 € est normal pour un loyer,
alarmant pour des courses. On modélise une base robuste par (utilisateur,
catégorie) — médiane et MAD glissantes **causales** (passé uniquement) — et un
**z-score robuste**.
"""),
    code("""
for fig in ["spend_trajectory.png", "zscore_separation.png"]:
    display(Image(ROOT / "reports" / "figures" / fig, width=760))
"""),
    md("""
**Lecture importante** — le z-score sur le montant ne sépare bien que les
anomalies **de montant** (pics, achats hors habitude). Les **doublons** et
**rafales** ont des montants normaux : ils exigent des features de
**fréquence/duplication**. D'où l'approche multivariée au §4.
"""),
    md("""
## 4. Détection d'anomalies — comparaison

Features couvrant les 4 mécanismes (z-score, montant, compteur intra-journalier,
doublon, rareté de catégorie), et trois détecteurs comparés à la vérité terrain :
**règle transparente**, **Isolation Forest**, **DBSCAN**.
"""),
    code("""
comp = pd.read_csv(ROOT / "reports" / "anomaly_comparison.csv")
by_type = pd.read_csv(ROOT / "reports" / "anomaly_recall_by_type.csv")
display(comp.round(3))
print("Rappel d'Isolation Forest par type d'anomalie :")
display(by_type.round(2))
display(Image(ROOT / "reports" / "figures" / "anomaly_pr_curves.png", width=620))
"""),
    md("""
**Lecture** — **Isolation Forest** l'emporte (PR-AUC 0,91, précision 0,90,
rappel 0,88). Le rappel par type est instructif : **100 %** des rafales, ~78 %
des pics de montant, ~71 % des achats hors habitude, mais seulement ~54 % des
**doublons** — les plus discrets (montant normal, seul le timing trahit).
DBSCAN est peu adapté ici (rappel faible). La règle transparente reste un bon
filet de base.
"""),
    md("""
## 5. Recommandations budgétaires & alertes

Budget mensuel robuste par catégorie (médiane des mois *normaux* + tolérance
MAD), projection fin de mois à partir du dépensé courant, et flux d'alertes
lisible.
"""),
    code("""
proj = pd.read_csv(ROOT / "reports" / "budget_projection.csv")
feed = pd.read_csv(ROOT / "reports" / "alert_feed.csv")
print("Catégories en dépassement projeté :")
display(proj[proj.breach].head(8)[["user", "category", "budget",
        "projected", "overshoot_pct"]].round(1))
print("Flux d'alertes (extrait) :")
display(feed.head(8))
display(Image(ROOT / "reports" / "figures" / "budget_vs_projection.png",
              width=680))
"""),
    md("""
## 6. Limites & éthique

- **Données synthétiques** : les patterns et anomalies sont *générés* selon nos
  règles ; les performances (surtout la catégorisation quasi-parfaite) seraient
  plus basses sur de vraies données bancaires, plus bruitées et ambiguës. Le
  code est prêt à recevoir un vrai jeu (ex. *Daily Household Transactions*).
- **Vie privée** : les données financières sont sensibles ; un déploiement réel
  impose consentement, minimisation et traitement local/chiffré.
- **Faux positifs** : une alerte injustifiée érode la confiance ; le seuil doit
  être réglable par l'utilisateur, et les alertes explicables (ce que fournit le
  flux typé).
- **Personnalisation vs démarrage à froid** : un nouvel utilisateur sans
  historique ne peut pas être profilé — prévoir des budgets par défaut le temps
  d'accumuler des données.
"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                              "name": "python3"}
    nb.cells = cells
    print(f"[notebook] executing {len(cells)} cells...")
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, NB_PATH)
    print(f"[notebook] written -> {NB_PATH}")


if __name__ == "__main__":
    main()
