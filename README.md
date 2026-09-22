# Factor Pricing ML

Détection et persistance des anomalies de pricing (factor investing) : comparaison d'une approche économétrique classique (OLS multi-facteurs, type Fama-French) avec des approches de Machine Learning (Lasso/Ridge, Random Forest, Gradient Boosting).

## Problématique

Les facteurs de risque identifiés par la littérature académique (marché, taille, valeur, momentum, etc.) expliquent-ils toujours les rendements hors échantillon ? Peut-on améliorer la prédiction des rendements en capturant des interactions non-linéaires entre facteurs grâce au ML, par rapport à une régression linéaire classique ?

## Données

- **Kenneth French Data Library** (gratuit) : rendements mensuels des facteurs Fama-French (3 et 5 facteurs), portefeuilles triés par taille/valeur.
- **Yahoo Finance** (via `yfinance`, gratuit) : prix et rendements d'actions ou d'indices pour construire des portefeuilles tests.

## Structure du projet

```
factor-pricing-ml/
├── data/               # Données brutes et transformées (non versionnées, voir .gitignore)
├── notebooks/          # Notebooks Jupyter d'exploration et de résultats
├── src/                # Code source réutilisable (scripts Python)
│   ├── data_loader.py  # Téléchargement et nettoyage des données
│   ├── models.py       # Modèles économétriques et ML
│   └── backtest.py     # Évaluation et backtest des stratégies
├── results/            # Graphiques et résultats générés
├── requirements.txt    # Dépendances Python
└── README.md
```

## Méthodologie

1. Télécharger les facteurs Fama-French et les rendements de portefeuilles tests.
2. Estimer un modèle OLS multi-facteurs (baseline économétrique).
3. Entraîner des modèles ML (Lasso, Ridge, Random Forest, XGBoost) sur les mêmes données.
4. Comparer les performances en split temporel (train / test hors échantillon).
5. Analyser l'importance des facteurs (coefficients OLS vs feature importance / SHAP).
6. Backtester une stratégie simple basée sur les prédictions et comparer le Sharpe ratio à la baseline.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Utilisation

```bash
python src/data_loader.py      # télécharge et prépare les données
jupyter notebook notebooks/    # exploration et résultats
```

## Résultats

*(à compléter au fur et à mesure des résultats obtenus)*

## Auteur

Charles Gavini — Master in Finance, EDHEC Business School
