"""
Modèles pour expliquer/prédire les rendements à partir des facteurs :
- Baseline économétrique : régression OLS multi-facteurs (type Fama-French)
- Modèles ML : Lasso, Ridge, Random Forest, XGBoost
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def fit_ols_baseline(y: pd.Series, X: pd.DataFrame):
    """
    Régression OLS classique : y (rendement excédentaire) ~ facteurs.
    Réplique l'approche Fama-French standard.
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const, missing="drop").fit()
    return model


def fit_ml_models(X_train, y_train, random_state: int = 42) -> dict:
    """
    Entraîne plusieurs modèles ML sur les mêmes features que l'OLS.

    Returns
    -------
    dict {nom_du_modèle: modèle entraîné}
    """
    models = {
        "Lasso": Lasso(alpha=0.001, random_state=random_state),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=5, random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=random_state
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models: dict, X_test, y_test, ols_model=None, X_test_ols=None) -> pd.DataFrame:
    """
    Compare les modèles ML et la baseline OLS sur le jeu de test
    (hors échantillon) via RMSE et R².
    """
    rows = []

    if ols_model is not None:
        X_test_const = sm.add_constant(X_test_ols, has_constant="add")
        pred_ols = ols_model.predict(X_test_const)
        rows.append({
            "model": "OLS (baseline)",
            "RMSE": np.sqrt(mean_squared_error(y_test, pred_ols)),
            "R2": r2_score(y_test, pred_ols),
        })

    for name, model in models.items():
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })

    return pd.DataFrame(rows).sort_values("RMSE")


def time_series_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Split temporel simple (pas de shuffle !) : les dernières observations
    servent de test hors échantillon, condition indispensable en finance.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test
