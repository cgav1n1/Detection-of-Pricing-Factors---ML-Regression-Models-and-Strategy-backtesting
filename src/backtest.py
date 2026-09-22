"""
Backtest simple : transformer les prédictions d'un modèle en stratégie
d'allocation binaire (long si prédiction > 0, cash sinon) et comparer
au buy-and-hold.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12, rf: float = 0.0) -> float:
    """Sharpe ratio annualisé à partir d'une série de rendements périodiques."""
    excess = returns - rf / periods_per_year
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def strategy_from_predictions(predictions: pd.Series, actual_returns: pd.Series) -> pd.Series:
    """
    Stratégie simple : être long l'actif quand la prédiction du modèle
    est positive, rester en cash (rendement 0) sinon.
    """
    signal = (predictions > 0).astype(int)
    strategy_returns = signal.values * actual_returns.values
    return pd.Series(strategy_returns, index=actual_returns.index)


def compare_to_buy_and_hold(predictions: pd.Series, actual_returns: pd.Series) -> pd.DataFrame:
    """
    Compare la stratégie basée sur les prédictions au buy-and-hold :
    rendement cumulé et Sharpe ratio des deux approches.
    """
    strat_returns = strategy_from_predictions(predictions, actual_returns)

    cum_strategy = (1 + strat_returns).cumprod() - 1
    cum_bh = (1 + actual_returns).cumprod() - 1

    summary = pd.DataFrame({
        "Stratégie (signal ML)": [strat_returns.mean(), strat_returns.std(),
                                   sharpe_ratio(strat_returns), cum_strategy.iloc[-1]],
        "Buy & Hold": [actual_returns.mean(), actual_returns.std(),
                        sharpe_ratio(actual_returns), cum_bh.iloc[-1]],
    }, index=["Rendement moyen", "Volatilité", "Sharpe ratio", "Rendement cumulé"])

    return summary
