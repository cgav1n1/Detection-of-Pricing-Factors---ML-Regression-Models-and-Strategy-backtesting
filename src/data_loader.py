"""
Module de récupération des données pour le projet Factor Pricing ML.

Sources :
- Kenneth French Data Library (facteurs Fama-French, gratuit)
- Yahoo Finance via yfinance (prix d'actions/indices, gratuit)
"""

import io
import zipfile
import requests
import pandas as pd
import yfinance as yf

FF_5FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
)


def load_fama_french_5factors(monthly: bool = True) -> pd.DataFrame:
    """
    Télécharge les 5 facteurs Fama-French (Mkt-RF, SMB, HML, RMW, CMA, RF)
    depuis la Kenneth French Data Library.

    Returns
    -------
    pd.DataFrame indexé par date (fin de mois), colonnes en décimal
    (déjà divisées par 100).
    """
    resp = requests.get(FF_5FACTORS_URL, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            raw = f.read().decode("utf-8", errors="ignore")

    # Le fichier contient un en-tête texte, puis les données mensuelles,
    # puis une section annuelle. On isole la partie mensuelle.
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("19") or l.strip().startswith("20"))
    end = next(
        i for i, l in enumerate(lines[start:], start=start)
        if l.strip() == "" or "Annual" in l
    )

    data_str = "\n".join(lines[start:end])
    df = pd.read_csv(
        io.StringIO(data_str),
        header=None,
        names=["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m")
    df = df.set_index("date")
    df = df.astype(float) / 100.0
    return df


def load_asset_returns(tickers, start="2000-01-01", end=None, freq="M") -> pd.DataFrame:
    """
    Télécharge les prix ajustés via yfinance et calcule les rendements.

    Parameters
    ----------
    tickers : list[str]
        Liste de tickers Yahoo Finance (ex: ["AAPL", "MSFT", "^GSPC"]).
    freq : str
        "M" pour rendements mensuels, "D" pour quotidiens.

    Returns
    -------
    pd.DataFrame des rendements, une colonne par ticker.
    """
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    if freq == "M":
        prices = prices.resample("ME").last()

    returns = prices.pct_change().dropna(how="all")
    return returns


if __name__ == "__main__":
    print("Téléchargement des facteurs Fama-French 5 facteurs...")
    ff = load_fama_french_5factors()
    print(ff.tail())
    ff.to_csv("data/ff5_factors.csv")
    print("Sauvegardé dans data/ff5_factors.csv")

    print("\nTéléchargement d'exemples de rendements (S&P 500)...")
    rets = load_asset_returns(["^GSPC"], start="2000-01-01")
    rets.to_csv("data/sp500_returns.csv")
    print("Sauvegardé dans data/sp500_returns.csv")
