"""Leitura das bases de dados."""
import pandas as pd

from .config import DATA_DIR


def load_auto() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "Auto.csv", na_values="?")
    # Algumas versões da base marcam horsepower ausente com "?".
    return df.dropna().reset_index(drop=True)


def load_boston() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Boston.csv")
