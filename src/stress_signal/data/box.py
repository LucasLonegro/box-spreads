"""Box-spread rates and convenience yields from the Diamond et al. public files.

All rates returned here are decimals, continuously compounded, annualized.
"""

from pathlib import Path

import pandas as pd

from stress_signal.config import BDG_FILE, DIAMOND_FILE

TENORS = ("3m", "6m", "1y", "2y")


def _dates(year: pd.Series, month: pd.Series, day: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day})))


def load_diamond_us(path: Path = DIAMOND_FILE) -> pd.DataFrame:
    """Diamond & Van Tassel daily US box, government and CY series (sheet `United States`).

    Weekend rows (box forward-filled by the source in 2010-19) and days without
    `gov_1y` are dropped, not filled.
    """
    raw = pd.read_excel(path, sheet_name="United States")
    df = raw.drop(columns=["date_year", "date_month", "date_day"])
    df.index = _dates(raw["date_year"], raw["date_month"], raw["date_day"])
    df.index.name = "date"
    df = df[df.index.dayofweek < 5]
    df = df[df["gov_1y"].notna()]
    return df.sort_index()


def load_bdg_2019(path: Path = BDG_FILE) -> pd.DataFrame:
    """van Binsbergen-Diamond-Grotteria (2022) box/gov rates, 6m/12m/18m, converted % -> decimal."""
    raw = pd.read_excel(path, sheet_name="box_gov_l")
    df = raw.drop(columns=["year", "month", "day"]) / 100.0
    df.index = _dates(raw["year"], raw["month"], raw["day"])
    df.index.name = "date"
    df = df[df.index.dayofweek < 5]
    for tenor in ("6m", "12m", "18m"):
        df[f"cy_{tenor}"] = df[f"box_{tenor}"] - df[f"gov_{tenor}"]
    return df.sort_index()
