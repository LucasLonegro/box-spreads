"""Treasury yields from the Fed GSW zero curve and FRED."""

import io

import numpy as np
import pandas as pd

from stress_signal.io import fetch_cached

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def load_gsw(columns: tuple[str, ...] = ("SVENY01", "SVENY02")) -> pd.DataFrame:
    """GSW zero-coupon yields. Returned as decimals (source: % continuously compounded)."""
    path = fetch_cached(GSW_URL, "feds200628.csv")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_row = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    df = pd.read_csv(
        io.StringIO("\n".join(lines[header_row:])),
        usecols=["Date", *columns],
        index_col="Date",
        parse_dates=["Date"],
        na_values=["NA", ""],
    )
    df.index.name = "date"
    return (df / 100.0).sort_index()


def load_fred(series: str) -> pd.Series:
    """A FRED daily series as published (percent). FRED rejects browser User-Agents, so none is sent."""
    path = fetch_cached(FRED_URL.format(series=series), f"fred_{series}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True, na_values=["."])
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(series)
    s.index.name = "date"
    return s.sort_index()


def bey_to_cc(y_pct: pd.Series | float) -> pd.Series | float:
    """Bond-equivalent (semi-annual) yield in % -> continuously compounded decimal."""
    return 2.0 * np.log1p(y_pct / 200.0)
