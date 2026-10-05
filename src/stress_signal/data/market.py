"""VIX (Cboe) and S&P 500 closes (Yahoo chart API)."""

import json

import pandas as pd

from stress_signal.config import SPX_FETCH_END, SPX_FETCH_START
from stress_signal.io import BROWSER_UA, fetch_cached

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1={p1}&period2={p2}&interval=1d"


def load_vix() -> pd.Series:
    """VIX daily close, in index points (annualized vol in %)."""
    path = fetch_cached(VIX_URL, "VIX_History.csv")
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    s = pd.Series(df["CLOSE"].to_numpy(dtype=float), index=pd.DatetimeIndex(idx, name="date"), name="vix")
    return s.sort_index()


def parse_yahoo_chart(payload: dict) -> pd.Series:
    result = payload["chart"]["result"][0]
    tz = result["meta"]["exchangeTimezoneName"]
    stamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(tz)
    dates = pd.DatetimeIndex(stamps.tz_localize(None).normalize(), name="date")
    close = pd.Series(result["indicators"]["quote"][0]["close"], index=dates, name="spx", dtype=float)
    close = close[~close.index.duplicated(keep="last")].dropna()
    return close.sort_index()


def load_spx(start: pd.Timestamp = SPX_FETCH_START, end: pd.Timestamp = SPX_FETCH_END) -> pd.Series:
    """S&P 500 daily close. Its dates define the trading-day calendar of the panel."""
    p1, p2 = int(start.timestamp()), int(end.timestamp())
    path = fetch_cached(YAHOO_URL.format(p1=p1, p2=p2), f"spx_{p1}_{p2}.json", headers=BROWSER_UA)
    return parse_yahoo_chart(json.loads(path.read_text()))
