"""Fixed paths and research parameters. Set before looking at results; not tuned."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PANEL_PATH = DATA_PROCESSED / "panel.parquet"

DIAMOND_FILE = DATA_RAW / "data_public_daily42023.xlsx"
BDG_FILE = DATA_RAW / "box_gov_07302019.xlsx"

H = 20
Z_WINDOW = 252
Z_MIN_PERIODS = 126
HAR_WINDOWS = {"d": 1, "w": 5, "m": 22}
TRADING_DAYS = 252
HAC_LAGS = H

SAMPLE_START = pd.Timestamp("2005-01-03")
SAMPLE_END = pd.Timestamp("2020-07-01")
# SPX is fetched beyond the CY sample: 2004 for HAR lags at the start, late 2020 for the forward target at the end.
SPX_FETCH_START = pd.Timestamp("2004-01-01")
SPX_FETCH_END = pd.Timestamp("2021-01-01")

OOS_STARTS = (pd.Timestamp("2008-01-02"), pd.Timestamp("2010-01-04"))
MIN_TRAIN_OBS = 250

EPISODES: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "GFC 2008-09 to 2009-03": (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31")),
    "US downgrade 2011-08": (pd.Timestamp("2011-08-01"), pd.Timestamp("2011-08-31")),
    "China deval. 2015-08": (pd.Timestamp("2015-08-01"), pd.Timestamp("2015-08-31")),
    "Volmageddon 2018-02": (pd.Timestamp("2018-02-01"), pd.Timestamp("2018-02-28")),
    "COVID 2020-03": (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-31")),
}
# Trading days before each episode start included when comparing CY and VIX peak timing.
EPISODE_LOOKBACK = 63
