"""Daily panel on the S&P 500 trading-day calendar."""

from pathlib import Path

import numpy as np
import pandas as pd

from stress_signal.config import PANEL_PATH, SAMPLE_END, SAMPLE_START
from stress_signal.data.box import load_diamond_us
from stress_signal.data.market import load_spx, load_vix
from stress_signal.data.treasury import bey_to_cc, load_fred, load_gsw


def build_panel(save: bool = True, path: Path = PANEL_PATH) -> pd.DataFrame:
    """Align all sources on SPX trading days.

    The calendar extends beyond the CY sample (see config.SPX_FETCH_*) so that
    trailing HAR components and the forward target are defined at both ends.
    Box/gov/CY columns are NaN outside the sample and on days the source lacks
    `gov_1y`; those rows are dropped at the modelling stage, never filled.
    Rates are decimals, continuously compounded; `vix` and `spx` are index points.
    """
    spx = load_spx()
    calendar = spx.index
    panel = pd.DataFrame(index=calendar)
    panel["spx"] = spx
    panel["vix"] = load_vix().reindex(calendar)

    diamond = load_diamond_us()
    diamond = diamond[(diamond.index >= SAMPLE_START) & (diamond.index <= SAMPLE_END)]
    panel = panel.join(diamond, how="left")

    gsw = load_gsw().rename(columns=str.lower)
    panel = panel.join(gsw, how="left")
    panel["cy_1y_gsw"] = panel["box_1y"] - panel["sveny01"]

    panel["dgs1_cc"] = bey_to_cc(load_fred("DGS1")).reindex(calendar)
    panel["cy_1y_fred"] = panel["box_1y"] - panel["dgs1_cc"]

    panel["tday"] = np.arange(len(panel))
    panel["in_sample"] = (panel.index >= SAMPLE_START) & (panel.index <= SAMPLE_END)
    panel.index.name = "date"

    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path)
    return panel


def load_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def alignment_report(panel: pd.DataFrame) -> dict[str, int]:
    """Counts describing how the Diamond series maps onto SPX trading days."""
    diamond = load_diamond_us()
    diamond = diamond[(diamond.index >= SAMPLE_START) & (diamond.index <= SAMPLE_END)]
    sample = panel[panel["in_sample"]]
    return {
        "diamond_weekday_rows_with_gov_1y": len(diamond),
        "spx_trading_days_in_sample": len(sample),
        "spx_days_with_cy_1y": int(sample["cy_1y"].notna().sum()),
        "spx_days_without_cy_1y": int(sample["cy_1y"].isna().sum()),
        "diamond_days_not_spx_trading_days": int((~diamond.index.isin(panel.index)).sum()),
        "spx_days_missing_vix": int(sample["vix"].isna().sum()),
    }
