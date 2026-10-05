import json

import numpy as np
import pandas as pd
import pytest

from stress_signal.config import DATA_CACHE, PANEL_PATH, SAMPLE_END, SAMPLE_START
from stress_signal.data.box import TENORS, load_bdg_2019, load_diamond_us
from stress_signal.data.market import parse_yahoo_chart
from stress_signal.data.treasury import bey_to_cc


@pytest.fixture(scope="module")
def diamond() -> pd.DataFrame:
    return load_diamond_us()


def test_diamond_cy_is_box_minus_gov(diamond):
    for tenor in TENORS:
        diff = diamond[f"cy_{tenor}"] - (diamond[f"box_{tenor}"] - diamond[f"gov_{tenor}"])
        assert diff.abs().max() < 1e-12


def test_diamond_values_are_decimals(diamond):
    rates = diamond.filter(regex="^(box|gov)_")
    assert rates.abs().max().max() < 0.1
    assert 0.005 < diamond["box_1y"].median() < 0.05
    assert 0.002 < diamond["cy_1y"].mean() < 0.005


def test_diamond_no_weekends_and_no_missing_gov(diamond):
    assert (diamond.index.dayofweek < 5).all()
    assert diamond["gov_1y"].notna().all()
    assert diamond.index.is_monotonic_increasing and diamond.index.is_unique


def test_diamond_date_range(diamond):
    assert diamond.index[0] == SAMPLE_START
    assert diamond.index[-1] == SAMPLE_END
    assert len(diamond) == 3877


def test_bdg_converted_to_decimals():
    bdg = load_bdg_2019()
    assert bdg.filter(regex="^(box|gov)_").abs().max().max() < 0.1
    assert bdg.index[0] == pd.Timestamp("2004-01-02")
    assert bdg.index[-1] == pd.Timestamp("2018-03-19")
    assert (bdg.index.dayofweek < 5).all()


def test_bey_to_cc():
    assert bey_to_cc(5.0) == pytest.approx(2 * np.log(1.025))
    s = bey_to_cc(pd.Series([0.0, 2.0]))
    assert s.iloc[0] == 0.0 and s.iloc[1] == pytest.approx(2 * np.log(1.01))


def test_parse_yahoo_chart_uses_exchange_dates():
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": [1104762600, 1104849000],  # 2005-01-03/04 09:30 ET
                    "indicators": {"quote": [{"close": [1202.08, None]}]},
                }
            ]
        }
    }
    s = parse_yahoo_chart(payload)
    assert list(s.index) == [pd.Timestamp("2005-01-03")]
    assert s.iloc[0] == 1202.08


needs_panel = pytest.mark.skipif(not PANEL_PATH.exists(), reason="run scripts/build_dataset.py first")


@needs_panel
def test_panel_calendar_and_units():
    panel = pd.read_parquet(PANEL_PATH)
    assert (panel.index.dayofweek < 5).all()
    sample = panel[panel["in_sample"]]
    assert sample.index[0] >= SAMPLE_START and sample.index[-1] <= SAMPLE_END
    assert sample["vix"].between(5, 100).all()
    assert (np.diff(panel["tday"]) == 1).all()
    cy = sample["cy_1y"].dropna()
    assert (cy - (sample.loc[cy.index, "box_1y"] - sample.loc[cy.index, "gov_1y"])).abs().max() < 1e-12


@needs_panel
def test_diamond_gov_leg_is_gsw():
    # Data fact: Diamond's US government leg is the GSW zero curve, so box - SVENY01 is not an independent check.
    panel = pd.read_parquet(PANEL_PATH)
    both = panel[["gov_1y", "sveny01"]].dropna()
    assert (both["gov_1y"] - both["sveny01"]).abs().max() < 1e-6  # GSW CSV is rounded to 1e-4 %


@pytest.mark.skipif(not (DATA_CACHE / "VIX_History.csv").exists(), reason="no cached VIX file")
def test_vix_loader():
    from stress_signal.data.market import load_vix

    vix = load_vix()
    assert vix.index.is_monotonic_increasing
    assert vix.loc["2008-11-20"] == pytest.approx(80.86)
