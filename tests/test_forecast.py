import numpy as np
import pandas as pd
import pytest

from stress_signal.features import build_features, model_frame, model_specs
from stress_signal.forecast import expanding_forecasts, refit_dates, training_mask
from test_features import perturb_after

H = 20


def test_training_mask_boundary():
    tday = np.arange(200)
    mask = training_mask(tday, 150, H)
    assert mask[130] and not mask[131]
    assert mask.sum() == 131


def toy_frame(n: int = 600, drop: slice | None = None) -> pd.DataFrame:
    idx = pd.bdate_range("2006-01-02", periods=n, name="date")
    frame = pd.DataFrame({"tday": np.arange(n), "y": np.arange(n, dtype=float)}, index=idx)
    if drop is not None:
        frame = frame.drop(frame.index[drop])
    return frame


def test_embargo_at_refit_date():
    frame = toy_frame()
    oos_start = pd.Timestamp("2007-03-01")
    fc = expanding_forecasts(frame, "y", {"mean": []}, oos_start, horizon=H, min_train=10)
    for d in refit_dates(fc.index):
        t = frame.at[d, "tday"]
        expected = frame.loc[frame["tday"] <= t - H, "y"].mean()
        assert fc.at[d, "mean"] == pytest.approx(expected)


def test_embargo_counts_trading_days_not_rows():
    frame = toy_frame(drop=slice(290, 300))
    oos_start = pd.Timestamp("2007-03-01")
    fc = expanding_forecasts(frame, "y", {"mean": []}, oos_start, horizon=H, min_train=10)
    d = refit_dates(fc.index)[0]
    t = frame.at[d, "tday"]
    assert fc.at[d, "mean"] == pytest.approx(frame.loc[frame["tday"] <= t - H, "y"].mean())


def test_parameters_held_within_month():
    frame = toy_frame()
    fc = expanding_forecasts(frame, "y", {"mean": []}, pd.Timestamp("2007-03-01"), horizon=H, min_train=10)
    for _, block in fc.groupby(fc.index.to_period("M")):
        assert block["mean"].nunique() == 1


def _forecasts(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = model_frame(build_features(panel))
    return expanding_forecasts(frame, "log_rv_fwd", model_specs("log"), pd.Timestamp("2006-12-01"), horizon=horizon)


def test_forecasts_do_not_look_ahead(synthetic_panel):
    # t is a few days after a refit date, so a missing embargo would leak perturbed targets.
    t = pd.Timestamp("2007-02-07")
    perturbed = perturb_after(synthetic_panel, t)
    fc0, fc1 = _forecasts(synthetic_panel, H), _forecasts(perturbed, H)
    models = [c for c in fc0.columns if c != "y"]
    pd.testing.assert_frame_equal(fc0.loc[:t, models], fc1.loc[:t, models])
    assert not fc0.loc[fc0.index > t, models].equals(fc1.loc[fc1.index > t, models])


def test_lookahead_test_detects_missing_embargo(synthetic_panel):
    t = pd.Timestamp("2007-02-07")
    perturbed = perturb_after(synthetic_panel, t)
    fc0, fc1 = _forecasts(synthetic_panel, 0), _forecasts(perturbed, 0)
    models = [c for c in fc0.columns if c != "y"]
    assert not np.allclose(fc0.loc[:t, models], fc1.loc[:t, models])
