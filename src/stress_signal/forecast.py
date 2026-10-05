"""Expanding-window out-of-sample forecasts with an embargo on overlapping targets."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from stress_signal.config import H, MIN_TRAIN_OBS


def training_mask(tday: np.ndarray, forecast_tday: int, horizon: int = H) -> np.ndarray:
    """Rows whose target window has closed by the forecast date.

    The target at s spans returns s+1..s+horizon, so it is observed at the close of
    s+horizon. Training at forecast date t may use only s <= t - horizon.
    """
    return tday <= forecast_tday - horizon


def refit_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First forecast date of each calendar month."""
    months = index.to_period("M")
    first = ~pd.Series(months, index=index).duplicated().to_numpy()
    return index[first]


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return beta


def ols_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def expanding_forecasts(
    frame: pd.DataFrame,
    target: str,
    models: Mapping[str, Sequence[str]],
    oos_start: pd.Timestamp,
    horizon: int = H,
    min_train: int = MIN_TRAIN_OBS,
) -> pd.DataFrame:
    """OOS forecasts of `target` for every row dated >= oos_start.

    Models are refit on the first forecast date of each month and held fixed for
    the rest of that month; the embargo is applied at the refit date, which is
    the earliest date the parameters are used, so it holds for the whole month.
    `frame` must hold a `tday` column with the trading-day ordinal of each row.
    Returns the realized target (`y`) and one column per model.
    """
    frame = frame.sort_index()
    tday = frame["tday"].to_numpy()
    y_all = frame[target].to_numpy(dtype=float)
    oos_index = frame.index[frame.index >= oos_start]
    out = pd.DataFrame(index=oos_index, columns=["y", *models], dtype=float)
    out["y"] = frame.loc[oos_index, target]

    starts = oos_index.get_indexer(refit_dates(oos_index))
    stops = [*starts[1:], len(oos_index)]
    for i0, i1 in zip(starts, stops):
        block = oos_index[i0:i1]
        start = block[0]
        train = training_mask(tday, int(frame.at[start, "tday"]), horizon)
        if train.sum() < min_train:
            raise ValueError(f"only {train.sum()} training rows at {start.date()}")
        for name, cols in models.items():
            X_train = frame.loc[train, list(cols)].to_numpy(dtype=float)
            beta = ols_fit(X_train, y_all[train])
            X_block = frame.loc[block, list(cols)].to_numpy(dtype=float)
            out.loc[block, name] = ols_predict(beta, X_block)
    out.index.name = "date"
    return out
