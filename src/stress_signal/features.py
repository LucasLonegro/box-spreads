"""Realized-vol components, the forward target and CY features.

Every feature at date t uses data dated <= t only. The target at t uses the
returns of t+1..t+H only. Shifts are in rows of the SPX trading-day calendar.
Volatilities are annualized and in percent, the same units as VIX.
"""

import numpy as np
import pandas as pd

from stress_signal.config import H, HAR_WINDOWS, SAMPLE_END, SAMPLE_START, TRADING_DAYS, Z_MIN_PERIODS, Z_WINDOW

# Floor for log RV: a zero daily return would otherwise give log(0) in the 1-day component.
RV_FLOOR = 0.1

BASELINE = {
    "log": ["log_vix", "log_rv_d", "log_rv_w", "log_rv_m"],
    "level": ["vix", "rv_d", "rv_w", "rv_m"],
}
TARGET = {"log": "log_rv_fwd", "level": "rv_fwd"}
CY_FEATURES = {
    "cy_level": ["cy_1y_bp"],
    "cy_d5": ["d5_cy_1y_bp"],
    "cy_z": ["z_cy_1y"],
    "cy_slope": ["slope_cy_bp"],
    "cy_all": ["cy_1y_bp", "d5_cy_1y_bp", "z_cy_1y", "slope_cy_bp"],
}
RATE_CONTROL = ["gov_1y_pct"]


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def trailing_rv(r: pd.Series, k: int) -> pd.Series:
    """sqrt(252/k * sum of r^2 over t-k+1..t), in % annualized."""
    return 100.0 * np.sqrt(TRADING_DAYS / k * (r**2).rolling(k, min_periods=k).sum())


def forward_rv(r: pd.Series, h: int = H) -> pd.Series:
    """sqrt(252/h * sum of r^2 over t+1..t+h), in % annualized. NaN if the window is incomplete."""
    return 100.0 * np.sqrt(TRADING_DAYS / h * (r**2).rolling(h, min_periods=h).sum().shift(-h))


def trailing_zscore(x: pd.Series, window: int = Z_WINDOW, min_periods: int = Z_MIN_PERIODS) -> pd.Series:
    roll = x.rolling(window, min_periods=min_periods)
    return (x - roll.mean()) / roll.std()


def build_features(panel: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """All model inputs and targets on the panel's full trading-day calendar."""
    r = log_returns(panel["spx"])
    f = pd.DataFrame(index=panel.index)
    f["tday"] = panel["tday"]
    f["ret"] = r
    f["vix"] = panel["vix"]
    f["log_vix"] = np.log(panel["vix"])
    for name, k in HAR_WINDOWS.items():
        f[f"rv_{name}"] = trailing_rv(r, k)
        f[f"log_rv_{name}"] = np.log(f[f"rv_{name}"].clip(lower=RV_FLOOR))
    f["rv_fwd"] = forward_rv(r, h)
    f["log_rv_fwd"] = np.log(f["rv_fwd"].clip(lower=RV_FLOOR))

    cy_bp = 1e4 * panel["cy_1y"]
    f["cy_1y_bp"] = cy_bp
    f["d5_cy_1y_bp"] = cy_bp - cy_bp.shift(5)
    f["z_cy_1y"] = trailing_zscore(cy_bp)
    f["slope_cy_bp"] = 1e4 * (panel["cy_1y"] - panel["cy_3m"])
    f["cy_1y_fred_bp"] = 1e4 * panel["cy_1y_fred"]
    f["gov_1y_pct"] = 100.0 * panel["gov_1y"]
    return f


def all_model_columns() -> list[str]:
    cols = {c for spec in BASELINE.values() for c in spec}
    cols |= {c for spec in CY_FEATURES.values() for c in spec}
    cols |= set(RATE_CONTROL) | {"cy_1y_fred_bp"} | set(TARGET.values())
    return sorted(cols)


def model_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Rows inside the CY sample where every model input and both targets exist.

    One common frame keeps all models on identical training and evaluation rows.
    """
    in_sample = (features.index >= SAMPLE_START) & (features.index <= SAMPLE_END)
    cols = ["tday", *all_model_columns()]
    return features.loc[in_sample, cols].dropna()


def model_specs(kind: str) -> dict[str, list[str]]:
    """Nested model specifications for target kind 'log' or 'level'. 'mean' is intercept-only."""
    base = BASELINE[kind]
    specs: dict[str, list[str]] = {"mean": [], "baseline": list(base)}
    for name, cols in CY_FEATURES.items():
        specs[name] = [*base, *cols]
    specs["cy_level_fred"] = [*base, "cy_1y_fred_bp"]
    specs["baseline_rate"] = [*base, *RATE_CONTROL]
    specs["cy_level_rate"] = [*base, *RATE_CONTROL, "cy_1y_bp"]
    specs["cy_all_rate"] = [*base, *RATE_CONTROL, *CY_FEATURES["cy_all"]]
    return specs


# Each augmented model and the nested benchmark it is tested against.
COMPARISONS: dict[str, str] = {
    "baseline": "mean",
    "cy_level": "baseline",
    "cy_d5": "baseline",
    "cy_z": "baseline",
    "cy_slope": "baseline",
    "cy_all": "baseline",
    "cy_level_fred": "baseline",
    "cy_level_rate": "baseline_rate",
    "cy_all_rate": "baseline_rate",
}

MODEL_LABELS: dict[str, str] = {
    "baseline": "Baseline (VIX + HAR) vs historical mean",
    "cy_level": "+ CY 1y level",
    "cy_d5": "+ 5-day change in CY",
    "cy_z": "+ CY 252-day z-score",
    "cy_slope": "+ CY slope (1y - 3m)",
    "cy_all": "+ all four CY features",
    "cy_level_fred": "+ CY level, FRED DGS1 leg",
    "cy_level_rate": "+ CY level, with 1y yield control",
    "cy_all_rate": "+ all CY, with 1y yield control",
}
