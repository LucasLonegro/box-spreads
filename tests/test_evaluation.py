import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS

from stress_signal.evaluation import clark_west, cumulative_dsse, newey_west_var, oos_r2

H = 20


def recursive_forecasts(y: np.ndarray, x: np.ndarray, n0: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window OLS forecasts (intercept-only vs intercept + x), trained on s <= t - h."""
    X = np.column_stack([np.ones_like(x), x])
    f_small, f_big = [], []
    for t in range(n0, len(y)):
        tr = slice(0, t - h + 1)
        f_small.append(y[tr].mean())
        beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        f_big.append(X[t] @ beta)
    return np.array(f_small), np.array(f_big)


def overlapping_dgp(n: int, beta: float, rng: np.random.Generator, h: int = H) -> tuple[np.ndarray, np.ndarray]:
    """y_t = beta * x_t + sum of the next h shocks: an MA(h-1) error, like a 20-day forward target."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.standard_normal()
    shocks = rng.standard_normal(n + h)
    noise = np.convolve(shocks, np.ones(h), mode="valid")[1 : n + 1]
    return beta * x + noise, x


def test_newey_west_matches_statsmodels():
    rng = np.random.default_rng(1)
    x = np.convolve(rng.standard_normal(520), np.ones(20), mode="valid")
    fit = OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": H, "use_correction": False})
    assert np.sqrt(newey_west_var(x, H) / len(x)) == pytest.approx(fit.bse[0], rel=1e-10)


def test_oos_r2_and_cumulative_dsse():
    y = np.array([1.0, 2.0, 3.0])
    assert oos_r2(y, np.zeros(3), y) == 1.0
    assert oos_r2(y, y + 1, y + 2) == pytest.approx(1 - 4)
    fc = pd.DataFrame({"y": y, "a": y + 1, "b": y})
    assert cumulative_dsse(fc, "a", "b").tolist() == [1.0, 2.0, 3.0]


def test_clark_west_detects_real_signal():
    rng = np.random.default_rng(7)
    y, x = overlapping_dgp(1500, beta=1.0, rng=rng)
    f_small, f_big = recursive_forecasts(y, x, n0=500, h=H)
    res = clark_west(y[500:], f_small, f_big, lags=H)
    assert oos_r2(y[500:], f_small, f_big) > 0
    assert res["cw_t"] > 2.33 and res["cw_p"] < 0.01


def test_clark_west_size_without_signal():
    rng = np.random.default_rng(11)
    pvals = []
    for _ in range(200):
        y, x = overlapping_dgp(800, beta=0.0, rng=rng)
        f_small, f_big = recursive_forecasts(y, x, n0=300, h=H)
        pvals.append(clark_west(y[300:], f_small, f_big, lags=H)["cw_p"])
    rejection = np.mean(np.array(pvals) < 0.05)
    assert rejection < 0.12
    assert np.median(pvals) > 0.2
