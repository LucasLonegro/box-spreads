"""Forecast evaluation: OOS R², Clark-West, in-sample HAC regressions, episodes."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy import stats

from stress_signal.config import EPISODE_LOOKBACK, EPISODES, HAC_LAGS
from stress_signal.features import trailing_zscore

TIMING_Z = 2.0


def oos_r2(y: np.ndarray, f_bench: np.ndarray, f_model: np.ndarray) -> float:
    """1 - SSE_model / SSE_bench."""
    y, f_bench, f_model = map(np.asarray, (y, f_bench, f_model))
    return 1.0 - np.sum((y - f_model) ** 2) / np.sum((y - f_bench) ** 2)


def newey_west_var(x: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance of x (demeaned)."""
    u = np.asarray(x, dtype=float) - np.mean(x)
    n = len(u)
    lrv = u @ u / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (u[k:] @ u[:-k]) / n
    return lrv


def clark_west(y: np.ndarray, f_bench: np.ndarray, f_model: np.ndarray, lags: int = HAC_LAGS) -> dict[str, float]:
    """Clark-West (2007) test that the larger nested model does not improve MSPE.

    f_t = e_bench^2 - (e_model^2 - (f_bench - f_model)^2); t = mean(f) / NW s.e.
    One-sided p-value; H1 is that the larger model forecasts better.
    """
    y, f_bench, f_model = map(np.asarray, (y, f_bench, f_model))
    adj = (y - f_bench) ** 2 - ((y - f_model) ** 2 - (f_bench - f_model) ** 2)
    se = np.sqrt(newey_west_var(adj, lags) / len(adj))
    t = float(np.mean(adj) / se)
    return {"cw_mean": float(np.mean(adj)), "cw_t": t, "cw_p": float(stats.norm.sf(t))}


def metrics_table(fc: pd.DataFrame, comparisons: Mapping[str, str], lags: int = HAC_LAGS) -> pd.DataFrame:
    rows = []
    y = fc["y"].to_numpy()
    for model, bench in comparisons.items():
        f_m, f_b = fc[model].to_numpy(), fc[bench].to_numpy()
        rows.append(
            {
                "model": model,
                "benchmark": bench,
                "n": len(y),
                "rmse_model": float(np.sqrt(np.mean((y - f_m) ** 2))),
                "rmse_bench": float(np.sqrt(np.mean((y - f_b) ** 2))),
                "oos_r2": oos_r2(y, f_b, f_m),
                **clark_west(y, f_b, f_m, lags),
            }
        )
    return pd.DataFrame(rows)


def cumulative_dsse(fc: pd.DataFrame, bench: str, model: str) -> pd.Series:
    """Running sum of e_bench^2 - e_model^2; rising means the model is gaining."""
    d = (fc["y"] - fc[bench]) ** 2 - (fc["y"] - fc[model]) ** 2
    return d.cumsum().rename(model)


def insample_table(
    frame: pd.DataFrame, target: str, baseline: Sequence[str], additions: Mapping[str, Sequence[str]], lags: int = HAC_LAGS
) -> pd.DataFrame:
    """Full-sample OLS of target on baseline + each addition, Newey-West s.e. Descriptive only."""
    base_fit = OLS(frame[target], add_constant(frame[list(baseline)])).fit()
    rows = []
    for name, cols in additions.items():
        X = add_constant(frame[[*baseline, *cols]])
        fit = OLS(frame[target], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        for c in cols:
            rows.append(
                {
                    "spec": name,
                    "variable": c,
                    "coef": fit.params[c],
                    "hac_t": fit.tvalues[c],
                    "hac_p": fit.pvalues[c],
                    "r2": fit.rsquared,
                    "r2_baseline": base_fit.rsquared,
                    "n": int(fit.nobs),
                }
            )
    return pd.DataFrame(rows)


def episode_errors(
    fc: pd.DataFrame, models: Sequence[str], episodes: Mapping[str, tuple] = EPISODES
) -> pd.DataFrame:
    """Mean error (realized - forecast) and RMSE by episode, dated by forecast date."""
    rows = []
    for name, (start, end) in episodes.items():
        sub = fc.loc[start:end]
        if sub.empty:
            continue
        for m in models:
            e = sub["y"] - sub[m]
            rows.append({"episode": name, "model": m, "n": len(sub), "mean_error": e.mean(), "rmse": np.sqrt((e**2).mean())})
    return pd.DataFrame(rows)


def _first_crossing(z: pd.Series, threshold: float) -> pd.Timestamp | None:
    hit = z[z > threshold]
    return hit.index[0] if len(hit) else None


def cy_vix_timing(
    features: pd.DataFrame,
    episodes: Mapping[str, tuple] = EPISODES,
    lookback: int = EPISODE_LOOKBACK,
    threshold: float = TIMING_Z,
) -> pd.DataFrame:
    """Did CY move before or after VIX around each episode?

    Window: `lookback` trading days before the episode start through its end.
    Reports each series' peak date and the first date its trailing 252-day
    z-score exceeds `threshold`. Leads are in trading days; positive means CY first.
    """
    cy = features["cy_1y_bp"]
    vix = features["vix"]
    z_cy = features["z_cy_1y"]
    z_vix = trailing_zscore(vix)
    pos = pd.Series(np.arange(len(features)), index=features.index)
    rows = []
    for name, (start, end) in episodes.items():
        i0 = max(pos.index.searchsorted(start) - lookback, 0)
        window = features.index[i0 : pos.index.searchsorted(end, side="right")]
        cy_w, vix_w = cy.loc[window].dropna(), vix.loc[window].dropna()
        cy_peak, vix_peak = cy_w.idxmax(), vix_w.idxmax()
        cy_x = _first_crossing(z_cy.loc[window], threshold)
        vix_x = _first_crossing(z_vix.loc[window], threshold)
        rows.append(
            {
                "episode": name,
                "cy_peak_date": cy_peak.date(),
                "cy_peak_bp": cy_w.max(),
                "vix_peak_date": vix_peak.date(),
                "vix_peak": vix_w.max(),
                "peak_lead_days": int(pos[vix_peak] - pos[cy_peak]),
                "cy_z_cross_date": cy_x.date() if cy_x is not None else None,
                "vix_z_cross_date": vix_x.date() if vix_x is not None else None,
                "cross_lead_days": int(pos[vix_x] - pos[cy_x]) if cy_x is not None and vix_x is not None else None,
            }
        )
    return pd.DataFrame(rows)


def series_agreement(a: pd.Series, b: pd.Series, scale: float = 1e4) -> dict[str, float]:
    """Overlap statistics between two rate series (differences reported in bp)."""
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    d = scale * (both.iloc[:, 0] - both.iloc[:, 1])
    return {
        "n": len(both),
        "start": both.index[0].date(),
        "end": both.index[-1].date(),
        "corr_level": both.iloc[:, 0].corr(both.iloc[:, 1]),
        "corr_20d_change": both.iloc[:, 0].diff(20).corr(both.iloc[:, 1].diff(20)),
        "mean_a_bp": scale * both.iloc[:, 0].mean(),
        "mean_b_bp": scale * both.iloc[:, 1].mean(),
        "mean_diff_bp": d.mean(),
        "sd_diff_bp": d.std(),
    }


def cy_rate_regression(features: pd.DataFrame, lags: int = HAC_LAGS) -> pd.DataFrame:
    """cy_1y (bp) on the 1y Treasury yield (%) with Newey-West s.e., full sample and pre/post 2009."""
    df = features[["cy_1y_bp", "gov_1y_pct"]].dropna()
    splits = {
        "2005-2020": df,
        "2005-2008": df.loc[:"2008-12-31"],
        "2009-2020": df.loc["2009-01-01":],
    }
    rows = []
    for name, sub in splits.items():
        fit = OLS(sub["cy_1y_bp"], add_constant(sub["gov_1y_pct"])).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        rows.append(
            {
                "period": name,
                "n": int(fit.nobs),
                "intercept_bp": fit.params["const"],
                "slope_bp_per_pct": fit.params["gov_1y_pct"],
                "hac_t": fit.tvalues["gov_1y_pct"],
                "r2": fit.rsquared,
                "mean_cy_bp": sub["cy_1y_bp"].mean(),
                "mean_gov_pct": sub["gov_1y_pct"].mean(),
            }
        )
    return pd.DataFrame(rows)


def format_metrics(metrics: pd.DataFrame, target: str, labels: Mapping[str, str]) -> pd.DataFrame:
    """One row per comparison, OOS R² (%) and Clark-West t / p side by side for each OOS start."""
    sub = metrics[metrics["target"] == target].copy()
    sub["OOS R² (%)"] = 100 * sub["oos_r2"]
    sub = sub.rename(columns={"cw_t": "CW t", "cw_p": "CW p"})
    wide = sub.pivot(index="model", columns="oos_start", values=["OOS R² (%)", "CW t", "CW p"])
    wide = wide.swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
    wide = wide.reindex(columns=["OOS R² (%)", "CW t", "CW p"], level=1)
    order = [m for m in labels if m in wide.index]
    wide = wide.loc[order]
    wide.index = [labels[m] for m in order]
    wide.columns = wide.columns.set_names(["OOS start", ""])
    return wide


def oos_r2_by_period(
    fc: pd.DataFrame, bench: str, models: Sequence[str], periods: Mapping[str, tuple[str, str]]
) -> pd.DataFrame:
    """OOS R² (%) of each model against `bench` within each (start, end) sub-window."""
    rows = {}
    for m in models:
        rows[m] = {
            name: 100 * oos_r2(sub["y"], sub[bench], sub[m])
            for name, (lo, hi) in periods.items()
            for sub in [fc.loc[lo:hi]]
        }
    return pd.DataFrame(rows).T
