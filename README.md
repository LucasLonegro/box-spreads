# stress-signal

Does the Treasury convenience yield (CY = box rate − Treasury yield, 1-year) improve out-of-sample forecasts of S&P 500 realized volatility over the next 20 trading days, beyond VIX and a HAR realized-volatility baseline?

**Sample limit.** Box-rate data are available only from **2005-01-03 to 2020-07-01**. The sample covers the 2008 crisis, August 2011, August 2015, February 2018 and March 2020, but nothing after mid-2020. Extending it requires raw SPX option quotes (e.g. via WRDS).

## Result in brief

- The VIX + HAR baseline has an out-of-sample R² of 58% for log RV relative to the historical mean (2008–2020).
- Adding the 1y CY level lowers the MSE by 2.4% when the evaluation starts in 2008 (Clark–West t = 3.2). With the evaluation starting in 2010 the gain is −0.8% (CW p = 0.07). Almost all of the gain comes from 2008–2009.
- The CY's 5-day change, 252-day z-score and 1y − 3m slope add nothing, alone or combined.
- The CY does not consistently move before VIX in the five stress episodes. In March 2020 it fell, briefly below zero, while volatility spiked.

Details, charts and limitations are in `notebooks/02_forecasting.ipynb`.

## Data

| Source | Use | Units as delivered |
|---|---|---|
| `data/raw/data_public_daily42023.xlsx`, sheet `United States` (Diamond & Van Tassel) | box, gov, CY at 3m/6m/1y/2y | decimals, continuously compounded |
| `data/raw/box_gov_07302019.xlsx`, sheet `box_gov_l` (van Binsbergen, Diamond & Grotteria) | cross-check only | percent, continuously compounded |
| Fed GSW zero curve `feds200628.csv` (`SVENY01/02`) | cross-check of the Treasury leg | percent, continuously compounded |
| FRED `DGS1` | alternative Treasury leg (robustness) | percent, bond-equivalent |
| Cboe `VIX_History.csv` | VIX close | index points |
| Yahoo chart API `^GSPC` | S&P 500 close; defines the trading-day calendar | index points |

Conventions in code: rates are decimals, continuously compounded. CY features are in bp. Volatilities are annualized percent, the same units as VIX. `DGS1` is converted with `2·ln(1 + y/2)`. Weekend rows in the box file (forward-filled by the source) are dropped. Days without `gov_1y` are dropped, not filled.

Data notes:
- Diamond's US government leg is the GSW zero curve (`gov_1y` = `SVENY01` to 0.005 bp), so `box − SVENY01` is not an independent check. The robustness leg is FRED `DGS1` instead, which raises the CY by about 3 bp on average (correlation 0.98).
- The 2019 BDG file matches the Diamond 1y series with a correlation of 0.997 and a mean difference of 0.07 bp. The exceptions are two outlier days in 2009.
- FRED refuses requests that carry a browser User-Agent. Yahoo requires one. `io.fetch_cached` handles both.

## Method

- Target: `log RV_fwd_t`, where `RV_fwd_t = sqrt(252/20 · Σ_{i=1..20} r²_{t+i})`. Levels are a robustness check.
- Baseline: VIX plus 1/5/22-day realized-vol components, in logs for the log target and levels for the level target.
- Expanding-window OLS, refit monthly. Embargo: at forecast date t, train only on rows with s ≤ t − 20 trading days.
- OOS R² against the nested benchmark. Clark–West test with Newey–West standard errors (20 lags). OOS starts 2008-01-02 and 2010-01-04.
- All windows and thresholds are fixed in `src/stress_signal/config.py`.

## Layout

```
src/stress_signal/   config, io (cached downloads), data/ loaders, panel, features, forecast, evaluation, plotting
scripts/             build_dataset.py -> data/processed/panel.parquet; run_backtest.py -> results/*.csv
notebooks/           01_convenience_yield.ipynb, 02_forecasting.ipynb (loading + plotting only)
tests/               loaders, features, no-lookahead, embargo, Clark-West
```

## Running

The box-rate files are not redistributed here. Download `data_public_daily42023.xlsx` and `box_gov_07302019.xlsx` from [William Diamond's papers page](https://williamdiamond.weebly.com/papers.html) into `data/raw/`. Everything else is downloaded automatically.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe scripts\build_dataset.py
.venv\Scripts\python.exe scripts\run_backtest.py
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_convenience_yield.ipynb notebooks\02_forecasting.ipynb
```

Downloads are cached in `data/cache/` and never expire (the sample is historical), so reruns work offline.
