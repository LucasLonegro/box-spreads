import numpy as np
import pandas as pd
import pytest


def make_synthetic_panel(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Random panel with the columns build_features expects, starting at the CY sample start."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n, name="date")
    vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.05, n)).clip(-1.5, 1.5))
    ret = vol * rng.standard_normal(n)
    cy_1y = 0.0035 + 0.001 * np.cumsum(rng.normal(0, 0.05, n))
    panel = pd.DataFrame(
        {
            "spx": 1000 * np.exp(np.cumsum(ret)),
            "vix": 100 * np.sqrt(252) * vol * np.exp(rng.normal(0, 0.1, n)),
            "cy_1y": cy_1y,
            "cy_3m": cy_1y + rng.normal(0, 0.0005, n),
            "cy_1y_fred": cy_1y + rng.normal(0, 0.0003, n),
            "gov_1y": 0.02 + 0.001 * np.cumsum(rng.normal(0, 0.05, n)),
            "tday": np.arange(n),
        },
        index=idx,
    )
    missing = rng.choice(np.arange(300, n), size=10, replace=False)
    panel.iloc[missing, panel.columns.get_indexer(["cy_1y", "cy_3m", "cy_1y_fred", "gov_1y"])] = np.nan
    return panel


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    return make_synthetic_panel()
