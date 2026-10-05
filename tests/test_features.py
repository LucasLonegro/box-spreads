import numpy as np
import pandas as pd
import pytest

from stress_signal.features import build_features, forward_rv, trailing_rv, trailing_zscore

TARGETS = ["rv_fwd", "log_rv_fwd"]


def test_forward_rv_by_hand():
    r = pd.Series([0.01, -0.02, 0.03, 0.00, 0.01, -0.01])
    got = forward_rv(r, h=3)
    expected_0 = 100 * np.sqrt(252 / 3 * (0.02**2 + 0.03**2 + 0.0**2))
    expected_2 = 100 * np.sqrt(252 / 3 * (0.0**2 + 0.01**2 + 0.01**2))
    assert got.iloc[0] == pytest.approx(expected_0)
    assert got.iloc[2] == pytest.approx(expected_2)
    assert got.iloc[3:].isna().all()


def test_trailing_rv_by_hand():
    r = pd.Series([0.01, -0.02, 0.03, 0.00])
    got = trailing_rv(r, k=2)
    assert np.isnan(got.iloc[0])
    assert got.iloc[1] == pytest.approx(100 * np.sqrt(252 / 2 * (0.01**2 + 0.02**2)))
    assert got.iloc[3] == pytest.approx(100 * np.sqrt(252 / 2 * 0.03**2))
    assert trailing_rv(r, k=1).iloc[1] == pytest.approx(100 * np.sqrt(252) * 0.02)


def test_trailing_zscore_uses_past_window_only():
    x = pd.Series(np.arange(10, dtype=float) ** 2)
    z = trailing_zscore(x, window=4, min_periods=3)
    w = x.iloc[3:7]
    assert z.iloc[6] == pytest.approx((x.iloc[6] - w.mean()) / w.std())
    assert z.iloc[:2].isna().all()


def test_build_features_target_matches_returns(synthetic_panel):
    f = build_features(synthetic_panel)
    r = np.log(synthetic_panel["spx"]).diff()
    t = 100
    expected = 100 * np.sqrt(252 / 20 * (r.iloc[t + 1 : t + 21] ** 2).sum())
    assert f["rv_fwd"].iloc[t] == pytest.approx(expected)
    assert f["cy_1y_bp"].iloc[t] == pytest.approx(1e4 * synthetic_panel["cy_1y"].iloc[t])


def perturb_after(panel: pd.DataFrame, t: pd.Timestamp, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = panel.copy()
    after = out.index > t
    k = int(after.sum())
    out.loc[after, "spx"] = out.loc[~after, "spx"].iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.03, k)))
    out.loc[after, "vix"] = rng.uniform(10, 80, k)
    for col in ["cy_1y", "cy_3m", "cy_1y_fred", "gov_1y"]:
        out.loc[after, col] = rng.normal(0.004, 0.003, k)
    return out


def test_features_do_not_look_ahead(synthetic_panel):
    t = synthetic_panel.index[600]
    f0 = build_features(synthetic_panel)
    f1 = build_features(perturb_after(synthetic_panel, t))
    cols = [c for c in f0.columns if c not in TARGETS]
    pd.testing.assert_frame_equal(f0.loc[:t, cols], f1.loc[:t, cols])
    assert not f0.loc[f0.index > t, cols].equals(f1.loc[f1.index > t, cols])


def test_target_uses_only_next_h_returns(synthetic_panel):
    t = synthetic_panel.index[600]
    f0 = build_features(synthetic_panel)
    f1 = build_features(perturb_after(synthetic_panel, t))
    closed = f0.index[: 600 - 20 + 1]
    pd.testing.assert_frame_equal(f0.loc[closed, TARGETS], f1.loc[closed, TARGETS])
    assert f0["rv_fwd"].iloc[600 - 19] != f1["rv_fwd"].iloc[600 - 19]
