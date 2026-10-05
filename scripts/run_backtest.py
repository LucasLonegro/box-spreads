"""Run the OOS backtests and write results/*.csv."""

import pandas as pd

from stress_signal.config import OOS_STARTS, PANEL_PATH, RESULTS
from stress_signal.evaluation import cy_rate_regression, cy_vix_timing, episode_errors, insample_table, metrics_table
from stress_signal.features import BASELINE, COMPARISONS, CY_FEATURES, TARGET, build_features, model_frame, model_specs
from stress_signal.forecast import expanding_forecasts
from stress_signal.panel import build_panel, load_panel

EPISODE_MODELS = ["baseline", "cy_level", "cy_all"]


def main() -> None:
    panel = load_panel() if PANEL_PATH.exists() else build_panel(save=True)
    features = build_features(panel)
    frame = model_frame(features)
    RESULTS.mkdir(parents=True, exist_ok=True)

    metrics, episodes, insample = [], [], []
    for kind, target in TARGET.items():
        for oos_start in OOS_STARTS:
            fc = expanding_forecasts(frame, target, model_specs(kind), oos_start)
            tag = f"{kind}_{oos_start:%Y%m%d}"
            fc.to_csv(RESULTS / f"forecasts_{tag}.csv")
            meta = {"target": kind, "oos_start": oos_start.date()}
            metrics.append(metrics_table(fc, COMPARISONS).assign(**meta))
            episodes.append(episode_errors(fc, EPISODE_MODELS).assign(**meta))
        insample.append(insample_table(frame, target, BASELINE[kind], CY_FEATURES).assign(target=kind))

    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df.to_csv(RESULTS / "metrics.csv", index=False)
    pd.concat(episodes, ignore_index=True).to_csv(RESULTS / "episodes.csv", index=False)
    pd.concat(insample, ignore_index=True).to_csv(RESULTS / "insample.csv", index=False)
    cy_vix_timing(features).to_csv(RESULTS / "timing.csv", index=False)
    cy_rate_regression(features).to_csv(RESULTS / "cy_rate.csv", index=False)

    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        cols = ["target", "oos_start", "model", "benchmark", "n", "oos_r2", "cw_t", "cw_p"]
        print(metrics_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
