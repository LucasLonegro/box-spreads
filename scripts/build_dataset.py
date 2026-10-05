"""Download (or reuse cached) sources and write data/processed/panel.parquet."""

from stress_signal.config import PANEL_PATH
from stress_signal.features import build_features, model_frame
from stress_signal.panel import alignment_report, build_panel


def main() -> None:
    panel = build_panel(save=True)
    print(f"panel: {len(panel)} SPX trading days {panel.index[0].date()} .. {panel.index[-1].date()} -> {PANEL_PATH}")
    for key, value in alignment_report(panel).items():
        print(f"  {key}: {value}")
    frame = model_frame(build_features(panel))
    print(f"model frame: {len(frame)} rows {frame.index[0].date()} .. {frame.index[-1].date()}")
    print(f"mean cy_1y: {frame['cy_1y_bp'].mean():.1f} bp")


if __name__ == "__main__":
    main()
