"""Shared matplotlib styling: one palette, recessive chrome, shaded stress episodes."""

from collections.abc import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from stress_signal.config import EPISODES

# Categorical slots, assigned in this fixed order (validated for CVD on adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EPISODE_FILL = "#e9e8e3"

EPISODE_SHORT = {
    "GFC 2008-09 to 2009-03": "GFC",
    "US downgrade 2011-08": "US downgrade",
    "China deval. 2015-08": "Aug 2015",
    "Volmageddon 2018-02": "Feb 2018",
    "COVID 2020-03": "Mar 2020",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
        }
    )


def shade_episodes(
    ax: Axes,
    episodes: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] = EPISODES,
    label: bool = True,
) -> None:
    """Neutral bands for the pre-specified stress episodes, labelled along the top edge."""
    lo, hi = ax.get_xlim()
    for name, (start, end) in episodes.items():
        s, e = mpl.dates.date2num(start), mpl.dates.date2num(end + pd.Timedelta(days=1))
        if e < lo or s > hi:
            continue
        ax.axvspan(start, end + pd.Timedelta(days=1), color=EPISODE_FILL, lw=0, zorder=0)
        if label:
            ax.annotate(
                EPISODE_SHORT.get(name, name),
                xy=(start, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                va="top",
                fontsize=8,
                color=MUTED,
            )
    ax.set_xlim(lo, hi)


def end_label(ax: Axes, series: pd.Series, text: str, dy: float = 0.0) -> None:
    """Direct label at the last point of a line, in secondary ink."""
    s = series.dropna()
    ax.annotate(
        text,
        xy=(s.index[-1], s.iloc[-1]),
        xytext=(4, dy),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=INK_2,
    )


def source_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, -0.01, text, fontsize=8, color=MUTED, ha="left", va="top")
