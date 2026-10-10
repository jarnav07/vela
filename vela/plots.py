"""Static report figures (matplotlib, PNG).

Colour roles follow one fixed categorical order so an entity keeps its colour
across every figure: strategy = slot 1 (blue), SPY = slot 2 (orange),
EW universe = slot 3 (aqua), extra comparison = slot 7 (violet). Legends are always
shown for two or more series; axes and grid are kept recessive.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ROLE = {"strategy": SLOTS[0], "SPY": SLOTS[1], "EW": SLOTS[2], "alt": SLOTS[6], "neutral": "#8a8985"}
POS, NEG = "#2a78d6", "#e34948"


def _style(ax, title: str | None = None, ylabel: str | None = None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)


def _fig(w=9, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def equity_curves(series: dict[str, pd.Series], path: Path, title: str, colors: dict[str, str] | None = None,
                  shade: tuple[str, str] | None = None):
    """Growth of $1 on a log axis; optional shaded span (e.g. the holdout period)."""
    fig, ax = _fig()
    _style(ax, title, "Growth of $1 (log scale)")
    colors = colors or {}
    for i, (name, r) in enumerate(series.items()):
        eq = (1 + r.fillna(0)).cumprod()
        c = colors.get(name, SLOTS[i % len(SLOTS)])
        ax.plot(eq.index, eq.values, color=c, linewidth=1.8 if i == 0 else 1.4, label=name)
        ax.annotate(f"{eq.iloc[-1]:.2f}", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=TEXT_2, fontsize=8, va="center")
    ax.set_yscale("log")
    if shade:
        ax.axvspan(pd.Timestamp(shade[0]), pd.Timestamp(shade[1]), color="#f1f0ec", zorder=0)
        ax.text(pd.Timestamp(shade[0]), ax.get_ylim()[1], " holdout", color=TEXT_2, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    _save(fig, path)


def drawdowns(series: dict[str, pd.Series], path: Path, title: str, colors: dict[str, str] | None = None):
    fig, ax = _fig(9, 3.2)
    _style(ax, title, "Drawdown")
    colors = colors or {}
    for i, (name, r) in enumerate(series.items()):
        eq = (1 + r.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        c = colors.get(name, SLOTS[i % len(SLOTS)])
        ax.plot(dd.index, dd.values, color=c, linewidth=1.3, label=f"{name} (max {dd.min():.0%})")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left")
    _save(fig, path)


def annual_returns(series: dict[str, pd.Series], path: Path, title: str, colors: dict[str, str] | None = None):
    df = pd.DataFrame({k: (1 + v.fillna(0)).groupby(v.index.year).prod() - 1 for k, v in series.items()})
    fig, ax = _fig(9, 3.8)
    _style(ax, title, "Calendar-year return")
    colors = colors or {}
    n = df.shape[1]
    width = 0.8 / n
    x = np.arange(len(df))
    for i, col in enumerate(df.columns):
        ax.bar(x + (i - (n - 1) / 2) * width, df[col].values, width=width * 0.92,
               color=colors.get(col, SLOTS[i % len(SLOTS)]), label=col)
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set_xticks(x, [str(y) for y in df.index], fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, ncol=n, loc="upper left")
    _save(fig, path)
    return df


def line_xy(x, ys: dict[str, np.ndarray], path: Path, title: str, xlabel: str, ylabel: str,
            hline: float | None = 0.0, vline: float | None = None, colors: dict[str, str] | None = None):
    fig, ax = _fig(7, 3.8)
    _style(ax, title, ylabel)
    colors = colors or {}
    for i, (name, y) in enumerate(ys.items()):
        ax.plot(x, y, color=colors.get(name, SLOTS[i % len(SLOTS)]), linewidth=2, marker="o", markersize=4,
                label=name)
    if hline is not None:
        ax.axhline(hline, color=TEXT_2, linewidth=0.8)
    if vline is not None:
        ax.axvline(vline, color=TEXT_2, linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    if len(ys) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    _save(fig, path)


def bars(labels, values, path: Path, title: str, ylabel: str, horizontal: bool = False, signed: bool = True,
         fmt: str = "{:.1f}"):
    values = np.asarray(values, dtype=float)
    colors = [POS if v >= 0 else NEG for v in values] if signed else [SLOTS[0]] * len(values)
    if horizontal:
        fig, ax = _fig(7, max(3, 0.28 * len(values) + 1))
        _style(ax, title)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        y = np.arange(len(values))
        ax.barh(y, values, color=colors, height=0.7)
        ax.set_yticks(y, labels, fontsize=8, color=TEXT)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel, color=TEXT_2, fontsize=9)
    else:
        fig, ax = _fig(8, 3.6)
        _style(ax, title, ylabel)
        x = np.arange(len(values))
        ax.bar(x, values, color=colors, width=0.7)
        ax.set_xticks(x, labels, fontsize=8)
        for xi, v in zip(x, values):
            ax.annotate(fmt.format(v), (xi, v), xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                        ha="center", fontsize=7, color=TEXT_2)
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
    _save(fig, path)
