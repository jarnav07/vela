"""Project-wide paths and the research calendar.

The calendar is fixed *before* any modelling so that the holdout period cannot
drift as results come in (docs/DECISIONS.md, D5).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("VELA_DATA_DIR", ROOT / "data"))
RAW_QLIB_DIR = Path(os.environ.get("VELA_QLIB_DIR", DATA_DIR / "qlib_us"))
PANEL_PATH = DATA_DIR / "panel.npz"
FEATURES_PATH = DATA_DIR / "features.npz"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

# Research calendar (fixed up front)
DATA_START = "2000-01-03"
FIRST_TEST_YEAR = 2006          # first walk-forward out-of-sample year
DEV_END = "2015-12-31"          # development (model-selection) period ends here
HOLDOUT_START = "2016-01-01"    # locked holdout: evaluated once with a frozen config
DATA_END = "2020-11-10"
