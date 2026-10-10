"""Build the cleaned, aligned research panel from the raw Qlib download.

    python scripts/fetch_data.sh          # once
    python scripts/build_dataset.py

Writes data/panel.npz and results/data_quality.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela.config import PANEL_PATH, RAW_QLIB_DIR, RESULTS_DIR  # noqa: E402
from vela.data.panel import build_panel  # noqa: E402
from vela.data.qlib_source import QlibSource  # noqa: E402
from vela.data.quality import clean_panel  # noqa: E402
from vela.universe import tradable_mask  # noqa: E402


def main() -> None:
    t0 = time.time()
    raw = build_panel(QlibSource(RAW_QLIB_DIR), market="sp500")
    panel, report = clean_panel(raw)
    panel.save(PANEL_PATH)

    tradable = tradable_mask(panel)
    n_member = panel.member.sum(1)
    n_trad = tradable.sum(1)
    years = panel.dates.year
    summary = {
        "dates": [str(panel.dates[0].date()), str(panel.dates[-1].date())],
        "n_days": int(len(panel.dates)),
        "n_tickers_with_data": int(panel.shape[1]),
        "membership_symbols": raw.meta["n_membership_symbols"],
        "membership_symbols_without_price_data": raw.meta["missing_symbols"],
        "members_per_day_median": float(np.median(n_member)),
        "tradable_per_day_by_year": {
            int(y): int(np.median(n_trad[years == y])) for y in np.unique(years)
        },
        **report,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "data_quality.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "membership_symbols_without_price_data"}, indent=2))
    print(f"panel saved to {PANEL_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
