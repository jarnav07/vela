"""Which feature families drive the GBM? Grouped, within-date permutation importance.

Trains the frozen GBM configuration on 2001-2012 (validation 2013) and measures the drop
in mean daily rank IC on 2014-2015 when each feature family is shuffled across stocks
within each date. Uses development-period data only (the holdout is never touched).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela import plots  # noqa: E402
from vela.analysis import grouped_permutation_importance  # noqa: E402
from vela.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from vela.models.sklearn_models import GBMModel  # noqa: E402
from vela.research import ResearchData  # noqa: E402
from vela.targets import make_target  # noqa: E402


def main(feature_set: str = "v1", horizon: int = 5) -> None:
    d = ResearchData.load(feature_set)
    y = make_target(d.fwd(horizon), d.tradable, "rank")
    ok = d.tradable & np.isfinite(y)
    yrs = d.dates.year
    purge = horizon + 2

    def rows(mask_dates, every=2):
        idx = np.flatnonzero(mask_dates)[::every]
        t, j = np.nonzero(ok[idx])
        return idx[t], j

    tr = (yrs >= 2001) & (yrs <= 2012)
    tr[np.flatnonzero(yrs == 2013)[0] - purge:] = False
    va = yrs == 2013
    va[np.flatnonzero(yrs == 2014)[0] - purge:] = False
    te = (yrs >= 2014) & (yrs <= 2015)
    F = d.features.values
    (a, b), (c, e), (g, h) = rows(tr), rows(va), rows(te, every=1)
    model = GBMModel().fit(np.asarray(F[a, b]), y[a, b], np.asarray(F[c, e]), y[c, e], c)
    imp = grouped_permutation_importance(model, np.asarray(F[g, h]), y[g, h], g, d.features.names)
    imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    print(imp.to_string(index=False))
    plots.bars(imp["family"].tolist(), (imp["ic_drop"] * 100).values, FIGURES_DIR / "feature_importance.png",
               "What the GBM uses: IC lost when a feature family is shuffled (2014-2015)",
               "drop in mean daily IC (x100)", horizontal=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
