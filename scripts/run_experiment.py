"""Run one or more walk-forward experiments on the development period and log them.

    python scripts/run_experiment.py baselines
    python scripts/run_experiment.py gbm_h5 ridge_h5 ...

Experiment presets live in vela/experiments.py. Every run is appended to
results/experiments.csv (the trial log used by the Deflated Sharpe Ratio), and the
out-of-sample scores are cached in data/scores/ for later analysis.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela.config import DEV_END  # noqa: E402
from vela.experiments import PRESETS  # noqa: E402
from vela.research import ResearchData, evaluate, log_experiment, run_walkforward, save_scores  # noqa: E402


def main(names: list[str]) -> None:
    loaded: dict[str, ResearchData] = {}
    for group in names:
        for cfg in PRESETS[group]():
            if cfg.feature_set not in loaded:
                loaded.clear()  # keep one feature cache in memory at a time
                loaded[cfg.feature_set] = ResearchData.load(cfg.feature_set)
            data = loaded[cfg.feature_set]
            t0 = time.time()
            scores, diags = run_walkforward(cfg, data)
            save_scores(cfg.name, scores)
            res = evaluate(scores, data, cfg.horizon, f"{cfg.first_test_year}-01-01", DEV_END)
            log_experiment(cfg, res, diags)
            keys = ["ic_mean_h1", f"ic_mean_h{cfg.horizon}", f"ic_t_h{cfg.horizon}", "decile_spread_bps",
                    "ls_sharpe_gross", "ls_sharpe_net", "ls_turnover", "lo_sharpe_net", "lo_ir_ew", "lo_ir_spy"]
            summary = " ".join(f"{k}={res[k]:.3f}" for k in keys if k in res)
            print(f"{cfg.name:<28} {summary}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["baselines"])
