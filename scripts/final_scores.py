"""Compute (but do not evaluate) full-period 2006-2020 walk-forward scores for the frozen models.

Separated from final_evaluation.py so the expensive training can run while other work continues,
without anyone looking at holdout performance before the freeze is complete.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela.config import DATA_DIR, DATA_END, RESULTS_DIR  # noqa: E402
from vela.research import ExperimentConfig, ResearchData, run_walkforward, save_scores  # noqa: E402


def main() -> None:
    frozen = json.loads((RESULTS_DIR / "frozen_config.json").read_text())
    data = ResearchData.load(frozen["feature_set"])
    names = {f.name for f in fields(ExperimentConfig)}
    for d in frozen["models"]:
        cfg = ExperimentConfig(**{k: v for k, v in d.items() if k in names})
        if (DATA_DIR / "scores" / f"final_{cfg.name}.npy").exists():
            print("cached", cfg.name)
            continue
        cfg.last_test_year, cfg.last_test_date = 2020, DATA_END
        scores, diags = run_walkforward(cfg, data)
        save_scores(f"final_{cfg.name}", scores)
        print("done", cfg.name, sum(x["secs"] for x in diags), "s", flush=True)


if __name__ == "__main__":
    main()
