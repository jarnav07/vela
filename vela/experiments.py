"""Named experiment presets (each returns a list of ExperimentConfig).

Kept in code, not ad-hoc notebooks, so every development trial is reproducible
and countable.
"""
from __future__ import annotations

from .research import ExperimentConfig as C


def baselines():
    out = []
    for h in (1, 5):
        out += [
            C(f"base_rev1_h{h}", model="baseline", baseline_feature="ret_1", baseline_sign=-1, horizon=h),
            C(f"base_rev5_h{h}", model="baseline", baseline_feature="ret_5", baseline_sign=-1, horizon=h),
            C(f"base_rev21_h{h}", model="baseline", baseline_feature="ret_21", baseline_sign=-1, horizon=h),
            C(f"base_mom12_h{h}", model="baseline", baseline_feature="mom_12_1", baseline_sign=1, horizon=h),
        ]
    return out


PRESETS = {
    "baselines": baselines,
}


def ml_v1():
    """First pass of the model ladder on all 53 features."""
    out = []
    for h in (5, 1):
        out += [
            C(f"ridge_h{h}", model="ridge", model_params={"alpha": 100.0}, horizon=h),
            C(f"gbm_h{h}", model="gbm", horizon=h),
            C(f"mlp_h{h}", model="mlp", horizon=h),
        ]
    return out


PRESETS["ml_v1"] = ml_v1
