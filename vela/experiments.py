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


def ml_v2():
    """Feature set v2 (statistical industries); slower horizon; volatility-scaled target."""
    return [
        C("gbm_v2_h5", model="gbm", horizon=5, feature_set="v2"),
        C("gbm_v2_h21", model="gbm", horizon=21, feature_set="v2"),
        C("gbm_v2_h5_vs", model="gbm", horizon=5, target="vol_scaled", feature_set="v2"),
    ]


PRESETS["ml_v2"] = ml_v2


def seeds():
    """Seed-noise floor: how much do GBM results move with the random seed alone?"""
    return [
        C("gbm_h5_s1", model="gbm", horizon=5, feature_set="v1", seed=1),
        C("gbm_v2_h5_s1", model="gbm", horizon=5, feature_set="v2", seed=1),
    ]


PRESETS["seeds"] = seeds


def lstm():
    """Sequence model on raw daily inputs (no hand-engineered features)."""
    # First attempt (seq_len 40, hidden 32, 2 seeds, expanding window) needed ~9 min for the smallest
    # fold on 4 CPU cores and early-stopped at epoch 2: resized to finish in ~30 min (DEVLOG, Phase 9).
    return [C("lstm_h5", model="lstm", horizon=5, feature_set="v1", sample_every=5, train_years=5,
              model_params={"seq_len": 30, "hidden": 24, "n_seeds": 1, "max_epochs": 6, "patience": 2})]


PRESETS["lstm"] = lstm


def horizons():
    """Label horizon matched to the measured ~3-day signal decay."""
    return [C("gbm_h3", model="gbm", horizon=3, feature_set="v1")]


PRESETS["horizons"] = horizons
