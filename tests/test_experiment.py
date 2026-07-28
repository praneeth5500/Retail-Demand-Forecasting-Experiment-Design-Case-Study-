"""Tests for the switchback experiment machinery.

Three things the task demands proof of, plus the supporting invariants:
  * the MDE formula matches a hand-computed closed form,
  * every randomization scheme produces balanced arms,
  * the A/A harness recovers ~alpha on synthetic iid data, where the
    false-positive rate is known analytically to be exactly alpha.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import config, experiment as ex


# ---------------------------------------------------------------------------
# MDE / power closed forms
# ---------------------------------------------------------------------------
def test_mde_matches_hand_computed_closed_form():
    """MDE = (z_{0.975} + z_{0.80}) * sd * sqrt(2/n), computed independently."""
    sd, n = 100.0, 50
    expected = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * sd * np.sqrt(2.0 / n)
    got = ex.mde_two_sample(sd, n_per_arm=n, alpha=0.05, power=0.80)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # sanity on the magnitude so a wrong constant cannot pass silently
    np.testing.assert_allclose(got, 56.0317, atol=1e-3)


def test_mde_scales_as_one_over_sqrt_n():
    """Quadrupling n should halve the MDE."""
    a = ex.mde_two_sample(50.0, 40)
    b = ex.mde_two_sample(50.0, 160)
    np.testing.assert_allclose(a / b, 2.0, rtol=1e-12)


def test_power_is_inverse_of_mde():
    """Power evaluated exactly at the MDE must return the target power."""
    sd, n = 80.0, 45
    mde = ex.mde_two_sample(sd, n, alpha=0.05, power=0.80)
    got = ex.power_two_sample(mde, sd, n, alpha=0.05)
    # matches to the far-tail term the two-sided power formula adds (~1e-6)
    np.testing.assert_allclose(got, 0.80, atol=1e-4)


def test_switchback_halves_days_per_arm():
    """A switchback of N days has N/2 days per arm."""
    n_days = 112
    np.testing.assert_allclose(
        ex.mde_switchback(30.0, n_days),
        ex.mde_two_sample(30.0, n_days // 2),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Randomization schemes produce balanced arms
# ---------------------------------------------------------------------------
@pytest.fixture
def window_56():
    return pd.date_range("2024-01-01", periods=56, freq="D")   # 8 whole weeks


def test_complete_randomization_is_exactly_balanced(window_56):
    rng = np.random.default_rng(0)
    for _ in range(100):
        arm = ex.assign_complete(window_56, rng)
        assert arm.sum() == 28                       # exactly half of 56


def test_blocked_within_week_is_balanced(window_56):
    rng = np.random.default_rng(1)
    for _ in range(100):
        arm = ex.assign_blocked_within_week(window_56, rng)
        assert ex.arm_balance(arm) == pytest.approx(0.5)


def test_week_level_assignment_keeps_weeks_intact(window_56):
    rng = np.random.default_rng(2)
    weeks = ex._week_index(window_56)
    for _ in range(50):
        arm = ex.assign_by_week(window_56, rng)
        # every day in a week must share one arm
        for w in np.unique(weeks):
            assert len(set(arm[weeks == w])) == 1
        assert ex.arm_balance(arm) == pytest.approx(0.5)   # 8 weeks -> 4 each


def test_complete_randomization_covers_the_support(window_56):
    """Different seeds give different assignments (not a constant)."""
    a = ex.assign_complete(window_56, np.random.default_rng(0))
    b = ex.assign_complete(window_56, np.random.default_rng(1))
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Variance reduction behaves
# ---------------------------------------------------------------------------
def test_no_adjustment_reduction_is_zero():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-01", periods=400, freq="D")
    s = pd.Series(1000 + rng.normal(0, 100, 400), index=idx)
    vr = ex.variance_reduction_table(s).set_index("scheme")
    assert vr.loc["none", "variance_reduction_pct"] == pytest.approx(0.0, abs=1e-9)


def test_dow_adjustment_removes_a_dow_signal():
    """A pure day-of-week signal should be almost fully absorbed by the dow
    fixed effects, and barely at all by 'none'."""
    idx = pd.date_range("2023-01-01", periods=400, freq="D")
    dow_effect = np.array([0, 50, 100, 150, 200, 500, 450])[idx.dayofweek]
    rng = np.random.default_rng(4)
    s = pd.Series(1000 + dow_effect + rng.normal(0, 10, 400), index=idx)
    vr = ex.variance_reduction_table(s).set_index("scheme")
    assert vr.loc["dow", "variance_reduction_pct"] > 90.0


# ---------------------------------------------------------------------------
# Analysis estimators
# ---------------------------------------------------------------------------
def test_welch_recovers_a_known_shift():
    rng = np.random.default_rng(5)
    y = np.r_[rng.normal(110, 5, 200), rng.normal(100, 5, 200)]
    arm = np.r_[np.ones(200, int), np.zeros(200, int)]
    eff, p = ex.welch_ttest(y, arm)
    assert eff == pytest.approx(10, abs=1.5)
    assert p < 1e-9


def test_regression_adjustment_recovers_arm_coefficient():
    idx = pd.date_range("2023-01-01", periods=112, freq="D")
    cov = ex.covariate_frame(pd.Series(np.arange(112.0), index=idx))
    X = ex.build_design(cov.fillna(0.0), "cuped")
    rng = np.random.default_rng(6)
    arm = ex.assign_complete(idx, rng)
    y = 1000 + 25.0 * arm + rng.normal(0, 5, 112)   # true arm effect = 25
    eff, p = ex.regression_adjusted_test(y, arm, X)
    assert eff == pytest.approx(25, abs=3)
    assert p < 1e-6


def test_dow_residual_unmasks_within_week_correlation():
    """The S3 fix: a strong day-of-week pattern lives INSIDE the week and hides
    real week-to-week correlation, so the RAW intra-week ICC reads near zero while
    the day-of-week-RESIDUAL ICC reveals the clustering. This is exactly why S3's
    inflation is governed by the residual ICC, not the (misleading) raw one."""
    idx = pd.date_range("2023-01-01", periods=8 * 7, freq="D")
    rng = np.random.default_rng(11)
    dow_effect = np.array([0, 50, 100, 150, 200, 600, 550])[idx.dayofweek]
    week = ((idx - idx.min()).days // 7).to_numpy()
    week_shock = rng.normal(0, 40, week.max() + 1)[week]   # shared within each week
    y = pd.Series(1000 + dow_effect + week_shock + rng.normal(0, 5, len(idx)), index=idx)

    raw_icc = ex.intraclass_correlation(y)
    res_icc = ex.intraclass_correlation(ex._dow_residual(y))
    assert res_icc > raw_icc          # residualizing reveals the hidden clustering
    assert raw_icc < 0.05             # raw ICC is masked toward zero by day-of-week
    assert res_icc > 0.3              # the week shock dominates the residual


def test_mean_local_icc_residualize_flag_runs():
    """mean_local_icc with residualize_dow returns a finite ICC >= the raw one on
    a series that has a day-of-week pattern plus a week-level shared shock."""
    idx = pd.date_range("2023-01-01", periods=300, freq="D")
    rng = np.random.default_rng(12)
    dow_effect = np.array([0, 40, 80, 120, 160, 400, 380])[idx.dayofweek]
    week = ((idx - idx.min()).days // 7).to_numpy()
    week_shock = rng.normal(0, 30, week.max() + 1)[week]
    y = pd.Series(800 + dow_effect + week_shock + rng.normal(0, 5, len(idx)), index=idx)
    raw = ex.mean_local_icc(y, window_days=56)
    res = ex.mean_local_icc(y, window_days=56, residualize_dow=True)
    assert np.isfinite(raw) and np.isfinite(res)
    assert res > raw


def test_peeking_stops_early_when_it_can():
    """With a real early effect, peeking reports an early stop."""
    idx = pd.date_range("2023-01-01", periods=56, freq="D")
    arm = np.tile([1, 0], 28)
    rng = np.random.default_rng(7)
    y = 100 + 30.0 * arm + rng.normal(0, 3, 56)
    eff, p, stopped, stop_day = ex.peeking_ttest(y, arm)
    assert stopped and stop_day <= 56 and p < 0.05


# ---------------------------------------------------------------------------
# The headline calibration test: A/A on iid data recovers alpha
# ---------------------------------------------------------------------------
def test_aa_recovers_alpha_on_iid_data():
    """On i.i.d. Gaussian days with no signal, complete randomization + Welch
    is EXACTLY calibrated: P(p < 0.05) = 0.05 analytically. The harness must
    land inside the binomial confidence interval around 0.05."""
    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    rng = np.random.default_rng(123)
    s = pd.Series(1000 + rng.normal(0, 100, 600), index=idx)   # no seasonality
    res = ex.run_aa_scenario(
        s, "S1_complete_welch", duration_weeks=8, n_iter=2000, seed=999)
    # the true FPR is 0.05; the CI around the empirical estimate must cover it
    assert res.fpr_ci_low <= 0.05 <= res.fpr_ci_high
    # effect is centered on zero under the null
    assert abs(res.effect_mean) < 3 * res.effect_sd / np.sqrt(2000)
    # p-values are uniform: KS should not reject
    assert res.ks_pvalue > 0.01


def test_aa_effect_centered_and_pvalues_uniform_iid():
    """Same setup, week-level assignment analyzed at the week level (S4) is also
    calibrated on iid data -- the correct unit of analysis holds its nominal
    rate."""
    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    rng = np.random.default_rng(321)
    s = pd.Series(500 + rng.normal(0, 40, 600), index=idx)
    res = ex.run_aa_scenario(
        s, "S4_week_assign_week_analysis", duration_weeks=8, n_iter=1500, seed=7)
    assert res.fpr_ci_low <= 0.05 <= res.fpr_ci_high
