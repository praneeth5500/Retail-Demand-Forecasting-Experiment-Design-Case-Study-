"""Statistical machinery for a DAY-LEVEL SWITCHBACK experiment design.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
No treatment was ever applied to this store. There is no experiment in this
data. Everything here is *offline design and validation* on historical
observational data: how big an effect a switchback could detect (power / MDE),
and whether the intended analysis pipeline holds its nominal false-positive
rate when it is fed a known null (the A/A test in scripts/07). Nothing in this
module should be read as evidence that any promotion works.

WHY A SWITCHBACK
----------------
The store has no customer identifiers and no loyalty program, so customers
cannot be assigned to treatment and control. The only unit that can be
randomized is TIME. A switchback randomizes days (or weeks) to treatment and
control and uses the store as its own control. Its Achilles heel is SUTVA:
a promo-day drink purchase can suppress the next day's purchase (stockpiling),
so treatment days can contaminate adjacent control days. The estimators here
assume that carryover is negligible; where it is not, effects are biased toward
zero and the day-level variance estimates understate the true uncertainty.

METHODS AND WHY THESE ONES
--------------------------
* MDE / power: the standard two-sample normal approximation. Daily revenue is
  averaged over dozens of days per arm, so by the CLT the arm means are
  near-normal and the normal quantile formula is accurate; it is also the form
  every reviewer recognises and can check by hand. The small-sample t
  correction is negligible at n >= 28 per arm and is noted where it applies.
* Variance reduction: OLS regression adjustment (CUPED is the pre-period-
  covariate special case). Chosen over stratified estimators because it
  composes -- day-of-week, trend, seasonality and a pre-period level all enter
  one design matrix -- and because the realised reduction is then simply
  1 - Var(resid)/Var(raw), measured on real data rather than assumed.
* A/A analysis: Welch's t-test (unequal-variance) as the default two-sample
  test; it is the correct default when the two arms' variances are not known to
  be equal, and here they need not be.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config


# ===========================================================================
# 1. Metric construction
# ===========================================================================
def department_panel_to_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide daily frame: one column per department plus the composite metrics.

    `panel` is the long date x department x revenue table from data_prep.
    Returns a date-indexed frame carrying every individual department, the
    primary `inside_merch` metric, and the two guardrails (`total_revenue`,
    `fuel_volume_proxy`).
    """
    wide = panel.pivot_table(
        index="date", columns="department", values="revenue", aggfunc="sum"
    ).sort_index()

    present = [d for d in config.INSIDE_MERCH_DEPARTMENTS if d in wide.columns]
    missing = set(config.INSIDE_MERCH_DEPARTMENTS) - set(present)
    if missing:
        raise ValueError(f"inside-merch departments absent from panel: {sorted(missing)}")

    out = wide.copy()
    out["inside_merch"] = wide[present].sum(axis=1)
    out["total_revenue"] = wide.sum(axis=1)
    fuel = [d for d in config.FUEL_DEPARTMENTS if d in wide.columns]
    out["fuel_volume_proxy"] = wide[fuel].sum(axis=1)
    return out


def recent_window(frame: pd.DataFrame, weeks: int = config.FEASIBILITY_WINDOW_WEEKS) -> pd.DataFrame:
    """The trailing `weeks` whole weeks of a date-indexed frame.

    Whole weeks (7*weeks days) so day-of-week is balanced in the window and the
    variance estimate is not tilted by a partial final week.
    """
    end = frame.index.max()
    start = end - pd.Timedelta(days=weeks * 7 - 1)
    return frame.loc[frame.index >= start]


# ===========================================================================
# 2. Descriptive feasibility triage
# ===========================================================================
def describe_metric(series: pd.Series, total: pd.Series) -> dict:
    """Daily mean, sd, coefficient of variation, and share of a total."""
    mean = float(series.mean())
    sd = float(series.std(ddof=1))
    return {
        "mean": mean,
        "sd": sd,
        "cv": sd / mean if mean != 0 else np.nan,
        "share": mean / float(total.mean()) if total.mean() != 0 else np.nan,
    }


# ===========================================================================
# 3. Power / minimum detectable effect
# ===========================================================================
def mde_two_sample(sd: float, n_per_arm: int,
                   alpha: float = config.ALPHA, power: float = config.POWER) -> float:
    """Minimum detectable effect for a balanced two-sample comparison of means.

    Normal approximation, equal per-arm standard deviation `sd`:

        MDE = (z_{1-alpha/2} + z_{power}) * sd * sqrt(2 / n_per_arm)

    This is the closed form validated in tests/test_experiment.py.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    se = sd * np.sqrt(2.0 / n_per_arm)
    return float((z_alpha + z_power) * se)


def mde_switchback(sd: float, n_days: int,
                   alpha: float = config.ALPHA, power: float = config.POWER) -> float:
    """MDE for a balanced day-level switchback of `n_days` total days.

    Half the days are treatment, half control, so n_per_arm = n_days // 2.
    """
    return mde_two_sample(sd, n_days // 2, alpha=alpha, power=power)


def power_two_sample(effect: float, sd: float, n_per_arm: int,
                     alpha: float = config.ALPHA) -> float:
    """Power to detect `effect` in a balanced two-sample test (normal approx)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se = sd * np.sqrt(2.0 / n_per_arm)
    ncp = abs(effect) / se
    # two-sided: reject in either tail
    return float(stats.norm.cdf(ncp - z_alpha) + stats.norm.cdf(-ncp - z_alpha))


# ===========================================================================
# 4. Variance reduction via regression adjustment
# ===========================================================================
# Adjustment schemes, from A2 of the task. Each name maps to the covariates
# added on top of the intercept. "trend" is a within-window normalised index,
# added at design-matrix build time because it is defined relative to the
# analysis window, not to the calendar.
ADJUSTMENT_SCHEMES = {
    "none": [],
    "dow": ["dow"],
    "dow_trend_fourier": ["dow", "trend", "fourier"],
    "cuped": ["dow", "trend", "fourier", "pretrend28"],
}
ADJUSTMENT_LABELS = {
    "none": "(i) no adjustment",
    "dow": "(ii) day-of-week fixed effects",
    "dow_trend_fourier": "(iii) day-of-week + trend + Fourier yearly",
    "cuped": "(iv) (iii) + pre-period 28d mean (CUPED)",
}


def covariate_frame(series: pd.Series) -> pd.DataFrame:
    """Date-indexed covariates for regression adjustment.

    The pre-period covariate is the trailing 28-day mean of the metric, shifted
    by one day so it uses only strictly-pre-period information -- the CUPED
    covariate must be untouched by the arm assignment, so it can never include
    the current day.
    """
    idx = series.index
    cov = pd.DataFrame(index=idx)
    cov["dow"] = idx.dayofweek
    doy = idx.dayofyear.to_numpy()
    for k in (1, 2):
        cov[f"yearly_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cov[f"yearly_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    cov["pretrend28"] = series.shift(1).rolling(28, min_periods=28).mean()
    return cov


def _fourier_cols(cov: pd.DataFrame) -> list[str]:
    return [c for c in cov.columns if c.startswith("yearly_")]


def build_design(cov: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Design matrix (no intercept; caller adds the constant) for a scheme.

    `cov` is a covariate_frame slice already restricted to the analysis window.
    """
    parts = []
    terms = ADJUSTMENT_SCHEMES[scheme]
    if "dow" in terms:
        parts.append(pd.get_dummies(cov["dow"], prefix="dow", drop_first=True).astype(float))
    if "trend" in terms:
        # normalised within-window linear trend in [0, 1]
        n = len(cov)
        parts.append(pd.DataFrame(
            {"trend": np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(n)}, index=cov.index))
    if "fourier" in terms:
        parts.append(cov[_fourier_cols(cov)])
    if "pretrend28" in terms:
        parts.append(cov[["pretrend28"]])
    if not parts:
        return pd.DataFrame(index=cov.index)
    return pd.concat(parts, axis=1)


def residual_std(y: np.ndarray, X: pd.DataFrame | None) -> float:
    """Residual standard deviation of y after OLS on X (intercept always in)."""
    y = np.asarray(y, dtype=float)
    if X is None or X.shape[1] == 0:
        return float(np.std(y, ddof=1))
    Xc = sm.add_constant(X.to_numpy(dtype=float), has_constant="add")
    res = sm.OLS(y, Xc).fit()
    dof = len(y) - Xc.shape[1]
    return float(np.sqrt(res.ssr / dof))


def variance_reduction_table(series: pd.Series) -> pd.DataFrame:
    """Realised residual sd and variance-reduction % for each scheme.

    Variance reduction is measured, not assumed:
        reduction = 1 - Var(resid_scheme) / Var(resid_none).
    Computed on real historical data for the given (windowed) series.
    """
    cov = covariate_frame(series)
    # CUPED needs 28 days of pre-history; drop rows where any covariate is NaN
    # so every scheme is compared on the SAME rows (a fair reduction ratio).
    common = cov.dropna().index.intersection(series.index)
    y = series.loc[common].to_numpy(dtype=float)
    cov = cov.loc[common]

    base_var = np.var(y, ddof=1)
    rows = []
    for scheme in ADJUSTMENT_SCHEMES:
        X = build_design(cov, scheme)
        rsd = residual_std(y, X if X.shape[1] else None)
        rows.append({
            "scheme": scheme,
            "label": ADJUSTMENT_LABELS[scheme],
            "residual_sd": rsd,
            "variance_reduction_pct": 100.0 * (1.0 - rsd**2 / base_var),
            "n": len(y),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 5. Randomization schemes  (all return an int arm array, 0 = control, 1 = treat)
# ===========================================================================
def _week_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """Consecutive 7-day blocks from the window start = 'weeks'.

    Windows are always a whole number of weeks long, so blocks are clean and
    this matches how a switchback schedule would actually be laid out.
    """
    offset = (dates - dates.min()).days
    return (offset // 7).to_numpy()


def assign_complete(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Complete randomization: exactly floor(n/2) days to treatment, all such
    assignments equally likely. Guarantees near-exact arm balance."""
    n = len(dates)
    arm = np.zeros(n, dtype=int)
    treat = rng.permutation(n)[: n // 2]
    arm[treat] = 1
    return arm


def assign_blocked_within_week(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Randomize days to arms WITHIN each week block.

    Each 7-day block contributes 3 or 4 treatment days (the odd day alternates
    block to block so the whole window stays balanced). Balances weekday mix
    across arms, which is the dominant source of daily variance.
    """
    weeks = _week_index(dates)
    arm = np.zeros(len(dates), dtype=int)
    odd_to_treat = True
    for w in np.unique(weeks):
        pos = np.where(weeks == w)[0]
        k = len(pos) // 2 + (len(pos) % 2 if odd_to_treat else 0)
        if len(pos) % 2:
            odd_to_treat = not odd_to_treat
        chosen = rng.permutation(len(pos))[:k]
        arm[pos[chosen]] = 1
    return arm


def assign_by_week(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Cluster (week-level) assignment: every day in a week shares one arm.

    Half the week blocks go to treatment. This is the honest switchback unit if
    a promo cannot be toggled day to day, and it is what scenarios S3/S4 stress.
    """
    weeks = _week_index(dates)
    uniq = np.unique(weeks)
    treat_weeks = set(uniq[rng.permutation(len(uniq))[: len(uniq) // 2]])
    return np.array([1 if w in treat_weeks else 0 for w in weeks], dtype=int)


def arm_balance(arm: np.ndarray) -> float:
    """Fraction of days in the treatment arm (0.5 == perfectly balanced)."""
    return float(np.mean(arm))


# ===========================================================================
# 6. Analysis estimators  -> (effect, p_value)
# ===========================================================================
def welch_ttest(y: np.ndarray, arm: np.ndarray) -> tuple[float, float]:
    """Welch (unequal-variance) two-sample t-test. effect = mean_T - mean_C."""
    y = np.asarray(y, dtype=float)
    a, b = y[arm == 1], y[arm == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p)


def week_level_ttest(y: np.ndarray, arm: np.ndarray,
                     dates: pd.DatetimeIndex) -> tuple[float, float]:
    """Aggregate each week to its mean, then Welch-test at the week level.

    The correct analysis for week-level (cluster) assignment: the independent
    unit is the week, not the day."""
    weeks = _week_index(dates)
    df = pd.DataFrame({"y": y, "arm": arm, "week": weeks})
    wk = df.groupby("week").agg(y=("y", "mean"), arm=("arm", "first"))
    return welch_ttest(wk["y"].to_numpy(), wk["arm"].to_numpy())


def regression_adjusted_test(y: np.ndarray, arm: np.ndarray,
                             X: pd.DataFrame) -> tuple[float, float]:
    """OLS of y on [arm, covariates]; report the arm coefficient and its p.

    This is the A2(iv)/CUPED analysis: the treatment effect is the arm
    coefficient after adjusting for day-of-week, trend, seasonality and the
    pre-period level."""
    design = pd.concat(
        [pd.Series(arm, index=X.index, name="arm").astype(float), X], axis=1)
    Xc = sm.add_constant(design.to_numpy(dtype=float), has_constant="add")
    res = sm.OLS(np.asarray(y, dtype=float), Xc).fit()
    # column 0 is the constant, column 1 is arm
    return float(res.params[1]), float(res.pvalues[1])


def peeking_ttest(y: np.ndarray, arm: np.ndarray,
                  start_day: int = 7, alpha: float = config.ALPHA
                  ) -> tuple[float, float, bool, int]:
    """Daily peeking: walk day by day, stop the FIRST time Welch p < alpha.

    The schedule (arm) is fixed up front; the analyst re-tests every day as data
    arrives and stops at the first 'significant' look. Returns the effect and p
    at the stopping day, whether it stopped early, and the stopping day index.
    This is the multiple-comparisons trap the A/A test is meant to expose."""
    n = len(y)
    last_effect, last_p = welch_ttest(y, arm)
    for t in range(start_day, n + 1):
        eff, p = welch_ttest(y[:t], arm[:t])
        if np.isnan(p):
            continue
        last_effect, last_p = eff, p
        if p < alpha:
            return eff, p, True, t
    return last_effect, last_p, False, n


# ===========================================================================
# 7. A/A harness
# ===========================================================================
# Randomization x analysis pairings from B2. Each scenario is a self-contained
# recipe the harness dispatches on.
AA_SCENARIOS = {
    "S1_complete_welch": "day-level complete randomization, Welch t-test",
    "S2_blocked_welch": "day-level randomization blocked within week, Welch t-test",
    "S3_week_assign_day_analysis": "week-level assignment, analyzed at day level (wrong unit)",
    "S4_week_assign_week_analysis": "week-level assignment, analyzed at week level",
    "S5_complete_regadj": "day-level complete randomization, CUPED regression adjustment",
    "S6_complete_peeking": "day-level complete randomization, daily peeking to first p<0.05",
}


@dataclass
class AAResult:
    scenario: str
    draws: pd.DataFrame          # per-iteration: effect, p_value, (stopped, stop_day for S6)
    false_positive_rate: float
    fpr_ci_low: float
    fpr_ci_high: float
    effect_mean: float
    effect_sd: float
    ks_stat: float
    ks_pvalue: float


def _valid_starts(index: pd.DatetimeIndex, window_days: int, min_prehistory: int) -> np.ndarray:
    """Start positions for a contiguous window that (a) fits and (b) leaves
    `min_prehistory` days before it, so the CUPED covariate is always defined.
    Every scenario draws from this same pool for a fair comparison."""
    n = len(index)
    lo, hi = min_prehistory, n - window_days
    if hi < lo:
        raise ValueError("series too short for the requested window")
    return np.arange(lo, hi + 1)


def _run_scenario_once(scenario: str, y: np.ndarray, dates: pd.DatetimeIndex,
                       cov_slice: pd.DataFrame, rng: np.random.Generator) -> dict:
    if scenario == "S1_complete_welch":
        arm = assign_complete(dates, rng)
        eff, p = welch_ttest(y, arm)
    elif scenario == "S2_blocked_welch":
        arm = assign_blocked_within_week(dates, rng)
        eff, p = welch_ttest(y, arm)
    elif scenario == "S3_week_assign_day_analysis":
        arm = assign_by_week(dates, rng)
        eff, p = welch_ttest(y, arm)
    elif scenario == "S4_week_assign_week_analysis":
        arm = assign_by_week(dates, rng)
        eff, p = week_level_ttest(y, arm, dates)
    elif scenario == "S5_complete_regadj":
        arm = assign_complete(dates, rng)
        X = build_design(cov_slice, "cuped")
        eff, p = regression_adjusted_test(y, arm, X)
    elif scenario == "S6_complete_peeking":
        arm = assign_complete(dates, rng)
        eff, p, stopped, stop_day = peeking_ttest(y, arm)
        return {"effect": eff, "p_value": p, "stopped_early": stopped, "stop_day": stop_day}
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return {"effect": eff, "p_value": p}


def run_aa_scenario(series: pd.Series, scenario: str,
                    duration_weeks: int = config.PLANNED_DURATION_WEEKS,
                    n_iter: int = config.AA_ITERATIONS,
                    seed: int = config.EXPERIMENT_SEED,
                    alpha: float = config.ALPHA) -> AAResult:
    """Run one draw-assign-analyze scenario `n_iter` times under the null.

    Because no treatment was applied, the true effect is exactly zero: every
    p < alpha is a false positive by construction. A well-calibrated pairing
    returns FPR ~= alpha and uniform p-values.
    """
    if scenario not in AA_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    window_days = duration_weeks * 7
    cov = covariate_frame(series)
    starts = _valid_starts(series.index, window_days, min_prehistory=28)
    rng = np.random.default_rng(seed)

    records = []
    for _ in range(n_iter):
        s0 = int(rng.choice(starts))
        widx = series.index[s0: s0 + window_days]
        y = series.loc[widx].to_numpy(dtype=float)
        records.append(_run_scenario_once(scenario, y, widx, cov.loc[widx], rng))

    draws = pd.DataFrame(records)
    p = draws["p_value"].to_numpy()
    p = p[~np.isnan(p)]

    n_sig = int(np.sum(p < alpha))
    fpr = n_sig / len(p)
    ci_low, ci_high = _binom_ci(n_sig, len(p))
    ks = stats.kstest(p, "uniform")
    return AAResult(
        scenario=scenario,
        draws=draws,
        false_positive_rate=fpr,
        fpr_ci_low=ci_low,
        fpr_ci_high=ci_high,
        effect_mean=float(draws["effect"].mean()),
        effect_sd=float(draws["effect"].std(ddof=1)),
        ks_stat=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
    )


def intraclass_correlation(series: pd.Series, dates: pd.DatetimeIndex | None = None) -> float:
    """One-way ICC of a daily series across 7-day (week) blocks.

    The design effect for week-level (cluster) assignment analyzed at the day
    level is 1 + (m-1)*ICC, with m the mean cluster size. A large ICC means the
    day-level Welch test badly understates the true SE; an ICC near zero means
    it barely does. This is the number that explains why S3 does or does not
    fail, so the A/A script can report it rather than assert a textbook figure.
    """
    x = series if dates is None else series.loc[dates]
    weeks = _week_index(pd.DatetimeIndex(x.index))
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "wk": weeks})
    grand = df["x"].mean()
    n, k = len(df), df["wk"].nunique()
    if k < 2 or n <= k:
        return float("nan")
    grp = df.groupby("wk")["x"]
    m_bar = grp.count().mean()
    ss_between = (grp.mean().sub(grand) ** 2 * grp.count()).sum()
    ss_within = grp.apply(lambda g: ((g - g.mean()) ** 2).sum()).sum()
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    return float((ms_between - ms_within) / (ms_between + (m_bar - 1) * ms_within))


def _dow_residual(x: pd.Series) -> pd.Series:
    """Residual of a daily series after removing day-of-week means (OLS)."""
    idx = pd.DatetimeIndex(x.index)
    dummies = pd.get_dummies(idx.dayofweek, prefix="d", drop_first=True).astype(float)
    Xc = sm.add_constant(dummies.to_numpy(), has_constant="add")
    res = sm.OLS(np.asarray(x, dtype=float), Xc).fit()
    return pd.Series(np.asarray(x, dtype=float) - res.predict(Xc), index=x.index)


def mean_local_icc(series: pd.Series, window_days: int, min_prehistory: int = 28,
                   residualize_dow: bool = False) -> float:
    """Average within-window ICC over every contiguous window of `window_days`.

    The A/A test draws *short contiguous windows*, so the ICC that governs S3 is
    the local one measured inside such a window, not the ICC of the full
    multi-year series (which is inflated by trend and seasonal drift absent at an
    8-week scale).

    Two versions matter, and they tell different halves of the story:

    * ``residualize_dow=False`` -- ICC of the RAW daily values. This comes out
      near zero, but that is misleading: within a week the huge day-of-week
      swing lives INSIDE the cluster, inflating the within-cluster variance and
      masking any real week-to-week correlation.
    * ``residualize_dow=True`` -- ICC AFTER removing the day-of-week means. This
      is the number that actually governs S3, because week-level assignment gives
      both arms a full set of weekdays, so day-of-week cancels in the arm
      contrast and only the residual correlation drives the inflation.
    """
    idx = series.index
    starts = _valid_starts(pd.DatetimeIndex(idx), window_days, min_prehistory)
    vals = []
    for s in starts:
        w = series.iloc[s: s + window_days]
        vals.append(intraclass_correlation(_dow_residual(w) if residualize_dow else w))
    return float(np.nanmean(vals))


def _binom_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) interval, which misbehaves for the small
    proportions (~0.05) we are testing here."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)
