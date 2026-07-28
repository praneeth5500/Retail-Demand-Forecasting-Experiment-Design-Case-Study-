# Experiment Design — Fuel-to-Store Conversion Switchback (Pre-Registration)

**Status: design document. No treatment has been applied to this store. This is
a pre-registered analysis plan written before any test, plus the offline power
and calibration work that justifies it. Nothing here is a result.**

This plan is deliberately written to be committed *before* data is collected, so
that the metric, the analysis, and the stopping rule cannot be chosen after
seeing the outcome. The numbers that motivate the choices below come from the
power analysis (`scripts/06_power_analysis.py`) and the A/A calibration
(`scripts/07_aa_test.py`), both of which are design-time work on historical data.

---

## 1. Hypothesis

A "discounted drink with fuel purchase" promotion increases **daily
inside-merchandise revenue** by converting fuel-only customers into
inside-store buyers.

- **H₀:** the promotion has no effect on daily inside-merchandise revenue
  (true mean difference = \$0/day).
- **H₁ (two-sided):** the promotion changes daily inside-merchandise revenue.

Two-sided is deliberate. Cannibalization — a discounted drink displacing a
full-price purchase, or pulling tomorrow's sale into today — could plausibly
make the effect *negative*, and the test must be able to see that.

## 2. Primary metric

**Daily inside-merchandise revenue** = sum of the DeptSale departments the store
actually controls:

> Cigarette, Beer, Soda, Sales Taxable, Sales Non-Tax, Fountain/Coffee,
> Autoparts, Cigarette Carton, Tobacco.

Defined in `config.INSIDE_MERCH_DEPARTMENTS`.

**Why this and not per-department revenue.** The feasibility triage (§7) shows
that *every individual department is too small or too noisy to test on its own*
at any duration up to 16 weeks: their minimum detectable effects all exceed any
lift the promotion could plausibly produce. The promotion targets the drink
categories, but those categories (Fountain/Coffee ≈ \$22/day, Soda ≈ \$225/day)
are exactly the ones with the worst signal-to-noise. The only metric with enough
signal to test is the aggregate. This is a finding, not a footnote.

**Excluded, with reasons.** Fuel (REG/PRE/MID) — the store does not set fuel
price, so it is a guardrail, not a target. Lottery (Lotto/Scratchoff) — recorded
as *gross* ticket sales on a ~5–6% commission, so a lottery dollar is not a
margin-bearing merchandise dollar. Air Vac — an OtherIncome line, not a DeptSale.

## 3. Guardrail metrics

Monitored for cannibalization; **not** the basis for declaring success:

- **Total daily revenue** — catches the promotion lifting inside sales purely by
  shifting dollars out of fuel or lottery.
- **Fuel volume proxy** = REG+PRE+MID revenue (`config.FUEL_DEPARTMENTS`) — a
  drink promotion tied to fuel should not *reduce* fuel purchases; if it does,
  that shows up here.

A guardrail breach (a statistically clear move in the wrong direction) blocks a
"ship" decision even if the primary metric is positive.

## 4. Randomization unit and scheme

**Unit: the day.** The store has no customer identifiers and no loyalty program,
so customers cannot be randomized. The only randomizable unit is time. The store
serves as its own control on the days it is not treated — a **switchback**.

**Scheme: day-level randomization, blocked within calendar week.** Each week,
days are randomly split so that treatment and control are balanced across
weekdays (`experiment.assign_blocked_within_week`). Blocking on weekday matters
because day-of-week is by far the largest source of daily variance here — it
alone explains ~23% of it (§8). Complete (unblocked) day-level randomization is
the pre-registered fallback; the A/A test confirms both hold their nominal error
rate.

**Rejected alternative — week-level (cluster) assignment.** Toggling the promo
weekly is operationally simpler, but the A/A test shows that if whole weeks are
assigned and then analyzed *as if days were independent*, the false-positive
rate inflates; and if analyzed correctly at the week level, an 8-week test has
only 4 week-observations per arm and is badly underpowered. Day-level assignment
avoids both problems.

## 5. Duration

**Planned length: 8 weeks (56 days, 28 per arm).** This is the length the A/A
test validates. The power analysis (§7, §9) is explicit that this is a
*minimum-viable* length, not a comfortable one:

| Duration | MDE (best adjustment) | as % of \$1,580/day base |
|---|---|---|
| 4 weeks | \$207/day | 13.1% |
| 6 weeks | \$169/day | 10.7% |
| **8 weeks** | **\$147/day** | **9.3%** |
| 12 weeks | \$120/day | 7.6% |
| 16 weeks | \$104/day | 6.6% |

The operational recommendation is to run the **longest duration the business can
tolerate** (12–16 weeks), because even 16 weeks only reaches a ~6.6% MDE, and a
realistically-sized aggregate lift from a single-category drink promo is well
below that (see §9, "Power reality check"). The team must decide *before
launch* whether an effect it can actually detect is one it would act on.

## 6. Analysis method

**Primary estimator: CUPED-style OLS regression adjustment.** Fit

```
inside_merch_day ~ arm + dow + linear_trend + Fourier_yearly + pre_period_28d_mean
```

and report the **coefficient on `arm`** as the treatment effect, with its
two-sided p-value and 95% confidence interval
(`experiment.regression_adjusted_test`). The pre-period covariate is the
trailing 28-day mean *before* the experiment window, so it is untouched by
assignment — the standard CUPED construction.

**Chosen over** a bare difference-in-means because regression adjustment on
pre-period covariates is unbiased under randomization and strictly reduces
variance; §8 measures the realized reduction on this store's data rather than
assuming a textbook figure.

**Secondary / robustness:** Welch's unequal-variance t-test on the raw daily
values (`experiment.welch_ttest`). Welch is the correct default two-sample test
when the arms' variances are not assumed equal. Agreement between the adjusted
and unadjusted estimates is a sanity check; the adjusted estimate is primary.

- **Significance:** α = 0.05, two-sided.
- **Target power:** 80%.

## 7. Stopping rule

**Fixed horizon. No peeking.** The end date is fixed in advance at the chosen
duration; the data is analyzed **once**, after the last day. There is no interim
"stop early if significant." The A/A test quantifies why (§9): daily peeking
inflates the false-positive rate roughly five-fold. If an interim look is ever
required for operational reasons, it must use a pre-specified alpha-spending
boundary (e.g. O'Brien–Fleming), not the naive 0.05 threshold — but the base
plan is a single analysis.

## 8. Variance reduction (pre-computed, not assumed)

Realized reduction from progressively richer adjustment, fit on the trailing 52
weeks of real data (`scripts/06`):

| Adjustment | Residual sd | Variance reduction |
|---|---|---|
| (i) none | \$227 | 0.0% |
| (ii) day-of-week | \$199 | 22.7% |
| (iii) + trend + Fourier yearly | \$198 | 24.0% |
| (iv) + pre-period 28-day mean (CUPED) | \$196 | 25.3% |

Nearly all the available reduction comes from day-of-week; the trend, seasonal,
and CUPED terms add little here, because once weekday is removed this store's
daily inside-merchandise revenue has weak short-horizon autocorrelation. The
adjustment is still pre-registered — it cannot hurt under randomization — but the
honest expectation is a ~25% variance reduction, i.e. a ~13% smaller MDE than the
unadjusted design, not the 40–50% sometimes quoted for CUPED elsewhere.

## 9. What would falsify the hypothesis

- The pre-registered `arm` coefficient is **not significantly different from
  zero** at α = 0.05 after the fixed 8-week (or longer) window → we fail to
  reject H₀; the promotion has no *detectable* effect on inside-merchandise
  revenue at the store's noise level.
- The `arm` coefficient is **significantly negative** → evidence the promotion
  *reduces* inside-merchandise revenue (cannibalization dominates).
- A **guardrail breach** (fuel volume proxy or total revenue moves clearly in the
  wrong direction) blocks a ship decision regardless of the primary result.

**Power reality check (load-bearing).** The best-adjusted MDE is ~9% of base at 8
weeks and ~7% at 16 weeks. A promotion that discounts one drink category will, at
the *aggregate* inside-merchandise level, plausibly move revenue by low single
digits, because the targeted categories are a small share of the total. It is
therefore likely that the true effect is **smaller than this single-store
switchback can detect at any feasible duration.** A null result must be read as
"no effect large enough to see here," not "no effect." This is stated up front so
it cannot be spun after the fact.

## 10. Load-bearing assumptions

- **Stationarity.** The power/MDE calculation assumes the daily mean and variance
  during the test resemble the recent historical window used to estimate them
  (trailing 52 weeks). A fuel-price shock, a new competitor, or a road closure
  during the test breaks this and invalidates the power figures.
- **SUTVA / carryover (stockpiling).** A promo-day drink purchase can *suppress*
  the next day's purchase. When the next day is a control day, treatment leaks
  into control: it biases the estimated effect toward zero and makes the
  day-level variance understate the truth. Blocking within week limits, but does
  not eliminate, adjacency between arms. If carryover is believed to be strong,
  the honest unit is the week, at a large power cost. This risk is not fully
  removable in a day-level switchback and is accepted knowingly.
- **No margin data.** The export contains revenue only. Every effect here is
  top-line revenue, not profit. A discounted-drink promo gives up margin on the
  discounted item, so the profit effect is strictly worse than the revenue effect
  the test measures.
- **Single store, no external validity.** One location. Nothing here generalizes
  to other sites, and a positive result at this store is not evidence the
  promotion works anywhere else.

---

*Reproduce:* `python scripts/06_power_analysis.py` (power, MDE, variance
reduction) and `python scripts/07_aa_test.py` (pipeline calibration). Results and
their interpretation are in [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md).
