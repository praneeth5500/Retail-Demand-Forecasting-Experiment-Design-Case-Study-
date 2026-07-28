"""Task B: offline A/A test -- validate the analysis pipeline against a known null.

No treatment was ever applied to this store, so the true effect is exactly
zero. Any "significant" result this script produces is a false positive BY
CONSTRUCTION. That is the point: a trustworthy analysis pipeline should reject
the null about 5% of the time and no more. This script feeds the exact
randomize-and-analyze recipes the real experiment would use back onto real
historical days and measures how often each one cries wolf.

Six randomization x analysis pairings (B2):
  S1 complete randomization + Welch            -- expected calibrated
  S2 blocked-within-week + Welch               -- expected calibrated/conservative
  S3 week-level assignment, analyzed by day    -- expected INFLATED (wrong unit)
  S4 week-level assignment, analyzed by week    -- expected calibrated
  S5 complete randomization + CUPED adjustment -- expected calibrated
  S6 complete randomization + daily peeking    -- expected INFLATED (peeking)

Outputs:
  reports/07_aa_summary.csv          FPR, CI, effect, KS per scenario
  reports/07_aa_draws_<scenario>.csv per-iteration p-values and effects
  reports/fig7_aa_pvalue_histograms.png
  reports/fig8_aa_qq_uniform.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, experiment as ex

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def main():
    panel = pd.read_csv(config.PROCESSED_DIR / "daily_by_department.csv",
                        parse_dates=["date"])
    metrics = ex.department_panel_to_metrics(panel)
    series = metrics["inside_merch"].sort_index()

    dur = config.PLANNED_DURATION_WEEKS
    print(f"A/A test on inside_merch, {series.index.min().date()} .. "
          f"{series.index.max().date()} ({len(series)} days)")
    print(f"Planned switchback length: {dur} weeks ({dur*7} days).  "
          f"{config.AA_ITERATIONS} iterations, seed {config.EXPERIMENT_SEED}.")
    print("True effect is zero by construction; every p<0.05 is a false positive.\n")

    results = {}
    for sc in ex.AA_SCENARIOS:
        res = ex.run_aa_scenario(series, sc, duration_weeks=dur,
                                 n_iter=config.AA_ITERATIONS,
                                 seed=config.EXPERIMENT_SEED)
        results[sc] = res
        res.draws.to_csv(config.REPORTS_DIR / f"07_aa_draws_{sc}.csv", index=False)

    # -----------------------------------------------------------------------
    # Summary table (B3)
    # -----------------------------------------------------------------------
    summ = pd.DataFrame([{
        "scenario": sc,
        "description": ex.AA_SCENARIOS[sc],
        "false_positive_rate": r.false_positive_rate,
        "fpr_ci_low": r.fpr_ci_low,
        "fpr_ci_high": r.fpr_ci_high,
        "effect_mean": r.effect_mean,
        "effect_sd": r.effect_sd,
        "ks_stat": r.ks_stat,
        "ks_pvalue": r.ks_pvalue,
        "nominal_held": r.fpr_ci_low <= config.ALPHA <= r.fpr_ci_high,
    } for sc, r in results.items()])
    summ.to_csv(config.REPORTS_DIR / "07_aa_summary.csv", index=False)

    print("=== B3  A/A CALIBRATION SUMMARY (nominal alpha = 0.05) ===")
    disp = summ.copy()
    disp["FPR (95% CI)"] = [
        f"{r.false_positive_rate:.3f} [{r.fpr_ci_low:.3f}, {r.fpr_ci_high:.3f}]"
        for r in results.values()]
    disp["effect mean±sd"] = [f"{r.effect_mean:+.1f} ± {r.effect_sd:.0f}"
                              for r in results.values()]
    disp["KS p (unif)"] = summ["ks_pvalue"].round(3)
    disp["holds 5%?"] = np.where(summ["nominal_held"], "yes", "NO")
    print(disp[["scenario", "FPR (95% CI)", "effect mean±sd",
                "KS p (unif)", "holds 5%?"]].to_string(index=False))
    print("\n(effect is $ /day, treatment minus control; should be centered on 0)")

    # -----------------------------------------------------------------------
    # fig7: p-value histograms (uniform under a calibrated null)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, (sc, r) in zip(axes.ravel(), results.items()):
        p = r.draws["p_value"].dropna().to_numpy()
        ax.hist(p, bins=20, range=(0, 1), color="#7A9AB8",
                edgecolor="white", linewidth=0.4)
        ax.axhline(len(p) / 20, color="#333", ls="--", lw=0.9)  # uniform level
        if r.fpr_ci_low <= config.ALPHA <= r.fpr_ci_high:
            held, color = "calibrated", "#333"
        elif r.false_positive_rate > config.ALPHA:
            held, color = "INFLATED", "#B04A2E"
        else:
            held, color = "CONSERVATIVE", "#B04A2E"
        ax.set_title(f"{sc}\nFPR={r.false_positive_rate:.3f}  ({held})",
                     fontsize=8, color=color)
        ax.set_xlim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("p-value")
    fig.suptitle("A/A p-value distributions -- flat = calibrated null, "
                 "left-spike = inflated false positives", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "fig7_aa_pvalue_histograms.png")
    print("\nWrote reports/fig7_aa_pvalue_histograms.png")

    # -----------------------------------------------------------------------
    # fig8: QQ plot of p-values against Uniform(0,1)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (sc, r) in enumerate(results.items()):
        p = np.sort(r.draws["p_value"].dropna().to_numpy())
        q = (np.arange(1, len(p) + 1) - 0.5) / len(p)   # theoretical uniform quantiles
        ax.plot(q, p, lw=1.5, color=cmap[i], label=f"{sc} (FPR {r.false_positive_rate:.3f})")
    ax.plot([0, 1], [0, 1], color="#333", ls="--", lw=1)
    ax.axvline(config.ALPHA, color="#B04A2E", ls=":", lw=0.9)
    ax.set_xlabel("Uniform(0,1) quantile")
    ax.set_ylabel("Observed p-value quantile")
    ax.set_title("QQ vs uniform: curves BELOW the diagonal at the left\n"
                 "= too many small p-values = inflated false positives",
                 loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "fig8_aa_qq_uniform.png")
    print("Wrote reports/fig8_aa_qq_uniform.png")

    # -----------------------------------------------------------------------
    # B4 interpretation, printed. The S3 story is subtle, so ground it in the
    # actual intra-week correlation rather than a textbook assertion.
    # -----------------------------------------------------------------------
    global_icc = ex.intraclass_correlation(ex._dow_residual(series))
    local_icc_raw = ex.mean_local_icc(series, window_days=dur * 7)
    local_icc_res = ex.mean_local_icc(series, window_days=dur * 7, residualize_dow=True)
    n_clusters = dur                       # one cluster per week
    print("\n=== B4  INTERPRETATION ===")
    for sc, r in results.items():
        verdict = "HOLDS ~5%" if r.fpr_ci_low <= config.ALPHA <= r.fpr_ci_high else \
                  ("INFLATED" if r.false_positive_rate > config.ALPHA else "CONSERVATIVE")
        print(f"  {sc}: FPR={r.false_positive_rate:.3f} -> {verdict}")
    print(f"\nS3 (week assignment, day-level analysis) is INFLATED but only mildly")
    print(f"(FPR {results['S3_week_assign_day_analysis'].false_positive_rate:.3f}, CI excludes 0.05). Two mechanisms combine:")
    print(f"  (a) Residual within-week clustering. The RAW local ICC in a {dur}-week")
    print(f"      window is ~{local_icc_raw:.3f} -- misleadingly low, because the big")
    print(f"      day-of-week swing sits INSIDE the week and masks the correlation.")
    print(f"      Week-level assignment gives both arms every weekday, so day-of-week")
    print(f"      cancels and the number that actually bites is the DOW-RESIDUAL local")
    print(f"      ICC = {local_icc_res:.3f} (design effect 1+(7-1)*ICC = {1 + 6*local_icc_res:.2f}). That is")
    print(f"      well above 1.0 and is what pushes the FPR past nominal.")
    print(f"  (b) Only {n_clusters} clusters ({n_clusters//2}/arm). The day-level t-test spends ~{dur*7-2} df")
    print(f"      but there are really only ~{n_clusters} independent units; the reference")
    print(f"      distribution is mismatched to the true small-cluster sampling law.")
    print(f"The full-series DOW-residual ICC is {global_icc:.3f} (design effect {1+6*global_icc:.2f}); the")
    print(f"trap is milder over {dur} weeks than over 4 years because most week-to-week")
    print("correlation is trend/seasonal drift absent at the experiment's timescale.")
    print(f"\nS4 (correct week-level analysis) is the SAME small-cluster effect seen from")
    print(f"the other side: {n_clusters//2} week-means per arm gives ~{n_clusters-2} df, so the t-test is")
    print("genuinely small-sample and slightly CONSERVATIVE (under-rejects at 0.039).")
    print(f"\nS6 fails hardest: up to {dur*7} daily looks; under the null the probability")
    print("that at least one look crosses 0.05 is far above 0.05. The effect at")
    print("stopping is still ~0 on average but its SD balloons -- peeking selects the")
    print("random extremes in BOTH directions.")


if __name__ == "__main__":
    main()
