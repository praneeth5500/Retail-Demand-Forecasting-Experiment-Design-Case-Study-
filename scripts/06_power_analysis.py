"""Task A: power / minimum-detectable-effect analysis for the switchback design.

OFFLINE design work on historical observational data. No treatment was applied;
nothing here is an experimental result. This script answers one question: IF the
store ran a day-level switchback, how small an effect on inside-merchandise
revenue could it detect, and for which departments is that even worth trying?

Outputs (all regenerable):
  reports/06_feasibility_triage.csv     A1 per-department stats + MDE
  reports/06_variance_reduction.csv     A2 realised variance reduction
  reports/06_economic_significance.csv  A4 MDE in annual dollars
  reports/fig5_power_curves.png         A3 power vs effect, 80% marked
  reports/fig6_mde_by_department.png    A1 MDE% by department
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

TREAT = "#B04A2E"
CTRL = "#7A9AB8"


def main():
    panel = pd.read_csv(config.PROCESSED_DIR / "daily_by_department.csv",
                        parse_dates=["date"])
    metrics = ex.department_panel_to_metrics(panel)
    window = ex.recent_window(metrics)
    w0, w1 = window.index.min().date(), window.index.max().date()
    total = window["total_revenue"]

    print(f"Feasibility window: {w0} .. {w1}  ({len(window)} days, "
          f"{config.FEASIBILITY_WINDOW_WEEKS} whole weeks)")
    print("Everything below is offline design on historical data; no promo was run.\n")

    # -----------------------------------------------------------------------
    # A1. Feasibility triage: per department + the composite metric
    # -----------------------------------------------------------------------
    targets = config.INSIDE_MERCH_DEPARTMENTS + ["inside_merch"]
    durations = config.POWER_DURATIONS_WEEKS

    rows = []
    for name in targets:
        s = window[name]
        d = ex.describe_metric(s, total)
        row = {"metric": name, **d}
        for wk in durations:
            mde = ex.mde_switchback(d["sd"], n_days=wk * 7)
            row[f"mde_abs_{wk}w"] = mde
            row[f"mde_pct_{wk}w"] = 100.0 * mde / d["mean"] if d["mean"] else np.nan
        # flag: even at the longest duration, is the MDE larger than any lift the
        # promo could plausibly produce?
        row["feasible"] = row[f"mde_pct_{max(durations)}w"] <= config.PLAUSIBLE_MAX_LIFT_PCT
        rows.append(row)

    triage = pd.DataFrame(rows)
    triage.to_csv(config.REPORTS_DIR / "06_feasibility_triage.csv", index=False)

    show = triage.copy()
    show["share_%"] = (show["share"] * 100).round(1)
    disp_cols = ["metric", "mean", "sd", "cv", "share_%"] + \
                [f"mde_pct_{w}w" for w in durations] + ["feasible"]
    fmt = show[disp_cols].copy()
    for c in ["mean", "sd"]:
        fmt[c] = fmt[c].round(0)
    fmt["cv"] = fmt["cv"].round(3)
    for w in durations:
        fmt[f"mde_pct_{w}w"] = fmt[f"mde_pct_{w}w"].round(1)
    print("=== A1  FEASIBILITY TRIAGE  (MDE as % of the metric's own base level) ===")
    print("    balanced day-level switchback, 80% power, alpha=0.05 two-sided, raw sd")
    print(fmt.to_string(index=False))

    infeasible = triage.loc[~triage["feasible"] & (triage["metric"] != "inside_merch"),
                            "metric"].tolist()
    print("\nFINDING (A1): at the plausible-lift ceiling of "
          f"{config.PLAUSIBLE_MAX_LIFT_PCT:.0f}% even a {max(durations)}-week switchback "
          "cannot detect an effect in:")
    print("  " + (", ".join(infeasible) if infeasible else "(none)"))
    print("  These departments are too small and/or too noisy (high CV) to test on")
    print("  their own. The primary metric must be the aggregate `inside_merch`.")

    # A business insight that fell out of the triage: "small" is not "static".
    # Fountain/Coffee is tiny but the fastest-growing category in the store, so
    # "too small to test" today is an argument for measuring it properly later,
    # not for dismissing it.
    growth = []
    for name in config.INSIDE_MERCH_DEPARTMENTS:
        life = float(metrics[name].mean())
        rec = float(window[name].mean())
        growth.append((name, life, rec, rec / life if life > 0 else np.nan))
    gdf = pd.DataFrame(growth, columns=["dept", "lifetime", "recent52w", "growth_x"]) \
        .sort_values("growth_x", ascending=False)
    top = gdf.iloc[0]
    print(f"\nNOTE (A1): fastest-growing inside category is {top['dept']}: "
          f"${top['lifetime']:.2f}/day lifetime -> ${top['recent52w']:.2f}/day in the")
    print(f"  last 52 weeks ({top['growth_x']:.1f}x). It is small today but growing fast, so")
    print("  the recent window (not the lifetime average) is the right basis for triage.")

    # -----------------------------------------------------------------------
    # A2. Variance reduction on the primary metric, fitted on real data
    # -----------------------------------------------------------------------
    vr = ex.variance_reduction_table(window["inside_merch"])
    vr.to_csv(config.REPORTS_DIR / "06_variance_reduction.csv", index=False)
    print("\n=== A2  VARIANCE REDUCTION on inside_merch (measured, not assumed) ===")
    print(vr[["label", "residual_sd", "variance_reduction_pct", "n"]]
          .round({"residual_sd": 1, "variance_reduction_pct": 1}).to_string(index=False))

    best = vr.sort_values("residual_sd").iloc[0]
    best_sd = float(best["residual_sd"])
    raw_sd = float(vr.loc[vr["scheme"] == "none", "residual_sd"].iloc[0])
    base = float(window["inside_merch"].mean())
    print(f"\nBest adjustment: {best['label']}  ->  residual sd "
          f"${best_sd:,.0f} (raw ${raw_sd:,.0f}).")

    # -----------------------------------------------------------------------
    # A3. Power curves under the best adjustment scheme
    # -----------------------------------------------------------------------
    effects = np.linspace(0, 0.12 * base, 300)      # 0 .. 12% of base level
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(durations)))
    for wk, color in zip(durations, cmap):
        n_per_arm = (wk * 7) // 2
        power = [ex.power_two_sample(e, best_sd, n_per_arm) for e in effects]
        ax.plot(effects, power, color=color, lw=1.8, label=f"{wk} weeks")
        mde = ex.mde_switchback(best_sd, wk * 7)
        ax.plot([mde], [config.POWER], "o", color=color, ms=6)
        ax.annotate(f"${mde:,.0f}\n({100*mde/base:.1f}%)",
                    xy=(mde, config.POWER), xytext=(6, -18),
                    textcoords="offset points", fontsize=7, color=color)
    ax.axhline(config.POWER, color="#333", ls="--", lw=0.9)
    ax.text(effects[-1], config.POWER + 0.01, "80% power", ha="right", fontsize=7)
    ax.set_xlabel("True daily effect on inside-merchandise revenue ($/day)")
    ax.set_ylabel("Statistical power")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Power vs effect, best adjustment ({best['label']}), "
                 f"base ${base:,.0f}/day", loc="left", fontsize=9)
    ax.legend(title="switchback length", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "fig5_power_curves.png")
    print("\nWrote reports/fig5_power_curves.png")

    # -----------------------------------------------------------------------
    # A1 figure: MDE% by department (8-week reference duration)
    # -----------------------------------------------------------------------
    ref_w = config.PLANNED_DURATION_WEEKS
    dep = triage[triage["metric"] != "inside_merch"].copy()
    dep = dep.sort_values(f"mde_pct_{ref_w}w")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [TREAT if not f else CTRL for f in dep["feasible"]]
    # Log x-axis: MDE% spans ~14% to >1000% (Cigarette Carton is ~$0/day), so a
    # linear axis would hide both the 10% ceiling line and the spread among the
    # near-feasible departments. Log keeps every department without dropping the
    # outlier.
    ax.barh(dep["metric"], dep[f"mde_pct_{ref_w}w"], color=colors, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(8, dep[f"mde_pct_{ref_w}w"].max() * 1.6)
    ax.axvline(config.PLAUSIBLE_MAX_LIFT_PCT, color="#333", ls="--", lw=1)
    ax.text(config.PLAUSIBLE_MAX_LIFT_PCT * 1.05, -0.4,
            f"plausible lift ceiling {config.PLAUSIBLE_MAX_LIFT_PCT:.0f}%",
            fontsize=7, va="bottom")
    for y, v in enumerate(dep[f"mde_pct_{ref_w}w"]):
        ax.text(v * 1.05, y, f"{v:.0f}%", va="center", fontsize=7)
    ax.set_xlabel(f"MDE as % of base level (log scale), "
                  f"{ref_w}-week switchback (80% power)")
    ax.set_title("Every department is individually untestable; "
                 "all bars sit past the 10% ceiling", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "fig6_mde_by_department.png")
    print("Wrote reports/fig6_mde_by_department.png")

    # -----------------------------------------------------------------------
    # A4. Economic significance -- MDE in annual dollars (revenue, not profit)
    # -----------------------------------------------------------------------
    econ = []
    for wk in durations:
        mde = ex.mde_switchback(best_sd, wk * 7)
        econ.append({
            "duration_weeks": wk,
            "days_per_arm": (wk * 7) // 2,
            "mde_abs_per_day": mde,
            "mde_pct_of_base": 100.0 * mde / base,
            # The MDE per day scaled to a year -- the smallest annual REVENUE
            # swing the test could detect. NOT a benefit, gain, or forecast.
            "mde_annualized_equiv": mde * 365.0,
        })
    econ = pd.DataFrame(econ)
    econ.to_csv(config.REPORTS_DIR / "06_economic_significance.csv", index=False)
    print("\n=== A4  ECONOMIC SIGNIFICANCE (best adjustment; REVENUE, not profit) ===")
    print(econ.round({"mde_abs_per_day": 0, "mde_pct_of_base": 1,
                      "mde_annualized_equiv": 0}).to_string(index=False))
    print(f"\nBase inside-merch revenue is ${base:,.0f}/day (~${base*365:,.0f}/yr).")
    print("READ THE LAST COLUMN CAREFULLY: 'mde_annualized_equiv' is the MDE per day")
    print("scaled to a year -- the smallest annual change the test could DETECT. It is")
    print("NOT a projected benefit, NOT a gain, and NOT a forecast of the promo's value.")
    print("And it is top-line REVENUE, not profit: the export has no margin data, and a")
    print("discounted-drink promo gives up margin, so the profit MDE is strictly worse.")


if __name__ == "__main__":
    main()
