"""Step 6: fit the chosen model on all history, forecast 30 days, make charts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, features, models

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def main():
    daily = pd.read_csv(config.PROCESSED_DIR / "daily_revenue.csv", parse_dates=["date"])
    series = daily.set_index("date")["revenue"].sort_index()
    summary = pd.read_csv(config.REPORTS_DIR / "03_model_comparison.csv")
    preds = pd.read_csv(config.REPORTS_DIR / "03_fold_predictions.csv", parse_dates=["date"])

    # --- Chart 1: the series, with the structural break called out ----------
    raw_all = pd.read_excel(config.RAW_XLSX)
    raw_all["Date"] = pd.to_datetime(raw_all["Date"], errors="coerce")
    all_daily = (raw_all[raw_all.Item_Category.isin(config.REVENUE_CATEGORIES)]
                 .groupby("Date")["Amount"].sum())
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(all_daily.index, all_daily.rolling(28).mean(), lw=1.2, color="#B04A2E")
    ax.axvline(pd.Timestamp(config.ANALYSIS_START), color="#333", ls="--", lw=1)
    ax.annotate("2021-09-01: fuel + 5 departments\nappear in the export",
                xy=(pd.Timestamp(config.ANALYSIS_START), all_daily.max() * 0.55),
                xytext=(10, 0), textcoords="offset points", fontsize=8)
    ax.set_title("Daily net revenue, 28-day moving average — the pre-2021-09 period "
                 "measures a narrower product scope", loc="left")
    ax.set_ylabel("$/day")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "fig1_structural_break.png")

    # --- Chart 2: backtest, actual vs forecast on held-out windows ----------
    keep = ["SARIMA(1,1,1)(1,1,1,7)", "LightGBM", "Seasonal naive (t-7)"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharey=True)
    for ax, (fold, grp) in zip(axes.ravel(), preds.groupby("fold")):
        ax.plot(grp[grp.model == keep[0]].date, grp[grp.model == keep[0]].y_true,
                color="black", lw=1.4, label="actual")
        for mdl, c in zip(keep, ["#B04A2E", "#2E6FB0", "#999"]):
            g = grp[grp.model == mdl]
            ax.plot(g.date, g.y_pred, lw=1, color=c, label=mdl)
        ax.set_title(f"fold {fold}", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Rolling-origin backtest: six independent 30-day forecasts", x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "fig2_backtest_folds.png")

    # --- Chart 3: honest model comparison ----------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.2))
    s = summary.sort_values("MAE")
    colors = ["#B04A2E" if m in ("SARIMA(1,1,1)(1,1,1,7)", "Ensemble (mean)")
              else "#7A9AB8" for m in s.model]
    ax.barh(s.model, s.MAE, xerr=s.MAE_std, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("MAE ($/day), mean over 6 folds, error bars = fold std")
    ax.set_title("No model beats a well-specified classical baseline", loc="left")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "fig3_model_comparison.png")

    # --- Final forecast: fit on ALL history, predict the next 30 days -------
    frame = features.build_direct_multistep_frame(daily)
    frame, _ = features.attach_fuel_prices(frame)
    cols = [c for c in features.FEATURE_COLUMNS if c in frame.columns]
    tr = frame.dropna(subset=cols + ["revenue"])
    lgbm = models.make_lightgbm().fit(tr[cols], tr["revenue"])

    last = series.index.max()
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=config.HORIZON)
    fut = pd.DataFrame({"date": future_dates})
    fut["days_ahead"] = np.arange(1, config.HORIZON + 1)
    fut["origin_date"] = last
    hist = features._origin_history_table(series)
    for c in hist.columns:
        fut[c] = hist.loc[last, c]
    for lag in (357, 364, 371):
        fut[f"lag_{lag}"] = series.reindex(future_dates - pd.Timedelta(days=lag)).values
    fut["lag_364_dow_mean"] = fut[["lag_357", "lag_364", "lag_371"]].mean(axis=1)
    fut = features.add_holiday_features(features.add_calendar_features(fut))

    fut["lgbm"] = lgbm.predict(fut[cols])
    fut["sarima"] = models.fit_predict_sarima(series, config.HORIZON)
    fut["dow_mean"] = models.baseline_dow_mean(series, config.HORIZON)
    fut["forecast"] = fut[["lgbm", "sarima", "dow_mean"]].mean(axis=1)

    # Empirical prediction interval from backtest residuals, by horizon.
    ens = preds[preds.model == "Ensemble (mean)"].copy()
    ens["err"] = ens.y_pred - ens.y_true
    band = ens.groupby("days_ahead")["err"].quantile([0.1, 0.9]).unstack()
    fut["lo80"] = fut["forecast"] - band[0.9].values
    fut["hi80"] = fut["forecast"] - band[0.1].values

    out = fut[["date", "forecast", "lo80", "hi80", "lgbm", "sarima", "dow_mean"]].round(2)
    out.to_csv(config.REPORTS_DIR / "04_forecast_next_30_days.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(series.index[-120:], series.iloc[-120:], color="black", lw=1, label="actual")
    ax.plot(out.date, out.forecast, color="#B04A2E", lw=1.6, label="forecast (ensemble)")
    ax.fill_between(out.date, out.lo80, out.hi80, color="#B04A2E", alpha=0.18,
                    label="80% interval (empirical)")
    ax.legend(fontsize=7)
    ax.set_title(f"Next {config.HORIZON} days: "
                 f"${out.forecast.sum():,.0f} total predicted revenue", loc="left")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "fig4_forecast.png")

    print(out.head(10).to_string(index=False))
    print(f"\n30-day total: ${out.forecast.sum():,.0f}  "
          f"(80% interval ${out.lo80.sum():,.0f} - ${out.hi80.sum():,.0f})")
    print("Charts written to reports/")


if __name__ == "__main__":
    main()
