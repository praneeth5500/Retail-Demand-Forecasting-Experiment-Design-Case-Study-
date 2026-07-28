"""Step 4-5: build features, run rolling-origin backtest, write the honest table."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import config, backtest, features

def main():
    daily = pd.read_csv(config.PROCESSED_DIR / "daily_revenue.csv", parse_dates=["date"])
    frame = features.build_direct_multistep_frame(daily)
    frame, has_fuel = features.attach_fuel_prices(frame)
    cols = list(features.FEATURE_COLUMNS)
    if has_fuel:
        cols += ["fuel_price", "fuel_price_chg_4w"]
        print("EIA fuel prices merged as external regressors.")
    else:
        print("No EIA cache found -> running without fuel prices "
              "(run scripts/02_fetch_eia.py to add them).")
    print(f"Supervised frame: {len(frame):,} rows x {len(cols)} features")

    t0 = time.time()
    results, preds = backtest.run_backtest(daily, frame, cols)
    results, preds = backtest.add_ensemble(
        results, preds,
        members=("SARIMA(1,1,1)(1,1,1,7)", "LightGBM", "Day-of-week mean (8w)"),
    )
    print(f"Backtest finished in {time.time()-t0:.0f}s")

    preds.to_csv(config.REPORTS_DIR / "03_fold_predictions.csv", index=False)
    results.to_csv(config.REPORTS_DIR / "03_fold_results.csv", index=False)
    summary = backtest.summarise(results)
    summary.to_csv(config.REPORTS_DIR / "03_model_comparison.csv", index=False)

    print("\n=== MODEL COMPARISON (mean over folds, 30-day horizon) ===")
    print(summary.to_string(index=False))
    print("\nOrigins used:", sorted(results.origin.unique()))
    print("\n=== PER-FOLD MAE ===")
    print(results.pivot_table(index="model", columns="fold", values="MAE").round(0).to_string())

if __name__ == "__main__":
    main()
