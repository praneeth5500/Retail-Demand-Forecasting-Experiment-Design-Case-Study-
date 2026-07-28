"""Step 1-2: audit the raw export, then build the clean daily series."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import config, data_prep

def main():
    raw = data_prep.load_raw()
    print(f"Raw rows: {len(raw):,}  |  {raw.Date.min().date()} -> {raw.Date.max().date()}")

    audit = data_prep.audit_raw(raw)
    audit.to_csv(config.REPORTS_DIR / "01_raw_inventory.csv", index=False)
    print("\n=== RAW INVENTORY ===")
    print(audit.to_string(index=False))

    print("\n=== IS 'Sales Tax' A DERIVED ROW? ===")
    print(data_prep.verify_sales_tax_is_derived(raw).round(4).to_string())

    grid = data_prep.find_structural_break(raw)
    grid.to_csv(config.REPORTS_DIR / "01_coverage_grid.csv")
    print("\n=== DEPARTMENT COVERAGE AROUND 2021-09 ===")
    print(grid.loc["2021-06":"2021-11"].to_string())

    print("\n=== HEADLINE NUMBER RECONCILIATION ===")
    naive_total = raw.groupby(raw.Date.dt.year).Amount.sum()
    clean = data_prep.clean_revenue_rows(raw)
    clean_total = clean.groupby(clean.Date.dt.year).Amount.sum()
    recon = pd.DataFrame({
        "v1_sum_everything": naive_total,
        "v2_net_revenue": clean_total,
    })
    recon["excluded"] = recon.v1_sum_everything - recon.v2_net_revenue.fillna(0)
    print(recon.round(0).to_string())
    recon.to_csv(config.REPORTS_DIR / "01_revenue_reconciliation.csv")

    daily = data_prep.build_daily_total(raw)
    daily.to_csv(config.PROCESSED_DIR / "daily_revenue.csv", index=False)
    panel = data_prep.build_daily_by_category(raw)
    panel.to_csv(config.PROCESSED_DIR / "daily_by_department.csv", index=False)
    print(f"\nDaily series: {len(daily)} rows, {daily.date.min().date()} -> {daily.date.max().date()}")
    print(daily.revenue.describe().round(2).to_string())

if __name__ == "__main__":
    main()
