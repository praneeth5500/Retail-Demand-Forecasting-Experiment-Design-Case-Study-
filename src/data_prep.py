"""Load the raw POS export and turn it into an auditable daily revenue series."""
from __future__ import annotations

import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Read the raw export exactly as-is, with minimal coercion."""
    df = pd.read_excel(config.RAW_XLSX)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    return df


def audit_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level inventory: what is in this file, and over what period.

    Run this BEFORE deciding what revenue means. It is what surfaced both the
    sales-tax double count and the 2021-09-01 structural break.
    """
    out = (
        df.groupby(["Item_Category", "Item_Description"], dropna=False)
        .agg(
            rows=("Amount", "size"),
            total=("Amount", "sum"),
            first_seen=("Date", "min"),
            last_seen=("Date", "max"),
        )
        .reset_index()
        .sort_values(["Item_Category", "total"], ascending=[True, False])
    )
    return out


def verify_sales_tax_is_derived(df: pd.DataFrame) -> pd.Series:
    """Check whether the SalesTax rows are derived from the department rows.

    Do not take my word for it, and do not take anyone else's. If the ratio is
    a tight band around a statutory rate, the row is derived and must not be
    counted as revenue.
    """
    taxable = [
        "Cigarette", "Beer", "Soda", "Sales Taxable",
        "Autoparts", "Fountain/Coffee", "Cigarette Carton", "Tobacco",
    ]
    wide = df.pivot_table(
        index="Date", columns="Item_Description", values="Amount", aggfunc="sum"
    ).fillna(0.0)
    wide = wide.loc[wide.index >= config.ANALYSIS_START]
    present = [c for c in taxable if c in wide.columns]
    ratio = wide["Sales Tax"] / wide[present].sum(axis=1)
    return ratio.describe()


def find_structural_break(df: pd.DataFrame) -> pd.DataFrame:
    """Month x department coverage grid. Rows appearing all at once = break."""
    tmp = df.dropna(subset=["Date"]).copy()
    tmp["month"] = tmp["Date"].dt.to_period("M")
    grid = tmp.pivot_table(
        index="month", columns="Item_Description", values="Amount", aggfunc="size"
    )
    return grid.notna().astype(int)


def clean_revenue_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the revenue definition from config and drop junk rows."""
    out = df.dropna(subset=["Date", "Amount"]).copy()
    out = out[out["Item_Category"].isin(config.REVENUE_CATEGORIES)]
    out = out.drop_duplicates(
        subset=["Date", "Item_Category", "Item_Description", "Amount"]
    )
    out = out[out["Date"] >= config.ANALYSIS_START]
    return out


def build_daily_total(df: pd.DataFrame) -> pd.DataFrame:
    """Daily total net revenue, reindexed to a gap-free calendar."""
    rev = clean_revenue_rows(df)
    s = rev.groupby("Date")["Amount"].sum().sort_index()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    # A truly missing day is different from a zero-sales day. There are no
    # missing days in this file; assert it rather than silently filling.
    assert s.isna().sum() == 0, f"{s.isna().sum()} calendar gaps found"
    return s.rename("revenue").rename_axis("date").reset_index()


def build_daily_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Long panel of date x department, gap-filled with zeros.

    Here a missing (date, department) pair genuinely means "sold nothing that
    day", so zero-filling is correct -- unlike a missing calendar date.
    """
    rev = clean_revenue_rows(df)
    panel = (
        rev.groupby(["Date", "Item_Description"])["Amount"].sum().unstack(fill_value=0.0)
    )
    full_idx = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    panel = panel.reindex(full_idx, fill_value=0.0)
    long = (
        panel.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="department", value_name="revenue")
        .sort_values(["department", "date"])
        .reset_index(drop=True)
    )
    return long
