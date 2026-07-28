"""Leakage-safe feature engineering for direct multi-step forecasting.

THE CORE PROBLEM v1 GOT WRONG
-----------------------------
If you want to forecast 30 days ahead, you do NOT have yesterday's revenue for
day 30. A model trained on `lag_1` scores beautifully in a naive backtest and
then collapses in production, because at prediction time `lag_1` does not exist.

The fix used here is DIRECT MULTI-STEP forecasting:
  * Every row is a (forecast_origin, target_date) pair.
  * `days_ahead` = target_date - origin is itself a feature.
  * Every recent-history feature is computed AT THE ORIGIN and then shifted
    forward, so it only ever uses information that existed when the forecast
    was made.
  * Long lags (364/371/357 days) are safe to attach to the target date
    directly, because 364 > 30: a year-ago value is already known at origin.

One model then serves all horizons 1..30.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    d = out[date_col]
    out["dow"] = d.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["day_of_month"] = d.dt.day
    out["month"] = d.dt.month
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["day_of_year"] = d.dt.dayofyear
    # Paydays drive convenience-store traffic.
    out["is_payday_window"] = d.dt.day.isin([1, 2, 3, 15, 16, 17]).astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)

    # Fourier terms let a linear or tree model express smooth yearly seasonality
    # without one-hot exploding 365 day-of-year levels.
    for k in (1, 2):
        out[f"yearly_sin_{k}"] = np.sin(2 * np.pi * k * out["day_of_year"] / 365.25)
        out[f"yearly_cos_{k}"] = np.cos(2 * np.pi * k * out["day_of_year"] / 365.25)

    # NOTE: no raw `year` column. Tree models cannot extrapolate; a `year`
    # feature is useless at best and harmful at worst when you predict a year
    # the model never saw in training. v1 of this project made that mistake.
    return out


def add_holiday_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    years = range(out[date_col].dt.year.min() - 1, out[date_col].dt.year.max() + 2)
    us_holidays = holidays.US(years=list(years), subdiv="FL")
    hol_dates = pd.to_datetime(sorted(us_holidays.keys()))

    out["is_holiday"] = out[date_col].isin(hol_dates).astype(int)

    # Distance to nearest holiday: travel-driven demand builds and decays
    # around holidays, it does not switch on for exactly one day.
    target = out[date_col].values.astype("datetime64[D]").astype(int)
    hol = hol_dates.values.astype("datetime64[D]").astype(int)
    diffs = target[:, None] - hol[None, :]
    out["days_to_nearest_holiday"] = np.clip(
        diffs[np.arange(len(target)), np.abs(diffs).argmin(axis=1)], -14, 14
    )
    return out


# ---------------------------------------------------------------------------
# Origin-based history features
# ---------------------------------------------------------------------------
def _origin_history_table(series: pd.Series) -> pd.DataFrame:
    """Statistics known as of each date, using that date and earlier only."""
    hist = pd.DataFrame(index=series.index)
    hist["origin_last"] = series
    for w in (7, 14, 28, 91):
        hist[f"origin_mean_{w}"] = series.rolling(w, min_periods=w).mean()
    hist["origin_std_28"] = series.rolling(28, min_periods=28).std()
    hist["origin_median_28"] = series.rolling(28, min_periods=28).median()
    # Short-vs-long ratio is a cheap, robust trend signal.
    hist["origin_trend_7_91"] = hist["origin_mean_7"] / hist["origin_mean_91"]
    # Same-weekday recent level, robust to the strong weekly cycle.
    hist["origin_dow_mean_4"] = (
        series.groupby(series.index.dayofweek)
        .transform(lambda s: s.rolling(4, min_periods=4).mean())
    )
    return hist


def build_direct_multistep_frame(
    daily: pd.DataFrame,
    horizon: int = config.HORIZON,
    value_col: str = "revenue",
) -> pd.DataFrame:
    """Expand a daily series into (origin, target) rows with safe features.

    Returns one row per (target_date, days_ahead) pair.
    """
    s = daily.set_index("date")[value_col].sort_index().astype(float)
    hist = _origin_history_table(s)

    # Year-ago lags are attached to the TARGET date. Safe because 357 > horizon.
    long_lags = pd.DataFrame(index=s.index)
    for lag in (357, 364, 371):
        long_lags[f"lag_{lag}"] = s.shift(lag)
    long_lags["lag_364_dow_mean"] = long_lags[["lag_357", "lag_364", "lag_371"]].mean(axis=1)

    blocks = []
    for k in range(1, horizon + 1):
        block = pd.DataFrame(index=s.index)
        block["date"] = s.index
        block["days_ahead"] = k
        block["origin_date"] = s.index - pd.Timedelta(days=k)
        block[value_col] = s.values
        # Shift origin stats forward by k: row for target T carries stats as of T-k.
        for col in hist.columns:
            block[col] = hist[col].shift(k).values
        for col in long_lags.columns:
            block[col] = long_lags[col].values
        blocks.append(block)

    frame = pd.concat(blocks, ignore_index=True)
    frame = add_calendar_features(frame)
    frame = add_holiday_features(frame)
    return frame.sort_values(["date", "days_ahead"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# External regressor: retail fuel prices
# ---------------------------------------------------------------------------
def attach_fuel_prices(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Merge weekly EIA retail gasoline prices if the cache exists.

    Weekly EIA series are published with a lag, so we merge the price known at
    the FORECAST ORIGIN (merge_asof on origin_date), never the price on the
    target date -- that would be information from the future.
    """
    path = config.EXTERNAL_DIR / "eia_fuel_prices.csv"
    if not path.exists():
        return frame, False

    fuel = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    fuel = fuel.rename(columns={"price": "fuel_price"})
    out = frame.sort_values("origin_date")
    out = pd.merge_asof(
        out, fuel[["date", "fuel_price"]].rename(columns={"date": "origin_date"}),
        on="origin_date", direction="backward",
    )
    fuel4 = fuel.set_index("date")["fuel_price"]
    chg = (fuel4 / fuel4.shift(4) - 1).rename("fuel_price_chg_4w").reset_index()
    out = pd.merge_asof(
        out, chg.rename(columns={"date": "origin_date"}),
        on="origin_date", direction="backward",
    )
    return out.sort_values(["date", "days_ahead"]).reset_index(drop=True), True


FEATURE_COLUMNS = [
    "days_ahead",
    "origin_last", "origin_mean_7", "origin_mean_14", "origin_mean_28",
    "origin_mean_91", "origin_std_28", "origin_median_28",
    "origin_trend_7_91", "origin_dow_mean_4",
    "lag_357", "lag_364", "lag_371", "lag_364_dow_mean",
    "dow", "is_weekend", "day_of_month", "month", "week_of_year",
    "is_payday_window", "is_month_end",
    "yearly_sin_1", "yearly_cos_1", "yearly_sin_2", "yearly_cos_2",
    "is_holiday", "days_to_nearest_holiday",
]
