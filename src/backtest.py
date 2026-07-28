"""Rolling-origin backtesting.

WHY NOT train_test_split
------------------------
`train_test_split` shuffles rows. On a time series that lets the model learn
from next month to predict last month. The score is inflated and meaningless.
v1 of this project split a time series with a random split.

The correct protocol: pick several forecast origins near the end of history.
For each one, train only on data up to that origin, forecast the next 30 days,
score against what actually happened, then roll the origin forward and repeat.
Six folds gives six independent 30-day forecasts -- a spread, not a single
lucky number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, models
from .features import FEATURE_COLUMNS


def make_origins(
    dates: pd.DatetimeIndex,
    n_folds: int = config.N_FOLDS,
    horizon: int = config.HORIZON,
    step: int = config.FOLD_STEP,
    min_train_days: int = config.MIN_TRAIN_DAYS,
) -> list[pd.Timestamp]:
    """Forecast origins, most recent last."""
    last_origin = dates.max() - pd.Timedelta(days=horizon)
    origins = [last_origin - pd.Timedelta(days=step * i) for i in range(n_folds)][::-1]
    earliest_ok = dates.min() + pd.Timedelta(days=min_train_days)
    origins = [o for o in origins if o >= earliest_ok]
    if not origins:
        raise ValueError("Not enough history for the requested folds.")
    return origins


def run_backtest(
    daily: pd.DataFrame,
    ml_frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
    horizon: int = config.HORIZON,
) -> pd.DataFrame:
    """Score every baseline, ML model and SARIMA on identical folds."""
    feature_cols = feature_cols or FEATURE_COLUMNS
    feature_cols = [c for c in feature_cols if c in ml_frame.columns]

    series = daily.set_index("date")["revenue"].sort_index()
    origins = make_origins(series.index, horizon=horizon)
    rows = []
    preds = []

    for fold, origin in enumerate(origins, start=1):
        train = series.loc[:origin]
        target_idx = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
        y_true = series.reindex(target_idx).values

        def record(name, y_pred):
            y_pred = np.asarray(y_pred, dtype=float)
            m = models.score_all(y_true, y_pred, train)
            m.update(fold=fold, origin=origin.date(), model=name)
            rows.append(m)
            preds.append(pd.DataFrame({
                "fold": fold, "origin": origin, "date": target_idx,
                "days_ahead": np.arange(1, horizon + 1),
                "model": name, "y_true": y_true, "y_pred": y_pred,
            }))

        # --- baselines -------------------------------------------------
        for name, fn in models.BASELINES.items():
            record(name, fn(train, horizon))

        # --- ML models: strictly past-only training rows ----------------
        tr = ml_frame[ml_frame["date"] <= origin].dropna(subset=feature_cols + ["revenue"])
        te = ml_frame[
            (ml_frame["origin_date"] == origin) & (ml_frame["days_ahead"].between(1, horizon))
        ].sort_values("days_ahead")

        if len(te) == horizon and len(tr) > 500:
            X_tr, y_tr = tr[feature_cols], tr["revenue"]
            X_te = te[feature_cols]
            for name, mk in (
                ("Random Forest", models.make_random_forest),
                ("LightGBM", models.make_lightgbm),
            ):
                est = mk()
                est.fit(X_tr, y_tr)
                record(name, est.predict(X_te))

        # --- classical --------------------------------------------------
        try:
            record("SARIMA(1,1,1)(1,1,1,7)", models.fit_predict_sarima(train, horizon))
        except Exception as exc:  # noqa: BLE001
            print(f"  [fold {fold}] SARIMA failed: {exc}")

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def add_ensemble(
    results: pd.DataFrame,
    preds: pd.DataFrame,
    members: tuple[str, ...],
    name: str = "Ensemble (mean)",
    season: int = config.SEASONAL_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the simple mean of several models' forecasts.

    Averaging models that make different kinds of mistakes is the cheapest
    variance reduction available, and it is what most production forecasting
    stacks actually ship.
    """
    sub = preds[preds["model"].isin(members)]
    ens = (
        sub.groupby(["fold", "origin", "date", "days_ahead"])
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        .reset_index()
    )
    ens["model"] = name
    new_rows = []
    for fold, grp in ens.groupby("fold"):
        origin = grp["origin"].iloc[0]
        train = preds.loc[preds["fold"] == fold, "y_true"]  # placeholder, replaced below
        m = {
            "MAE": models.mae(grp.y_true, grp.y_pred),
            "RMSE": models.rmse(grp.y_true, grp.y_pred),
            "MAPE": models.mape(grp.y_true, grp.y_pred),
            "sMAPE": models.smape(grp.y_true, grp.y_pred),
        }
        # Reuse the seasonal-naive scale already implied by the stored results.
        ref = results[(results.fold == fold) & (results.model == "Seasonal naive (t-7)")]
        m["MASE"] = m["MAE"] / (ref["MAE"].iloc[0] / ref["MASE"].iloc[0])
        m.update(fold=fold, origin=pd.Timestamp(origin).date(), model=name)
        new_rows.append(m)
    return (
        pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True),
        pd.concat([preds, ens], ignore_index=True),
    )


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean across folds, plus the fold-to-fold spread of MAE."""
    agg = (
        results.groupby("model")
        .agg(
            MAE=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE=("RMSE", "mean"),
            MAPE=("MAPE", "mean"),
            MASE=("MASE", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values("MAE")
        .round(3)
    )
    return agg.reset_index()
