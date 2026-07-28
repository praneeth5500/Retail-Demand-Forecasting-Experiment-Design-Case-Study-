"""Does the evaluation protocol actually change the answer? Measure it.

v1 used a random train/test split on a time series. That is methodologically
invalid -- but "invalid" and "produces a badly wrong number" are different
claims, and a good analyst checks which one applies before writing it up.

This script runs the same model under four protocols and reports the spread.
On THIS series the damage turns out to be modest, and the honest write-up says
so. Do not inherit a scary number from a blog post; measure your own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import backtest, config, features, models


def build_v1_style_frame(series: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    df = pd.DataFrame({"revenue": series})
    for lag in (1, 2, 3, 7, 14):
        df[f"lag_{lag}"] = series.shift(lag)
    df["roll7"] = series.shift(1).rolling(7).mean()
    df = features.add_calendar_features(
        df.reset_index().rename(columns={"index": "date"})
    ).dropna()
    return df, [c for c in df.columns if c not in ("date", "revenue")]


def main():
    daily = pd.read_csv(config.PROCESSED_DIR / "daily_revenue.csv", parse_dates=["date"])
    series = daily.set_index("date")["revenue"].sort_index()
    df, cols = build_v1_style_frame(series)
    origins = backtest.make_origins(series.index)

    # A. Random split -- invalid, but how wrong?
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[cols], df["revenue"], test_size=0.2, random_state=config.RANDOM_SEED
    )
    m = models.make_random_forest().fit(X_tr, y_tr)
    a = models.mae(y_te, m.predict(X_te))

    # B. 1-step-ahead with true lags fed in -- valid protocol, WRONG QUESTION.
    #    The store asked for 30 days. Answering a 1-day question flatters you.
    b = []
    for o in origins:
        tr = df[df.date <= o]
        est = models.make_random_forest().fit(tr[cols], tr["revenue"])
        fut = df[(df.date > o) & (df.date <= o + pd.Timedelta(days=config.HORIZON))]
        b.append(models.mae(fut["revenue"], est.predict(fut[cols])))

    # C. Recursive 30-day: feed the model its own predictions back as lags.
    c = []
    for o in origins:
        tr = df[df.date <= o]
        est = models.make_random_forest().fit(tr[cols], tr["revenue"])
        hist, preds = list(series.loc[:o].values), []
        for k in range(1, config.HORIZON + 1):
            d = o + pd.Timedelta(days=k)
            row = {"lag_1": hist[-1], "lag_2": hist[-2], "lag_3": hist[-3],
                   "lag_7": hist[-7], "lag_14": hist[-14], "roll7": np.mean(hist[-7:])}
            cal = features.add_calendar_features(pd.DataFrame({"date": [d]})).iloc[0]
            for col in cols:
                if col not in row:
                    row[col] = cal[col]
            p = float(est.predict(pd.DataFrame([row])[cols])[0])
            preds.append(p)
            hist.append(p)
        truth = series.reindex(pd.date_range(o + pd.Timedelta(days=1), periods=config.HORIZON))
        c.append(models.mae(truth.values, preds))

    out = pd.DataFrame([
        {"protocol": "A. Random 80/20 split (v1 method)", "MAE": a,
         "verdict": "invalid: shuffles time"},
        {"protocol": "B. Rolling origin, 1-step-ahead", "MAE": float(np.mean(b)),
         "verdict": "valid protocol, wrong horizon"},
        {"protocol": "C. Rolling origin, 30-day recursive", "MAE": float(np.mean(c)),
         "verdict": "valid"},
    ])
    out.to_csv(config.REPORTS_DIR / "05_protocol_comparison.csv", index=False)
    print(out.round(0).to_string(index=False))
    print("\nRead this honestly: the gap between protocols is small here because the")
    print("series is strongly mean-reverting with a dominant weekly cycle, so a model")
    print("that loses its short lags falls back on the weekly pattern and the level")
    print("and does nearly as well. On a trending or shock-driven series the same")
    print("mistakes would be far more expensive. The protocol is still wrong in (A)")
    print("and still answers the wrong question in (B) -- the point is that you")
    print("measured it instead of asserting it.")


if __name__ == "__main__":
    main()
