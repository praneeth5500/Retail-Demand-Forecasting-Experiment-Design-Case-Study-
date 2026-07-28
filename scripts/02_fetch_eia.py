"""Fetch weekly Florida retail gasoline prices from the EIA API (free key).

Removes the "external factors such as fuel price fluctuations are not
included" limitation you listed in v1.

Setup:
    1. Free key: https://www.eia.gov/opendata/register.php
    2. export EIA_API_KEY=...
    3. python scripts/02_fetch_eia.py

Series: PET.EMM_EPM0_PTE_SFL_DPG.W  (FL all-grades all-formulations retail,
weekly, dollars per gallon). Swap the series id for a different geography.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from src import config

SERIES = "PET.EMM_EPM0_PTE_SFL_DPG.W"
URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"


def main():
    key = os.environ.get("EIA_API_KEY")
    if not key:
        sys.exit("Set EIA_API_KEY first (https://www.eia.gov/opendata/register.php)")

    params = {
        "api_key": key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": SERIES,
        "start": "2021-01-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    if not rows:
        sys.exit("EIA returned no rows -- check the series id in the API browser.")

    df = pd.DataFrame(rows)[["period", "value"]]
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna().sort_values("date")

    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EXTERNAL_DIR / "eia_fuel_prices.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} weekly prices -> {out}")
    print(df.tail().to_string(index=False))
    print("\nNow re-run scripts/03_backtest.py. It will merge these automatically")
    print("and you can compare the model comparison table with and without them.")


if __name__ == "__main__":
    main()
