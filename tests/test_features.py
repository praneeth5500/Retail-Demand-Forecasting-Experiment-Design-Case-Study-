"""Tests that would have caught the v1 bugs.

Two of these encode the exact mistakes v1 made. That is what a test suite is
for on a portfolio project: proof you understand your own failure modes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import config, data_prep, features, models


@pytest.fixture
def toy_series():
    idx = pd.date_range("2022-01-01", periods=800, freq="D")
    rng = np.random.default_rng(0)
    vals = 1000 + 200 * np.sin(np.arange(800) * 2 * np.pi / 7) + rng.normal(0, 50, 800)
    return pd.DataFrame({"date": idx, "revenue": vals})


def test_no_future_information_in_features(toy_series):
    """Every feature on a row must be computable at that row's origin_date."""
    frame = features.build_direct_multistep_frame(toy_series, horizon=30)
    assert (frame["origin_date"] < frame["date"]).all()
    assert (frame["date"] - frame["origin_date"]).dt.days.between(1, 30).all()


def test_origin_features_match_manual_computation(toy_series):
    """origin_last on a row must equal the actual revenue at origin_date."""
    frame = features.build_direct_multistep_frame(toy_series, horizon=30)
    truth = toy_series.set_index("date")["revenue"]
    sample = frame.dropna(subset=["origin_last"]).sample(50, random_state=0)
    expected = truth.reindex(sample["origin_date"]).values
    np.testing.assert_allclose(sample["origin_last"].values, expected, rtol=1e-9)


def test_no_year_feature():
    """A raw `year` column is the v1 extrapolation bug. Keep it out."""
    assert "year" not in features.FEATURE_COLUMNS


def test_seasonal_naive_is_actually_seasonal(toy_series):
    s = toy_series.set_index("date")["revenue"]
    pred = models.baseline_seasonal_naive(s.iloc[:-30], 30, season=7)
    assert len(pred) == 30
    np.testing.assert_allclose(pred[:7], s.iloc[-37:-30].values)


def test_mase_of_seasonal_naive_is_near_one(toy_series):
    """Sanity check on the metric itself: the baseline should score ~1."""
    s = toy_series.set_index("date")["revenue"]
    train, test = s.iloc[:-30], s.iloc[-30:]
    pred = models.baseline_seasonal_naive(train, 30)
    assert 0.5 < models.mase(test.values, pred, train) < 2.0


def test_mape_handles_zeros():
    assert not np.isnan(models.mape([0.0, 100.0], [1.0, 90.0]))
    assert np.isnan(models.mape([0.0, 0.0], [1.0, 1.0]))


def test_revenue_definition_excludes_sales_tax():
    assert "SalesTax" in config.EXCLUDED_CATEGORIES
    assert "SalesTax" not in config.REVENUE_CATEGORIES


@pytest.mark.skipif(not config.RAW_XLSX.exists(), reason="raw file not present")
def test_daily_series_has_no_calendar_gaps():
    raw = data_prep.load_raw()
    daily = data_prep.build_daily_total(raw)
    expected = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    assert len(daily) == len(expected)
