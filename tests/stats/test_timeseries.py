import numpy as np
import pandas as pd
import pytest

from climate_risk.stats.timeseries import adf_test_summary, make_var_names, stl_deviation


@pytest.mark.parametrize(
    ("n_lags", "regression", "expected"),
    [
        (0, "n", ["L1.x"]),
        (1, "n", ["L1.x", "D1L1.x"]),
        (1, "c", ["L1.x", "D1L1.x", "Constant"]),
        (1, "ct", ["L1.x", "D1L1.x", "Constant", "Trend"]),
    ],
)
def test_var_names_follow_the_regression_terms(n_lags, regression, expected):
    """The names label ADF output columns positionally, so an extra or missing one misaligns them."""
    assert make_var_names("x", n_lags, regression) == expected


def test_missing_data_is_refused_by_default():
    """statsmodels would otherwise return a silent nan for the whole test."""
    with pytest.raises(ValueError, match="missing data"):
        adf_test_summary(pd.DataFrame({"x": [1.0, np.nan, 3.0]}))


def test_removing_the_trend_leaves_a_series_with_no_drift():
    """A rising series detrends to something centered on zero; keeping the trend is the failure."""
    years = pd.date_range("1990", periods=40, freq="YS")
    rising = pd.Series(np.arange(40, dtype=float) + np.tile([1.0, -1.0, 0.0, 0.5], 10), index=years)

    residual = stl_deviation(rising)

    assert abs(residual.mean()) < 0.5, "a trend survived the decomposition"
    assert residual.max() - residual.min() < rising.max() - rising.min()


def test_the_deviation_keeps_the_index_it_was_given():
    """The result is subtracted from and plotted against the original, so a reindex breaks both."""
    years = pd.date_range("1990", periods=30, freq="YS")
    series = pd.Series(np.linspace(0.0, 10.0, 30), index=years)

    assert stl_deviation(series).index.equals(series.index)


def test_the_period_changes_what_counts_as_trend():
    """Period is the one modeling choice here, and a default that ignored it would go unnoticed."""
    years = pd.date_range("1990", periods=48, freq="YS")
    seasonal = pd.Series(np.tile([2.0, -2.0, 0.0, 1.0], 12) + np.arange(48) * 0.1, index=years)

    assert not stl_deviation(seasonal, period=4).equals(stl_deviation(seasonal, period=3))
