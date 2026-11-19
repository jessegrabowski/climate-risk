import numpy as np
import pandas as pd
import pytest

from climate_risk.data.bis import transform_effective_exchange_rate


def test_the_rate_is_indexed_by_month_with_blank_months_dropped():
    """The panel averages complete quarters, so a blank month has to be absent rather than read as a value."""
    raw = pd.DataFrame(
        {("M", "R", "B", "IN"): [97.7, np.nan, 98.1]}, index=pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"])
    )

    rate = transform_effective_exchange_rate(raw)

    assert rate["real_effective_exchange_rate"].to_dict() == {pd.Period("2021-01"): 97.7, pd.Period("2021-03"): 98.1}


def test_a_selection_matching_several_series_raises():
    """An unrestricted dimension returns one column per code, and taking the first would pick one silently."""
    raw = pd.DataFrame(
        {("M", "R", "B", "IN"): [1.0], ("M", "R", "N", "IN"): [2.0]}, index=pd.to_datetime(["2021-01-01"])
    )

    with pytest.raises(ValueError, match="one BIS series"):
        transform_effective_exchange_rate(raw)
