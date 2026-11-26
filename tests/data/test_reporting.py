import numpy as np
import polars as pl
import xarray as xr

from climate_risk.data.global_temperature import transform_global_temperature
from climate_risk.data.reporting import zero_before_first_observation


def test_only_early_blanks_before_the_first_observation_become_zero():
    raw = pl.DataFrame(
        {
            "country_code": ["AAA"] * 4 + ["BBB"] * 3,
            "year": [1989, 1990, 1991, 1992, 1993, 1996, 1997],
            "internet_users_pct": [None, 1.0, None, 3.0, None, None, 5.0],
            "mobile_per_100": [None, None, None, None, None, None, None],
        },
        schema_overrides={"mobile_per_100": pl.Float64},
    )
    filled = zero_before_first_observation(raw)

    assert filled["internet_users_pct"].to_list() == [0.0, 1.0, None, 3.0, 0.0, None, 5.0]
    assert filled["mobile_per_100"].to_list() == [0.0, 0.0, 0.0, 0.0, 0.0, None, None]


def test_the_temperature_series_keeps_year_and_best_estimate():
    time = np.array(["1991-07-02", "1990-07-02"], dtype="datetime64[ns]")
    raw = xr.Dataset({"tas_mean": ("time", [0.2, 0.1]), "tas_lower": ("time", [0.0, 0.0])}, coords={"time": time})

    series = transform_global_temperature(raw)

    assert series.to_dict(as_series=False) == {"year": [1990, 1991], "gmst": [0.1, 0.2]}
