import numpy as np
import pandas as pd
import pytest

from pyproj import CRS

from climate_risk.geo.crs import GEOGRAPHIC_CRS, PROJECTED_CRS, to_km


def test_the_projected_crs_measures_in_meters():
    """Every distance in the library is meters because of this CRS; a non-metric one lies silently."""
    axes = CRS.from_user_input(PROJECTED_CRS).axis_info

    # PROJ reports the EPSG unit name, which is spelled "metre" upstream.
    assert {axis.unit_name for axis in axes} == {"metre"}


def test_the_geographic_crs_is_lat_lon():
    """Points are built from raw lat/lon, so a projected CRS here would place them nowhere."""
    crs = CRS.from_user_input(GEOGRAPHIC_CRS)

    assert crs.is_geographic
    assert {axis.unit_name for axis in crs.axis_info} == {"degree"}


@pytest.mark.parametrize("distance", [1500.0, pd.Series([1500.0])], ids=type)
def test_meters_convert_to_kilometers(distance):
    assert np.all(to_km(distance) == 1.5)
