import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from shapely.geometry import box

from climate_risk.data.resample import atom_year_table


@pytest.fixture
def grids(tmp_path):
    """One ERA5-like year on a 0.25-degree lattice whose value is the longitude, plus a flat population."""
    lon = np.arange(10.125, 12.0, 0.25)
    lat = np.arange(5.875, 4.0, -0.25)
    field = np.tile(lon, (len(lat), 1))
    data = xr.Dataset(
        {"precip_total": (("latitude", "longitude"), field), "rx5day": (("latitude", "longitude"), field * 2)},
        coords={"latitude": lat, "longitude": lon},
    )
    data.to_netcdf(tmp_path / "era5_daily_2000.nc")
    np.save(tmp_path / "population_era5_daily_2000.npy", np.ones(field.shape))
    return tmp_path


def test_atom_mean_is_the_mean_of_the_pixels_it_covers(grids):
    atoms = gpd.GeoSeries([box(10.0, 4.0, 11.0, 6.0), box(11.0, 4.0, 12.0, 6.0)], crs="EPSG:4326")
    table = atom_year_table(atoms, ["west", "east"], [2000], grids, sources=["era5_daily"])

    assert table["atom_id"].tolist() == ["west", "east"]
    assert table["precip_era5_mean"].to_numpy() == pytest.approx([10.5, 11.5], abs=1e-6)
    assert table["rx5day_mean"].to_numpy() == pytest.approx([21.0, 23.0], abs=1e-6)
    # Uniform population weights reduce the population-weighted mean to the area-weighted one.
    assert table["precip_era5_popmean"].to_numpy() == pytest.approx(table["precip_era5_mean"].to_numpy(), abs=1e-6)


def test_a_year_without_a_file_is_nan_not_dropped(grids):
    atoms = gpd.GeoSeries([box(10.0, 4.0, 11.0, 6.0)], crs="EPSG:4326")
    table = atom_year_table(atoms, ["a"], [2000, 2001], grids, sources=["era5_daily"])

    assert table["year"].tolist() == [2000, 2001]
    assert np.isfinite(table["rx5day_mean"].iloc[0])
    assert np.isnan(table["rx5day_mean"].iloc[1])


def test_the_number_of_ids_must_match_the_atoms(grids):
    atoms = gpd.GeoSeries([box(10.0, 4.0, 11.0, 6.0)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="ids"):
        atom_year_table(atoms, ["a", "b"], [2000], grids)
