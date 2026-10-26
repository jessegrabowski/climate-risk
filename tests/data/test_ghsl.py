import zipfile

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import requests

from exactextract import exact_extract
from exactextract.raster import RasterioRasterSource
from rasterio.transform import from_bounds
from shapely.geometry import box

from climate_risk.data.gadm import load_units_in_country
from climate_risk.data.ghsl import (
    POPULATION_EPOCHS,
    population_on_cells,
    population_raster,
    population_source,
)
from climate_risk.exceptions import DataValidationError
from climate_risk.geo.raster import build_cell_grid, dissolve_place_boundary


def write_archive(cache_dir, epoch, members):
    """Write the zip GHSL publishes for an epoch, holding whatever members a test needs."""
    directory = cache_dir / "ghsl"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / population_source(epoch).filename
    with zipfile.ZipFile(archive, "w") as writing:
        for name, payload in members.items():
            writing.writestr(name, payload)

    return archive


def test_an_epoch_the_release_does_not_publish_is_an_error():
    """The URL is built from the epoch, so a wrong one would 404 halfway through a download rather
    than failing where the caller can see why."""
    with pytest.raises(DataValidationError, match="1976"):
        population_source(1976)


@pytest.mark.parametrize("epoch", [1975, 1990, 2020])
def test_the_declared_url_names_the_epoch_it_was_asked_for(epoch):
    """One template serves twelve epochs, so the year has to reach both the path and the filename.
    Asking for one epoch only would not catch a template that had the year baked in."""
    source = population_source(epoch)

    assert f"E{epoch}" in source.url
    assert f"E{epoch}" in source.filename
    assert source.url.endswith(source.filename)


def test_the_raster_is_addressed_inside_its_archive(tmp_path):
    """Half a gigabyte per epoch, twelve epochs: extracting each one doubles the cost for a file
    GDAL can read where it lies. The metadata beside it must not be mistaken for the raster."""
    archive = write_archive(tmp_path, 2020, {"GHS_POP_E2020.tif": b"raster bytes", "GHS_POP_E2020.txt": b"metadata"})

    uri = population_raster(2020, tmp_path)

    assert uri == f"zip://{archive}!/GHS_POP_E2020.tif"
    assert not list(tmp_path.glob("**/*.tif")), "nothing is extracted"


def test_an_archive_holding_no_single_raster_is_an_error(tmp_path):
    """GHSL ships metadata beside the raster, so the member is found by extension. Two of them means
    the archive is not the product this loader was written against."""
    write_archive(tmp_path, 2020, {"first.tif": b"a", "second.tif": b"b"})

    with pytest.raises(DataValidationError, match="2 rasters"):
        population_raster(2020, tmp_path)


def test_population_is_counted_once_per_grid(tmp_path, monkeypatch):
    """Clipping every cell to a detailed coastline takes minutes over a region, so a second call
    reads the first one's count, and a grid at another resolution is counted afresh."""
    raster_path = tmp_path / "people.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        transform=from_bounds(0.0, 0.0, 1.0, 1.0, 8, 8),
    ) as writing:
        writing.write(np.ones((8, 8)), 1)
    write_archive(tmp_path, 2020, {"GHS_POP_E2020.tif": raster_path.read_bytes()})
    boundary = dissolve_place_boundary(
        gpd.GeoDataFrame({"ISO_A3": ["AAA"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    )
    grid = build_cell_grid(boundary, resolution_km=30.0)
    coarser = build_cell_grid(boundary, resolution_km=50.0)

    counted = population_on_cells(grid, 2020, tmp_path)

    def uncounted(*args, **kwargs):
        raise AssertionError("the population was counted again")

    # The count itself is patched out, since the sampler's own source is part of the cache key.
    monkeypatch.setattr("climate_risk.geo.raster.exact_extract", uncounted)
    np.testing.assert_array_equal(population_on_cells(grid, 2020, tmp_path), counted)
    with pytest.raises(AssertionError, match="counted again"):
        population_on_cells(coarser, 2020, tmp_path)


def test_a_warm_count_does_not_reach_the_network(tmp_path, monkeypatch):
    """The archive is half a gigabyte per epoch, so a cached count must be read without it, even
    once the archive has been deleted."""
    raster_path = tmp_path / "people.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        transform=from_bounds(0.0, 0.0, 1.0, 1.0, 8, 8),
    ) as writing:
        writing.write(np.ones((8, 8)), 1)
    archive = write_archive(tmp_path, 2020, {"GHS_POP_E2020.tif": raster_path.read_bytes()})
    boundary = dissolve_place_boundary(
        gpd.GeoDataFrame({"ISO_A3": ["AAA"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    )
    grid = build_cell_grid(boundary, resolution_km=30.0)

    counted = population_on_cells(grid, 2020, tmp_path)
    archive.unlink()

    def offline(*args, **kwargs):
        raise AssertionError("reached the network on a warm cache")

    monkeypatch.setattr(requests, "get", offline)
    np.testing.assert_array_equal(population_on_cells(grid, 2020, tmp_path), counted)


def test_the_epochs_reach_back_past_the_study_window():
    """Exposure is interpolated between epochs, so the window's 1981 opening has to sit inside them.
    An epoch list starting later would extrapolate population backwards over the whole first decade."""
    assert min(POPULATION_EPOCHS) <= 1981


# Half a gigabyte per epoch, so it is absent on CI and on a fresh clone.
REAL_CACHE_DIR = Path(__file__).parents[2] / "data"
REAL_ARCHIVE = REAL_CACHE_DIR / "ghsl" / population_source(2020).filename


@pytest.mark.requires_ghsl
@pytest.mark.requires_gadm
@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="needs the GHS-POP archive")
@pytest.mark.skipif(not (REAL_CACHE_DIR / "gadm" / "gadm_410.gpkg").exists(), reason="needs the GADM GeoPackage")
@pytest.mark.parametrize(("iso", "published"), [("LAO", 7.4e6), ("KHM", 16.4e6)], ids=["Laos", "Cambodia"])
def test_the_raster_totals_a_country_to_its_published_population(iso, published):
    """A synthetic fixture cannot catch the wrong product, the wrong units, or a grid read against
    the wrong transform. Summing real polygons against a number from outside the repo can."""
    units = load_units_in_country(iso, 1, REAL_CACHE_DIR)
    with rasterio.open(population_raster(2020, REAL_CACHE_DIR)) as source:
        totals = exact_extract(RasterioRasterSource(source), units, ["sum"], output="pandas")

    assert totals["sum"].sum() == pytest.approx(published, rel=0.05)


@pytest.mark.slow
@pytest.mark.requires_ghsl
@pytest.mark.requires_gadm
@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="needs the GHS-POP archive")
@pytest.mark.skipif(not (REAL_CACHE_DIR / "gadm" / "gadm_410.gpkg").exists(), reason="needs the GADM GeoPackage")
def test_the_epochs_form_one_series_on_one_grid():
    """The exposure offset interpolates between epochs, so a swapped or mislabeled year would move
    population backwards in time and no single-epoch check would see it. Every grid must also share
    a shape, since one operator is built against all of them."""
    units = load_units_in_country("LAO", 1, REAL_CACHE_DIR)
    grids, totals = set(), []

    for epoch in POPULATION_EPOCHS:
        uri = population_raster(epoch, REAL_CACHE_DIR)
        if not Path(uri.removeprefix("zip://").split("!/")[0]).exists():
            pytest.skip(f"epoch {epoch} is not in the cache")
        with rasterio.open(uri) as source:
            grids.add((source.shape, source.crs.to_string()))
            totals.append(exact_extract(RasterioRasterSource(source), units, ["sum"], output="pandas")["sum"].sum())

    assert len(grids) == 1, "the epochs are read against different grids"
    assert totals == sorted(totals), "Laos population does not rise through the series"


@pytest.mark.slow
@pytest.mark.requires_ghsl
@pytest.mark.requires_gadm
@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="needs the GHS-POP archive")
@pytest.mark.skipif(not (REAL_CACHE_DIR / "gadm" / "gadm_410.gpkg").exists(), reason="needs the GADM GeoPackage")
def test_gridded_population_holds_the_country_total_at_any_resolution():
    """The exposure offset must not depend on the grid it is measured on. Cells overhang the border,
    so an unclipped count would pick up the neighbours and grow as the cells got coarser."""
    units = load_units_in_country("LAO", 1, REAL_CACHE_DIR).assign(ISO_A3="LAO")
    boundary = dissolve_place_boundary(units)

    totals = [
        population_on_cells(build_cell_grid(boundary, resolution_km=km), 2020, REAL_CACHE_DIR).sum()
        for km in (10.0, 50.0)
    ]

    assert totals[0] == pytest.approx(totals[1], rel=0.001)
    assert totals[0] == pytest.approx(7.4e6, rel=0.05)


def test_force_reload_downloads_the_epoch_again_and_recounts_it(tmp_path, serve_downloads):
    def raster_of(people_per_pixel):
        raster_path = tmp_path / "people.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="float64",
            crs="EPSG:4326",
            transform=from_bounds(0.0, 0.0, 1.0, 1.0, 8, 8),
        ) as writing:
            writing.write(np.full((8, 8), people_per_pixel), 1)
        return raster_path.read_bytes()

    write_archive(tmp_path, 2020, {"GHS_POP_E2020.tif": raster_of(1.0)})
    boundary = dissolve_place_boundary(
        gpd.GeoDataFrame({"ISO_A3": ["AAA"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    )
    grid = build_cell_grid(boundary, resolution_km=30.0)
    counted = population_on_cells(grid, 2020, tmp_path)

    upstream = write_archive(tmp_path / "upstream", 2020, {"GHS_POP_E2020.tif": raster_of(2.0)})
    requested = serve_downloads({population_source(2020).url: upstream.read_bytes()})
    recounted = population_on_cells(grid, 2020, tmp_path, force_reload=True)

    np.testing.assert_allclose(recounted, 2 * counted)
    assert requested == [population_source(2020).url]
