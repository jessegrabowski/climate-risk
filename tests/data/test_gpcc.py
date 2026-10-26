import gzip
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr

from shapely.geometry import box

from climate_risk.data import gpcc
from climate_risk.data.gpcc import (
    GPCC_PRODUCTS,
    PRECIPITATION,
    GriddedProduct,
    _as_timestamps,
    _boxes_holding,
    load_gpcc_data,
    precipitation_on_cells,
    transform_gpcc,
)
from climate_risk.data.source import DataSource
from climate_risk.geo.raster import build_cell_grid, dissolve_place_boundary
from tests.conftest import TOY_ARCHIVES, toy_gpcc_products, toy_world

# Every parameter but the reading digest is spelled out, so a key that lost one fails rather than
# agreeing with whatever the loader happened to write. The span is the toy manifest's.
UNREPAIRED_CACHE = "gpcc__coverage=1981-2021__precision=float64__reading=*__repaired_iso=False.parquet"


def gridded(rows, gauges=1.0) -> pd.DataFrame:
    """Rows of time, lat, lon and precipitation, with a uniform station count unless one is given."""
    return pd.DataFrame(rows, columns=["time", "lat", "lon", "precip"]).assign(
        time=lambda frame: pd.to_datetime(frame["time"]), gauges=gauges
    )


def one_country(geometry) -> gpd.GeoDataFrame:
    """A world holding a single country, shaped as the test needs."""
    return gpd.GeoDataFrame(
        {
            "ISO_A3": ["AAA"],
            "FORMAL_EN": ["Aland"],
            "CONTINENT": ["Asia"],
            "REGION_UN": ["Asia"],
            "geometry": [geometry],
        },
        crs="EPSG:4326",
    )


def extracted_name(archive: str) -> str:
    return archive.removesuffix(".gz")


def test_interrupted_extraction_resumes(write_gpcc_archives, write_shapefile_cache):
    """One extracted archive once stood in for all of them, so an interrupted run never recovered."""
    cache_dir = write_gpcc_archives(extracted=[TOY_ARCHIVES[0]])
    write_shapefile_cache("world", toy_world())

    load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False)

    assert all((cache_dir / "gpcc" / extracted_name(archive)).exists() for archive in TOY_ARCHIVES)


def test_cold_run_aggregates_precipitation_by_country(write_gpcc_archives, write_shapefile_cache):
    cache_dir = write_gpcc_archives()
    write_shapefile_cache("world", toy_world())

    frame = load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False)

    assert frame.index.names == ["country_code", "time"]
    assert sorted(frame.index.get_level_values("country_code").unique()) == ["AAA", "BBB", "CCC"]


def test_each_product_is_read_under_its_own_grid_names(write_gpcc_archives, write_shapefile_cache):
    """The full-data archives name the grids `precip` and `numgauge`, the monitoring ones `p` and
    `s`. Wiring either product to the wrong pair swaps its two columns without raising, because
    both grids are read and both are floats.
    """
    cache_dir = write_gpcc_archives()
    write_shapefile_cache("world", toy_world())

    frame = load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False)
    full_data = frame.xs(pd.Timestamp("1981-01-01"), level="time")
    monitoring = frame.xs(pd.Timestamp("2021-01-01"), level="time")

    assert full_data.loc["AAA", "precip"] == pytest.approx(0.0)
    assert full_data.loc["AAA", "gauges"] == pytest.approx(2.0)
    assert monitoring.loc["CCC", "precip"] == pytest.approx(2.0)
    assert monitoring.loc["CCC", "gauges"] == pytest.approx(4.0)


def test_the_cache_is_written_in_double_precision(write_gpcc_archives, write_shapefile_cache):
    """The archives are float32, and every total taken from this cache adds partial results.

    Whichever order the threads add them in has to reach the same answer, which float32 across a
    wide spread of values does not.
    """
    cache_dir = write_gpcc_archives()
    write_shapefile_cache("world", toy_world())

    gpcc = load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False)

    assert gpcc[PRECIPITATION].dtype == "float64"


def test_the_cold_run_writes_the_cache_it_will_read(write_gpcc_archives, write_shapefile_cache):
    """A key spelled one way on write and another on read is the bug this replaces."""
    cache_dir = write_gpcc_archives()
    write_shapefile_cache("world", toy_world())

    load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False)

    assert len(list(cache_dir.glob(UNREPAIRED_CACHE))) == 1


def test_the_monitoring_dates_are_decoded_rather_than_read_as_numbers(write_gpcc_archives, write_shapefile_cache):
    """The monitoring archives date rows as YYYYMMDD floats; taken as numbers they all land in 1970."""
    cache_dir = write_gpcc_archives()
    write_shapefile_cache("world", toy_world())

    frame = load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False)
    months = pd.DatetimeIndex(frame.index.get_level_values("time"))

    assert months.min() == pd.Timestamp("1981-01-01")
    assert months.max() == pd.Timestamp("2021-01-01")


def test_a_wider_manifest_rebuilds_rather_than_reading_the_narrower_cache(write_gpcc_archives, write_shapefile_cache):
    """Extending the record must not serve the shorter frame a previous run left on disk."""
    cache_dir = write_gpcc_archives()
    write_shapefile_cache("world", toy_world())
    full_data, monitoring = toy_gpcc_products()

    narrow = load_gpcc_data(cache_dir, products=(full_data,), repair_ISO_codes=False)
    wide = load_gpcc_data(cache_dir, products=(full_data, monitoring), repair_ISO_codes=False)

    assert len(wide) > len(narrow)


def test_the_published_manifest_lists_an_archive_for_every_period_it_claims():
    """A decade or a month left off the list is a hole in the record that nothing downstream sees."""
    full_data, monitoring = GPCC_PRODUCTS

    assert len(full_data.sources) * 10 == full_data.last_year - full_data.first_year + 1
    assert len(monitoring.sources) == (monitoring.last_year - monitoring.first_year + 1) * 12


def test_the_two_products_meet_without_a_gap_or_an_overlap():
    """An overlap would total a month twice; a gap would leave a year out of the panel entirely."""
    full_data, monitoring = GPCC_PRODUCTS

    assert monitoring.first_year == full_data.last_year + 1


def test_a_time_axis_in_an_unknown_encoding_is_refused():
    """Reading an unrecognized numeric axis as-is would date the record silently, not loudly."""
    time = xr.DataArray([20210101.0], dims="time", attrs={"units": "days since 1900-01-01"})

    with pytest.raises(ValueError, match="neither a decoded datetime"):
        _as_timestamps(time)


def test_a_warm_cache_does_not_touch_the_archives(write_gpcc_archives, write_shapefile_cache):
    """The archives are gigabytes; a warm processed cache must not read or extract them."""
    cache_dir = write_gpcc_archives()
    write_shapefile_cache("world", toy_world())
    load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False)

    for archive in TOY_ARCHIVES:
        (cache_dir / "gpcc" / archive).unlink()
        (cache_dir / "gpcc" / extracted_name(archive)).unlink()

    assert not load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False).empty


def test_cells_are_averaged_per_country_and_month():
    """Half-degree cells on one row, so the two inside AAA sit at one latitude and weigh the same."""
    grid = gridded(
        [
            ("1981-01-01", 0.5, 0.25, 4.0),
            ("1981-01-01", 0.5, 0.75, 6.0),
            ("1981-01-01", 0.5, 1.25, 0.0),
            ("1981-01-01", 0.5, 1.75, 0.0),
            ("1981-01-01", 0.5, 2.25, 100.0),
            ("1981-01-01", 0.5, 2.75, 100.0),
        ]
    )

    monthly = transform_gpcc([grid], toy_world())

    assert monthly.loc[("AAA", pd.Timestamp("1981-01-01")), "precip"] == pytest.approx(5.0)
    assert monthly.loc[("BBB", pd.Timestamp("1981-01-01")), "precip"] == pytest.approx(100.0)


def test_a_country_smaller_than_a_cell_still_gets_a_value():
    """No cell center falls inside a country this small, so joining on centers leaves it with no
    precipitation at all rather than with the reading over the ground it sits on.
    """
    tiny = one_country(box(0.1, 0.1, 0.3, 0.3))
    grid = gridded(
        [
            ("1981-01-01", 0.5, 0.5, 7.0),
            ("1981-01-01", 0.5, 1.5, 99.0),
            ("1981-01-01", 1.5, 0.5, 99.0),
            ("1981-01-01", 1.5, 1.5, 99.0),
        ]
    )

    monthly = transform_gpcc([grid], tiny)

    assert monthly.loc[("AAA", pd.Timestamp("1981-01-01")), "precip"] == pytest.approx(7.0)


def test_a_cell_counts_only_for_the_land_it_holds():
    """The country fills one cell and half of the next, so the fuller cell carries twice the weight.
    An unweighted mean of the two would read 1.5 instead.
    """
    straddling = one_country(box(0.0, 0.0, 1.5, 1.0))
    grid = gridded([("1981-01-01", 0.5, 0.5, 0.0), ("1981-01-01", 0.5, 1.5, 3.0)])

    monthly = transform_gpcc([grid], straddling)

    assert monthly.loc[("AAA", pd.Timestamp("1981-01-01")), "precip"] == pytest.approx(1.0)


def test_an_unevenly_spaced_grid_is_refused():
    """A weight is measured over a cell the lattice places by index, so an axis with uneven gaps
    would credit a country with ground the reading never covered.
    """
    grid = gridded(
        [
            ("1981-01-01", 0.5, 0.5, 1.0),
            ("1981-01-01", 0.5, 1.5, 1.0),
            ("1981-01-01", 0.5, 9.5, 1.0),
        ]
    )

    with pytest.raises(ValueError, match="not a grid"):
        transform_gpcc([grid], toy_world())


def test_station_counts_are_weighted_like_the_precipitation():
    """The count says how much gauge evidence stands behind a reading, so a cell contributing half
    the land contributes half its stations to the country's figure.
    """
    straddling = one_country(box(0.0, 0.0, 1.5, 1.0))
    grid = gridded([("1981-01-01", 0.5, 0.5, 0.0), ("1981-01-01", 0.5, 1.5, 0.0)], gauges=[6.0, 0.0])

    monthly = transform_gpcc([grid], straddling)

    assert monthly.loc[("AAA", pd.Timestamp("1981-01-01")), "gauges"] == pytest.approx(4.0)


def test_cells_over_the_ocean_are_dropped():
    """The countries of `toy_world` all lie below one degree north, so the upper row is open water."""
    grid = gridded(
        [
            ("1981-01-01", 0.5, 0.5, 4.0),
            ("1981-01-01", 0.5, 1.5, 999.0),
            ("1981-01-01", 1.5, 0.5, 999.0),
            ("1981-01-01", 1.5, 1.5, 999.0),
        ]
    )

    monthly = transform_gpcc([grid], toy_world())

    assert monthly["precip"].tolist() == [4.0]


def test_every_archive_reaches_the_result():
    """The archives are read one at a time; dropping one loses its years without an error."""
    grids = [gridded([("1981-01-01", 0.5, 0.5, 1.0)]), gridded([("1991-01-01", 0.5, 0.5, 2.0)])]

    monthly = transform_gpcc(grids, toy_world())

    assert pd.DatetimeIndex(monthly.index.get_level_values("time")).year.tolist() == [1981, 1991]


def test_an_interrupted_extraction_leaves_nothing_to_trust(write_gpcc_archives, write_shapefile_cache, monkeypatch):
    """A truncated .nc would satisfy the exists() check and be read as complete on every later run."""
    cache_dir = write_gpcc_archives()
    write_shapefile_cache("world", toy_world())

    def die_partway(source, target, length=0):
        target.write(b"\x89HDF truncated")
        raise OSError("no space left on device")

    monkeypatch.setattr(shutil, "copyfileobj", die_partway)
    with pytest.raises(OSError, match="no space left"):
        load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False)

    monkeypatch.undo()
    assert not (cache_dir / "gpcc" / extracted_name(TOY_ARCHIVES[0])).exists()
    assert not load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False).empty


def test_the_archives_are_decompressed_before_reading(write_gpcc_archives, write_shapefile_cache):
    """The published files are gzipped NetCDF; reading one without decompressing raises."""
    cache_dir = write_gpcc_archives()
    archive = cache_dir / "gpcc" / TOY_ARCHIVES[0]

    assert gzip.decompress(archive.read_bytes())[1:4] == b"HDF"

    write_shapefile_cache("world", toy_world())
    assert not load_gpcc_data(cache_dir, products=toy_gpcc_products(), repair_ISO_codes=False).empty


def test_the_world_boundaries_decide_the_countries():
    """Passing the boundaries in is what lets this run without the 606MB shapefile."""
    grid = gridded([("1981-01-01", 0.5, 0.5, 4.0)])
    one_country = gpd.GeoDataFrame(toy_world().iloc[:1])

    monthly = transform_gpcc([grid], one_country)

    assert monthly.index.get_level_values("country_code").tolist() == ["AAA"]


# Two 5-degree boxes side by side, monthly over 2000 and 2001 in the full product and over 2002 in the monitoring one.
FULL_ARCHIVE = "full_data_monthly_v2022_2000_2001_10.nc.gz"
MONITORING_ARCHIVE = "monitoring_v2022_10_2002.nc.gz"


def two_box_products() -> tuple[GriddedProduct, ...]:
    """One archive from each product, spelling its grids and dating its rows as that product does."""

    def source(filename: str) -> DataSource:
        return DataSource(
            url=f"https://opendata.dwd.de/climate_environment/GPCC/{filename}",
            filename=filename,
            license="CC BY 4.0",
            citation="GPCC",
            retrieved="2026-08-07",
        )

    return (
        GriddedProduct(
            variable="precip", gauges="numgauge", first_year=2000, last_year=2001, sources=(source(FULL_ARCHIVE),)
        ),
        GriddedProduct(
            variable="p", gauges="s", first_year=2002, last_year=2002, sources=(source(MONITORING_ARCHIVE),)
        ),
    )


def write_two_boxes(cache_dir, *, west, east, west_gauges=None, monitored=None):
    """Write both archives, holding ``west`` and ``east`` per month of the full product, then ``monitored`` in both
    boxes through 2002."""
    gpcc_dir = cache_dir / "gpcc"
    gpcc_dir.mkdir(parents=True, exist_ok=True)
    gauges = np.ones(24) if west_gauges is None else west_gauges
    full = xr.Dataset(
        {
            "precip": (("time", "lat", "lon"), np.stack([west, east], axis=-1)[:, None, :].astype("float32")),
            "numgauge": (
                ("time", "lat", "lon"),
                np.stack([gauges, np.ones(24)], axis=-1)[:, None, :].astype("float32"),
            ),
        },
        coords={"time": pd.date_range("2000-01-01", periods=24, freq="MS"), "lat": [2.5], "lon": [2.5, 7.5]},
    )
    monitored = np.full(12, 9.0) if monitored is None else monitored
    monitoring = xr.Dataset(
        {
            "p": (("time", "lat", "lon"), np.repeat(monitored[:, None, None], 2, axis=-1).astype("float32")),
            "s": (("time", "lat", "lon"), np.ones((12, 1, 2), dtype="float32")),
        },
        coords={
            "time": ("time", [float(f"2002{month:02d}01") for month in range(1, 13)], {"units": "day as %Y%m%d.%f"}),
            "lat": [2.5],
            "lon": [2.5, 7.5],
        },
    )
    for name, dataset in ((FULL_ARCHIVE, full), (MONITORING_ARCHIVE, monitoring)):
        (gpcc_dir / name).write_bytes(gzip.compress(bytes(dataset.to_netcdf())))


@pytest.fixture
def grid():
    """Cells over the two boxes, with a column of centers on their shared edge at 5 east."""
    boundary = dissolve_place_boundary(
        gpd.GeoDataFrame({"ISO_A3": ["AAA"], "geometry": [box(0, 0, 10, 5)]}, crs="EPSG:4326")
    )

    return build_cell_grid(boundary, resolution_km=100.0)


def read_with_longitudes(grid, cache_dir):
    return precipitation_on_cells(grid, cache_dir, products=two_box_products()).merge(
        grid.cells[["cell_id", "lon"]], on="cell_id"
    )


def test_each_cell_takes_the_annual_total_of_the_box_its_center_falls_in(tmp_path, grid):
    """The western box rains 1 a month through 2000 and 3 through 2001, the eastern 5 throughout. A center on the
    shared edge belongs to the eastern box."""
    write_two_boxes(tmp_path, west=np.repeat([1.0, 3.0], 12), east=np.full(24, 5.0))

    precipitation = read_with_longitudes(grid, tmp_path)
    western = precipitation["lon"] < 5.0

    assert precipitation.loc[western & (precipitation["year"] == 2000), "precip"].unique().tolist() == [12.0]
    assert precipitation.loc[western & (precipitation["year"] == 2001), "precip"].unique().tolist() == [36.0]
    assert precipitation.loc[~western & (precipitation["year"] < 2002), "precip"].unique().tolist() == [60.0]
    assert len(precipitation) == 3 * len(grid.cells)


def test_the_station_count_is_the_mean_over_the_year(tmp_path, grid):
    """A total of station counts would read twelve months of one gauge as twelve gauges."""
    write_two_boxes(tmp_path, west=np.ones(24), east=np.ones(24), west_gauges=np.repeat([2.0, 4.0], [6, 18]))

    precipitation = read_with_longitudes(grid, tmp_path)
    western = precipitation["lon"] < 5.0

    assert precipitation.loc[western & (precipitation["year"] == 2000), "gauges"].unique().tolist() == [3.0]
    assert precipitation.loc[western & (precipitation["year"] == 2001), "gauges"].unique().tolist() == [4.0]


def test_a_year_missing_a_month_is_missing(tmp_path, grid):
    """Eleven months summed read as a drier year than it was, so the year is missing instead."""
    east = np.full(24, 5.0)
    east[13] = np.nan
    write_two_boxes(tmp_path, west=np.ones(24), east=east)

    precipitation = read_with_longitudes(grid, tmp_path)
    eastern_2001 = (precipitation["lon"] >= 5.0) & (precipitation["year"] == 2001)

    assert precipitation.loc[eastern_2001, ["precip", "gauges"]].isna().all().all()
    assert precipitation.loc[~eastern_2001, ["precip", "gauges"]].notna().all().all()


def test_the_monitoring_product_continues_the_record(tmp_path, grid):
    """The monitoring archive names its grids differently and dates its rows as YYYYMMDD floats."""
    write_two_boxes(tmp_path, west=np.ones(24), east=np.ones(24), monitored=np.arange(1.0, 13.0))

    precipitation = read_with_longitudes(grid, tmp_path)

    assert precipitation.loc[precipitation["year"] == 2002, "precip"].unique().tolist() == [78.0]


def test_a_warm_read_needs_neither_the_archives_nor_the_network(tmp_path, grid, monkeypatch):
    write_two_boxes(tmp_path, west=np.ones(24), east=np.ones(24))
    first = precipitation_on_cells(grid, tmp_path, products=two_box_products())
    shutil.rmtree(tmp_path / "gpcc")

    def offline(*args, **kwargs):
        raise AssertionError("reached the network on a warm cache")

    monkeypatch.setattr(requests, "get", offline)
    pd.testing.assert_frame_equal(precipitation_on_cells(grid, tmp_path, products=two_box_products()), first)


def test_changing_how_box_spacing_is_read_turns_the_cached_cells_over(tmp_path, grid, monkeypatch):
    """The spacing decides which box a cell reads, so a cache that cannot see it change keeps serving cells read
    from the wrong boxes."""
    write_two_boxes(tmp_path, west=np.ones(24), east=np.full(24, 5.0))
    five_degrees = read_with_longitudes(grid, tmp_path)

    read_steps = gpcc._axis_steps
    monkeypatch.setattr(gpcc, "_axis_steps", lambda *axes: tuple(2 * step for step in read_steps(*axes)))
    ten_degrees = read_with_longitudes(grid, tmp_path)

    assert not five_degrees.equals(ten_degrees)


def test_boxes_are_found_on_a_descending_axis():
    """GPCC lists latitude north to south. A center on the edge at 1 north belongs to the box above it, at 1.5."""
    boxes = _boxes_holding(np.array([1.0, -1.0, 0.2]), np.array([1.5, 0.5, -0.5, -1.5]), 1.0)

    assert boxes.tolist() == [0, 2, 1]
