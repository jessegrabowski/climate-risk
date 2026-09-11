import gzip
import shutil

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from shapely.geometry import box

from climate_risk.data.gpcc import GPCC_PRODUCTS, PRECIPITATION, _as_timestamps, load_gpcc_data, transform_gpcc
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
