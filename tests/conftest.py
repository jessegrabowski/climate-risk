import gzip
import socket
import zipfile

from datetime import date
from functools import partial
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from shapely.geometry import LineString, Point, box

from climate_risk.data import world_bank
from climate_risk.data.gpcc import GriddedProduct
from climate_risk.data.ocean_heat import OCEAN_HEAT, OCEAN_HEAT_BASELINE_OFFSET
from climate_risk.data.osm import LOOKUP_COLUMNS
from climate_risk.data.source import DataSource

# One event that clears every downstream filter: deaths above 100, affected above 1000, and a start
# year inside both the 1970 and 1980 cutoffs. Tests override only the field under examination.
EMDAT_EVENT_DEFAULTS = {
    "DisNo.": "1990-0001-AAA",
    "Country": "Testland",
    "ISO": "AAA",
    "Region": "Asia",
    "Subregion": "South-eastern Asia",
    "Disaster Type": "Flood",
    "Start Year": 1990,
    "End Year": 1990,
    "Total Deaths": 500,
    "No. Injured": 10,
    "No. Affected": 5_000,
    "No. Homeless": 10,
    "Total Affected": 5_000,
    "Total Damage ('000 US$)": 100,
    "Total Damage, Adjusted ('000 US$)": 120,
    "Reconstruction Costs ('000 US$)": 1,
    "Reconstruction Costs, Adjusted ('000 US$)": 1,
    "Insured Damage ('000 US$)": 1,
    "Insured Damage, Adjusted ('000 US$)": 1,
    "Latitude": 18.0,
    "Longitude": 102.0,
    "River Basin": "Mekong",
    "Location": "Somewhere",
    # Present on every real row, empty on the two thirds EM-DAT never geocoded.
    "GADM Admin Units": "",
}


def emdat_event(overrides=None):
    return EMDAT_EVENT_DEFAULTS | (overrides or {})


def toy_world():
    """Three square countries: two in Asia, one in Africa, with synthetic ISO codes."""
    return gpd.GeoDataFrame(
        {
            "ISO_A3": ["AAA", "BBB", "CCC"],
            "FORMAL_EN": ["Aland", "Beeland", "Ceeland"],
            "CONTINENT": ["Asia", "Asia", "Africa"],
            "REGION_UN": ["Asia", "Asia", "Africa"],
            "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)],
        },
        crs="EPSG:4326",
    )


def toy_world_needing_repair():
    """A world file shaped like the real one: unlabeled sovereigns and territories to drop.

    Carries one row per entry in DROPPED_TERRITORIES, DROPPED_ISO_CODES and ISO_CODE_REPAIRS; adding
    to any of those tables without adding a row here fails every repair test on the missing entry.
    """
    rows = [
        ("France", "-99", "Europe"),
        ("Norway", "-99", "Europe"),
        ("Kosovo", "-99", "Europe"),
        ("Tokelau (NZ)", "NZL", "Oceania"),
        ("Guantanamo Bay (US)", "-99", "North America"),
        ("Clipperton Island (Fr.)", "-99", "North America"),
        ("Cocos (Keeling) Islands (Aus.)", "-99", "Asia"),
        ("Christmas Island (Aus.)", "-99", "Asia"),
        ("Bonaire (Neth.)", "NLD", "North America"),
        ("Sint Eustatius (Neth.)", "NLD", "North America"),
        ("Saba (Neth.)", "NLD", "North America"),
        ("Johnston Atoll (US)", "UMI", "Oceania"),
        ("Netherlands", "NLD", "Europe"),
        ("New Zealand", "NZL", "Oceania"),
    ]
    return gpd.GeoDataFrame(
        {
            "WB_NAME": [name for name, _, _ in rows],
            "ISO_A3": [iso for _, iso, _ in rows],
            "CONTINENT": [continent for _, _, continent in rows],
            "geometry": [box(i, 0, i + 0.5, 1) for i in range(len(rows))],
        },
        crs="EPSG:4326",
    )


def toy_world_with_places():
    """`toy_world_needing_repair` plus the three shipped countries, so a load can still be repaired.

    Costa Rica straddles `toy_coastline`; Laos and Zambia sit well inland of it.
    """
    countries = gpd.GeoDataFrame(
        {
            "WB_NAME": ["Lao PDR", "Zambia", "Costa Rica"],
            "ISO_A3": ["LAO", "ZMB", "CRI"],
            "CONTINENT": ["Asia", "Africa", "North America"],
            "geometry": [box(20, 0, 21, 1), box(23, 0, 24, 1), box(29.5, 0, 30.5, 1)],
        },
        crs="EPSG:4326",
    )

    return gpd.GeoDataFrame(pd.concat([toy_world_needing_repair(), countries], ignore_index=True), crs="EPSG:4326")


def toy_coastline():
    return gpd.GeoDataFrame({"geometry": [LineString([(30, -1), (30, 2)])]}, crs="EPSG:4326")


def toy_rivers():
    """ORD_FLOW spans the thresholds both loaders filter on: < 5 is big, < 6 adds medium."""
    return gpd.GeoDataFrame(
        {
            "ORD_FLOW": [4, 5, 6],
            "HYRIV_ID": [1, 2, 3],
            "geometry": [
                LineString([(0, 0), (0, 1)]),
                LineString([(2, 0), (2, 1)]),
                LineString([(4, 0), (4, 1)]),
            ],
        },
        crs="EPSG:4326",
    )


# Archive and directory names as they appear upstream, stated independently of the loader so a
# change to either constant fails a test rather than silently agreeing with itself.
# Archive name and the path each one unpacks to. The world archive holds a directory; the Laos
# archive is flat, one file per admin level.
UPSTREAM_SHAPEFILE_LAYOUT = {
    "world": ("wb_countries_admin0_10m.zip", "WB_countries_Admin0_10m/WB_countries_Admin0_10m.shp"),
    "laos": ("lao_admin_boundaries.shp.zip", "lao_admin2.shp"),
    "coastline": ("gshhg-shp-2.3.7.zip", "GSHHS_shp/f/GSHHS_f_L1.shp"),
}


def toy_precipitation(year_range):
    """A full-data grid with one point inside each country of `toy_world`.

    Single precision, as the archives publish it, so a loader that keeps that dtype is visible here.
    """
    start = int(year_range.split("_")[0])
    return xr.Dataset(
        {"precip": (("time", "lat", "lon"), np.arange(3.0, dtype="float32").reshape(1, 1, 3))},
        coords={
            "time": np.array([f"{start}-01-01"], dtype="datetime64[ns]"),
            "lat": [0.5],
            "lon": [0.5, 2.5, 4.5],
        },
    )


def toy_monitoring(year, month):
    """A monitoring grid, which names its variable ``p`` and dates it with a YYYYMMDD float."""
    return xr.Dataset(
        {"p": (("time", "lat", "lon"), np.arange(3.0, dtype="float32").reshape(1, 1, 3))},
        coords={
            "time": ("time", [float(f"{year}{month:02d}01")], {"units": "day as %Y%m%d.%f"}),
            "lat": [0.5],
            "lon": [0.5, 2.5, 4.5],
        },
    )


# Archive names as they appear upstream, stated literally so a wrong path fails rather than agreeing
# with the loader. One archive per product is the whole manifest a test needs: nothing the loader
# does varies with how many are listed.
TOY_ARCHIVES = ("full_data_monthly_v2022_1981_1990_10.nc.gz", "monitoring_v2022_10_2021_01.nc.gz")

# The processed cache the published manifest writes. The key carries the span, so extending the
# record renames the entry instead of shadowing it.
GPCC_CACHE_FILE = "gpcc__coverage=1891-2025__precision=float64__repaired_iso=True.parquet"


def toy_gpcc_products() -> tuple[GriddedProduct, ...]:
    """The published manifest cut to one archive from each product, which differ in both fields."""

    def source(filename: str) -> DataSource:
        return DataSource(
            url=f"https://opendata.dwd.de/climate_environment/GPCC/{filename}",
            filename=filename,
            license="CC BY 4.0",
            citation="GPCC",
            retrieved="2026-08-07",
        )

    full_data, monitoring = TOY_ARCHIVES

    return (
        GriddedProduct(variable="precip", first_year=1981, last_year=1990, sources=(source(full_data),)),
        GriddedProduct(variable="p", first_year=2021, last_year=2021, sources=(source(monitoring),)),
    )


@pytest.fixture
def write_gpcc_archives(tmp_path):
    """Return a callable writing one gzipped archive per product, some already extracted."""

    def write(extracted=()):
        gpcc_dir = tmp_path / "gpcc"
        gpcc_dir.mkdir(parents=True, exist_ok=True)

        grids = zip(TOY_ARCHIVES, (toy_precipitation("1981_1990"), toy_monitoring(2021, 1)), strict=True)
        for name, grid in grids:
            raw = bytes(grid.to_netcdf())
            (gpcc_dir / name).write_bytes(gzip.compress(raw))
            if name in extracted:
                (gpcc_dir / name.removesuffix(".gz")).write_bytes(raw)
        return tmp_path

    return write


def seed_world_bank_cache(cache_dir, rows):
    """
    Write ``rows`` into the World Bank cache the way the loader will look for them.

    The panel is placed by running the loader over a stubbed download rather than by writing a file
    named by hand, because the entry is keyed on a fingerprint of how it was built. The stub replaces
    the transform as well as the download: these rows are already in the panel's shape, and the codes
    they use are not names the World Bank publishes.
    """
    panel = pl.DataFrame(
        rows, schema=["country_code", "year", "gdp_per_cap", "population_density", "Population"], orient="row"
    )
    with (
        mock.patch.object(world_bank.wb, "download", lambda **kwargs: pl.DataFrame()),
        mock.patch.object(world_bank, "transform_world_bank", lambda raw, indicator_names: panel),
    ):
        world_bank.load_wb_data(cache_dir)


def seed_ocean_heat_cache(cache_dir, annual):
    """
    Write ``annual`` into the ocean heat cache, as the seasonal file NCEI publishes.

    The processed entry is keyed on a fingerprint of how it was built, so it cannot be written by
    name. Seeding the download the loader reads instead lets it build its own entry, which also
    exercises the headerless read. Each year is four identical seasons, so their mean is the value
    asked for once the baseline offset is added back.
    """
    seasons = [
        f"{row['Date'].year}-{month},{row['Temp'] - OCEAN_HEAT_BASELINE_OFFSET}"
        for row in annual.iter_rows(named=True)
        for month in (3, 6, 9, 12)
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    OCEAN_HEAT.path(cache_dir).write_text("\n".join(seasons) + "\n")


def write_merge_cache(cache_dir):
    """Seed every cache `load_all_data` reads, so the whole merge runs offline.

    AAA and BBB appear in every source. CCC is EM-DAT only and DDD World Bank only, so the first
    reconciliation has something to drop from each side. EEE has both but no precipitation, so it
    survives that pass and is dropped only from the GPCC frame. FFF has precipitation and
    nothing else, so the GPCC pass has something of its own to drop.
    """
    events = [
        emdat_event({"ISO": iso, "DisNo.": f"{iso}-{year}", "Start Year": year, "End Year": year})
        for iso in ("AAA", "BBB", "CCC", "EEE")
        for year in (1990, 1991)
    ]
    # A climatological event, so the hydro/clim damage split has something on both sides.
    events.append(
        emdat_event(
            {
                "ISO": "AAA",
                "DisNo.": "AAA-drought",
                "Start Year": 1990,
                "End Year": 1990,
                "Disaster Type": "Drought",
            }
        )
    )
    # One country has a disaster type in a single year, so a per-year column mistaken for a
    # country constant shows up as a duplicated country.
    events.append(
        emdat_event(
            {
                "ISO": "AAA",
                "DisNo.": "AAA-landslide",
                "Start Year": 1991,
                "End Year": 1991,
                "Disaster Type": "Mass movement (wet)",
            }
        )
    )
    write_emdat_workbook(cache_dir, events)
    seed_world_bank_cache(
        cache_dir,
        [
            ("AAA", 1990, 1000.0, 10.0, 1000000),
            ("AAA", 1991, 1100.0, 11.0, 1010000),
            ("BBB", 1990, 2000.0, 20.0, 2000000),
            ("BBB", 1991, 2200.0, 22.0, 2020000),
            ("DDD", 1990, 3000.0, 30.0, 3000000),
            ("DDD", 1991, 3300.0, 33.0, 3030000),
            ("EEE", 1990, 4000.0, 40.0, 4000000),
            ("EEE", 1991, 4400.0, 44.0, 4040000),
        ],
    )
    # The cache key is stated literally, so a wrong one fails rather than agreeing with itself.
    pl.DataFrame({"Date": [date(1990, 1, 1), date(1991, 1, 1)], "co2": [354.0, 355.0]}).write_parquet(
        cache_dir / "co2.parquet"
    )
    seed_ocean_heat_cache(cache_dir, pl.DataFrame({"Date": [date(1990, 1, 1), date(1991, 1, 1)], "Temp": [1.0, 2.0]}))
    # GPCC publishes monthly, and only whole years survive the annual total, so each year here
    # carries all twelve months. A total stays distinguishable from an average.
    pd.DataFrame(
        [
            (iso, pd.Timestamp(f"{year}-{month:02d}-01"), base + 1000.0 * (year - 1990) + month)
            for iso, base in (("AAA", 100.0), ("BBB", 200.0), ("FFF", 300.0))
            for year in (1990, 1991)
            for month in range(1, 13)
        ],
        columns=["country_code", "time", "precip"],
    ).set_index(["country_code", "time"]).to_parquet(cache_dir / GPCC_CACHE_FILE)
    return cache_dir


@pytest.fixture
def write_full_cache(tmp_path):
    """Return a callable seeding every cache `load_all_data` reads, and giving back its directory."""
    return partial(write_merge_cache, tmp_path)


@pytest.fixture
def rivers_clear_of_the_grid():
    """Offset from the countries, so every grid point sits a strictly positive distance away."""
    return gpd.GeoDataFrame(
        {
            "ORD_FLOW": [4, 5],
            "HYRIV_ID": [1, 2],
            "geometry": [LineString([(-2, -1), (-2, 2)]), LineString([(8, -1), (8, 2)])],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def rivers_through_the_grid():
    """Runs up the western edge of the toy world, so the grid points along it measure zero."""
    return gpd.GeoDataFrame(
        {
            "ORD_FLOW": [4, 5],
            "HYRIV_ID": [1, 2],
            "geometry": [LineString([(0, -1), (0, 2)]), LineString([(8, -1), (8, 2)])],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def coastline_through_the_grid():
    """Ends on a grid point; `create_grid_from_shape` measures to the boundary, which is that endpoint."""
    return gpd.GeoDataFrame({"geometry": [LineString([(5, 0), (5, 2)])]}, crs="EPSG:4326")


@pytest.fixture
def coastline():
    return gpd.GeoDataFrame({"geometry": [LineString([(6, -1), (6, 2)])]}, crs="EPSG:4326")


@pytest.fixture
def grid_points():
    return gpd.GeoDataFrame({"geometry": [Point(0.5, 0.5), Point(2.5, 0.5)]}, crs="EPSG:4326")


@pytest.fixture
def write_shapefile_cache(tmp_path):
    """Return a callable laying out a shapefile the way a completed download would have."""

    def write(which, gdf):
        archive_name, member = UPSTREAM_SHAPEFILE_LAYOUT[which]
        # The layout is stated literally, not derived from the loader, so a wrong cache path fails.
        shapefile = tmp_path / "shapefiles" / member
        shapefile.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(shapefile)

        archive = tmp_path / "shapefiles" / archive_name
        # Only the member's own sidecars, so an archive written beside an earlier one never
        # contains it, or itself.
        sidecars = sorted(part for part in shapefile.parent.iterdir() if part.stem == shapefile.stem)
        with zipfile.ZipFile(archive, "w") as bundle:
            for part in sidecars:
                bundle.write(part, str(part.relative_to(tmp_path / "shapefiles")))

        return tmp_path

    return write


def toy_gadm() -> gpd.GeoDataFrame:
    """A GeoPackage shaped like GADM: one row per finest unit, coarser levels spanning several rows.

    `LAO.1_1` is split across two districts, so reading it back as one polygon exercises the union
    rather than returning whichever row came first.

    Ghana is included because GADM numbers it unlike everywhere else — `GHA11_2` for a province and
    `GHA7.13_2` for a district, with no dot after the country code. Any code inferring the level
    from the shape of the id gets Ghana wrong.

    `LAO.1.2.1_1` is the seat named after the district holding it, which is what most ambiguous
    mentions turn out to be: candidates on one nesting chain rather than places in two locations.

    Ethiopia and Eritrea stand in for a country that lost territory and the state that holds it now.

    Poland stands in for a language that writes a unit as an adjective built from its seat's name.

    Canada carries a unit whose own name joins two places with `and`, and Czechia and Slovakia
    stand in for the successors of a state GADM no longer models.

    Ukraine carries GADM's placeholder identifier, which is the string `?` at both levels and
    therefore parents itself. Walking a unit's containers without a visited set never terminates.

    `VARNAME` carries the alternative spellings GADM publishes, pipe-separated, and is empty for
    most units. A name lookup that reads only `NAME` misses whichever spelling the mention used.
    """
    rows = [
        (
            "LAO",
            "Laos",
            "LAO.1_1",
            "Attapu",
            "Attopeu",
            "LAO.1.1_1",
            "Sanamxay",
            "",
            "LAO.1.1.1_1",
            "Ban Mai",
            "",
            "",
            box(0, 0, 1, 1),
        ),
        (
            "LAO",
            "Laos",
            "LAO.1_1",
            "Attapu",
            "Attopeu",
            "LAO.1.2_1",
            "Samakhixay",
            "",
            "LAO.1.2.1_1",
            "Samakhixay",
            "",
            "",
            box(1, 0, 2, 1),
        ),
        (
            "LAO",
            "Laos",
            "LAO.2_1",
            "Bokeo",
            "",
            "LAO.2.1_1",
            "Houayxay",
            "Ban Houayxay|Houei Sai",
            "LAO.2.1.1_1",
            "Ban Mai",
            "",
            "",
            box(3, 0, 4, 1),
        ),
        # A district sharing its name with a province elsewhere, which is the common homonym: 94 of
        # 111 ambiguous mentions in the workbook are a province and a same-named district.
        ("LAO", "Laos", "LAO.2_1", "Bokeo", "", "LAO.2.2_1", "Attapu", "", "", "", "", "", box(4, 0, 5, 1)),
        ("ZMB", "Zambia", "ZMB.1_1", "Central", "", "ZMB.1.1_1", "Kabwe", "", "", "", "", "", box(6, 0, 7, 1)),
        # A district whose name is short enough to be a syllable of another, which is what a dash
        # split has to refuse: `Ali-Shan` is one mountain and both halves are Chinese counties.
        ("LAO", "Laos", "LAO.2_1", "Bokeo", "", "LAO.2.4_1", "Xay", "", "", "", "", "", box(7, 0, 8, 1)),
        # `Nam Bay` is one edit from this district and names a bay, which is the collision an
        # approximate match has to refuse: `Manila Bay` reaches a barangay called Manlabay.
        ("LAO", "Laos", "LAO.2_1", "Bokeo", "", "LAO.2.3_1", "Nambak", "", "", "", "", "", box(5, 0, 6, 1)),
        # 75 GADM units carry a conjunction in their own name, which a split on `and` destroys.
        (
            "CAN",
            "Canada",
            "CAN.5_1",
            "Newfoundland and Labrador",
            "",
            "CAN.5.1_1",
            "Division No. 1",
            "",
            "",
            "",
            "",
            "",
            box(12, 0, 13, 1),
        ),
        # Two successors of a dissolved state, so a historical event has something to choose between.
        ("CZE", "Czechia", "CZE.1_1", "Praha", "", "CZE.1.1_1", "Praha 1", "", "", "", "", "", box(14, 0, 15, 1)),
        (
            "SVK",
            "Slovakia",
            "SVK.1_1",
            "Bratislavsky",
            "",
            "SVK.1.1_1",
            "Bratislava I",
            "",
            "",
            "",
            "",
            "",
            box(16, 0, 17, 1),
        ),
        # A name both successors publish, which is a tie rather than an answer.
        ("CZE", "Czechia", "CZE.1_1", "Praha", "", "CZE.1.2_1", "Nove Mesto", "", "", "", "", "", box(14, 1, 15, 2)),
        (
            "SVK",
            "Slovakia",
            "SVK.1_1",
            "Bratislavsky",
            "",
            "SVK.1.2_1",
            "Nove Mesto",
            "",
            "",
            "",
            "",
            "",
            box(16, 1, 17, 2),
        ),
        # One successor of a second dissolved state, so a lone candidate placing nothing is still no answer.
        ("HRV", "Croatia", "HRV.1_1", "Zagreb", "", "HRV.1.1_1", "Zagreb", "", "", "", "", "", box(18, 0, 19, 1)),
        # Kashmir is filed under a code of its own, with GADM naming the country administering it.
        ("IND", "India", "IND.1_1", "Kerala", "", "IND.1.1_1", "Kochi", "", "", "", "", "", box(20, 0, 21, 1)),
        (
            "Z01",
            "India",
            "Z01.1_1",
            "Jammu and Kashmir",
            "",
            "Z01.1.1_1",
            "Srinagar",
            "",
            "",
            "",
            "",
            "",
            box(22, 0, 23, 1),
        ),
        # A country that lost territory, and the state that holds it now.
        ("ETH", "Ethiopia", "ETH.1_1", "Tigray", "", "ETH.1.1_1", "Mekele", "", "", "", "", "", box(24, 0, 25, 1)),
        ("ERI", "Eritrea", "ERI.1_1", "Maekel", "", "ERI.1.1_1", "Asmara", "", "", "", "", "", box(26, 0, 27, 1)),
        # GADM writes an unnamed Ukrainian unit as `?` at both levels, so it comes out its own parent.
        ("UKR", "Ukraine", "?", "?", "", "?", "?", "", "", "", "", "", box(10, 0, 11, 1)),
        ("GHA", "Ghana", "GHA11_2", "Savannah", "", "GHA7.13_2", "Ga Central", "", "", "", "", "", box(8, 0, 9, 1)),
        # Poland, where a mention is the adjective built from the seat GADM publishes. `Rybnik`
        # and `Rybno` share the stem the adjective leaves, so one adjective settles nothing.
        (
            "POL",
            "Poland",
            "POL.1_1",
            "Podkarpackie",
            "",
            "POL.1.1_1",
            "Tarnobrzeg",
            "",
            "",
            "",
            "",
            "",
            box(28, 0, 29, 1),
        ),
        ("POL", "Poland", "POL.2_1", "Slaskie", "", "POL.2.1_1", "Rybnik", "", "", "", "", "", box(30, 0, 31, 1)),
        ("POL", "Poland", "POL.2_1", "Slaskie", "", "POL.2.2_1", "Rybno", "", "", "", "", "", box(31, 0, 32, 1)),
    ]
    columns = (
        "GID_0",
        "COUNTRY",
        "GID_1",
        "NAME_1",
        "VARNAME_1",
        "GID_2",
        "NAME_2",
        "VARNAME_2",
        "GID_3",
        "NAME_3",
        "GID_4",
        "NAME_4",
        "geometry",
    )

    return gpd.GeoDataFrame(dict(zip(columns, zip(*rows, strict=True), strict=True)), crs="EPSG:4326")


@pytest.fixture
def write_gadm_cache(tmp_path):
    """Return a callable writing a GADM-shaped GeoPackage into the cache, and giving back the root."""

    def write(units=None):
        directory = tmp_path / "gadm"
        directory.mkdir(exist_ok=True)
        (units if units is not None else toy_gadm()).to_file(directory / "gadm_410.gpkg", layer="gadm_410")

        return tmp_path

    return write


def toy_geo_disasters() -> gpd.GeoDataFrame:
    """A GeoPackage shaped like Geo-Disasters: one row per affected unit, keyed on ``DisNo.``.

    Two Laos events, one geocoded to provinces and one to districts, so a reader taking the name
    from a fixed column returns nothing for half the rows. The Zambian event is there for the ISO
    filter to exclude.
    """
    rows = [
        ("1991-0761-LAO", "LAO", 1, 2, "Savannakhet", None, box(0, 0, 1, 1)),
        ("1991-0761-LAO", "LAO", 1, 2, "Khammouan", None, box(1, 0, 2, 1)),
        ("2018-0339-LAO", "LAO", 2, 1, "Attapu", "Sanamxay", box(2, 0, 3, 1)),
        ("2007-0225-ZMB", "ZMB", 1, 1, "Central", None, box(6, 0, 7, 1)),
    ]
    columns = ("DisNo.", "ISO", "admin_level", "geocoding_q", "ADM1_NAME", "ADM2_NAME", "geometry")

    return gpd.GeoDataFrame(dict(zip(columns, zip(*rows, strict=True), strict=True)), crs="EPSG:4326")


@pytest.fixture
def write_geo_disasters_cache(tmp_path):
    """Return a callable writing a Geo-Disasters-shaped GeoPackage into the cache."""

    def write(locations=None):
        directory = tmp_path / "geo_disasters"
        directory.mkdir(exist_ok=True)
        frame = locations if locations is not None else toy_geo_disasters()
        frame.to_file(directory / "disaster_subnational_90_23.gpkg", layer="disaster_subnational_90_23")

        return tmp_path

    return write


@pytest.fixture
def write_rivers_cache(tmp_path):
    """Return a callable writing the processed river network a warm cache would hold."""

    def write(gdf, include_medium=False):
        rivers_dir = tmp_path / "rivers"
        rivers_dir.mkdir(parents=True, exist_ok=True)
        # The cache key is stated literally, so a wrong one fails rather than agreeing with itself.
        cutoff = 6 if include_medium else 5
        gdf.to_parquet(rivers_dir / f"rivers__stream_order_below={cutoff}.parquet")
        return tmp_path

    return write


class NetworkAccessError(RuntimeError):
    """Raised when a test that is not marked `network` opens a connection."""


def write_emdat_workbook(cache_dir, events):
    """Write the given events to a synthetic EM-DAT workbook in ``cache_dir``."""
    with pd.ExcelWriter(cache_dir / "emdat.xlsx") as writer:
        pd.DataFrame(list(events)).to_excel(writer, sheet_name="EM-DAT Data", index=False)

    return cache_dir


@pytest.fixture
def write_emdat_cache(tmp_path):
    """Return a callable writing the given events to a synthetic EM-DAT workbook in the cache."""
    return partial(write_emdat_workbook, tmp_path)


OUTBOUND_SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")


def _describe_target(args):
    for arg in args:
        if isinstance(arg, tuple) and arg and isinstance(arg[0], str):
            return arg[0]
        if isinstance(arg, str):
            return arg
    return "an external host"


def _refuse_outbound(self, *args, **kwargs):
    target = _describe_target(args)

    # socket.create_connection only closes on OSError, so an unclosed socket would surface as a
    # ResourceWarning and, under filterwarnings=error, mask this exception.
    self.close()

    raise NetworkAccessError(
        f"network access to {target} in a test not marked `network`. Write the file the loader expects "
        f"into its cache directory, or mark the test `network` and run with --run-network."
    )


def _refuse_lookup(*args, **kwargs):
    raise NetworkAccessError(
        f"DNS lookup of {_describe_target(args)} in a test not marked `network`. Write the file the "
        f"loader expects into its cache directory, or mark the test `network` and run with --run-network."
    )


# Captured before anything is patched, so a `network` test can be handed the real ones back.
CONNECTED_SOCKET_METHODS = {method: getattr(socket.socket, method) for method in OUTBOUND_SOCKET_METHODS}
RESOLVE = socket.getaddrinfo


@pytest.fixture(autouse=True, scope="session")
def block_network():
    """Refuse the socket for the whole session.

    Session scope rather than function scope because a module- or session-scoped fixture is set up
    before any function-scoped one, so a guard at function scope leaves fixture bodies free to reach
    upstream. A seeded cache that stops matching then downloads instead of failing.
    """
    with pytest.MonkeyPatch.context() as patch:
        for method in OUTBOUND_SOCKET_METHODS:
            patch.setattr(socket.socket, method, _refuse_outbound)

        # Refusing name resolution turns an offline run's DNS timeout into an immediate error.
        patch.setattr(socket, "getaddrinfo", _refuse_lookup)

        yield


@pytest.fixture(autouse=True)
def allow_network_when_marked(request, monkeypatch):
    """Hand the real socket back to a test marked `network`, which the session guard has taken."""
    if "network" not in request.keywords:
        return

    for method, connect in CONNECTED_SOCKET_METHODS.items():
        monkeypatch.setattr(socket.socket, method, connect)

    monkeypatch.setattr(socket, "getaddrinfo", RESOLVE)


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked `network`, which download from real upstream sources.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return

    skip_network = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


def toy_geonames() -> str:
    """Rows shaped like a GeoNames country dump: headerless, tab-separated, nineteen fields.

    Bacolod is published under three spellings and shares its name with a far smaller barangay,
    which is the collision that decides whether a written mention reaches the city or the hamlet.
    """
    rows = [
        (
            "1",
            "Bacolod",
            "Bacolod",
            "Bacolod City,Bakolod",
            "10.667",
            "122.95",
            "P",
            "PPL",
            "PH",
            "",
            "",
            "",
            "",
            "",
            "561875",
        ),
        ("2", "Bacolod", "Bacolod", "", "8.5", "124.1", "P", "PPL", "PH", "", "", "", "", "", "0"),
        ("3", "Iloilo", "Iloilo", "Ilo-ilo", "10.7", "122.567", "P", "PPLA", "PH", "", "", "", "", "", "457626"),
        ("4", "Sulu Sea", "Sulu Sea", "", "8.0", "120.0", "H", "SEA", "PH", "", "", "", "", "", "0"),
    ]
    padding = ("", "", "", "")

    return "\n".join("\t".join(row + padding) for row in rows) + "\n"


COUNTRY_INFO_HEADER = "#ISO\tISO3\tISO-Numeric\tfips\tCountry\n"


@pytest.fixture
def write_geonames_cache(tmp_path):
    """Return a callable writing a GeoNames-shaped dump into the cache, and giving back the root."""

    def write(rows=None, *, alpha2="PH", alpha3="PHL"):
        directory = tmp_path / "geonames"
        directory.mkdir(exist_ok=True)
        (directory / "countryInfo.txt").write_text(
            COUNTRY_INFO_HEADER + f"{alpha2}\t{alpha3}\t608\tRP\tPhilippines\n", encoding="utf-8"
        )
        with zipfile.ZipFile(directory / f"{alpha2}.zip", "w") as archive:
            archive.writestr(f"{alpha2}.txt", toy_geonames() if rows is None else rows)

        return tmp_path

    return write


@pytest.fixture
def write_osm_cache(tmp_path):
    """Return a callable writing cached Nominatim answers, and giving back the cache root.

    The layout is stated literally rather than derived from the loader, so a wrong cache path fails
    instead of agreeing with itself."""

    def write(rows, *, iso="LAO"):
        directory = tmp_path / "osm"
        directory.mkdir(exist_ok=True)
        pl.DataFrame(rows, schema=LOOKUP_COLUMNS, orient="row").write_parquet(directory / f"lookups__iso={iso}.parquet")

        return tmp_path

    return write
