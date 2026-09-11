import re
import shutil
import zipfile

import geopandas as gpd
import pandas as pd
import pytest

from shapely.geometry import box

from climate_risk import load_shapefile
from climate_risk.config.registry import CONFIG_ROOT, load_place
from climate_risk.config.schema import CountryConfig, RegionConfig
from climate_risk.data_functions.shapefiles_data_loader import load_place_boundary, repair_iso_codes
from climate_risk.exceptions import DataValidationError, ISOCodeValidationError
from climate_risk.geo.crs import GEOGRAPHIC_CRS
from tests.conftest import toy_coastline, toy_world, toy_world_needing_repair, toy_world_with_places


def test_unknown_shapefile_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="which should be one of"):
        load_shapefile("atlantis", tmp_path)


def test_warm_cache_reads_without_downloading(write_shapefile_cache):
    """Both the download and the extract step must no-op when the cache is already populated."""
    cache_dir = write_shapefile_cache("world", toy_world())

    world = load_shapefile("world", cache_dir, repair_ISO_codes=False)

    assert len(world) == 3


def test_only_the_archive_on_disk_is_enough_to_load(write_shapefile_cache):
    """A half-populated cache is the normal state after an interrupted run; extraction must resume."""
    cache_dir = write_shapefile_cache("world", toy_world())
    # Stated literally, not derived from the loader, so a wrong layout leaves the real files in
    # place and fails here rather than agreeing with itself.
    shutil.rmtree(cache_dir / "shapefiles" / "WB_countries_Admin0_10m")

    world = load_shapefile("world", cache_dir, repair_ISO_codes=False)

    assert len(world) == 3


def test_an_archive_holds_only_its_own_layer(write_shapefile_cache):
    """A boundary archive is written into the directory it archives, so it can swallow its neighbours.

    Including itself: zipping a growing file into itself never reaches EOF, and fills the disk
    instead of failing.
    """
    cache_dir = write_shapefile_cache("world", toy_world())
    write_shapefile_cache("laos", toy_world())

    with zipfile.ZipFile(cache_dir / "shapefiles" / "lao_admin_boundaries.shp.zip") as bundle:
        archived = bundle.namelist()

    assert all(name.startswith("lao_admin2.") for name in archived), archived


def test_the_repair_does_not_depend_on_how_the_caller_capitalised_it(write_shapefile_cache):
    """Every other branch lowercases `which`, so a capitalised call silently skipped the repair."""
    cache_dir = write_shapefile_cache("world", toy_world_needing_repair())

    world = load_shapefile("World", cache_dir)

    assert world.loc[world["WB_NAME"] == "France", "ISO_A3"].tolist() == ["FRA"]
    assert "-99" not in set(world["ISO_A3"])


def test_the_coastline_reads_the_continental_layer_not_whichever_comes_first(write_shapefile_cache):
    """The real archive ships six layers and the driver enumerates the Antarctic one first.

    A single-layer cache cannot show this: with one file in the directory, any layer index finds it.
    """
    cache_dir = write_shapefile_cache("coastline", toy_coastline())
    antarctic_ice = gpd.GeoDataFrame(geometry=[box(-180, -90, 180, -63)], crs=GEOGRAPHIC_CRS)
    antarctic_ice.to_file(cache_dir / "shapefiles" / "GSHHS_shp" / "f" / "GSHHS_f_L6.shp")

    coast = load_shapefile("coastline", cache_dir)

    _, southernmost, _, _ = coast.total_bounds
    assert southernmost > -60.0, "read an Antarctic layer instead of the shoreline"


def test_repair_supplies_the_missing_sovereign_codes():
    repaired = repair_iso_codes(toy_world_needing_repair())

    codes = dict(zip(repaired["WB_NAME"], repaired["ISO_A3"], strict=True))
    assert codes["France"] == "FRA"
    assert codes["Norway"] == "NOR"
    assert codes["Kosovo"] == "XKX"


def test_repair_drops_territories_that_would_double_count_their_owner():
    """Bonaire is labeled NLD; keeping it would give the Netherlands two geometries."""
    repaired = repair_iso_codes(toy_world_needing_repair())

    assert "Bonaire (Neth.)" not in set(repaired["WB_NAME"])
    assert "Johnston Atoll (US)" not in set(repaired["WB_NAME"])
    assert sorted(repaired["ISO_A3"]) == ["FRA", "NLD", "NOR", "NZL", "XKX"]


def test_repair_reindexes_from_zero():
    repaired = repair_iso_codes(toy_world_needing_repair())

    assert repaired.index.tolist() == list(range(len(repaired)))


def test_a_frame_without_the_naming_columns_is_an_error():
    """An upstream schema change should name the missing column, not fail on a bare KeyError."""
    with pytest.raises(DataValidationError, match="WB_NAME"):
        repair_iso_codes(toy_world())


@pytest.mark.parametrize(
    ("column", "renamed", "expected"),
    [("WB_NAME", {"Kosovo": "Republic of Kosovo"}, "Kosovo"), ("ISO_A3", {"UMI": "USA"}, "UMI")],
    ids=["by-name", "by-code"],
)
def test_a_repair_matching_nothing_is_an_error(column, renamed, expected):
    """Upstream renaming anything a repair targets must fail loudly, not silently skip it."""
    world = toy_world_needing_repair().replace({column: renamed})

    with pytest.raises(DataValidationError, match=expected):
        repair_iso_codes(world)


def test_duplicate_iso_codes_are_an_error():
    """One code per geometry is what every downstream join assumes."""
    world = toy_world_needing_repair()
    second_netherlands = world[world["WB_NAME"] == "Netherlands"].assign(WB_NAME="Netherlands (again)")
    doubled = pd.concat([world, second_netherlands], ignore_index=True)

    with pytest.raises(ISOCodeValidationError, match="NLD"):
        repair_iso_codes(doubled)


def test_a_place_reads_the_boundary_layer_its_config_names(write_shapefile_cache):
    """The archive unpacks flat, so reading the wrong admin level means reading a different file.

    Only the boundary archive is cached here, so falling back to the world slice cannot quietly work.
    """
    cache_dir = write_shapefile_cache("laos", toy_world())
    (cache_dir / "shapefiles" / "lao_admin0.shp").write_text("not a shapefile")

    boundary = load_place_boundary(load_place("lao"), cache_dir)

    assert sorted(boundary["ISO_A3"]) == ["AAA", "BBB", "CCC"]


def test_a_country_without_a_boundary_is_sliced_out_of_the_repaired_world(write_shapefile_cache):
    """France is unlabeled upstream, so a place named FRA is only findable once the repair has run."""
    cache_dir = write_shapefile_cache("world", toy_world_needing_repair())
    place = CountryConfig(iso3="FRA", name="France")

    boundary = load_place_boundary(place, cache_dir)

    assert boundary["WB_NAME"].tolist() == ["France"]


def test_a_region_is_sliced_by_every_member(write_shapefile_cache):
    cache_dir = write_shapefile_cache("world", toy_world_needing_repair())
    place = RegionConfig(key="north", name="North", members=("NOR", "NLD"))

    boundary = load_place_boundary(place, cache_dir)

    assert sorted(boundary["WB_NAME"]) == ["Netherlands", "Norway"]


def test_a_place_the_world_file_does_not_know_is_an_error(write_shapefile_cache):
    """An empty slice grids to zero points, and the empty panel only surfaces much further downstream."""
    cache_dir = write_shapefile_cache("world", toy_world_needing_repair())
    place = CountryConfig(iso3="ZWE", name="Zimbabwe")

    with pytest.raises(DataValidationError, match=re.escape("no geometry for ['ZWE']")):
        load_place_boundary(place, cache_dir)


SHIPPED_PLACE_KEYS = sorted(path.stem for path in (CONFIG_ROOT / "places").glob("*.toml"))

# Where `toy_world_with_places` puts each shipped country. Stated here rather than read from the
# fixture, so a place resolving to the wrong ISO fails instead of agreeing with itself.
SHIPPED_BOUNDS = {"lao": (20.0, 21.0), "zmb": (23.0, 24.0), "cri": (29.5, 30.5)}


@pytest.mark.parametrize("key", SHIPPED_PLACE_KEYS)
def test_every_shipped_country_resolves_to_geometry_from_its_config_alone(write_shapefile_cache, key):
    """A country the loaders have never heard of has to work from its config file and nothing else.

    Laos reads its own boundary archive and the other two are sliced out of the world file, so this
    covers both paths. Checking the bounds is what makes it more than a smoke test: a place resolving
    to the wrong ISO still returns geometry, just somewhere else.
    """
    cache_dir = write_shapefile_cache("world", toy_world_with_places())
    # Laos declares its own archive, so seed that too rather than letting the fetch reach the network.
    write_shapefile_cache("laos", toy_world_with_places())

    place = load_place(key)

    boundary = load_place_boundary(place, cache_dir)

    assert not boundary.empty
    if isinstance(place, CountryConfig) and place.boundary is None:
        west, east = SHIPPED_BOUNDS[key]
        assert boundary.total_bounds[0] >= west
        assert boundary.total_bounds[2] <= east
