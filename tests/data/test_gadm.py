import json
import re
import sqlite3

from pathlib import Path

import geopandas as gpd
import polars as pl
import pytest

from climate_risk.data import gadm
from climate_risk.data.gadm import (
    GADM,
    administered_territories,
    gadm_dir,
    gadm_path,
    load_admin_units,
    load_units_in_country,
)
from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.exceptions import DataValidationError


def test_the_geopackage_is_looked_for_under_the_cache(tmp_path):
    assert gadm_dir(tmp_path) == tmp_path / "gadm"


def test_an_absent_geopackage_names_the_path_and_where_to_get_it(tmp_path):
    """The license forbids fetching it, so the error is the user's only instruction."""
    with pytest.raises(NotImplementedError) as raised:
        gadm_path(tmp_path)

    message = str(raised.value)
    assert str(tmp_path / "gadm" / "gadm_410.gpkg") in message
    assert "https://gadm.org" in message


def test_a_placed_geopackage_is_returned(tmp_path):
    placed = tmp_path / "gadm" / "gadm_410.gpkg"
    placed.parent.mkdir()
    placed.touch()

    assert gadm_path(tmp_path) == placed


def test_the_declaration_carries_the_non_commercial_restriction():
    """Generic checks live in test_source; what is specific here is the restriction itself, which
    binds every figure built from these boundaries."""
    assert "non-commercial" in GADM.license.lower()
    assert "gadm.org" in GADM.citation


def test_a_unit_above_the_finest_level_is_read_as_one_polygon(write_gadm_cache):
    """GADM stores one row per district, so a province is the union of several rows.

    Attapu spans two districts in the fixture; taking the first row would return half its area.
    """
    cache_dir = write_gadm_cache()

    units = load_admin_units([("LAO.1_1", 1)], cache_dir)

    assert units["gid"].tolist() == ["LAO.1_1"]
    assert units["name"].tolist() == ["Attapu"]
    assert units["admin_level"].tolist() == [1]
    # The two district boxes span x 0..2; either row alone would be half of it.
    assert units.geometry.iloc[0].bounds == (0.0, 0.0, 2.0, 1.0)


def test_levels_may_be_mixed_in_one_request(write_gadm_cache):
    """EM-DAT codes some events to provinces and others to districts, often within one country."""
    cache_dir = write_gadm_cache()

    units = load_admin_units([("LAO.2_1", 1), ("LAO.1.1_1", 2)], cache_dir).set_index("gid")

    assert units.loc["LAO.2_1", "admin_level"] == 1
    assert units.loc["LAO.1.1_1", "admin_level"] == 2
    assert units.loc["LAO.1.1_1", "name"] == "Sanamxay"


def test_an_id_gadm_does_not_hold_is_an_error_not_a_dropped_row(write_gadm_cache):
    """Returning fewer rows than asked for would quietly shrink an event's footprint."""
    cache_dir = write_gadm_cache()

    with pytest.raises(DataValidationError, match=re.escape("LAO.99_1")):
        load_admin_units([("LAO.1_1", 1), ("LAO.99_1", 1)], cache_dir)


def test_a_level_gadm_is_not_read_at_is_rejected(write_gadm_cache):
    """Units are read at levels 1 through 4; anything else would silently match nothing."""
    cache_dir = write_gadm_cache()

    with pytest.raises(DataValidationError, match="read at levels"):
        load_admin_units([("LAO", 0)], cache_dir)


def test_the_same_id_asked_for_twice_is_read_once(write_gadm_cache):
    cache_dir = write_gadm_cache()

    units = load_admin_units([("LAO.1_1", 1), ("LAO.1_1", 1)], cache_dir)

    assert len(units) == 1


def test_a_country_reads_back_every_unit_it_holds_at_a_level(write_gadm_cache):
    """A search by geometry needs the candidates, which is the opposite question to `by id`."""
    cache_dir = write_gadm_cache()

    provinces = load_units_in_country("LAO", 1, cache_dir)

    assert sorted(provinces["gid"]) == ["LAO.1_1", "LAO.2_1"]
    assert set(provinces["admin_level"]) == {1}


def test_a_country_reads_back_the_villages_below_its_districts(write_gadm_cache):
    """Written locations name units below adm2 — 18% of the places that resolve are adm3 or adm4 —
    and scoring a point against one needs its polygon."""
    cache_dir = write_gadm_cache()

    villages = load_units_in_country("LAO", 3, cache_dir)

    assert sorted(villages["gid"]) == ["LAO.1.1.1_1", "LAO.1.2.1_1", "LAO.2.1.1_1"]


def test_a_country_not_divided_at_a_level_reads_back_empty(write_gadm_cache):
    """GADM stores a blank identifier rather than omitting the row, so dissolving without dropping
    those yields one phantom unit whose id is the empty string and whose polygon is the country."""
    cache_dir = write_gadm_cache()

    assert load_units_in_country("LAO", 4, cache_dir).empty


def test_a_unit_below_adm2_resolves_to_its_polygon(write_gadm_cache):
    cache_dir = write_gadm_cache()

    units = load_admin_units([("LAO.1.1.1_1", 3)], cache_dir)

    assert list(units["name"]) == ["Ban Mai"]


def test_a_country_reads_back_its_districts(write_gadm_cache):
    """Level decides which column is read, and reading the wrong one returns the wrong geography."""
    cache_dir = write_gadm_cache()

    districts = load_units_in_country("LAO", 2, cache_dir)

    assert sorted(districts["name"]) == ["Attapu", "Houayxay", "Nambak", "Samakhixay", "Sanamxay", "Xay"]


def test_a_province_spanning_several_districts_is_one_row(write_gadm_cache):
    """The table stores districts, so Attapu is two rows that have to become one polygon."""
    cache_dir = write_gadm_cache()

    attapu = load_units_in_country("LAO", 1, cache_dir).set_index("gid").loc["LAO.1_1"]

    assert attapu["geometry"].bounds == (0.0, 0.0, 2.0, 1.0)


def test_only_the_country_asked_for_comes_back(write_gadm_cache):
    """Matching a footprint against another country's units places it somewhere else entirely."""
    cache_dir = write_gadm_cache()

    assert load_units_in_country("ZMB", 1, cache_dir)["gid"].tolist() == ["ZMB.1_1"]


def test_a_country_gadm_does_not_hold_is_empty_but_still_carries_the_crs(write_gadm_cache):
    cache_dir = write_gadm_cache()

    units = load_units_in_country("CRI", 1, cache_dir)

    assert units.empty
    assert units.crs == "EPSG:4326"


def test_a_level_gadm_is_not_read_at_is_rejected_for_a_country(write_gadm_cache):
    cache_dir = write_gadm_cache()

    with pytest.raises(DataValidationError, match="read at levels"):
        load_units_in_country("LAO", 0, cache_dir)


# The GeoPackage is non-commercial and hand-placed, so it is absent on CI and on a fresh clone.
REAL_CACHE_DIR = Path(__file__).parents[2] / "data"


@pytest.mark.requires_gadm
@pytest.mark.requires_emdat
@pytest.mark.skipif(not (REAL_CACHE_DIR / "gadm" / "gadm_410.gpkg").exists(), reason="needs the GADM GeoPackage")
@pytest.mark.skipif(not (REAL_CACHE_DIR / "emdat.xlsx").exists(), reason="needs the licensed EM-DAT workbook")
def test_every_gid_em_dat_references_resolves_against_gadm():
    """The whole event-unit table rests on this: a gid that does not resolve is a lost footprint.

    Synthetic fixtures cannot catch a version mismatch, because they are written to agree. This is
    the only check that the GeoPackage on disk is the one EM-DAT coded against.
    """
    events = load_emdat_events(REAL_CACHE_DIR).filter(pl.col("GADM Admin Units").str.strip_chars().str.len_chars() > 2)
    referenced = {
        (unit["gid_2"], 2) if "gid_2" in unit else (unit["gid_1"], 1)
        for raw in events["GADM Admin Units"]
        for unit in json.loads(raw)
    }
    assert len(referenced) > 10_000, "the workbook should reference thousands of units"

    # Set membership, not geometry: the claim is that the ids exist, and assembling twelve thousand
    # polygons to answer it turns a two-second check into a four-minute one.
    with sqlite3.connect(REAL_CACHE_DIR / "gadm" / "gadm_410.gpkg") as gadm:
        held = {
            (gid, level)
            for level, column in ((1, "GID_1"), (2, "GID_2"))
            for (gid,) in gadm.execute(f"SELECT DISTINCT {column} FROM gadm_410 WHERE {column} IS NOT NULL")
        }

    assert referenced <= held, f"GADM holds no unit for {sorted(referenced - held)[:5]}"


@pytest.mark.parametrize(("gid", "level", "name"), [("GHA11_2", 1, "Savannah"), ("GHA7.13_2", 2, "Ga Central")])
def test_an_id_that_does_not_state_its_own_level_still_resolves(write_gadm_cache, gid, level, name):
    """GADM numbers Ghana without the dot every other country has, so the id cannot imply the level.

    Inferring it from the string reads `GHA11_2` as a country and drops a real province.
    """
    cache_dir = write_gadm_cache()

    units = load_admin_units([(gid, level)], cache_dir)

    assert units["name"].tolist() == [name]
    assert units["admin_level"].tolist() == [level]


def test_a_request_spanning_several_chunks_returns_all_of_them(write_gadm_cache, monkeypatch):
    """A real request names thousands of ids and is split across reads; only the last would survive
    a concat that overwrote rather than appended, and no fixture is large enough to notice."""
    monkeypatch.setattr("climate_risk.data.gadm.GID_CHUNK", 1)
    cache_dir = write_gadm_cache()

    units = load_admin_units([("LAO.1_1", 1), ("LAO.2_1", 1), ("GHA11_2", 1)], cache_dir)

    assert sorted(units["gid"]) == ["GHA11_2", "LAO.1_1", "LAO.2_1"]


def test_the_units_carry_the_geopackage_crs(write_gadm_cache):
    """Geometry with no CRS reprojects silently to the wrong place when joined to anything else."""
    cache_dir = write_gadm_cache()

    units = load_admin_units([("LAO.1_1", 1)], cache_dir)

    assert units.crs == "EPSG:4326"


def test_an_empty_request_still_carries_the_crs(write_gadm_cache):
    """A CRS-less frame concatenates with projected geometry silently and reprojects it wrongly.

    The empty result has to be usable in the same places the populated one is.
    """
    cache_dir = write_gadm_cache()

    units = load_admin_units([], cache_dir)

    assert units.empty
    assert units.crs == "EPSG:4326"
    assert list(units.columns) == ["gid", "name", "admin_level", "geometry"]


def test_a_country_names_the_disputed_territory_it_administers(write_gadm_cache):
    """Kashmir is filed under a code of its own, so an Indian event named there reaches nothing
    unless the country's own code is read together with what it administers."""
    assert administered_territories("IND", write_gadm_cache()) == ("Z01",)


def test_a_country_administering_nothing_names_nothing(write_gadm_cache):
    assert administered_territories("LAO", write_gadm_cache()) == ()


def test_a_country_read_twice_reads_the_same_units(write_gadm_cache):
    """Reading a country's polygons out of the GeoPackage takes seconds, so the result is cached;
    a cache hit has to carry the same units and the same geometry as the build."""
    cache_dir = write_gadm_cache()

    built = load_units_in_country("LAO", 2, cache_dir, force_reload=True)
    from_cache = load_units_in_country("LAO", 2, cache_dir)

    assert list(from_cache["gid"]) == list(built["gid"])
    assert from_cache.geometry.equals(built.geometry)
    assert from_cache.crs == built.crs


def test_one_layer_does_not_answer_for_another(write_gadm_cache):
    """`layer` chooses which table is read, so two layers are two different questions. Keyed alike
    they collide, and the second caller is handed the first one's geometry."""
    cache_dir = write_gadm_cache()
    archive = cache_dir / "gadm" / "gadm_410.gpkg"
    everything = gpd.read_file(archive, layer="gadm_410")
    everything[everything["GID_0"] == "ZMB"].to_file(archive, layer="zambia_only")

    from_the_whole_world = load_units_in_country("LAO", 2, cache_dir)
    from_zambia_only = load_units_in_country("LAO", 2, cache_dir, layer="zambia_only")

    assert not from_the_whole_world.empty
    assert from_zambia_only.empty, "a layer holding no Laos must not answer with another layer's units"


def test_changing_which_columns_a_level_reads_turns_the_cached_geometry_over(write_gadm_cache, monkeypatch):
    """The columns each level is read from live in a table the builder consults, which its own source
    does not show. Pointing level 2 at level 1's columns returns provinces where districts were
    cached, and nothing else in the key would say so."""
    cache_dir = write_gadm_cache()
    districts = load_units_in_country("LAO", 2, cache_dir)

    monkeypatch.setattr(gadm, "GID_COLUMNS", {**gadm.GID_COLUMNS, 2: ("GID_1", "NAME_1")})
    reread = load_units_in_country("LAO", 2, cache_dir)

    assert sorted(reread["gid"]) != sorted(districts["gid"])
