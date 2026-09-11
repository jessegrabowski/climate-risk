import re

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from shapely.geometry import Polygon, box

from climate_risk.data.geo_disasters import (
    AGREEMENT_LEVELS,
    GEO_DISASTERS,
    RESOLVED_COLUMNS,
    RESOLVED_DTYPES,
    compare_event_units,
    event_unit_ids,
    geo_disasters_dir,
    geo_disasters_path,
    load_event_footprints,
    load_event_locations,
    load_resolved_units,
    normalize_unit_name,
    resolve_to_gadm,
    unit_names,
)
from climate_risk.exceptions import DataValidationError
from tests.conftest import toy_gadm, toy_geo_disasters


def test_the_geopackage_is_looked_for_under_the_cache(tmp_path):
    assert geo_disasters_dir(tmp_path) == tmp_path / "geo_disasters"


def test_an_absent_geopackage_names_the_path_and_where_to_get_it(tmp_path):
    """Nothing may fetch this, so the error is the user's only instruction."""
    with pytest.raises(NotImplementedError) as raised:
        geo_disasters_path(tmp_path)

    message = str(raised.value)
    assert str(tmp_path / "geo_disasters" / "disaster_subnational_90_23.gpkg") in message
    assert "10.5281/zenodo.15487667" in message


def test_the_declaration_splits_the_license_by_what_it_covers():
    """The geometry is non-commercial and the attributes are not; a figure built from one is
    constrained where the other is free, so both halves have to survive in the declaration."""
    license = GEO_DISASTERS.license.lower()

    assert "non-commercial" in license
    assert "cc-by-4.0" in license
    assert "zenodo.15487667" in GEO_DISASTERS.citation


def test_locations_come_back_keyed_on_the_em_dat_id(write_geo_disasters_cache):
    """`DisNo.` is the only key shared with EM-DAT, and one event spans several rows."""
    cache_dir = write_geo_disasters_cache()

    locations = load_event_locations(cache_dir)

    assert sorted(locations["DisNo."].unique()) == ["1991-0761-LAO", "2007-0225-ZMB", "2018-0339-LAO"]
    assert (locations["DisNo."] == "1991-0761-LAO").sum() == 2


def test_one_country_can_be_read_without_the_rest(write_geo_disasters_cache):
    """The real table is 45,000 rows across 206 countries, and a comparison wants one of them."""
    cache_dir = write_geo_disasters_cache()

    locations = load_event_locations(cache_dir, iso="LAO")

    assert set(locations["ISO"]) == {"LAO"}
    assert len(locations) == 3


def test_an_iso_the_table_does_not_hold_reads_back_empty(write_geo_disasters_cache):
    """A country with no geocoded events is the normal case, not an error."""
    cache_dir = write_geo_disasters_cache()

    locations = load_event_locations(cache_dir, iso="CRI")

    assert locations.empty
    assert list(locations.columns) == ["DisNo.", "ISO", "admin_level", "geocoding_q", "ADM1_NAME", "ADM2_NAME"]


def test_a_unit_is_named_at_the_level_it_was_geocoded_to(write_geo_disasters_cache):
    """A district row carries its province name too, so reading ADM1_NAME everywhere silently
    coarsens every level-2 location to the province containing it."""
    cache_dir = write_geo_disasters_cache()
    locations = load_event_locations(cache_dir, iso="LAO")

    named = unit_names(locations)

    by_event = dict(zip(locations["DisNo."], named, strict=True))
    assert by_event["2018-0339-LAO"] == "Sanamxay"
    assert set(named) == {"Savannakhet", "Khammouan", "Sanamxay"}


def test_geometry_is_left_on_disk(write_geo_disasters_cache):
    """The polygons are the non-commercial half of the license, and nothing here needs them."""
    cache_dir = write_geo_disasters_cache()

    locations = load_event_locations(cache_dir)

    assert "geometry" not in locations.columns


@pytest.mark.parametrize(
    ("published", "other"),
    [
        ("Attapu", "attapu"),
        ("Bolikhamxai", "Bolikhamxai "),
        ("Xekong", "Xékong"),
        ("Xai-somboun", "Xaisomboun"),
        ("Da Nang", "Đà Nẵng"),
        ("Lodz", "Łódź"),
    ],
    ids=["casing", "whitespace", "accent", "hyphen", "d-with-stroke", "l-with-stroke"],
)
def test_the_same_unit_spelled_two_ways_normalizes_alike(published, other):
    """GADM and GAUL differ in casing, spacing, accents and hyphens for units that are the same one,
    and a literal comparison would report every one of these as the sources disagreeing. A stroked
    letter carries no combining mark, so stripping accents leaves `Đ` and `ł` standing and `Đà Nẵng`
    never meets `Da Nang`. GADM holds both, in different scripts, so one language's fix is not enough."""
    assert normalize_unit_name(published) == normalize_unit_name(other)


def test_units_that_are_genuinely_different_stay_different():
    """Normalization that collapses too far reports agreement everywhere and validates nothing."""
    assert normalize_unit_name("Savannakhet") != normalize_unit_name("Khammouan")


def test_a_different_romanisation_is_not_reconciled():
    """Normalization handles typography, not spelling. `Xiangkhouang` and `Xiangkhoang` are one
    province under two romanisations, and normalizing leaves them different — which is why the
    agreement report matches identifiers and this function is not in its path."""
    assert normalize_unit_name("Xiangkhouang") != normalize_unit_name("Xiangkhoang")


def test_unit_ids_are_collected_per_event():
    """Both sides of the comparison come through here, and an event spans several units. The point
    and country tiers of `event_geography` carry no unit at all, and those events contribute an
    empty set rather than a set holding a null that would match nothing."""
    units = pd.DataFrame(
        {
            "DisNo.": ["1991-0761-LAO", "1991-0761-LAO", "2018-0339-LAO", "2020-0001-LAO"],
            "gid": ["LAO.15_1", "LAO.6_1", "LAO.1.1_1", None],
        }
    )

    assert event_unit_ids(units) == {
        "1991-0761-LAO": {"LAO.15_1", "LAO.6_1"},
        "2018-0339-LAO": {"LAO.1.1_1"},
        "2020-0001-LAO": set(),
    }


@pytest.mark.parametrize(
    ("em_dat", "geo_disasters", "expected"),
    [
        ({"LAO.6_1", "LAO.7_1"}, {"LAO.6_1", "LAO.7_1"}, "exact"),
        ({"LAO.15_1", "LAO.17_1"}, {"LAO.15_1"}, "partial"),
        ({"LAO.15_1"}, {"LAO.1_1"}, "disjoint"),
        (set(), {"LAO.16_1", "LAO.1_1"}, "gained"),
        ({"LAO.2_1"}, set(), "unmatched"),
    ],
    ids=["exact", "partial", "disjoint", "gained", "unmatched"],
)
def test_an_event_is_classified_by_how_the_two_geocodings_overlap(em_dat, geo_disasters, expected):
    report = compare_event_units(em_dat={"E": em_dat}, geo_disasters={"E": geo_disasters})

    assert report["agreement"].tolist() == [expected]


def test_an_event_neither_source_geocoded_is_left_out():
    """Two thirds of EM-DAT carries no units at all; reporting them would bury the comparison."""
    report = compare_event_units(em_dat={"E": set()}, geo_disasters={"E": set()})

    assert report.empty
    assert list(report.columns) == ["DisNo.", "agreement", "em_dat_units", "geo_disasters_units", "shared_units"]


def test_identifiers_are_compared_literally():
    """`normalize_unit_name` strips the punctuation that separates a GADM identifier's levels, so
    the district `LAO.1.1_1` and the province `LAO.11_1` both reduce to `lao111`. Putting the
    identifiers through it would merge two different units into an exact agreement."""
    report = compare_event_units(em_dat={"E": {"LAO.11_1"}}, geo_disasters={"E": {"LAO.1.1_1"}})

    assert normalize_unit_name("LAO.11_1") == normalize_unit_name("LAO.1.1_1"), "the collision is real"
    assert report["agreement"].tolist() == ["disjoint"]


def test_a_province_and_a_district_inside_it_do_not_match():
    """An identifier names a level as well as a place. Every event the two sources describe at
    different resolutions lands here, and it reads as `disjoint` — the same verdict as two genuinely
    different provinces, which is a limit of the report rather than a claim about the event."""
    report = compare_event_units(em_dat={"E": {"PHL.19_1"}}, geo_disasters={"E": {"PHL.19.5_1"}})

    assert report["agreement"].tolist() == ["disjoint"]
    assert report["shared_units"].tolist() == [0]


def test_the_counts_describe_each_side_and_the_overlap():
    report = compare_event_units(em_dat={"E": {"LAO.15_1", "LAO.17_1"}}, geo_disasters={"E": {"LAO.15_1", "LAO.1_1"}})

    assert report.loc[0, "em_dat_units"] == 2
    assert report.loc[0, "geo_disasters_units"] == 2
    assert report.loc[0, "shared_units"] == 1


def test_every_event_either_source_geocoded_appears_once():
    """A join done the wrong way round silently drops whichever side is absent."""
    report = compare_event_units(
        em_dat={"A": {"LAO.1_1"}, "B": {"LAO.2_1"}},
        geo_disasters={"B": {"LAO.2_1"}, "C": {"LAO.9_1"}},
    )

    assert report["DisNo."].tolist() == ["A", "B", "C"]


# The GeoPackage is non-commercial and hand-placed, so it is absent on CI and on a fresh clone.
REAL_CACHE_DIR = Path(__file__).parents[2] / "data"
REAL_GEOPACKAGE = REAL_CACHE_DIR / "geo_disasters" / "disaster_subnational_90_23.gpkg"


def test_footprints_spanning_two_countries_are_refused(tmp_path):
    """Only the first country's units would be read, and every other country's footprint would be
    placed against them — wrong geography, with nothing to show for it."""
    with pytest.raises(DataValidationError, match=re.escape("['LAO', 'ZMB']")):
        resolve_to_gadm(toy_geo_disasters(), tmp_path)


def test_no_footprints_resolve_to_an_empty_frame_of_the_right_shape(tmp_path):
    """A country Geo-Disasters never geocoded is ordinary, and the caller concatenates the result —
    so the empty frame needs the dtypes a populated one has, or it drags every column it is
    concatenated with back to object."""
    empty = toy_geo_disasters().iloc[:0]

    resolved = resolve_to_gadm(empty, tmp_path)

    assert resolved.empty
    assert list(resolved.columns) == RESOLVED_COLUMNS
    assert dict(resolved.dtypes.astype(str)) == RESOLVED_DTYPES


def test_footprints_come_back_with_their_geometry(write_geo_disasters_cache):
    """The counterpart to the attribute-only read: resolving needs the polygons."""
    cache_dir = write_geo_disasters_cache()

    footprints = load_event_footprints(cache_dir, iso="LAO")

    assert not footprints.geometry.isna().any()
    assert footprints.crs == "EPSG:4326"


def test_every_classification_is_one_of_the_declared_levels():
    """`AGREEMENT_LEVELS` is the published vocabulary; a verdict outside it is one no caller can
    branch on."""
    report = compare_event_units(
        em_dat={
            "exact": {"LAO.1_1"},
            "partial": {"LAO.1_1", "LAO.2_1"},
            "disjoint": {"LAO.1_1"},
            "unmatched": {"LAO.1_1"},
        },
        geo_disasters={"exact": {"LAO.1_1"}, "partial": {"LAO.1_1"}, "disjoint": {"LAO.9_1"}, "gained": {"LAO.1_1"}},
    )

    assert set(report["agreement"]) == set(AGREEMENT_LEVELS)


def test_the_resolved_dtypes_do_not_depend_on_which_levels_matched(write_gadm_cache):
    """An admin level that places nothing contributes an all-object empty frame to the concatenation,
    which would leave `admin_level` an object column for one country and an integer for the next —
    a schema that varies with the data, and only where a level happened to come back empty."""
    cache_dir = write_gadm_cache()
    lao = toy_geo_disasters().query("ISO == 'LAO'")

    both_levels = resolve_to_gadm(lao, cache_dir)
    provinces_only = resolve_to_gadm(lao[lao["admin_level"] == 1], cache_dir)

    assert not both_levels.empty and not provinces_only.empty, "both cases must place something"
    pd.testing.assert_series_equal(both_levels.dtypes, provinces_only.dtypes)
    assert both_levels["admin_level"].dtype == "Int64", "an admin level is a number the model compares"


def footprints_over(*rectangles, level=2, disno="1999-0001-LAO", quality=None):
    """One event's Geo-Disasters footprints, covering arbitrary rectangles of the toy GADM country."""
    return gpd.GeoDataFrame(
        {
            "DisNo.": [disno] * len(rectangles),
            "ISO": ["LAO"] * len(rectangles),
            "admin_level": [level] * len(rectangles),
            "geocoding_q": list(quality or [1] * len(rectangles)),
            "ADM1_NAME": ["Attapu"] * len(rectangles),
            "ADM2_NAME": ["Sanamxay"] * len(rectangles),
            "geometry": [box(*rectangle) for rectangle in rectangles],
        },
        crs="EPSG:4326",
    )


def test_a_footprint_spanning_two_units_is_placed_in_both(write_gadm_cache):
    """One GAUL unit routinely covers several GADM ones — where GAUL's admin-1 is a region and
    GADM's is a province, a single footprint holds five or six. Naming only the largest credits the
    whole event to one unit and records a zero for the others."""
    cache_dir = write_gadm_cache()

    resolved = resolve_to_gadm(footprints_over((0, 0, 2, 1)), cache_dir)

    assert set(resolved["gid"]) == {"LAO.1.1_1", "LAO.1.2_1"}, "both districts lie inside the footprint"


@pytest.mark.parametrize(
    ("east_edge", "expected"),
    [(1.6, {"LAO.1.1_1", "LAO.1.2_1"}), (1.4, {"LAO.1.1_1"})],
    ids=["60% of the second district", "40% of the second district"],
)
def test_a_unit_joins_when_most_of_it_lies_inside(write_gadm_cache, east_edge, expected):
    """Membership is the share of the unit inside the footprint, not the share of the footprint the
    unit covers. The second district is the same size either way; what changes is how much of it the
    footprint holds."""
    cache_dir = write_gadm_cache()

    resolved = resolve_to_gadm(footprints_over((0, 0, east_edge, 1)), cache_dir)

    assert set(resolved["gid"]) == expected


def test_a_unit_straddling_two_footprints_of_one_event_is_placed(write_gadm_cache):
    """The two halves belong to one event, so the unit is measured against their union. Testing each
    footprint alone finds 40% twice and places the unit nowhere, losing geography the event has."""
    cache_dir = write_gadm_cache()
    # Two slices of LAO.1.2_1, box(1, 0, 2, 1), covering 40% of it each and 80% together.
    straddling = footprints_over((1.0, 0, 1.4, 1), (1.6, 0, 2.0, 1))

    resolved = resolve_to_gadm(straddling, cache_dir)

    assert set(resolved["gid"]) == {"LAO.1.2_1"}


def test_a_unit_inside_two_footprints_of_one_event_is_named_once(write_gadm_cache):
    """`build_aggregation_from_overlaps` refuses a repeated unit outright, because the same overlap
    listed twice doubles the ground it stands for."""
    cache_dir = write_gadm_cache()
    covered_twice = footprints_over((0, 0, 1, 1), (0, 0, 0.9, 1))

    resolved = resolve_to_gadm(covered_twice, cache_dir)

    assert resolved["gid"].tolist() == ["LAO.1.1_1"]
    assert not resolved.duplicated(subset=["DisNo.", "gid"]).any()


def test_a_footprint_that_projects_to_an_invalid_polygon_is_repaired(write_gadm_cache):
    """A footprint crossing the antimeridian is valid in lat/lon and self-intersecting once
    projected, and 19 of Fiji's 108 are. Unioning one raises out of GEOS rather than returning a
    poor answer, so a whole country's geography is lost to a topology error."""
    cache_dir = write_gadm_cache()
    # One event, two footprints, the first tracing the districts' extent twice over so its ring
    # doubles back on itself. Unioning it with the second is what GEOS refuses.
    self_overlapping = footprints_over((0, 0, 2, 1), (0, 0, 1, 1))
    self_overlapping.geometry = [
        Polygon([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0.5), (2, 0.5), (2, 1), (0, 1)]),
        box(0, 0, 1, 1),
    ]
    assert not self_overlapping.geometry.is_valid.all(), "the fixture has to be invalid to test anything"

    resolved = resolve_to_gadm(self_overlapping, cache_dir)

    assert set(resolved["gid"]) == {"LAO.1.1_1", "LAO.1.2_1"}


def test_an_event_geocoded_at_both_levels_keeps_both(write_gadm_cache):
    """Footprints are dissolved per level, not per event. Dissolving across levels would union a
    province with a district somewhere else and place the pair against whichever level ran first."""
    cache_dir = write_gadm_cache()
    at_both = pd.concat([footprints_over((3, 0, 5, 1), level=1), footprints_over((0, 0, 1, 1), level=2)])

    resolved = resolve_to_gadm(gpd.GeoDataFrame(at_both, crs="EPSG:4326"), cache_dir)

    assert dict(zip(resolved["gid"], resolved["admin_level"], strict=True)) == {"LAO.2_1": 1, "LAO.1.1_1": 2}


def test_the_worse_quality_flag_survives_the_dissolve(write_gadm_cache):
    """Geo-Disasters propagates its per-location flag worst-case to the event, and collapsing the
    footprints must not quietly improve it."""
    cache_dir = write_gadm_cache()
    mixed = footprints_over((0, 0, 1, 1), (0, 0, 0.9, 1), quality=[1, 3])

    resolved = resolve_to_gadm(mixed, cache_dir)

    assert resolved["geocoding_q"].tolist() == [3]


def test_overlap_is_the_share_of_the_footprint_each_unit_covers(write_gadm_cache):
    """The two ratios answer different questions and only one is membership. `overlap` reports how
    much of the event this unit accounts for; how much of the unit the event reached decides whether
    it is named at all."""
    cache_dir = write_gadm_cache()

    resolved = resolve_to_gadm(footprints_over((0, 0, 2, 1)), cache_dir).set_index("gid")

    assert resolved["overlap"].sum() == pytest.approx(1.0), "two equal halves of the footprint"
    assert resolved.loc["LAO.1.1_1", "overlap"] == pytest.approx(0.5)


def test_a_footprint_that_only_borders_a_unit_is_not_placed_in_it(write_gadm_cache):
    """Two polygons meeting along an edge intersect in a line of zero area. Taking the largest
    overlap regardless would answer with a unit the footprint lies wholly outside, and answer it as
    confidently as a real match."""
    cache_dir = write_gadm_cache()
    units = toy_gadm()
    houayxay = units[units["GID_2"] == "LAO.2.1_1"].geometry.iloc[0]
    beside_houayxay = gpd.GeoDataFrame(
        {
            "DisNo.": ["1999-0001-LAO"],
            "ISO": ["LAO"],
            "admin_level": [2],
            "geocoding_q": [1],
            "ADM1_NAME": ["Bokeo"],
            "ADM2_NAME": ["Houayxay"],
            "geometry": [box(2, 0, 3, 1)],
        },
        crs="EPSG:4326",
    )
    footprint = beside_houayxay.geometry.iloc[0]
    assert footprint.touches(houayxay) and footprint.intersection(houayxay).area == 0.0, (
        "the footprint must share an edge and no area, or this is the ordinary disjoint case"
    )

    resolved = resolve_to_gadm(beside_houayxay, cache_dir)

    assert resolved.empty, "a footprint sharing only an edge covers no unit"


def test_resolved_units_span_every_country_asked_for(write_gadm_cache, write_geo_disasters_cache):
    write_gadm_cache()
    cache_dir = write_geo_disasters_cache()

    resolved = load_resolved_units(["ZMB", "LAO"], cache_dir)

    assert set(resolved["ISO"]) == {"LAO", "ZMB"}
    assert (resolved["gid"].str[:3] == resolved["ISO"]).all(), "a unit belongs to the country it was read for"


def test_a_country_is_resolved_once_and_read_back(write_gadm_cache, write_geo_disasters_cache):
    """The overlay runs against every GADM unit in the country and takes a hundred seconds over a
    region. A second read that touches the GeoPackages has not cached anything."""
    write_gadm_cache()
    cache_dir = write_geo_disasters_cache()
    first = load_resolved_units(["LAO"], cache_dir)

    (cache_dir / "geo_disasters" / "disaster_subnational_90_23.gpkg").unlink()
    (cache_dir / "gadm" / "gadm_410.gpkg").unlink()

    pd.testing.assert_frame_equal(load_resolved_units(["LAO"], cache_dir), first)


def test_changing_the_placement_rule_rebuilds_rather_than_serving_the_old_units(
    monkeypatch, write_gadm_cache, write_geo_disasters_cache
):
    """A cached entry outlives the rule that placed it. Keyed on the country alone, every warm cache
    would keep answering with units the current rule would not have chosen, and nothing would say so."""
    write_gadm_cache()
    cache_dir = write_geo_disasters_cache()
    load_resolved_units(["LAO"], cache_dir)
    (cache_dir / "geo_disasters" / "disaster_subnational_90_23.gpkg").unlink()

    monkeypatch.setattr("climate_risk.data.geo_disasters.MIN_CONTAINMENT", 0.9)

    # Reading the archive again is the point: under a rule it was not built with, the entry is unusable.
    with pytest.raises(NotImplementedError):
        load_resolved_units(["LAO"], cache_dir)


def test_each_country_caches_under_its_own_key(write_gadm_cache, write_geo_disasters_cache):
    """One entry per region would make `sea` and a global run rebuild each other's members, and a
    shared key would serve Laos' units for a request about Zambia."""
    write_gadm_cache()
    cache_dir = write_geo_disasters_cache()

    load_resolved_units(["LAO"], cache_dir)

    assert set(load_resolved_units(["ZMB"], cache_dir)["gid"]) == {"ZMB.1_1"}


def test_a_country_named_twice_is_read_once(write_gadm_cache, write_geo_disasters_cache):
    """The argument is a set of countries, not a list of reads. A region listing a member twice —
    or two overlapping regions concatenated — would otherwise duplicate every one of that country's
    rows, and the event counts built on them come out twice as large."""
    write_gadm_cache()
    cache_dir = write_geo_disasters_cache()

    once = load_resolved_units(["LAO", "ZMB"], cache_dir)
    repeated_and_reordered = load_resolved_units(["ZMB", "LAO", "LAO"], cache_dir)

    pd.testing.assert_frame_equal(once, repeated_and_reordered)
    # Set iteration order over strings varies between processes, so row order would too.
    assert once["ISO"].is_monotonic_increasing, "countries concatenate in a fixed order"


def test_asking_for_no_countries_gives_the_empty_frame(tmp_path):
    """A place whose members are all absent from Geo-Disasters reaches here as an empty list, and
    the caller concatenates whatever comes back."""
    resolved = load_resolved_units([], tmp_path)

    assert resolved.empty
    assert list(resolved.columns) == RESOLVED_COLUMNS
    assert dict(resolved.dtypes.astype(str)) == RESOLVED_DTYPES


REAL_GADM = REAL_CACHE_DIR / "gadm" / "gadm_410.gpkg"

needs_both = pytest.mark.skipif(
    not (REAL_GEOPACKAGE.exists() and REAL_GADM.exists()), reason="needs the Geo-Disasters and GADM GeoPackages"
)


@pytest.mark.requires_geo_disasters
@pytest.mark.requires_gadm
@needs_both
def test_a_footprint_resolves_to_the_gadm_unit_it_covers():
    """The two gazetteers share no identifier and spell the same province differently, so a
    footprint is placed by where it is. Laos is small enough to check every row."""
    footprints = load_event_footprints(REAL_CACHE_DIR, iso="LAO")

    resolved = resolve_to_gadm(footprints, REAL_CACHE_DIR)

    assert set(resolved["DisNo."]) == set(footprints["DisNo."]), "every Laos event keeps its geography"
    assert len(resolved) >= len(footprints), "a footprint spanning several units contributes several rows"
    assert set(resolved["geometry_source"]) == {"geo_disasters"}
    assert resolved["gid"].str.startswith("LAO").all()


@pytest.mark.requires_geo_disasters
@pytest.mark.requires_gadm
@needs_both
def test_a_footprint_is_matched_at_its_own_admin_level():
    """Matching a district against provinces returns the province containing it, which silently
    coarsens the footprint to something several times its size."""
    footprints = load_event_footprints(REAL_CACHE_DIR, iso="LAO")

    resolved = resolve_to_gadm(footprints, REAL_CACHE_DIR).set_index("DisNo.")

    at_level_2 = resolved[resolved["admin_level"] == 2]
    assert not at_level_2.empty, "Laos has district-level footprints to match"
    assert at_level_2["gid"].str.count(r"\.").eq(2).all(), "a level-2 gid names a district"


@pytest.mark.requires_geo_disasters
@pytest.mark.requires_gadm
@needs_both
def test_the_names_corroborate_the_geometry():
    """Geometry decides the match, so nothing checks the names — and a systematic mis-join would
    look perfectly healthy. Most matched pairs should still agree on the name."""
    footprints = load_event_footprints(REAL_CACHE_DIR, iso="LAO")

    resolved = resolve_to_gadm(footprints, REAL_CACHE_DIR)

    published = unit_names(footprints).map(normalize_unit_name)
    matched = resolved["name"].map(normalize_unit_name)
    agreeing = sum(name in set(published) for name in matched)
    assert agreeing / len(matched) > 0.8, f"only {agreeing} of {len(matched)} matched units kept their name"


@pytest.mark.requires_geo_disasters
@pytest.mark.skipif(not REAL_GEOPACKAGE.exists(), reason="needs the Geo-Disasters GeoPackage")
def test_the_published_table_is_the_one_the_loader_expects():
    """A synthetic fixture agrees with the loader by construction. This is the only check that the
    real archive still has the schema and the scale the paper describes."""
    locations = load_event_locations(REAL_CACHE_DIR)

    assert len(locations) == 45_121
    assert locations["DisNo."].nunique() == 9_217
    assert set(locations["admin_level"]) == {1, 2}
    assert locations["geocoding_q"].between(1, 4).all()
    assert unit_names(locations).notna().all(), "every location must be named at its own level"
