import json

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from climate_risk.config.registry import CONFIG_ROOT, load_place, resolve_isos
from climate_risk.config.schema import EventFilters
from climate_risk.data_functions.emdat_processing import (
    DISASTER_TYPES,
    EMDAT_WINDOW_START,
    GEOMETRY_SOURCES,
    RESOLVED_UNIT_SCHEMA,
    TEXT_RESOLUTION_SCHEMA,
    TEXT_UNIT_SCHEMA,
    NamedPlace,
    count_events_by_type,
    country_year_grid,
    event_filter,
    event_geography,
    event_units,
    events_missing_units,
    load_emdat_events,
    named_places,
    total_damage,
)
from climate_risk.exceptions import DataValidationError
from tests.conftest import emdat_event


def selected_events(cache_dir, *, filters=None):
    """The events a place's thresholds keep."""
    return load_emdat_events(cache_dir).filter(event_filter(filters or EventFilters()))


def count_panel(cache_dir, *, filters=None, window_start=EMDAT_WINDOW_START):
    """Those events counted over the full country-year grid, which is what the panel is built on."""
    raw = load_emdat_events(cache_dir)
    events = raw.filter(event_filter(filters or EventFilters()))

    return count_events_by_type(events, country_year_grid(raw, window_start=window_start))


WINDOW_START = date(1969, 1, 1)


def row(frame: pl.DataFrame, iso: str, year: str) -> dict:
    """The one row for a country and year, as a plain mapping."""
    match = frame.filter((pl.col("ISO") == iso) & (pl.col("Start_Year") == date.fromisoformat(year)))

    assert len(match) == 1, f"expected one row for {iso} {year}, got {len(match)}"
    return match.to_dicts()[0]


def test_missing_workbook_names_the_path_it_wants(tmp_path):
    """EM-DAT is licensed and hand-placed, so the error is the user's only instruction."""
    with pytest.raises(NotImplementedError, match=str(tmp_path / "emdat.xlsx")):
        load_emdat_events(tmp_path)


def test_empty_workbook_is_rejected(write_emdat_cache):
    with pytest.raises(ValueError, match="has no rows"):
        load_emdat_events(write_emdat_cache([]))


def test_a_cost_written_as_text_reads_back_as_a_number(write_emdat_cache):
    """EM-DAT stores these columns as text, empties included, so an undeclared reader types the
    whole column as string and the totals built on it concatenate instead of adding.
    """
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "unquantified", "Reconstruction Costs ('000 US$)": ""}),
            emdat_event({"DisNo.": "quantified", "Reconstruction Costs ('000 US$)": "1234.5"}),
        ]
    )

    costs = load_emdat_events(cache_dir)["Reconstruction_Costs"]

    assert costs.dtype.is_numeric(), costs.dtype
    assert costs.drop_nulls().to_list() == [1234.5]


def test_workbook_missing_a_column_names_it(write_emdat_cache):
    """Nothing else detects upstream schema drift, so the loader has to."""
    event = emdat_event()
    del event["Total Affected"]

    with pytest.raises(ValueError, match=r"missing \['Total Affected'\]"):
        load_emdat_events(write_emdat_cache([event]))


def test_every_country_year_appears_even_without_events(write_emdat_cache):
    """Count models need zero-event country-years present as rows, not absent."""
    cache_dir = write_emdat_cache([emdat_event({"ISO": "AAA"}), emdat_event({"ISO": "BBB", "DisNo.": "1990-0002-BBB"})])

    events = count_panel(cache_dir)
    years = range(WINDOW_START.year, 1991)

    assert events.columns[:2] == ["ISO", "Start_Year"]
    assert set(events["ISO"]) == {"AAA", "BBB"}
    assert events["Start_Year"].n_unique() == len(years)
    assert len(events) == 2 * len(years)


def test_window_start_can_be_overridden(write_emdat_cache):
    cache_dir = write_emdat_cache([emdat_event({"Start Year": 1990})])

    years = count_panel(cache_dir, window_start=date(1985, 1, 1))

    assert years["Start_Year"].min() == date(1985, 1, 1)


def test_window_start_after_the_newest_event_is_rejected(write_emdat_cache):
    """An empty window would otherwise return every frame with zero rows and no error."""
    cache_dir = write_emdat_cache([emdat_event({"Start Year": 1990})])

    with pytest.raises(ValueError, match="every output frame would be empty"):
        count_panel(cache_dir, window_start=date(2001, 1, 1))


def test_a_year_without_events_is_nan_not_zero(write_emdat_cache):
    """The replication panel relies on this to distinguish no-data from no-disasters."""
    cache_dir = write_emdat_cache(
        [emdat_event({"Start Year": 1990}), emdat_event({"DisNo.": "later", "Start Year": 1995})]
    )

    events = count_panel(cache_dir)

    assert row(events, "AAA", "1990-01-01")["Flood"] == 1
    assert row(events, "AAA", "1991-01-01")["Flood"] is None


def test_region_metadata_survives_reindexing(write_emdat_cache):
    cache_dir = write_emdat_cache(
        [
            emdat_event({"Region": "Africa", "Subregion": "Eastern Africa"}),
            emdat_event({"DisNo.": "later", "Start Year": 2001, "Region": "Africa", "Subregion": "Eastern Africa"}),
        ]
    )

    events = count_panel(cache_dir)
    quiet_year = row(events, "AAA", "1995-01-01")

    assert quiet_year["Region"] == "Africa"
    assert quiet_year["Subregion"] == "Eastern Africa"


@pytest.mark.parametrize(
    ("disaster_type", "expected_class"),
    [
        ("Flood", "Hydrometeorological"),
        ("Storm", "Hydrometeorological"),
        ("Drought", "Climatological"),
        ("Wildfire", "Climatological"),
        ("Extreme temperature", "Climatological"),
        ("Mass movement (wet)", "Hydrometeorological"),
    ],
)
def test_disaster_types_map_to_their_class(write_emdat_cache, disaster_type, expected_class):
    cache_dir = write_emdat_cache([emdat_event({"Disaster Type": disaster_type})])

    classes = load_emdat_events(cache_dir)["disaster_class"]

    assert classes.to_list() == [expected_class]


def test_an_unclassified_disaster_type_becomes_null(write_emdat_cache):
    """Geophysical and biological types are out of scope, and are mapped to null rather than guessed at."""
    cache_dir = write_emdat_cache([emdat_event({"Disaster Type": "Earthquake"})])

    classes = load_emdat_events(cache_dir)["disaster_class"]

    assert classes.to_list() == [None]


def test_adjusted_filter_ignores_deaths_but_starts_in_1981(write_emdat_cache):
    """1980 itself is excluded, so the boundary is pinned on both sides rather than approximately."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "kept-no-deaths", "Start Year": 1990, "Total Deaths": 0}),
            emdat_event({"DisNo.": "first-year", "Start Year": 1981}),
            emdat_event({"DisNo.": "boundary-year", "Start Year": 1980}),
            emdat_event({"DisNo.": "too-early", "Start Year": 1975}),
        ]
    )

    kept = selected_events(cache_dir)["DisNo."]

    assert sorted(kept.to_list()) == ["first-year", "kept-no-deaths"]


def test_an_event_exactly_on_the_affected_threshold_does_not_count(write_emdat_cache):
    """The threshold is strict, and an event sitting on it is the only way to tell that apart."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "over", "Start Year": 1995, "Total Affected": 1_001}),
            emdat_event({"DisNo.": "exactly-on-it", "Start Year": 1995, "Total Affected": 1_000}),
        ]
    )

    kept = selected_events(cache_dir, filters=EventFilters(min_total_affected=1_000))["DisNo."]

    assert kept.to_list() == ["over"]


def test_an_event_whose_reach_was_never_recorded_counts_by_default(write_emdat_cache):
    """Comparing a null against a floor drops it, so a default floor would discard events for being
    unquantified rather than for being small."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "counted", "Start Year": 1995, "Total Affected": 5_000}),
            emdat_event({"DisNo.": "unquantified", "Start Year": 1995, "Total Affected": None}),
        ]
    )

    by_default = selected_events(cache_dir)["DisNo."]
    with_a_floor = selected_events(cache_dir, filters=EventFilters(min_total_affected=1))["DisNo."]

    assert sorted(by_default.to_list()) == ["counted", "unquantified"]
    assert with_a_floor.to_list() == ["counted"], "a floor still drops what it cannot compare"


def test_a_small_event_counts_unless_a_place_sets_a_floor(write_emdat_cache):
    """Small events carry the finest geography, so a floor biases the panel towards coarse units."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "large", "Start Year": 1995, "Total Affected": 5_000}),
            emdat_event({"DisNo.": "small", "Start Year": 1995, "Total Affected": 30}),
        ]
    )

    by_default = selected_events(cache_dir)["DisNo."]
    with_a_floor = selected_events(cache_dir, filters=EventFilters(min_total_affected=1_000))["DisNo."]

    assert by_default.to_list() == ["large", "small"]
    assert with_a_floor.to_list() == ["large"]


def test_a_deaths_threshold_drops_an_event_whose_deaths_were_never_recorded(write_emdat_cache):
    """Every threshold compares against a possibly-null column, so the rule that an unrecorded
    severity fails any threshold set has to hold for deaths as much as for reach."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "counted", "Start Year": 1995, "Total Deaths": 500}),
            emdat_event({"DisNo.": "unquantified", "Start Year": 1995, "Total Deaths": None}),
        ]
    )

    by_default = selected_events(cache_dir)["DisNo."]
    with_a_floor = selected_events(cache_dir, filters=EventFilters(min_deaths=1))["DisNo."]

    assert sorted(by_default.to_list()) == ["counted", "unquantified"]
    assert with_a_floor.to_list() == ["counted"]


def test_the_adjusted_view_follows_the_filters_it_is_given(write_emdat_cache):
    """The window is a place's setting, so a place with a shorter one must get a shorter panel."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "in-window", "Start Year": 1995}),
            emdat_event({"DisNo.": "before-window", "Start Year": 1985}),
        ]
    )

    kept = selected_events(cache_dir, filters=EventFilters(start_year=1990))["DisNo."]

    assert kept.to_list() == ["in-window"]


def test_a_deaths_threshold_applies_only_when_a_place_sets_one(write_emdat_cache):
    """The default counts an event on reach alone, so a deadly-but-small event needs an explicit floor."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "deadly", "Start Year": 1995, "Total Deaths": 500}),
            emdat_event({"DisNo.": "harmless", "Start Year": 1995, "Total Deaths": 0}),
        ]
    )

    by_default = selected_events(cache_dir)["DisNo."]
    with_a_floor = selected_events(cache_dir, filters=EventFilters(min_deaths=100))["DisNo."]

    assert by_default.to_list() == ["deadly", "harmless"]
    assert with_a_floor.to_list() == ["deadly"]


def test_a_place_can_close_its_window_before_the_newest_event(write_emdat_cache):
    """`end_year` is the only filter that trims the recent end, and it counts its own year."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "in-window", "Start Year": 1995}),
            emdat_event({"DisNo.": "last-year", "Start Year": 2000}),
            emdat_event({"DisNo.": "after-window", "Start Year": 2005}),
        ]
    )

    kept = selected_events(cache_dir, filters=EventFilters(end_year=2000))["DisNo."]

    assert kept.to_list() == ["in-window", "last-year"]


def test_damage_totals_sum_within_a_country_year_and_leave_empty_ones_null(write_emdat_cache):
    """A country-year with no events reads as null, which is what the panel's outer joins rely on.

    Zero would be a claim that nothing happened; null is the absence of a record.
    """
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "first", "Start Year": 1995, "Total Deaths": 10}),
            emdat_event({"DisNo.": "second", "Start Year": 1995, "Total Deaths": 5}),
        ]
    )
    raw = load_emdat_events(cache_dir)

    damage = total_damage(raw.filter(event_filter(EventFilters())), country_year_grid(raw))

    assert damage.filter(pl.col("Start_Year") == date(1995, 1, 1))["Deaths"].to_list() == [15.0]

    # Asserted to exist first: `is_null().all()` is vacuously true on a row the grid dropped.
    quiet_year = damage.filter(pl.col("Start_Year") == date(1994, 1, 1))
    assert len(quiet_year) == 1
    assert quiet_year["Deaths"].is_null().all()


def test_the_window_extends_to_the_newest_event(write_emdat_cache):
    """A refreshed download must extend the panel, not silently lose its newest years."""
    cache_dir = write_emdat_cache([emdat_event({"Start Year": 2025})])

    years = count_panel(cache_dir)["Start_Year"]

    assert years.max() == date(2025, 1, 1)
    assert years.min() == WINDOW_START


def test_mass_movement_events_reach_the_count_frames(write_emdat_cache):
    cache_dir = write_emdat_cache([emdat_event({"Disaster Type": "Mass movement (wet)"})])

    events = count_panel(cache_dir)

    assert row(events, "AAA", "1990-01-01")["Mass movement (wet)"] == 1


def test_a_workbook_with_no_usable_year_is_rejected(write_emdat_cache):
    """A null Start_Year leaves the window with no end, which would otherwise fail deep in a join."""
    cache_dir = write_emdat_cache([emdat_event({"Start Year": None})])

    with pytest.raises(ValueError, match="Every Start_Year in the workbook is missing"):
        country_year_grid(load_emdat_events(cache_dir))


def test_every_disaster_type_gets_a_column_in_a_stable_order(write_emdat_cache):
    """Downstream selects by name, and a column order that follows the data is not a contract."""
    cache_dir = write_emdat_cache([emdat_event({"Disaster Type": "Flood"})])

    events = count_panel(cache_dir)

    assert events.columns == ["ISO", "Start_Year", "Region", "Subregion", *DISASTER_TYPES]


# The licensed workbook, which cannot be committed or fetched. Absent on CI and on a fresh clone.
REAL_CACHE_DIR = Path(__file__).parents[2] / "data"

SHIPPED_PLACE_KEYS = sorted(path.stem for path in (CONFIG_ROOT / "places").glob("*.toml"))


def test_an_event_carrying_units_is_not_offered_for_resolution(write_emdat_cache):
    """Two thirds of the workbook has no units, and re-resolving the third that does would override
    EM-DAT's own coding with a name match."""
    events = load_emdat_events(
        write_emdat_cache(
            [
                emdat_event(
                    {"DisNo.": "2018-0001-LAO", "Location": "Attapu", "GADM Admin Units": '[{"gid_1": "LAO.1_1"}]'}
                ),
                emdat_event({"DisNo.": "2018-0002-LAO", "Location": "Bokeo", "GADM Admin Units": ""}),
            ]
        )
    )

    assert [event.disno for event in events_missing_units(events)] == ["2018-0002-LAO"]


def test_an_event_whose_text_names_nothing_is_left_out(write_emdat_cache):
    """`Countrywide` and `N.A. on the source` reach no gazetteer, and carrying them makes a caller
    fetch an archive for a country it can place nothing in."""
    events = load_emdat_events(
        write_emdat_cache(
            [
                emdat_event({"DisNo.": "2018-0003-LAO", "Location": "()", "GADM Admin Units": ""}),
                emdat_event({"DisNo.": "2018-0004-LAO", "Location": "Bokeo", "GADM Admin Units": ""}),
            ]
        )
    )

    assert [event.disno for event in events_missing_units(events)] == ["2018-0004-LAO"]


@pytest.mark.requires_emdat
@pytest.mark.skipif(not (REAL_CACHE_DIR / "emdat.xlsx").exists(), reason="needs the licensed EM-DAT workbook")
@pytest.mark.parametrize("key", SHIPPED_PLACE_KEYS)
def test_every_shipped_country_has_events_clearing_its_own_filters(key):
    """A country whose filters admit nothing yields an empty panel, and every later test passes on it.

    Synthetic fixtures cannot catch this: they contain whatever events the fixture author wrote.
    """
    place = load_place(key)
    events = selected_events(REAL_CACHE_DIR, filters=place.events)

    assert len(events.filter(pl.col("ISO").is_in(resolve_isos(place)))) > 0


def units_json(*units) -> str:
    """The `GADM Admin Units` column as EM-DAT writes it: a JSON array of per-unit objects."""
    return json.dumps(list(units))


def test_an_event_with_no_units_contributes_no_rows(write_emdat_cache):
    """Half the workbook has no geography; those events must not appear with an empty unit."""
    cache_dir = write_emdat_cache([emdat_event({"GADM Admin Units": ""}), emdat_event({"DisNo.": "b"})])

    units = event_units(load_emdat_events(cache_dir))

    assert units.is_empty()
    assert units.columns == ["DisNo.", "gid", "name", "admin_level", "migration_method"]


def test_an_event_naming_several_units_yields_a_row_for_each(write_emdat_cache):
    """73% of located events name more than one unit; collapsing them loses the footprint."""
    cache_dir = write_emdat_cache(
        [
            emdat_event(
                {
                    "DisNo.": "many",
                    "GADM Admin Units": units_json(
                        {"gid_1": "LAO.1_1", "name_1": "Attapu", "migration_method": "jaccard1"},
                        {"gid_1": "LAO.2_1", "name_1": "Bokeo", "migration_method": "jaccard1"},
                        {"gid_1": "LAO.4_1", "name_1": "Champasak", "migration_method": "puzzle1_1"},
                    ),
                }
            ),
            emdat_event({"DisNo.": "one", "GADM Admin Units": units_json({"gid_1": "ZMB.1_1", "name_1": "Central"})}),
        ]
    )

    units = event_units(load_emdat_events(cache_dir))

    assert units.group_by("DisNo.").len().sort("DisNo.")["len"].to_list() == [3, 1]
    assert units.filter(pl.col("DisNo.") == "many")["gid"].to_list() == ["LAO.1_1", "LAO.2_1", "LAO.4_1"]


def test_the_level_comes_from_the_key_not_the_shape_of_the_id(write_emdat_cache):
    """GADM numbers Ghana `GHA11_2` and `GHA7.13_2`, so the id cannot be parsed for its level."""
    cache_dir = write_emdat_cache(
        [
            emdat_event(
                {
                    "GADM Admin Units": units_json(
                        {"gid_1": "GHA11_2", "name_1": "Savannah", "migration_method": "puzzle1_1"},
                        {"gid_2": "GHA7.13_2", "name_2": "Ga Central", "migration_method": "jaccard2"},
                        {"gid_2": "LAO.1.1_1", "name_2": "Sanamxay", "migration_method": "jaccard2"},
                    )
                }
            )
        ]
    )

    units = event_units(load_emdat_events(cache_dir))

    assert units["admin_level"].to_list() == [1, 2, 2]


def test_a_unit_with_no_name_or_no_method_still_yields_a_row(write_emdat_cache):
    """EM-DAT omits the name where GADM has none, and the method where nothing was migrated.

    Requiring either would silently drop 1,370 native-GADM units and 574 unnamed ones.
    """
    cache_dir = write_emdat_cache(
        [
            emdat_event(
                {
                    "GADM Admin Units": units_json(
                        {"gid_2": "GBR.1.26_1", "migration_method": "jaccard2"},
                        {"gid_1": "COL.27_1", "name_1": "San Andrés y Providencia"},
                    )
                }
            )
        ]
    )

    units = event_units(load_emdat_events(cache_dir))

    assert units["gid"].to_list() == ["GBR.1.26_1", "COL.27_1"]
    assert units["name"].to_list() == [None, "San Andrés y Providencia"]
    assert units["migration_method"].to_list() == ["jaccard2", None]


def test_a_unit_carrying_no_gid_names_the_event_it_came_from(write_emdat_cache):
    """An unidentifiable unit is a schema change upstream, and the error has to be findable."""
    cache_dir = write_emdat_cache(
        [emdat_event({"DisNo.": "broken", "GADM Admin Units": units_json({"name_1": "Nowhere"})})]
    )

    with pytest.raises(ValueError, match="broken"):
        event_units(load_emdat_events(cache_dir))


def test_a_flat_list_of_places_explodes_to_one_row_each():
    assert named_places("Cagayan, Ilocos Norte, Kalin-Aapayo province") == [
        NamedPlace("Cagayan", None),
        NamedPlace("Ilocos Norte", None),
        NamedPlace("Kalin-Aapayo province", None),
    ]


def test_a_container_applies_to_every_place_since_the_last_one():
    """EM-DAT writes a run of districts then the province holding all of them. Attaching the parent
    to the nearest place alone would leave the rest unconstrained, and the parent is what tells a
    repeated district name which province it is in."""
    places = named_places("Muang Nakhon Si Thammarat, Hua Sai, Pak Phanang districts (Nakhon Si Thammarat province)")

    assert [place.parent for place in places] == ["Nakhon Si Thammarat province"] * 3


def test_a_second_container_starts_a_new_group():
    """One event names districts of several provinces in sequence. A parent that leaked past its
    own group would put every later district in the first province named."""
    places = named_places("Ilocos Norte districts (Region II province), Batanes, Cagayan (Region III province)")

    assert places == [
        NamedPlace("Ilocos Norte districts", "Region II province"),
        NamedPlace("Batanes", "Region III province"),
        NamedPlace("Cagayan", "Region III province"),
    ]


def test_a_nested_parenthesis_glosses_the_container_rather_than_nesting_inside_it():
    """`Region I (Ilocos region)` is one place under two names, not a place inside a place. Reading
    the inner group as a further parent invents a level of hierarchy the prose does not claim."""
    assert named_places("Ilocos Norte district (Region I (Ilocos region) province)") == [
        NamedPlace("Ilocos Norte district", "Region I province")
    ]


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("Maradu, Long Lama (Northern Sarawak", [("Maradu", "Northern Sarawak"), ("Long Lama", "Northern Sarawak")]),
        ("Kakinada (Andhra Pradesh state))", [("Kakinada", "Andhra Pradesh state")]),
    ],
    ids=["never closed", "closed twice"],
)
def test_unbalanced_parentheses_still_yield_the_container(location, expected):
    """Fifty of the workbook's location strings are truncated or double-closed. A parser that gave
    up on them would drop the container that disambiguates the places it does read."""
    assert named_places(location) == [NamedPlace(*pair) for pair in expected]


def test_a_preposition_joining_a_place_to_its_container_is_dropped():
    """EM-DAT writes `Montgomery County in (Tennessee state)`, and the split leaves the preposition
    on the end of a name that would otherwise match."""
    assert named_places("Clarksville City, Montgomery County in (Tennesee state)") == [
        NamedPlace("Clarksville City", "Tennesee state"),
        NamedPlace("Montgomery County", "Tennesee state"),
    ]


def test_a_leaked_spreadsheet_label_is_dropped_from_the_name():
    """EM-DAT writes `Level 1 = Uttar Pradesh` in the location column, and the label reaches the
    matcher as part of the place name."""
    assert named_places("Level 1 = Uttar Pradesh; Bihar") == [
        NamedPlace("Uttar Pradesh", None),
        NamedPlace("Bihar", None),
    ]


def test_a_place_named_twice_appears_once():
    """`Savannakhet, Kham Muane, Savannakhet` would otherwise weight one province double in
    whatever counts the places."""
    assert named_places("Savannakhet, Kham Muane, Savannakhet") == [
        NamedPlace("Savannakhet", None),
        NamedPlace("Kham Muane", None),
    ]


def test_a_conjunction_is_not_a_separator():
    """Seventy-five GADM units are named like this. Splitting on `and` would destroy exactly the
    names the match is looking for, so a span is left whole for the resolver to interpret."""
    assert named_places("Newfoundland and Labrador") == [NamedPlace("Newfoundland and Labrador", None)]


def test_a_conjunction_stranded_by_the_split_is_dropped():
    """`X, and Y` splits on the comma and leaves the conjunction on the second place. No GADM unit
    is named with a leading `and`, so the fragment would match nothing and lose the place."""
    assert named_places("Aceh, and Sumatera Utara provinces") == [
        NamedPlace("Aceh", None),
        NamedPlace("Sumatera Utara provinces", None),
    ]


@pytest.mark.parametrize("location", [None, "", "   ", "()"], ids=["null", "empty", "whitespace", "empty group"])
def test_text_naming_nothing_yields_nothing(location):
    """Two thirds of the workbook has no location text, and a country-tier event reaches here."""
    assert named_places(location) == []


@pytest.mark.requires_emdat
@pytest.mark.skipif(not (REAL_CACHE_DIR / "emdat.xlsx").exists(), reason="needs the licensed EM-DAT workbook")
def test_the_unit_table_keeps_every_event_that_has_geography():
    """Synthetic fixtures hold whatever the author wrote; only the workbook shows the real shape.

    Counts are asserted as properties rather than literals, because a re-download moves them.
    """
    events = load_emdat_events(REAL_CACHE_DIR)
    units = event_units(events)

    with_geography = events.filter(pl.col("GADM Admin Units").str.strip_chars().str.len_chars() > 2)
    assert units["DisNo."].n_unique() == len(with_geography)

    assert units["gid"].null_count() == 0
    assert set(units["admin_level"].unique()) == {1, 2}

    # Most located events name several units, which is why the table is long rather than one row
    # per event. A collapse to one unit each would leave both of these at zero.
    per_event = units.group_by("DisNo.").len()
    assert per_event.filter(pl.col("len") > 1).height > per_event.height / 2
    assert per_event.filter(pl.col("len") > 100).height > 0


def test_every_event_appears_with_a_known_source(write_emdat_cache):
    """An event missing from the table would read as having no geography rather than saying so."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "LAO.1_1", "name_1": "Attapu"})}),
            emdat_event({"DisNo.": "point-only", "Latitude": 18.0, "Longitude": 102.0}),
            emdat_event({"DisNo.": "nothing", "Latitude": None, "Longitude": None}),
        ]
    )

    geography = event_geography(load_emdat_events(cache_dir))

    assert set(geography["DisNo."]) == {"coded", "point-only", "nothing"}
    assert set(geography["geometry_source"]) <= set(GEOMETRY_SOURCES)
    assert geography["geometry_source"].null_count() == 0
    assert dict(zip(geography["DisNo."], geography["geometry_source"], strict=True)) == {
        "coded": "gadm",
        "point-only": "emdat_point",
        "nothing": "country",
    }


def test_a_coded_event_keeps_one_row_per_unit(write_emdat_cache):
    """The footprint is the reason the table exists; one row per event would discard it."""
    cache_dir = write_emdat_cache(
        [
            emdat_event(
                {
                    "DisNo.": "many",
                    "GADM Admin Units": units_json(
                        {"gid_1": "LAO.1_1", "name_1": "Attapu"}, {"gid_2": "LAO.2.1_1", "name_2": "Houayxay"}
                    ),
                }
            )
        ]
    )

    geography = event_geography(load_emdat_events(cache_dir))

    assert len(geography) == 2
    assert geography["gid"].to_list() == ["LAO.1_1", "LAO.2.1_1"]
    assert geography["admin_level"].to_list() == [1, 2]
    assert set(geography["geometry_source"]) == {"gadm"}


def test_a_coded_event_that_also_has_a_point_keeps_both(write_emdat_cache):
    """Units and a coordinate are different claims; dropping either decides something this does not.

    The event is `gadm` because the polygon is the better geometry, but the point stays visible.
    """
    cache_dir = write_emdat_cache(
        [
            emdat_event(
                {
                    "DisNo.": "both",
                    "Latitude": 18.0,
                    "Longitude": 102.0,
                    "GADM Admin Units": units_json({"gid_1": "LAO.1_1", "name_1": "Attapu"}),
                }
            )
        ]
    )

    geography = event_geography(load_emdat_events(cache_dir))

    assert geography["geometry_source"].to_list() == ["gadm"]
    assert geography["Latitude"].to_list() == [18.0]


def test_the_unit_columns_are_empty_outside_the_gadm_tier(write_emdat_cache):
    """A stale gid on a country row would let a join attach geometry the event never had."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "nothing", "Latitude": None, "Longitude": None})])

    geography = event_geography(load_emdat_events(cache_dir))

    assert geography["gid"].null_count() == 1
    assert geography["admin_level"].null_count() == 1
    assert geography["migration_method"].null_count() == 1


@pytest.mark.requires_emdat
@pytest.mark.skipif(not (REAL_CACHE_DIR / "emdat.xlsx").exists(), reason="needs the licensed EM-DAT workbook")
def test_no_tier_silently_swallows_the_others():
    """A bug collapsing everything to `country` still produces a full, plausible-looking table.

    Bounds rather than counts, which move on re-download; what must not change is that all three
    tiers stay populated and `gadm` stays the biggest by row count.
    """
    events = load_emdat_events(REAL_CACHE_DIR)
    geography = event_geography(events)

    assert geography["DisNo."].n_unique() == len(events)

    per_source = dict(geography.group_by("geometry_source").len().iter_rows())
    # The workbook supplies these three. `geo_disasters` is resolved from a separate archive and
    # cannot appear here.
    assert set(per_source) == {"gadm", "emdat_point", "country"}
    assert set(per_source) < set(GEOMETRY_SOURCES)
    assert all(count > 1_000 for count in per_source.values())
    # Coded events carry several units each, so the gadm tier is the longest despite being a
    # minority of events.
    assert per_source["gadm"] > per_source["country"]


def text_units(rows):
    """Units an event's own prose reaches, in the shape `event_geography` takes them."""
    return pl.DataFrame(rows, schema=TEXT_UNIT_SCHEMA, orient="row")


def resolution_counts(rows):
    """How much of each event's prose resolved, per event."""
    return pl.DataFrame(rows, schema=TEXT_RESOLUTION_SCHEMA, orient="row")


def resolved_units(rows):
    """Units another gazetteer places, in the shape `event_geography` takes them."""
    return pl.DataFrame(rows, schema=RESOLVED_UNIT_SCHEMA, orient="row")


def test_an_event_the_workbook_places_nowhere_takes_the_overlay_units(write_emdat_cache):
    """The 209 events this tier exists for: geography is available from a second gazetteer and the
    table recorded none of it, because nothing produced the tier `GEOMETRY_SOURCES` advertised."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "uncoded", "Latitude": None, "Longitude": None})])

    geography = event_geography(
        load_emdat_events(cache_dir), resolved=resolved_units([("uncoded", "AAA.1_1", "Somewhere", 1, 3, 0.8)])
    )

    assert geography["geometry_source"].to_list() == ["geo_disasters"]
    assert geography["gid"].to_list() == ["AAA.1_1"]
    assert geography["admin_level"].to_list() == [1]
    assert geography["geocoding_q"].to_list() == [3]
    assert geography["overlap"].to_list() == [0.8]


def test_an_overlay_unit_inside_one_the_workbook_codes_is_dropped(write_emdat_cache):
    """The same ground at a second resolution. Keeping both would put a province and a district
    inside it into one observation, which reads as more geography than either source claims."""
    cache_dir = write_emdat_cache(
        [emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "AAA.1_1"})})]
    )

    geography = event_geography(
        load_emdat_events(cache_dir), resolved=resolved_units([("coded", "AAA.1.2_1", "Inside it", 2, 1, 0.9)])
    )

    assert set(geography["geometry_source"]) == {"gadm"}
    assert geography["gid"].to_list() == ["AAA.1_1"]


def test_an_overlay_unit_the_workbook_already_codes_is_dropped(write_emdat_cache):
    """The commonest case by far — the two sources mostly name the same units. Counting a repeat as
    new ground gives the event the same unit twice under two tiers, which reads as twice the
    footprint and double-counts it in anything aggregating over units."""
    cache_dir = write_emdat_cache(
        [emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "AAA.1_1"})})]
    )

    geography = event_geography(
        load_emdat_events(cache_dir), resolved=resolved_units([("coded", "AAA.1_1", "The same one", 1, 1, 0.9)])
    )

    assert geography["gid"].to_list() == ["AAA.1_1"]
    assert geography["geometry_source"].to_list() == ["gadm"]


def test_an_overlay_unit_containing_one_the_workbook_codes_is_dropped(write_emdat_cache):
    """Nesting the other way round, which happens wherever EM-DAT codes finer than the overlay
    resolves. Still one piece of ground named twice."""
    cache_dir = write_emdat_cache(
        [emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_2": "AAA.1.2_1"})})]
    )

    geography = event_geography(
        load_emdat_events(cache_dir), resolved=resolved_units([("coded", "AAA.1_1", "Around it", 1, 1, 0.9)])
    )

    assert set(geography["geometry_source"]) == {"gadm"}
    assert geography["gid"].to_list() == ["AAA.1.2_1"]


def test_an_overlay_unit_nesting_with_none_of_the_coded_ones_is_kept(write_emdat_cache):
    """Area EM-DAT never coded. Across the region this is 596 units over 90 events, and dropping it
    loses geography one source holds and nothing else records."""
    cache_dir = write_emdat_cache(
        [emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "AAA.1_1"})})]
    )

    geography = event_geography(
        load_emdat_events(cache_dir), resolved=resolved_units([("coded", "AAA.9_1", "Elsewhere", 1, 2, 0.6)])
    )

    assert sorted(geography["gid"]) == ["AAA.1_1", "AAA.9_1"]
    assert set(geography["geometry_source"]) == {"gadm", "geo_disasters"}
    assert geography.filter(pl.col("gid") == "AAA.9_1")["geocoding_q"].to_list() == [2]


def test_the_overlay_columns_are_null_on_every_other_tier(write_emdat_cache):
    """`geocoding_q` and `overlap` say how one gazetteer placed a unit, so a value on a row it did
    not place reads as its judgement of a placement it never made. The tier has to be populated for
    this to mean anything — with no overlay at all every row is null trivially."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "AAA.1_1"})}),
            emdat_event({"DisNo.": "overlaid", "Latitude": None, "Longitude": None}),
            emdat_event({"DisNo.": "point", "Latitude": 1.0, "Longitude": 2.0}),
            emdat_event({"DisNo.": "nothing", "Latitude": None, "Longitude": None}),
        ]
    )

    geography = event_geography(
        load_emdat_events(cache_dir), resolved=resolved_units([("overlaid", "AAA.2_1", "Somewhere", 2, 4, 0.6)])
    )

    scored = geography.filter(pl.col("geometry_source") == "geo_disasters")
    elsewhere = geography.filter(pl.col("geometry_source") != "geo_disasters")

    assert scored["geocoding_q"].to_list() == [4]
    assert elsewhere["geocoding_q"].null_count() == len(elsewhere)
    assert elsewhere["overlap"].null_count() == len(elsewhere)


def test_an_overlay_naming_an_event_the_workbook_does_not_have_is_ignored(write_emdat_cache):
    """The overlay is resolved per country, so it carries events this frame has filtered out. One
    arriving as its own row would be geography for an event nothing else in the table knows."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "nothing", "Latitude": None, "Longitude": None})])

    geography = event_geography(
        load_emdat_events(cache_dir),
        resolved=resolved_units([("a-different-event", "AAA.1_1", "Somewhere", 1, 1, 0.5)]),
    )

    assert geography["DisNo."].to_list() == ["nothing"]
    assert geography["geometry_source"].to_list() == ["country"]


def test_an_overlay_placing_an_event_in_several_units_contributes_a_row_each(write_emdat_cache):
    """An event spanning three provinces is three rows, the same as when the workbook codes it.
    Collapsing them would make a wide event look like a point one."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "wide", "Latitude": None, "Longitude": None})])

    geography = event_geography(
        load_emdat_events(cache_dir),
        resolved=resolved_units(
            [
                ("wide", "AAA.1_1", "First", 1, 1, 0.9),
                ("wide", "AAA.2_1", "Second", 1, 1, 0.7),
                ("wide", "AAA.3_1", "Third", 1, 2, 0.4),
            ]
        ),
    )

    assert sorted(geography["gid"]) == ["AAA.1_1", "AAA.2_1", "AAA.3_1"]
    assert set(geography["geometry_source"]) == {"geo_disasters"}


def test_every_geometry_source_the_table_advertises_can_be_produced(write_emdat_cache):
    """`GEOMETRY_SOURCES` is what a model reads to choose the tiers it accepts, so a value nothing
    emits is a tier a caller can filter on forever and never see."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "AAA.1_1"})}),
            emdat_event({"DisNo.": "overlaid", "Latitude": None, "Longitude": None}),
            emdat_event({"DisNo.": "spoken", "Latitude": None, "Longitude": None}),
            emdat_event({"DisNo.": "point", "Latitude": 1.0, "Longitude": 2.0}),
            emdat_event({"DisNo.": "nothing", "Latitude": None, "Longitude": None}),
        ]
    )

    geography = event_geography(
        load_emdat_events(cache_dir),
        resolved=resolved_units([("overlaid", "AAA.2_1", "Somewhere", 1, 1, 0.5)]),
        from_text=text_units([("spoken", "AAA.3_1", "Elsewhere", 1)]),
    )

    assert set(geography["geometry_source"]) == set(GEOMETRY_SOURCES)


def test_an_event_only_its_prose_places_takes_the_text_units(write_emdat_cache):
    """The 125 events of the 1980s the study window turns on: EM-DAT codes none of them and the
    overlay starts in 1990, so the location text is the only thing that reaches them."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "spoken", "Latitude": None, "Longitude": None})])

    geography = event_geography(
        load_emdat_events(cache_dir), from_text=text_units([("spoken", "AAA.1_1", "Somewhere", 1)])
    )

    assert geography["geometry_source"].to_list() == ["location_text"]
    assert geography["gid"].to_list() == ["AAA.1_1"]


def test_prose_naming_ground_the_workbook_already_codes_adds_nothing(write_emdat_cache):
    """Each source answers only for ground the ones before it left unnamed, and the prose is read
    last of the three that name units. A district inside a coded province is that province again."""
    cache_dir = write_emdat_cache(
        [emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "AAA.1_1"})})]
    )

    geography = event_geography(
        load_emdat_events(cache_dir), from_text=text_units([("coded", "AAA.1.2_1", "Inside it", 2)])
    )

    assert set(geography["geometry_source"]) == {"gadm"}
    assert geography["gid"].to_list() == ["AAA.1_1"]


def test_prose_naming_ground_the_overlay_already_placed_adds_nothing(write_emdat_cache):
    """The overlay is read before the prose, so ground it names is taken. Testing only against the
    workbook's own units would leave the middle source out of the ordering."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "both", "Latitude": None, "Longitude": None})])

    geography = event_geography(
        load_emdat_events(cache_dir),
        resolved=resolved_units([("both", "AAA.1_1", "Somewhere", 1, 1, 0.8)]),
        from_text=text_units([("both", "AAA.1.2_1", "Inside it", 2)]),
    )

    assert set(geography["geometry_source"]) == {"geo_disasters"}
    assert geography["gid"].to_list() == ["AAA.1_1"]


def test_how_much_of_the_prose_resolved_is_recorded_on_every_row_of_the_event(write_emdat_cache):
    """An event that named six places and reached five enters looking located while under-counting
    the sixth, and those failures cluster by country and era. The counts say which rows are whole."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "partly", "Latitude": None, "Longitude": None})])

    geography = event_geography(
        load_emdat_events(cache_dir),
        from_text=text_units([("partly", "AAA.1_1", "One", 1), ("partly", "AAA.2_1", "Two", 1)]),
        text_resolution=resolution_counts([("partly", 6, 2)]),
    )

    assert geography["names_written"].to_list() == [6, 6]
    assert geography["names_reached"].to_list() == [2, 2]


def test_an_event_whose_prose_reached_nothing_still_records_that_it_wrote_names(write_emdat_cache):
    """The case the counts exist for. An event at country tier because every name failed is not the
    same observation as one that carried no text at all, and nothing else in the table tells them
    apart."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "failed", "Latitude": None, "Longitude": None})])

    geography = event_geography(load_emdat_events(cache_dir), text_resolution=resolution_counts([("failed", 5, 0)]))

    assert geography["geometry_source"].to_list() == ["country"]
    assert geography["names_written"].to_list() == [5]
    assert geography["names_reached"].to_list() == [0]


def test_an_event_carrying_no_text_leaves_the_resolution_columns_null(write_emdat_cache):
    """Zero names written would be a claim the prose was read and held nothing; null is the absence
    of a reading. The two have to sit in one table for the difference to mean anything."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "read", "Latitude": None, "Longitude": None}),
            emdat_event({"DisNo.": "silent", "Latitude": None, "Longitude": None}),
        ]
    )

    geography = event_geography(load_emdat_events(cache_dir), text_resolution=resolution_counts([("read", 4, 0)]))

    was_read = geography.filter(pl.col("DisNo.") == "read")
    silent = geography.filter(pl.col("DisNo.") == "silent")

    assert was_read["names_written"].to_list() == [4]
    assert silent["names_written"].null_count() == len(silent)
    assert silent["names_reached"].null_count() == len(silent)


def test_the_window_counts_the_year_it_opens_in(write_emdat_cache):
    """1981 rather than 1990 is worth 125 located events, and every one of them is an off-by-one
    away from being discarded. `end_year` is tested for the same inclusivity at the other end."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "the-year-before", "Start Year": 1980}),
            emdat_event({"DisNo.": "the-opening-year", "Start Year": 1981}),
            emdat_event({"DisNo.": "after", "Start Year": 1982}),
        ]
    )

    kept = selected_events(cache_dir, filters=EventFilters(start_year=1981))["DisNo."]

    assert kept.to_list() == ["the-opening-year", "after"]


def test_a_place_that_names_no_window_opens_in_1981(write_emdat_cache):
    """The default is the decision, so it is worth one assertion: 1990 would silently drop the
    1980s, which are the events only the location text reaches."""
    cache_dir = write_emdat_cache(
        [
            emdat_event({"DisNo.": "eighties", "Start Year": 1985}),
            emdat_event({"DisNo.": "nineties", "Start Year": 1995}),
        ]
    )

    assert EventFilters().start_year == 1981
    assert selected_events(cache_dir)["DisNo."].to_list() == ["eighties", "nineties"]


def test_prose_naming_ground_nothing_else_reaches_is_added_to_a_coded_event(write_emdat_cache):
    """The positive half of the ordering rule. An event can be coded and still have prose naming
    somewhere the coding missed, which is 596 units across the region for the overlay and is the
    same argument for the text."""
    cache_dir = write_emdat_cache(
        [emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "AAA.1_1"})})]
    )

    geography = event_geography(
        load_emdat_events(cache_dir), from_text=text_units([("coded", "AAA.9_1", "Elsewhere", 1)])
    )

    assert sorted(geography["gid"]) == ["AAA.1_1", "AAA.9_1"]
    assert set(geography["geometry_source"]) == {"gadm", "location_text"}


def test_the_resolution_counts_reach_rows_of_a_tier_the_prose_did_not_place(write_emdat_cache):
    """The counts describe the prose, not the tier, so they belong on an event EM-DAT coded as much
    as on one the text placed — that is what makes them usable as a completeness covariate rather
    than a property of one tier."""
    cache_dir = write_emdat_cache(
        [emdat_event({"DisNo.": "coded", "GADM Admin Units": units_json({"gid_1": "AAA.1_1"})})]
    )

    geography = event_geography(load_emdat_events(cache_dir), text_resolution=resolution_counts([("coded", 3, 1)]))

    assert geography["geometry_source"].to_list() == ["gadm"]
    assert geography["names_written"].to_list() == [3]
    assert geography["names_reached"].to_list() == [1]


def test_resolution_counts_naming_an_event_twice_are_refused(write_emdat_cache):
    """They are attached by a join on the event alone, so a repeated event fans out every geography
    row it has — including rows of tiers the reading had nothing to do with. Silently doubling a
    table is worse than refusing to build one."""
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "twice", "Latitude": None, "Longitude": None})])

    with pytest.raises(DataValidationError, match="twice"):
        event_geography(
            load_emdat_events(cache_dir), text_resolution=resolution_counts([("twice", 5, 2), ("twice", 5, 2)])
        )
