import geopandas as gpd
import numpy as np
import pytest

from shapely.geometry import LineString

from climate_risk.config.schema import EventFilters
from climate_risk.data_functions import rivers_damage
from climate_risk.data_functions.rivers_damage import (
    create_floods_rivers_damage,
    create_hydro_rivers_damage,
)
from tests.conftest import emdat_event, toy_world_needing_repair

# A Laos flood whose coordinates the loader overrides from LAOS_LOCATION_DICTIONARY.
PATCHED_EVENT = "2013-0338-LAO"
HALF_LOCATED_LATITUDE = 20.5
UNDAMAGED_LATITUDE = 17.25
PATCHED_LATITUDE = 19.5
PATCHED_LONGITUDE = 103.5
UNPATCHED_LATITUDE = 1.25
FLOOD_LATITUDE = 18.25
STORM_LATITUDE = 18.5
DROUGHT_LATITUDE = 18.75
BARELY_FELT_LATITUDE = 17.75

VARIANTS = {
    "hydro": (create_hydro_rivers_damage, "Total_Damage_Hydro", "log_affected_hydro"),
    "floods": (create_floods_rivers_damage, "Total_Damage_Flood", "log_affected_floods"),
}


@pytest.fixture
def rivers_two_degrees_east():
    """A single big river offset from every event, so distances are positive and comparable."""
    return gpd.GeoDataFrame(
        {"ORD_FLOW": [4], "HYRIV_ID": [1], "geometry": [LineString([(104.0, 10.0), (104.0, 25.0)])]},
        crs="EPSG:4326",
    )


@pytest.fixture
def damage_cache(tmp_path, write_emdat_cache, write_shapefile_cache, write_rivers_cache, rivers_two_degrees_east):
    write_shapefile_cache("world", toy_world_needing_repair())
    write_rivers_cache(rivers_two_degrees_east)
    write_emdat_cache(
        [
            # Fractional coordinates, as EM-DAT supplies them: whole numbers make the column
            # integral through the workbook, and the coordinate patch then cannot write into it.
            emdat_event(
                {
                    "ISO": "LAO",
                    "DisNo.": "LAO-flood",
                    "Disaster Type": "Flood",
                    "Latitude": FLOOD_LATITUDE,
                    "Longitude": 102.25,
                }
            ),
            emdat_event(
                {
                    "ISO": "LAO",
                    "DisNo.": "LAO-storm",
                    "Disaster Type": "Storm",
                    "Latitude": STORM_LATITUDE,
                    "Longitude": 102.5,
                }
            ),
            emdat_event(
                {
                    "ISO": "LAO",
                    "DisNo.": "LAO-drought",
                    "Disaster Type": "Drought",
                    "Latitude": DROUGHT_LATITUDE,
                    "Longitude": 102.75,
                }
            ),
            # No longitude, so the event cannot be placed and must not reach the frame.
            emdat_event(
                {
                    "ISO": "LAO",
                    "DisNo.": "LAO-halfplaced",
                    "Disaster Type": "Flood",
                    "Latitude": HALF_LOCATED_LATITUDE,
                    "Longitude": None,
                }
            ),
            # Zero damage means unrecorded rather than free, so the total becomes missing.
            emdat_event(
                {
                    "ISO": "LAO",
                    "DisNo.": "LAO-undamaged",
                    "Disaster Type": "Flood",
                    "Latitude": UNDAMAGED_LATITUDE,
                    "Longitude": 101.25,
                    "Total Damage ('000 US$)": 0,
                }
            ),
            # Below the damage panel's reach floor, so it must not enter the regressions.
            emdat_event(
                {
                    "ISO": "LAO",
                    "DisNo.": "LAO-barelyfelt",
                    "Disaster Type": "Flood",
                    "Latitude": BARELY_FELT_LATITUDE,
                    "Longitude": 101.75,
                    "Total Affected": 30,
                }
            ),
            emdat_event(
                {
                    "ISO": "LAO",
                    "DisNo.": PATCHED_EVENT,
                    "Disaster Type": "Flood",
                    "Latitude": UNPATCHED_LATITUDE,
                    "Longitude": 1.75,
                }
            ),
        ]
    )
    return tmp_path


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
def test_the_cached_frame_matches_the_one_that_wrote_it(damage_cache, variant):
    create, _, _ = VARIANTS[variant]

    cold = create(damage_cache)
    warm = create(damage_cache)

    assert list(warm.columns) == list(cold.columns)
    assert (warm.dtypes == cold.dtypes).all()


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
def test_the_totals_and_their_logs_are_suffixed_per_variant(damage_cache, variant):
    """The suffixes are the only thing separating the two frames' columns, so a consumer can hold both."""
    create, total, log = VARIANTS[variant]

    damage = create(damage_cache)

    assert total in damage.columns
    assert log in damage.columns


def test_distance_to_the_nearest_river_is_in_kilometers(damage_cache):
    """The column feeds a regression in kilometers; meters would inflate it a thousandfold."""
    damage = create_floods_rivers_damage(damage_cache)

    # Every event sits one to three degrees off the river, so tens to hundreds of kilometers. The
    # lower bound is what catches a scale factor applied the wrong way.
    assert damage["closest_river"].between(10, 1_000).all()


@pytest.mark.parametrize(
    ("create", "selected"),
    [
        (create_hydro_rivers_damage, {FLOOD_LATITUDE, STORM_LATITUDE, UNPATCHED_LATITUDE, UNDAMAGED_LATITUDE}),
        (create_floods_rivers_damage, {FLOOD_LATITUDE, PATCHED_LATITUDE, UNDAMAGED_LATITUDE}),
    ],
    ids=["hydro", "floods"],
)
def test_each_variant_selects_its_own_events(damage_cache, create, selected):
    """The drought is climatological and the storm is not a flood; each frame must exclude them."""
    damage = create(damage_cache)

    assert set(damage["Latitude"]) == selected


def test_laos_flood_coordinates_are_overridden(damage_cache):
    """EM-DAT records these events against the wrong place, so the loader forces the coordinates."""
    damage = create_floods_rivers_damage(damage_cache)

    patched = damage[damage["Latitude"] == PATCHED_LATITUDE]

    assert len(patched) == 1
    assert patched["Longitude"].item() == PATCHED_LONGITUDE
    assert UNPATCHED_LATITUDE not in set(damage["Latitude"])


def test_the_year_survives_the_round_trip_as_a_timestamp(damage_cache):
    """Downstream reaches for `.dt`, which a string year cannot answer."""
    create_hydro_rivers_damage(damage_cache)
    warm = create_hydro_rivers_damage(damage_cache)

    assert set(warm["year"].dt.year) == {1990}


def test_an_event_missing_one_coordinate_is_dropped(damage_cache):
    """A half-located event yields POINT (NaN lat), which measures a distance to nothing."""
    damage = create_floods_rivers_damage(damage_cache)

    assert HALF_LOCATED_LATITUDE not in set(damage["Latitude"])
    assert damage.geometry.is_valid.all()


def test_an_event_below_the_reach_floor_stays_out_of_the_damage_frame(damage_cache):
    """The damage panel sets its own reach floor, so its regressions are fitted on the same events
    whatever thresholds a place carries."""
    damage = create_floods_rivers_damage(damage_cache)

    assert BARELY_FELT_LATITUDE not in set(damage["Latitude"])


def test_zero_damage_becomes_missing_rather_than_a_log_of_zero(damage_cache):
    """Zero means unrecorded here, and log(0) would put -inf into the regressor."""
    damage = create_floods_rivers_damage(damage_cache)

    undamaged = damage[damage["Latitude"] == UNDAMAGED_LATITUDE]

    assert len(undamaged) == 1
    assert undamaged["Total_Damage_Flood"].isna().all()
    assert not np.isneginf(damage["log_damage_floods"]).any()


def test_a_changed_reach_floor_rebuilds_rather_than_reading_the_old_frame(damage_cache, monkeypatch):
    """The floor is read by the builder and named nowhere in the cache key, so without a fingerprint
    over it the entry from the old floor reads back as a hit."""
    strict = create_floods_rivers_damage(damage_cache)

    monkeypatch.setattr(rivers_damage, "REPLICATION_FILTERS", EventFilters(min_total_affected=10))
    permissive = create_floods_rivers_damage(damage_cache)

    assert BARELY_FELT_LATITUDE not in set(strict["Latitude"])
    assert BARELY_FELT_LATITUDE in set(permissive["Latitude"])


def test_only_the_floods_frame_carries_the_location_text(damage_cache):
    """`Location` reaches the frame through extra_columns, which only the floods variant asks for."""
    floods = create_floods_rivers_damage(damage_cache)
    hydro = create_hydro_rivers_damage(damage_cache)

    assert "Location" in floods.columns
    assert "Location" not in hydro.columns
