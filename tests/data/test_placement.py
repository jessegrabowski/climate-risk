from climate_risk.data.placement import available_geocoders, resolve_location_text
from climate_risk.data_functions.emdat_processing import load_emdat_events
from tests.conftest import emdat_event


def test_a_gazetteer_of_settlements_is_asked_before_a_gazetteer_of_everything(write_geonames_cache, write_osm_cache):
    """GeoNames rows are populated places, so a name it knows is a settlement. Nominatim answers
    with whatever carries the name, which is worth having only where GeoNames has nothing."""
    write_geonames_cache()
    cache_dir = write_osm_cache([("Bacolod", 0.0, 0.0, "place", "village")], iso="PHL")

    first, _ = available_geocoders("PHL", cache_dir)

    assert first("PHL", "Bacolod") == (122.95, 10.667), "the GeoNames city, not the OSM point"


def test_a_name_geonames_has_no_row_for_reaches_the_openstreetmap_point(write_geonames_cache, write_osm_cache):
    """The whole reason for the second source: two thirds of what stays unplaced has no GeoNames
    row at all."""
    write_geonames_cache()
    cache_dir = write_osm_cache([("Dzidzole", 1.23, 6.16, "place", "village")], iso="PHL")

    answers = [point for locate in available_geocoders("PHL", cache_dir) if (point := locate("PHL", "Dzidzole"))]

    assert answers == [(1.23, 6.16)]


def test_a_country_with_no_geonames_dump_still_offers_openstreetmap(write_geonames_cache, write_osm_cache):
    """GeoNames files its dumps by country and does not publish one for every country. A source
    that cannot answer has to step aside, not take the country's other points down with it."""
    write_geonames_cache(alpha2="PH", alpha3="PHL")
    cache_dir = write_osm_cache([("Dzidzole", 1.23, 6.16, "place", "village")], iso="LAO")

    (only,) = available_geocoders("LAO", cache_dir)

    assert only("LAO", "Dzidzole") == (1.23, 6.16)


def lao_event(disno, location, **overrides):
    """A Lao event whose prose is the only geography it carries."""
    return emdat_event({"DisNo.": disno, "ISO": "LAO", "Location": location, **overrides})


def test_prose_that_names_a_unit_reaches_it_with_the_level_the_gazetteer_gives(
    write_gadm_cache, write_emdat_cache, write_geonames_cache
):
    """The whole point of reading the text: EM-DAT coded no units for two thirds of the workbook,
    and the prose names them."""
    write_gadm_cache()
    write_geonames_cache()
    cache_dir = write_emdat_cache([lao_event("prose", "Sanamxay district")])

    from_text, _ = resolve_location_text(load_emdat_events(cache_dir), cache_dir)

    assert from_text["gid"].to_list() == ["LAO.1.1_1"]
    assert from_text["admin_level"].to_list() == [2], "the level comes off the gazetteer, not the identifier"


def test_a_name_that_can_only_be_a_feature_counts_in_neither_column(
    write_gadm_cache, write_emdat_cache, write_geonames_cache
):
    """A sea is not a unit and never will be, so counting it as a name that failed to resolve would
    make the prose look worse than it read."""
    write_gadm_cache()
    write_geonames_cache()
    cache_dir = write_emdat_cache([lao_event("feature", "Sanamxay, Java Sea")])

    _, resolution = resolve_location_text(load_emdat_events(cache_dir), cache_dir)

    assert resolution["names_written"].to_list() == [1]
    assert resolution["names_reached"].to_list() == [1]


def test_an_event_em_dat_already_coded_is_left_alone(write_gadm_cache, write_emdat_cache, write_geonames_cache):
    """The coded units are better evidence than the prose, and reading the text anyway would put a
    second tier of rows behind the one that already won."""
    write_gadm_cache()
    write_geonames_cache()
    cache_dir = write_emdat_cache(
        [
            lao_event("coded", "Sanamxay", **{"GADM Admin Units": '[{"gid_1": "LAO.1_1"}]'}),
            lao_event("uncoded", "Sanamxay"),
        ]
    )

    from_text, resolution = resolve_location_text(load_emdat_events(cache_dir), cache_dir)

    assert from_text["DisNo."].unique().to_list() == ["uncoded"]
    assert resolution["DisNo."].to_list() == ["uncoded"]


def test_a_country_with_no_gazetteer_is_skipped_rather_than_raised_over(
    write_gadm_cache, write_emdat_cache, write_geonames_cache
):
    """GADM does not cover every ISO code EM-DAT files under, and one missing country must not stop
    the other hundred and seventy."""
    write_gadm_cache()
    write_geonames_cache()
    cache_dir = write_emdat_cache(
        [
            lao_event("known", "Sanamxay"),
            emdat_event({"DisNo.": "nowhere", "ISO": "XKX", "Location": "Somewhere"}),
        ]
    )

    from_text, _ = resolve_location_text(load_emdat_events(cache_dir), cache_dir)

    assert from_text["DisNo."].unique().to_list() == ["known"]


def test_the_level_is_read_off_the_gazetteer_rather_than_counted_in_the_identifier(
    write_gadm_cache, write_emdat_cache, write_geonames_cache
):
    """GADM does not key every country the same way. Ghana's level-1 identifiers carry no dot at
    all, so counting separators there reports every unit one level too coarse."""
    write_gadm_cache()
    write_geonames_cache()
    cache_dir = write_emdat_cache([emdat_event({"DisNo.": "ghana", "ISO": "GHA", "Location": "Savannah"})])

    from_text, _ = resolve_location_text(load_emdat_events(cache_dir), cache_dir)

    assert from_text["gid"].to_list() == ["GHA11_2"]
    assert from_text["admin_level"].to_list() == [1], "no dot in the identifier, but still level one"


def test_a_partly_resolved_event_records_both_counts(write_gadm_cache, write_emdat_cache, write_geonames_cache):
    """One in ten events resolves some of its names and not others. The pair of counts is what tells
    a reader whether a window is the whole of what the prose said or only the part that landed."""
    write_gadm_cache()
    write_geonames_cache()
    cache_dir = write_emdat_cache([lao_event("partial", "Sanamxay, Blargville")])

    from_text, resolution = resolve_location_text(load_emdat_events(cache_dir), cache_dir)

    assert resolution["names_written"].to_list() == [2]
    assert resolution["names_reached"].to_list() == [1]
    assert from_text["gid"].to_list() == ["LAO.1.1_1"], "only the name that landed reaches a unit"
