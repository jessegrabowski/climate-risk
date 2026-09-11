import re

from climate_risk.data import geonames, place_names
from climate_risk.data.geonames import (
    GEONAMES_LICENSE,
    country_dump,
    geonames_geocoder,
    load_place_points,
    read_country_codes,
)


def test_a_country_is_looked_up_by_the_code_geonames_files_it_under(write_geonames_cache):
    """EM-DAT keys on alpha-3 and GeoNames publishes on alpha-2, so every lookup crosses that gap."""
    codes = read_country_codes(write_geonames_cache())

    assert codes["PHL"] == "PH"


def test_a_dump_is_declared_under_the_license_it_is_published_with():
    """The archive is redistributable only with attribution, and the declaration is what carries it."""
    assert country_dump("PH").license == GEONAMES_LICENSE


def test_every_spelling_a_place_is_published_under_reaches_it(write_geonames_cache):
    """Written mentions use whichever spelling the reporter had; a lookup on the primary name alone
    misses the alternates, which is where exonyms and older names live."""
    locate = geonames_geocoder("PHL", write_geonames_cache())

    assert locate("PHL", "Bacolod") == locate("PHL", "Bakolod") == (122.95, 10.667)


def test_the_most_populous_place_keeps_a_shared_name(write_geonames_cache):
    """A city and a hamlet share a name far more often than not. Taking whichever was read first
    puts a national-scale disaster in a village."""
    locate = geonames_geocoder("PHL", write_geonames_cache())

    assert locate("PHL", "Bacolod") == (122.95, 10.667), "the city, not the barangay at 8.5N"


def test_a_place_the_dump_does_not_carry_answers_with_nothing(write_geonames_cache):
    """Two fifths of written places are misspelled, relational or not places at all, and the
    scoring rig counts a silence apart from a wrong answer."""
    locate = geonames_geocoder("PHL", write_geonames_cache())

    assert locate("PHL", "Bocolod") is None


def test_a_mention_carrying_a_unit_noun_still_reaches_the_place(write_geonames_cache):
    """EM-DAT writes `Iloilo city` where GeoNames writes `Iloilo`, so both sides have to be keyed
    the same way for the two to meet."""
    locate = geonames_geocoder("PHL", write_geonames_cache())

    assert locate("PHL", "Iloilo city") == (122.567, 10.7)


def test_a_dump_row_that_is_not_a_settlement_is_still_indexed(write_geonames_cache):
    """Seas and straits are 8% of what fails to resolve. They belong to no administrative unit, but
    a point in one still says where an event was."""
    locate = geonames_geocoder("PHL", write_geonames_cache())

    assert locate("PHL", "Sulu Sea") == (120.0, 8.0)


def test_the_places_table_carries_one_row_for_each_distinct_name(write_geonames_cache):
    """Two spellings of one place are two ways in; two places sharing a spelling are one row."""
    points = load_place_points("PHL", write_geonames_cache())

    assert sorted(points["key"]) == ["bacolod", "bakolod", "iloilo", "sulusea"]


def test_changing_how_a_name_is_keyed_turns_the_cached_places_over(write_geonames_cache, monkeypatch):
    """The places table is keyed by `match_key`, the same as the gazetteer it is matched against.
    Cached on the country alone it serves keys built under rules that no longer apply, and every
    mention whose key moved stops reaching its point."""
    cache_dir = write_geonames_cache()
    load_place_points("PHL", cache_dir)

    monkeypatch.setattr(place_names, "UNIT_NOUNS", re.compile(r"\b(bacolod)\b", re.IGNORECASE))
    rekeyed = load_place_points("PHL", cache_dir)

    assert "bacolod" not in rekeyed["key"].to_list(), "the stripped word cannot survive as a key"


def test_changing_which_columns_the_dump_is_read_from_turns_the_cached_places_over(write_geonames_cache, monkeypatch):
    """The dump is headerless, so which field is latitude lives in a table the builder consults and
    its own source never shows. Reading them the other way round puts every place in the wrong
    hemisphere, and a cache that cannot see that table change keeps serving the old points."""
    cache_dir = write_geonames_cache()
    upright = load_place_points("PHL", cache_dir)

    monkeypatch.setattr(geonames, "DUMP_FIELDS", {**geonames.DUMP_FIELDS, 4: "lon", 5: "lat"})
    swapped = load_place_points("PHL", cache_dir)

    assert swapped.sort("key")["lat"].to_list() != upright.sort("key")["lat"].to_list()
