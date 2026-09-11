import contextlib

import polars as pl
import pytest
import requests

from climate_risk.data import osm
from climate_risk.data.osm import (
    NOMINATIM,
    PLACE_CATEGORIES,
    NominatimAnswer,
    osm_geocoder,
    read_osm_places,
    record_lookups,
)
from climate_risk.exceptions import UpstreamUnavailableError

# Nominatim answers over the wire, where every field is a string.
A_VILLAGE = {"lon": "1.23", "lat": "6.16", "category": "place", "type": "village"}


class _FakeResponse:
    """Stands in for what `requests` hands back, so no socket is opened."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_a_place_reaches_the_point_osm_gave_it(write_osm_cache):
    """The whole point of the cache: a name GADM never published still reaches a coordinate."""
    cache_dir = write_osm_cache([("Dzidzole", 1.23, 6.16, "place", "village")])

    assert osm_geocoder("LAO", cache_dir)("LAO", "Dzidzole") == (1.23, 6.16)


def test_a_pharmacy_named_after_a_village_is_not_offered_as_the_village(write_osm_cache):
    """Nominatim answers with whatever carries the name. Of 30 hits on the residual, over half were
    a building, a road or an amenity — each of which lands a point somewhere plausible and wrong."""
    cache_dir = write_osm_cache([("Dzidzole", 9.99, 9.99, "amenity", "pharmacy")])

    assert osm_geocoder("LAO", cache_dir)("LAO", "Dzidzole") is None


def test_a_name_osm_could_not_find_is_remembered_as_unfound(write_osm_cache):
    """A miss is a result. Without keeping it, every rebuild asks Nominatim the whole question again,
    which is the one thing its usage policy forbids."""
    cache_dir = write_osm_cache([("Nowhere At All", None, None, None, None)])

    assert osm_geocoder("LAO", cache_dir)("LAO", "Nowhere At All") is None
    assert read_osm_places("LAO", cache_dir)["written"].to_list() == ["Nowhere At All"]


def test_a_mention_carrying_a_unit_noun_still_reaches_the_point(write_osm_cache):
    """EM-DAT says what kind of unit it means; the crawl asked under the name alone."""
    cache_dir = write_osm_cache([("Dzidzole", 1.23, 6.16, "place", "village")])

    assert osm_geocoder("LAO", cache_dir)("LAO", "Dzidzole village") == (1.23, 6.16)


def test_a_country_nobody_has_crawled_answers_with_nothing(tmp_path):
    """A cold cache is a country not yet asked about, not an error: the resolver falls through to
    every other reading exactly as it does when GeoNames has no dump."""
    assert osm_geocoder("LAO", tmp_path)("LAO", "Dzidzole") is None


def test_answers_recorded_twice_keep_the_later_one(write_osm_cache):
    """A recrawl after Nominatim improves its data must not leave two rows for one name, which
    would resolve on file order."""
    cache_dir = write_osm_cache([("Dzidzole", 0.0, 0.0, "amenity", "pharmacy")])

    record_lookups("LAO", cache_dir, {"Dzidzole": NominatimAnswer(1.23, 6.16, "place", "village")})

    assert read_osm_places("LAO", cache_dir).height == 1
    assert osm_geocoder("LAO", cache_dir)("LAO", "Dzidzole") == (1.23, 6.16)


def test_recording_an_answer_leaves_the_names_already_asked_about_alone(write_osm_cache):
    """The crawl runs country by country over many sittings, and each one has to add to the cache
    rather than replace it."""
    cache_dir = write_osm_cache([("Kegue", 1.22, 6.19, "place", "quarter")])

    record_lookups(
        "LAO", cache_dir, {"Dzidzole": NominatimAnswer(1.23, 6.16, "place", "village"), "Nowhere At All": None}
    )

    assert sorted(read_osm_places("LAO", cache_dir)["written"].to_list()) == ["Dzidzole", "Kegue", "Nowhere At All"]


def test_a_miss_is_recorded_with_no_coordinates(write_osm_cache):
    cache_dir = write_osm_cache([])

    recorded = record_lookups("LAO", cache_dir, {"Nowhere At All": None})

    assert recorded.filter(pl.col("written") == "Nowhere At All")["lon"].to_list() == [None]


def test_one_country_does_not_answer_for_another(write_osm_cache):
    """Nominatim was asked with the country fixed, so an answer is only good for that country."""
    cache_dir = write_osm_cache([("Dzidzole", 1.23, 6.16, "place", "village")], iso="TGO")

    assert osm_geocoder("TGO", cache_dir)("TGO", "Dzidzole") == (1.23, 6.16)
    assert osm_geocoder("LAO", cache_dir)("LAO", "Dzidzole") is None


def test_openstreetmap_is_declared_under_the_license_it_is_published_with():
    """ODbL carries attribution and share-alike obligations that the other sources here do not, so
    the terms travel with the data rather than living in a README."""
    assert NOMINATIM.license == "ODbL 1.0"


@pytest.mark.parametrize("category", PLACE_CATEGORIES)
def test_every_trusted_category_is_actually_offered(write_osm_cache, category):
    """The filter is the only thing standing between a point and a wrong province, so a category
    named in it that the geocoder then drops would be silently useless."""
    cache_dir = write_osm_cache([("Somewhere", 1.0, 2.0, category, "whatever")])

    assert osm_geocoder("LAO", cache_dir)("LAO", "Somewhere") == (1.0, 2.0)


def test_a_request_that_never_reached_nominatim_is_not_an_answer(monkeypatch):
    """A failed request and a name Nominatim has no place for are different facts. Conflating them
    caches a permanent absence for a name nobody ever managed to ask about, and nothing afterwards
    can tell it from a real one — which is the whole value of caching the misses."""
    monkeypatch.setattr(osm.requests, "get", lambda *_, **__: (_ for _ in ()).throw(requests.ConnectionError()))
    monkeypatch.setattr(osm.time, "sleep", lambda _: None)

    with pytest.raises(UpstreamUnavailableError, match="Dzidzole"):
        osm.search_nominatim("Dzidzole", "tg")


def test_a_name_nominatim_has_no_place_for_is_an_answer(monkeypatch):
    """The other half of the same contract: an empty result is a fact worth caching, not a failure."""
    monkeypatch.setattr(osm.requests, "get", lambda *_, **__: _FakeResponse([]))
    monkeypatch.setattr(osm.time, "sleep", lambda _: None)

    assert osm.search_nominatim("Nowhere At All", "tg") is None


@pytest.mark.parametrize(
    "answer",
    [lambda *a, **k: _FakeResponse([A_VILLAGE]), lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError())],
    ids=["an answer", "a failure"],
)
def test_every_request_waits_out_the_rate_limit(monkeypatch, answer):
    """One request a second is a condition of using the public instance, not a tuning knob. A crawl
    that skips the wait after a failed request gets the project blocked, so the wait is in the
    `finally` and both paths are held to it."""
    waited: list[float] = []
    monkeypatch.setattr(osm.requests, "get", answer)
    monkeypatch.setattr(osm.time, "sleep", waited.append)

    with contextlib.suppress(UpstreamUnavailableError):
        osm.search_nominatim("Dzidzole", "tg")

    assert waited == [osm.RATE_LIMIT_SECONDS]


def test_a_change_to_the_keying_rules_does_not_orphan_a_crawl(write_osm_cache, monkeypatch):
    """The written name is what Nominatim was asked about, and the key is derived from it. Storing
    the key instead would strand every answer the moment the keying rules moved — and a re-crawl
    could not repair it, because the crawl skips names it has already asked about."""
    cache_dir = write_osm_cache([("Dzidzole town", 1.23, 6.16, "place", "village")])
    monkeypatch.setattr(osm, "match_key", lambda name: name.replace(" ", "").upper())

    assert osm.osm_geocoder("LAO", cache_dir)("LAO", "Dzidzole town") == (1.23, 6.16)
