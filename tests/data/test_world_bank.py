import logging

import polars as pl
import pytest
import requests

from polars.testing import assert_frame_equal

from climate_risk.data import world_bank
from climate_risk.data.world_bank import (
    COUNTRIES_FILE,
    COUNTRY_CODE_BY_NAME,
    INDICATOR_NAMES,
    REQUESTED_COUNTRY_CODES,
    WB_INDICATORS,
    WORLD_BANK,
    load_wb_data,
    transform_world_bank,
)


def downloaded(rows) -> pl.DataFrame:
    """Indicators shaped as kuznets returns them tidy, with the year dated rather than numbered."""
    frame = pl.DataFrame(rows, schema=["country", "year", *WB_INDICATORS], orient="row")

    return frame.with_columns(pl.col("year").str.to_datetime("%Y"))


def row(country: str = "Aruba", year: str = "1990", **values: float) -> tuple:
    """One tidy row, each indicator numbered by its position unless named under its readable name."""
    numbered = {name: float(position) for position, name in enumerate(INDICATOR_NAMES.values())}

    return (country, year, *(values.get(name, numbered[name]) for name in INDICATOR_NAMES.values()))


@pytest.fixture
def serves(monkeypatch):
    """Answer wb.download from a frame instead of the network, recording the arguments used."""

    def serve(frame):
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return frame

        monkeypatch.setattr(world_bank.wb, "download", fake_download)
        return calls

    return serve


def test_indicators_are_keyed_by_iso_code_and_year():
    frame = transform_world_bank(downloaded([row()]), INDICATOR_NAMES)

    assert frame.columns[:2] == ["country_code", "year"]
    assert frame.select("country_code", "year").rows() == [("ABW", 1990)]


def test_the_dated_year_becomes_an_integer():
    """Upstream dates the year; casting that date rather than reading it yields microseconds."""
    frame = transform_world_bank(downloaded([row()]), INDICATOR_NAMES)

    assert frame.schema["year"] == pl.Int64


def test_indicator_codes_become_readable_names():
    frame = transform_world_bank(downloaded([row(gdp_per_cap_usd=1000.0)]), INDICATOR_NAMES)

    assert "NY.GDP.PCAP.KD" not in frame.columns
    assert frame["gdp_per_cap_usd"].to_list() == [1000.0]


def test_a_constant_price_series_says_which_currency_in_its_name():
    """`downloaded` builds its columns from WB_INDICATORS, so a renaming test agrees with whatever
    code is listed and cannot see a wrong one. The codes carry the units: KD is constant 2015 US$
    and KN is constant local currency. Ratios formed within a country need one of them and levels
    compared across countries need the other, so a name that does not say which produces a mix
    nothing downstream can see. The check runs both ways, since a KN code named `_usd` is as wrong
    as a KD code named `_lcu`.
    """
    dollars = {code for code in INDICATOR_NAMES if code.endswith(".KD")}
    local_currency = {code for code in INDICATOR_NAMES if code.endswith(".KN")}

    assert dollars and local_currency
    assert dollars == {code for code, name in INDICATOR_NAMES.items() if name.endswith("_usd")}
    assert local_currency == {code for code, name in INDICATOR_NAMES.items() if name.endswith("_lcu")}


def test_no_two_indicators_share_a_name():
    """The names become columns, so a repeated one would silently drop an indicator from the panel.
    Case is folded first: two columns differing only in case read as one quantity to a reader, and
    the panel carries both a population count and a population density.
    """
    names = [name.lower() for name in INDICATOR_NAMES.values()]

    assert len(names) == len(set(names))


def test_every_current_price_quantity_has_a_constant_price_counterpart():
    """The pair's ratio is the deflator, which is what carries import prices when no import price
    index exists. A current-price series whose constant-price counterpart is missing carries none.
    """
    current = {code.removesuffix(".CN") for code in INDICATOR_NAMES if code.endswith(".CN")}
    constant = {code.removesuffix(".KN") for code in INDICATOR_NAMES if code.endswith(".KN")}

    assert current, "no current-price quantities found"
    assert current <= constant, sorted(current - constant)


def test_a_country_with_no_iso_code_is_dropped():
    """An unmatched name would otherwise key a row on a null and survive into the panel."""
    raw = downloaded([row(), row(country="Not A Country")])

    frame = transform_world_bank(raw, INDICATOR_NAMES)

    assert frame["country_code"].to_list() == ["ABW"]


def test_the_result_is_sorted_by_country_and_year():
    raw = downloaded([row(country="Zimbabwe", year="1991"), row(year="1991"), row(year="1990")])

    frame = transform_world_bank(raw, INDICATOR_NAMES)

    assert frame.select("country_code", "year").rows() == [("ABW", 1990), ("ABW", 1991), ("ZWE", 1991)]


def test_a_warm_cache_does_not_download(tmp_path, serves):
    """The download is hundreds of requests; a present cache must not trigger it."""
    calls = serves(downloaded([row()]))

    load_wb_data(tmp_path)
    frame = load_wb_data(tmp_path)

    assert len(calls) == 1
    assert frame.select("country_code", "year").rows() == [("ABW", 1990)]


def test_the_cold_run_writes_the_cache_it_will_read(tmp_path, serves):
    """A key spelled one way on write and another on read is the bug this replaces."""
    serves(downloaded([row()]))

    load_wb_data(tmp_path)

    assert len(list(tmp_path.glob("world_bank__*.parquet"))) == 1


def test_the_cold_and_warm_frames_agree(tmp_path, serves):
    """The hand-rolled cache this replaces returned a string year cold and an integer year warm."""
    serves(downloaded([row()]))

    cold = load_wb_data(tmp_path)
    warm = load_wb_data(tmp_path)

    assert_frame_equal(cold, warm)


def test_forcing_a_reload_downloads_again(tmp_path, serves):
    calls = serves(downloaded([row()]))

    load_wb_data(tmp_path)
    load_wb_data(tmp_path, force_reload=True)

    assert len(calls) == 2


def test_the_download_asks_for_every_indicator(tmp_path, serves):
    """One download serves the whole panel, so a code left out of the request is a missing column."""
    calls = serves(downloaded([row()]))

    load_wb_data(tmp_path)

    assert calls[0]["indicator"] == WB_INDICATORS


def test_the_download_reaches_back_before_any_indicator_starts(tmp_path, serves):
    """A later start year would silently shorten every series in the panel."""
    calls = serves(downloaded([row()]))

    load_wb_data(tmp_path)

    assert calls[0]["start"] == 1900


def test_the_download_covers_every_requested_country(tmp_path, serves):
    """The model is estimated per country, so the panel must not be narrowed to any one of them."""
    calls = serves(downloaded([row()]))

    load_wb_data(tmp_path)

    assert calls[0]["country"] == REQUESTED_COUNTRY_CODES


def test_every_country_code_is_an_iso_alpha_3():
    """The table is hand-edited; a lower-case or truncated code would fail only on a cold run."""
    malformed = [code for code in COUNTRY_CODE_BY_NAME.values() if not (len(code) == 3 and code.isupper())]

    assert malformed == []


def test_no_two_countries_share_a_code():
    """Two names on one code would silently collapse rows when the download is keyed by code."""
    codes = list(COUNTRY_CODE_BY_NAME.values())

    assert len(codes) == len(set(codes))


def test_kosovo_is_requested_under_the_code_the_world_bank_uses():
    """The World Bank serves Kosovo under a code no ISO 3166-1 standard assigns."""
    assert COUNTRY_CODE_BY_NAME["Kosovo"] == "XKX"
    assert "XKX" in REQUESTED_COUNTRY_CODES


def test_the_aggregates_are_not_requested():
    """The Bank returns regional and income aggregates; asking for them would double-count."""
    assert "ARB" not in REQUESTED_COUNTRY_CODES
    assert COUNTRY_CODE_BY_NAME["Arab World"] == "ARB"


def test_no_two_countries_share_a_name():
    """The mapping is built with dict(zip(...)), so a repeated name would quietly overwrite a code."""
    table = pl.read_csv(COUNTRIES_FILE)

    assert len(COUNTRY_CODE_BY_NAME) == len(table)


def test_dropping_an_unmatched_country_says_which_one(caplog):
    """Silent dropping is the failure mode; the warning is the only trace it leaves."""
    raw = downloaded([row(), row(country="Not A Country")])

    with caplog.at_level(logging.WARNING, logger="climate_risk.data.world_bank"):
        transform_world_bank(raw, INDICATOR_NAMES)

    assert "Not A Country" in caplog.text


def test_a_backend_other_than_polars_is_rejected(tmp_path, monkeypatch):
    """The transform is polars-only, so a frame from another backend must fail here, not in a select."""
    monkeypatch.setattr(world_bank.wb, "download", lambda **kwargs: object())

    with pytest.raises(TypeError, match="output_type='polars'"):
        load_wb_data(tmp_path)


def test_an_indicator_the_bank_no_longer_serves_is_named():
    """kuznets warns and omits the column rather than raising, so without this the failure surfaces
    from the select as a polars error naming a column, after the whole panel has been downloaded.
    """
    retired = downloaded([row()]).drop("AG.SRF.TOTL.K2")

    with pytest.raises(ValueError, match=r"AG\.SRF\.TOTL\.K2"):
        transform_world_bank(retired, INDICATOR_NAMES)


@pytest.mark.network
def test_every_requested_country_is_listed_under_the_name_the_bank_serves():
    """The download is keyed by ISO code and the panel is keyed by the name the answer carries, so a
    country the Bank renames is asked for, returned, and then dropped for having no code. Nothing
    offline can see this: the mapping and the panel both come from `COUNTRY_CODE_BY_NAME`, and they
    agree with each other whatever the Bank calls the country.
    """
    response = requests.get(WORLD_BANK.url, timeout=30, params={"format": "json", "per_page": "400"})
    response.raise_for_status()
    published = {country["id"]: country["name"] for country in response.json()[1]}

    requested = set(REQUESTED_COUNTRY_CODES)
    renamed = {
        code: (name, published.get(code))
        for name, code in COUNTRY_CODE_BY_NAME.items()
        if code in requested and published.get(code) != name
    }

    assert renamed == {}


def test_editing_the_country_table_turns_the_cache_over(tmp_path, serves, monkeypatch):
    """The panel's rows are the country table's rows, so a table edit that did not reach the key
    would read back the panel the old table produced, which is a country quietly still missing.
    """
    serves(downloaded([row()]))
    load_wb_data(tmp_path)

    monkeypatch.setattr(world_bank, "REQUESTED_COUNTRY_CODES", [*REQUESTED_COUNTRY_CODES, "ZZZ"])
    load_wb_data(tmp_path)

    assert len(list(tmp_path.glob("world_bank__*.parquet"))) == 2
