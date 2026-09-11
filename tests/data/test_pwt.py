import polars as pl
import pytest

from climate_risk.data.pwt import PWT, PWT_COLUMNS, transform_pwt

SHEET_COLUMNS = ["countrycode", "year", *PWT_COLUMNS]

PWT_ROW_DEFAULTS = {
    "countrycode": "CRI",
    "year": 1980,
    "rgdpna": 22265.0,
    "rkna": 47132.0,
    "emp": 0.754,
    "pop": 2.39,
    "labsh": 0.5755,
    "delta": 0.0567,
    "ctfp": 0.8727,
}


def pwt_row(overrides=None):
    return PWT_ROW_DEFAULTS | (overrides or {})


def sheet(rows) -> pl.DataFrame:
    """The workbook columns the loader reads, in the order it names them."""
    return pl.DataFrame([[row[name] for name in SHEET_COLUMNS] for row in rows], schema=SHEET_COLUMNS, orient="row")


def test_the_quantity_series_are_national_accounts_not_purchasing_power():
    """The `o` and `e` families compare levels across countries; mixing families gives a growth
    rate neither family reports. A test on the renaming agrees with whatever code is listed, so
    the codes themselves are the claim.
    """
    assert PWT_COLUMNS["rgdpna"] == "pwt_real_gdp"
    assert PWT_COLUMNS["rkna"] == "capital"

    quantities = [code for code in PWT_COLUMNS if code.startswith(("rgdp", "rk"))]
    assert all(code.endswith("na") for code in quantities), quantities


def test_the_cross_country_productivity_level_is_the_ppp_series():
    """`rtfpna` is an index on a national base and reads 1.0 for every country in its base year, so
    a gap taken from it is an artifact. `ctfp` is the level with the United States at 1.
    """
    assert "rtfpna" not in PWT_COLUMNS
    assert PWT_COLUMNS["ctfp"] == "tfp_relative_to_usa"


def test_each_series_lands_under_its_own_name():
    """Renaming is the loader's whole job and a swapped alias is silent, since every value is a
    plausible float. The mapping is spelled out here rather than read from ``PWT_COLUMNS``, which
    would agree with whatever the loader currently says.
    """
    raw = sheet(
        [pwt_row({"rgdpna": 1.0, "rkna": 2.0, "emp": 3.0, "pop": 4.0, "labsh": 5.0, "delta": 6.0, "ctfp": 7.0})]
    )

    frame = transform_pwt(raw)

    assert frame.select(
        "pwt_real_gdp", "capital", "employment", "pwt_population", "labour_share", "depreciation", "tfp_relative_to_usa"
    ).rows() == [(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)]


def test_a_renamed_column_is_named_in_the_error():
    """A new release dropping or renaming a series must fail loudly, not yield a null column."""
    raw = sheet([pwt_row()]).drop("rkna")

    with pytest.raises(ValueError, match="rkna"):
        transform_pwt(raw)


def test_rows_are_keyed_by_iso_code_and_year():
    frame = transform_pwt(sheet([pwt_row()]))

    assert frame.columns[:2] == ["country_code", "year"]
    assert frame.select("country_code", "year").rows() == [("CRI", 1980)]


def test_the_result_is_sorted_by_country_and_year():
    raw = sheet(
        [
            pwt_row({"countrycode": "CRI", "year": 1981}),
            pwt_row({"countrycode": "ARG", "year": 1981}),
            pwt_row({"countrycode": "ARG", "year": 1980}),
        ]
    )

    frame = transform_pwt(raw)

    assert frame.select("country_code", "year").rows() == [("ARG", 1980), ("ARG", 1981), ("CRI", 1981)]


def test_a_country_with_no_code_is_dropped():
    """An unkeyed row would survive into the panel and join against nothing."""
    raw = sheet([pwt_row(), pwt_row({"countrycode": None})])

    frame = transform_pwt(raw)

    assert frame["country_code"].to_list() == ["CRI"]


def test_a_gap_in_a_series_stays_a_gap():
    """Coverage ends in 2019 while the panel runs later, and the model imputes the tail. A null
    silently filled here would be read as a measurement.
    """
    raw = sheet([pwt_row({"year": 2019}), pwt_row({"year": 2020, "rkna": None})])

    frame = transform_pwt(raw)

    assert frame.filter(pl.col("year") == 2020)["capital"].to_list() == [None]


def test_the_workbook_must_be_placed_by_hand(tmp_path):
    """It is not fetchable, so the error has to say where to get it and where to put it."""
    with pytest.raises(NotImplementedError, match=r"pwt100\.xlsx"):
        PWT.require(tmp_path)
