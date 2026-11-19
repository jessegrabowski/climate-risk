import re

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from climate_risk.data import rbi
from climate_risk.data.rbi import (
    fiscal_year_start,
    load_rbi_cpi,
    load_rbi_monthly_trade,
    load_rbi_quarterly_gdp,
    sum_complete_years,
    transform_all_india_cpi,
    transform_current_account,
    transform_monthly_trade,
    transform_quarterly_gdp,
)

GDP_HEADER = [
    "Item/ Year",
    "Quarter",
    "PFCE",
    "GFCE",
    "GFCF",
    "Change in Stock",
    "Valuables",
    "Export of goods & services",
    "Import of goods & services",
    "Discrepancies",
    "Gross Domestic Product",
]


def workbook_sheet(header, rows, title_rows=3):
    """A sheet as read with no header: title rows, the header row, then the data, all offset by a blank first column."""
    width = len(header) + 1
    blank = [np.nan] * width
    return pd.DataFrame([blank] * title_rows + [[np.nan, *header]] + [[np.nan, *row] for row in rows])


def gdp_row(fiscal_year, quarter, gdp=100.0):
    return [fiscal_year, quarter, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, gdp]


def test_fiscal_quarters_land_on_calendar_quarters():
    """India's fiscal year starts in April, so fiscal Q1 is the second calendar quarter and fiscal Q4 the first
    quarter of the next calendar year. A one-quarter shift would misalign every RBI series against FRED and the
    World Bank, and nothing downstream would notice.
    """
    sheet = workbook_sheet(GDP_HEADER, [gdp_row("2009-10", "Q1"), gdp_row(np.nan, "Q4")])

    gdp = transform_quarterly_gdp({"NAS - 2011-12": sheet})

    assert list(gdp.index.get_level_values("date")) == [pd.Period("2009Q2"), pd.Period("2010Q1")]


def test_gdp_components_land_under_their_names():
    sheet = workbook_sheet(GDP_HEADER, [gdp_row("2009-10", "Q1", gdp=9.0)])

    gdp = transform_quarterly_gdp({"NAS - 2011-12": sheet})

    components = gdp.iloc[0][["consumption", "government", "investment", "exports", "imports", "gdp"]].to_dict()
    assert components == {
        "consumption": 1.0,
        "government": 2.0,
        "investment": 3.0,
        "exports": 6.0,
        "imports": 7.0,
        "gdp": 9.0,
    }


def test_each_gdp_sheet_is_kept_under_its_base_year():
    """The bases break in level, so a series spliced across them has to know which base each value came from."""
    sheets = {
        "NAS - 2004-05": workbook_sheet(GDP_HEADER, [gdp_row("2009-10", "Q1")]),
        "NAS - 2011-12": workbook_sheet(GDP_HEADER, [gdp_row("2009-10", "Q1")]),
    }

    gdp = transform_quarterly_gdp(sheets)

    assert sorted(gdp.index.unique("base")) == ["2004-05", "2011-12"]


def test_a_sheet_without_its_header_row_is_named_in_the_error():
    """A workbook re-laid out by the RBI fails here, and the message is the only pointer to which sheet moved."""
    sheet = workbook_sheet([label.replace("Quarter", "Qtr") for label in GDP_HEADER], [gdp_row("2009-10", "Q1")])

    with pytest.raises(ValueError, match="the GDP sheet 'NAS - 2011-12'"):
        transform_quarterly_gdp({"NAS - 2011-12": sheet})


def test_an_unknown_gdp_column_is_named_in_the_error():
    header = [*GDP_HEADER[:-1], "Net Indirect Taxes"]
    sheet = workbook_sheet(header, [gdp_row("2009-10", "Q1")])

    with pytest.raises(ValueError, match="Net Indirect Taxes"):
        transform_quarterly_gdp({"NAS - 2011-12": sheet})


BOP_HEADER = [
    "Year / Item",
    "Quarter",
    "Transaction Type",
    "Overall balance",
    "1 CURRENT ACCOUNT (1.1+ 1.2)",
    "1.1 MERCHANDISE",
    "1.2.1 Services",
    "1.2.2 Transfers",
    "1.2.3 Income",
]


def bop_rows(fiscal_year, quarter, net):
    return [
        [fiscal_year, quarter, "Credit", 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        [fiscal_year, quarter, "Debit", 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        [fiscal_year, quarter, "Net", 0.0, net, 1.0, 2.0, 3.0, 4.0],
    ]


def test_only_net_flows_are_kept():
    """Each quarter appears three times, as credit, debit and net. Keeping a gross row would double the quarter."""
    sheet = workbook_sheet(BOP_HEADER, bop_rows("2000-01", "Q1", net=-5.0))

    current_account = transform_current_account({"2000-01 Q1 Onwards": sheet})

    assert current_account.to_dict("records") == [
        {"current account": -5.0, "goods": 1.0, "services": 2.0, "transfers": 3.0, "primary income": 4.0}
    ]
    assert list(current_account.index) == [pd.Period("2000Q2")]


def test_overlapping_balance_of_payments_sheets_raise():
    """The workbook splits quarters across sheets by decade. A quarter in both would be counted twice."""
    sheets = {
        "first": workbook_sheet(BOP_HEADER, bop_rows("1999-00", "Q4", net=1.0)),
        "second": workbook_sheet(BOP_HEADER, bop_rows("1999-00", "Q4", net=1.0)),
    }

    with pytest.raises(ValueError, match="overlap"):
        transform_current_account(sheets)


def cpi_sheet(months, combined_general):
    """A CPI sheet with a food group decoy beside the general index, and a state row beside each all-India row."""
    groups = [
        "Month",
        "State / Union Teritory",
        "Provisional/ Final",
        "A.1) Food",
        np.nan,
        "General Index",
        np.nan,
        "General Index",
        np.nan,
    ]
    sectors = [np.nan, np.nan, np.nan, "Combined", np.nan, "Rural", np.nan, "Combined", np.nan]
    measures = [np.nan, np.nan, np.nan, "Index", "Inflation (%)", "Index", "Inflation (%)", "Index", "Inflation (%)"]
    rows = []
    for month, value in zip(months, combined_general, strict=True):
        rows.append([month, "ALL INDIA", "Final", 999.0, 9.0, 888.0, 8.0, value, 1.0])
        rows.append([month, "KERALA", "Final", 777.0, 7.0, 666.0, 6.0, 555.0, 5.0])

    return workbook_sheet(groups, [sectors, measures, *rows])


def test_the_cpi_is_the_all_india_combined_general_index():
    """The sheet carries every state, the rural, urban and combined sectors, and every item group side by side."""
    months = pd.date_range("2013-01-31", periods=2, freq="ME")

    cpi = transform_all_india_cpi({"CPI - 2012=100": cpi_sheet(months, [100.0, 101.0])})

    assert cpi["index"].tolist() == [100.0, 101.0]
    assert list(cpi.index) == [("2012", pd.Period("2013-01")), ("2012", pd.Period("2013-02"))]


def test_known_bad_cpi_months_are_masked_and_inflation_recomputed():
    """The 2024-base sheet reports June and July 2026 at twice May's level, which would read as 100% inflation."""
    months = pd.date_range("2025-05-31", periods=15, freq="ME")
    values = [100.0 + index for index in range(15)]

    cpi = transform_all_india_cpi({"CPI - 2024=100": cpi_sheet(months, values)})

    index, inflation = cpi["index"].to_dict(), cpi["computed_inflation"].to_dict()
    june, may = ("2024", pd.Period("2026-06")), ("2024", pd.Period("2026-05"))
    assert np.isnan(index[june]) and np.isnan(inflation[june])
    assert inflation[may] == pytest.approx(100 * (112.0 / 100.0 - 1))


TRADE_HEADER = ["Year", "Month", "Exports", "Imports", "Trade Balance"]
# The trade sheet numbers its columns in the row under the header, and only rows with a fiscal year are data.
TRADE_COLUMN_NUMBERS = [1, 2, 3, 4, 5]


def test_trade_months_land_in_their_calendar_year():
    """January to March belong to the fiscal year that started the previous April."""
    rows = [TRADE_COLUMN_NUMBERS, ["2026-27", "January", 1.0, 2.0, -1.0], ["2026-27", "April", 3.0, 4.0, -1.0]]

    trade = transform_monthly_trade(workbook_sheet(TRADE_HEADER, rows))

    assert list(trade.index) == [pd.Period("2026-04"), pd.Period("2027-01")]
    assert trade["exports"].to_dict()[pd.Period("2027-01")] == 1.0


def test_fiscal_year_start_labels_january_to_march_by_the_previous_year():
    periods = pd.PeriodIndex(["2010Q1", "2010Q2"], freq="Q")

    assert fiscal_year_start(periods).tolist() == [2009, 2010]


def test_incomplete_years_are_dropped_from_annual_sums():
    """A year with a missing quarter would sum to three quarters of a year and read as a collapse."""
    frame = pd.DataFrame({"flow": [1.0, 1.0, 1.0, 1.0, 1.0]})

    totals = sum_complete_years(frame, year=np.array([2010, 2010, 2010, 2010, 2011]), periods_per_year=4)

    assert totals["flow"].to_dict() == {2010: 4.0}


def test_a_missing_workbook_names_where_it_belongs(tmp_path):
    """The error is the only place a collaborator learns the directory and the exact name, trailing spaces included."""
    expected = (
        tmp_path
        / "rbi"
        / "Quarterly Estimates of Gross Domestic Product (At Constant Prices) New Series (Base _ 2022-23).xlsx"
    )

    with pytest.raises(NotImplementedError, match=re.escape(str(expected))):
        load_rbi_quarterly_gdp(tmp_path)


def write_trade_workbook(cache_dir):
    rows = [TRADE_COLUMN_NUMBERS, ["2026-27", "April", 3.0, 4.0, -1.0]]
    (cache_dir / "rbi").mkdir()
    workbook_sheet(TRADE_HEADER, rows).to_excel(
        cache_dir / "rbi" / "Indias Foreign Trade - US Dollar.xlsx", header=False, index=False
    )


def test_the_loader_reads_the_workbook_from_the_rbi_directory(tmp_path, monkeypatch):
    """The workbooks sit in an `rbi` directory of the cache, under the names DBIE exports them with."""
    monkeypatch.setattr(rbi, "FOREIGN_TRADE", replace(rbi.FOREIGN_TRADE, sha256=None))
    write_trade_workbook(tmp_path)

    trade = load_rbi_monthly_trade(tmp_path)

    assert trade["exports"].tolist() == [3.0]


def test_a_workbook_other_than_the_pinned_release_is_refused(tmp_path):
    """DBIE reissues each workbook under the same name, and a rebuild from a later export is a different panel."""
    write_trade_workbook(tmp_path)

    with pytest.raises(ValueError, match="not the release"):
        load_rbi_monthly_trade(tmp_path)


def test_editing_the_cpi_mask_rebuilds_the_cached_cpi(tmp_path, monkeypatch):
    """A mask lifted once the RBI corrects the months must reach the frame, not read back the masked cache."""
    monkeypatch.setattr(rbi, "CONSUMER_PRICES", replace(rbi.CONSUMER_PRICES, sha256=None))
    (tmp_path / "rbi").mkdir()
    months = pd.date_range("2025-05-31", periods=15, freq="ME")
    sheet = cpi_sheet(months, [100.0 + index for index in range(15)])
    with pd.ExcelWriter(
        tmp_path / "rbi" / "Consumer Price Index - Rural, Urban, Combined (State-Wise) .xlsx"
    ) as writer:
        sheet.to_excel(writer, sheet_name="CPI - 2024=100", header=False, index=False)
    june = ("2024", pd.Period("2026-06"))

    masked = load_rbi_cpi(tmp_path)
    monkeypatch.setattr(rbi, "CPI_BAD_MONTHS", {})
    unmasked = load_rbi_cpi(tmp_path)

    assert np.isnan(masked["index"].to_dict()[june])
    assert unmasked["index"].to_dict()[june] == 113.0
