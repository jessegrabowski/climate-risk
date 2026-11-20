import logging
import warnings

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from climate_risk.data.cache import builder_fingerprint, cached, pandas_parquet
from climate_risk.data.source import ManualSource

_log = logging.getLogger(__name__)

# The workbooks sit together in this subdirectory of the cache, under the names DBIE exports them with.
RBI_DIRECTORY = "rbi"

_HOMEPAGE = "https://data.rbi.org.in/"
_LICENSE = "Reserve Bank of India, Database on Indian Economy terms of use. Not redistributed with this package."
_CITATION = "Reserve Bank of India, Database on Indian Economy (DBIE), https://data.rbi.org.in/."
_RETRIEVED = "2026-09-24"

# Each workbook is pinned to the release it was retrieved as, since DBIE reissues a workbook under its name as data
# are revised.
QUARTERLY_GDP = ManualSource(
    filename="Quarterly Estimates of Gross Domestic Product (At Constant Prices) New Series (Base _ 2022-23).xlsx",
    homepage=_HOMEPAGE,
    license=_LICENSE,
    citation=_CITATION,
    retrieved=_RETRIEVED,
    sha256="add54e626715792941752a49387c672ee209b9cfc6a9bbb84509f4f773dc2830",
)
BALANCE_OF_PAYMENTS = ManualSource(
    filename="Indias Overall Balance of Payments - Quarterly - US Dollar.xlsx",
    homepage=_HOMEPAGE,
    license=_LICENSE,
    citation=_CITATION,
    retrieved=_RETRIEVED,
    sha256="aff0e409df57b25204140d33c6dcd5e182af99d45137f828cfde3df2dc2631b8",
)
CONSUMER_PRICES = ManualSource(
    filename="Consumer Price Index - Rural, Urban, Combined (State-Wise) .xlsx",
    homepage=_HOMEPAGE,
    license=_LICENSE,
    citation=_CITATION,
    retrieved=_RETRIEVED,
    sha256="5f4cd758100dd9f465c13ddb06bccf225bbfcc171b05fd9dacfc14346d6ab5e8",
)
FOREIGN_TRADE = ManualSource(
    filename="Indias Foreign Trade - US Dollar.xlsx",
    homepage=_HOMEPAGE,
    license=_LICENSE,
    citation=_CITATION,
    retrieved=_RETRIEVED,
    sha256="0e1843b3c0401d57976f10e07167854807ea2b0cae675bf7604f771e5e4ab75f",
)

_BANKING_RETRIEVED = "2026-09-25"
DISTRICT_DEPOSITS_AND_CREDIT = ManualSource(
    filename=(
        "_Statement No. 4A _ District wise Number of Reporting Offices, Aggregate Deposit and Bank Credit of "
        "Scheduled Commercial Banks (SCBs).xlsx"
    ),
    homepage=_HOMEPAGE,
    license=_LICENSE,
    citation=_CITATION,
    retrieved=_BANKING_RETRIEVED,
    sha256="c9b69334fcd5cdbfb7372d1a911ddd87f29368676fab27ab5d70766e79d9feab",
)
DISTRICT_DEPOSIT_TYPES = ManualSource(
    filename=(
        "_Statement No. 4B _ District wise and Type-wise Distribution of Aggregate deposits with Scheduled "
        "Commercial Banks (SCBs).xlsx"
    ),
    homepage=_HOMEPAGE,
    license=_LICENSE,
    citation=_CITATION,
    retrieved=_BANKING_RETRIEVED,
    sha256="355346a3c481a3e5b0689964da7c1a3cc937383748e535d8ea0e49a0b9d79987",
)
DISTRICT_BANK_CREDIT = ManualSource(
    filename=("_Bank Credit of SCBs - Bank Group, Population Group, Occupation (Sector), District Wise - Annual.xlsx"),
    homepage=_HOMEPAGE,
    license=_LICENSE,
    citation=_CITATION,
    retrieved=_BANKING_RETRIEVED,
    sha256="9491292dc9caac78ad6f10d026ac59728e070ee1f4f3f8a2f646f443e33255fb",
)
NBFC_SECTOR_CREDIT = ManualSource(
    filename="Table VI.5_ Credit to Various Sectors by NBFCs.xlsx",
    homepage=_HOMEPAGE,
    license=_LICENSE,
    citation=_CITATION,
    retrieved=_BANKING_RETRIEVED,
    sha256="b54c90fc6f791b828f0fd71bbb797175cbc825442d88727da9b92c31ef4e01e8",
)

# Built for this package from the district names in Statement 4A and BSR-1, and shipped with it.
DISTRICT_GADM_CROSSWALK = Path(__file__).parent / "rbi_district_gadm.csv"

# Each expenditure component's column header contains its key.
GDP_COMPONENTS = {
    "PFCE": "consumption",
    "GFCE": "government",
    "GFCF": "investment",
    "Change in Stock": "inventories",
    "Valuables": "valuables",
    "Export": "exports",
    "Import": "imports",
    "Discrepancies": "discrepancy",
    "Gross Domestic Product": "gdp",
}

# Each item's column header starts with its key.
CURRENT_ACCOUNT_ITEMS = {
    "1 CURRENT ACCOUNT": "current account",
    "1.1 MERCHANDISE": "goods",
    "1.2.1 Services": "services",
    "1.2.2 Transfers": "transfers",
    "1.2.3 Income": "primary income",
}

# Item labels of the district statements, with line breaks and padding collapsed to single spaces.
DISTRICT_STATEMENT_ITEMS = {
    "Number of Reporting Offices": "reporting_offices",
    "Deposit": "deposits",
    "Credit": "credit",
}
DEPOSIT_TYPES = {
    "Current": "current_deposits",
    "Saving": "savings_deposits",
    "Term": "term_deposits",
    "Total": "deposits",
}

# Headers of the BSR-1 data sheets.
BANK_CREDIT_COLUMNS = {
    "Year": "year",
    "Region (Utilization)": "region",
    "State (Utilization)": "state",
    "District (Utilization)": "district",
    "Population Group - Utilization": "population_group",
    "Bank Group New": "bank_group",
    "Occupation Group": "occupation_group",
    "Occupation Sub- Group": "occupation_subgroup",
    "No. of Accounts": "accounts",
    "Credit Limit (in Rs Crores)": "credit_limit",
    "Amount Outstanding (in Rs Crores)": "amount_outstanding",
}
# Labels the BSR-1 records spell two ways across years, after upper-casing, mapped to the spelling kept.
BANK_CREDIT_LABEL_SYNONYMS = {
    "SEMI URBAN": "SEMI-URBAN",
    "SMALL FINANCE BANK": "SMALL FINANCE BANKS",
}

FISCAL_YEAR_PATTERN = r"\d{4}-\d{2}"

# The 2024-base sheet reports June and July 2026 at roughly twice the level of May, with a matching 109% year-on-year
# figure in the RBI table itself.
CPI_BAD_MONTHS = {"2024": pd.period_range("2026-06", "2026-07", freq="M")}


def rbi_directory(cache_dir: Path) -> Path:
    """Return the directory inside ``cache_dir`` that holds the RBI workbooks."""
    return cache_dir / RBI_DIRECTORY


def fiscal_quarter_to_period(fiscal_year: pd.Series, quarter: pd.Series) -> pd.PeriodIndex:
    """
    Convert Indian fiscal years and quarters to calendar quarters.

    The fiscal year runs April to March, so fiscal Q1 is April to June of the year's first calendar year and fiscal Q4
    is January to March of the next.

    Parameters
    ----------
    fiscal_year : Series of str
        Fiscal years written as ``"2009-10"``.
    quarter : Series of str
        Fiscal quarters written as ``"Q1"`` to ``"Q4"``.

    Returns
    -------
    quarters : PeriodIndex
        The calendar quarter of each row.
    """
    fiscal_start = fiscal_year.str[:4].astype(int)
    quarter_number = quarter.str[-1].astype(int)

    return _periods(year=fiscal_start + (quarter_number == 4), month=(quarter_number % 4) * 3 + 1, freq="Q")


def fiscal_year_start(periods: pd.PeriodIndex) -> np.ndarray:
    """
    Return the calendar year in which the April to March fiscal year containing each period begins.

    Parameters
    ----------
    periods : PeriodIndex
        Monthly or quarterly periods.

    Returns
    -------
    years : ndarray of int
        The fiscal year of each period, labeled by its first calendar year.
    """
    months = periods.to_timestamp().month

    return np.where(months <= 3, periods.year - 1, periods.year)


def sum_complete_years(frame: pd.DataFrame, year: np.ndarray, periods_per_year: int) -> pd.DataFrame:
    """
    Sum each column over the periods of each year, keeping only years with every period present.

    Parameters
    ----------
    frame : DataFrame
        Flows at a monthly or quarterly frequency.
    year : ndarray of int
        The year each row belongs to, calendar or fiscal.
    periods_per_year : int
        How many rows make a complete year.

    Returns
    -------
    totals : DataFrame
        One row per complete year.
    """
    grouped = frame.groupby(year)

    return grouped.sum().loc[grouped.size() == periods_per_year]


def _periods(year: pd.Series, month: pd.Series, freq: str) -> pd.PeriodIndex:
    """The periods of frequency ``freq`` holding the first day of each year and month."""
    starts = pd.to_datetime(pd.DataFrame({"year": year.to_numpy(), "month": month.to_numpy(), "day": 1}))

    return pd.PeriodIndex(starts, freq=freq)


def _find_header_row(sheet: pd.DataFrame, label: str, where: str) -> int:
    """Return the row of ``sheet`` with a cell reading ``label``, case and spaces aside; ``where`` names the sheet."""
    matches = sheet.apply(lambda column: column.astype(str).str.strip().str.lower() == label)
    rows = sheet.index[matches.any(axis=1)]
    if rows.empty:
        raise ValueError(f"No cell in {where} reads {label!r}; the RBI may have changed the workbook's layout.")

    return int(rows[0])


def _component_name(header: object, where: str) -> str:
    """Return the GDP component a column header names."""
    for key, name in GDP_COMPONENTS.items():
        if key in str(header):
            return name

    raise ValueError(f"The column {header!r} in {where} matches none of {list(GDP_COMPONENTS)}.")


def _parse_quarterly_gdp(sheet: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Parse one base-year sheet of the quarterly GDP workbook."""
    where = f"the GDP sheet {sheet_name!r}"
    header_row = _find_header_row(sheet, label="quarter", where=where)
    table = sheet.iloc[header_row + 1 :, 1:]
    table.columns = [
        "fiscal_year",
        "quarter",
        *[_component_name(header, where=where) for header in sheet.iloc[header_row, 3:]],
    ]
    table = table.assign(
        fiscal_year=table["fiscal_year"].ffill().astype(str).str.strip(),
        quarter=table["quarter"].astype(str).str.strip(),
    )
    table = table[table["quarter"].str.fullmatch(r"Q[1-4]") & table["fiscal_year"].str.fullmatch(FISCAL_YEAR_PATTERN)]

    return (
        table.drop(columns=["fiscal_year", "quarter"])
        .astype(float)
        .assign(
            base=sheet_name.split()[-1],
            date=fiscal_quarter_to_period(fiscal_year=table["fiscal_year"], quarter=table["quarter"]),
        )
        .set_index(["base", "date"])
        .sort_index()
    )


def transform_quarterly_gdp(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Stack every base-year sheet of the RBI quarterly GDP workbook.

    Parameters
    ----------
    sheets : dict of str to DataFrame
        Each sheet read with no header, keyed by its name, which ends in the base year, as ``"NAS - 2011-12"``.

    Returns
    -------
    gdp : DataFrame
        Expenditure components and GDP at constant prices in rupees crore, indexed by base year and calendar quarter.
    """
    return pd.concat([_parse_quarterly_gdp(sheet, sheet_name=name) for name, sheet in sheets.items()])


def _parse_balance_of_payments(sheet: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Parse the net rows of one sheet of the balance of payments workbook."""
    header_row = _find_header_row(sheet, label="quarter", where=f"the balance of payments sheet {sheet_name!r}")
    table = sheet.iloc[header_row + 1 :, 1:]
    item_names = [str(header).strip() for header in sheet.iloc[header_row, 4:]]
    table.columns = ["fiscal_year", "quarter", "transaction", *item_names]
    table = table.assign(
        fiscal_year=table["fiscal_year"].ffill().astype(str).str.strip(),
        quarter=table["quarter"].ffill().astype(str).str.strip(),
    )
    table = table[
        table["quarter"].str.fullmatch(r"Q[1-4]")
        & table["fiscal_year"].str.fullmatch(FISCAL_YEAR_PATTERN)
        & (table["transaction"].astype(str).str.strip() == "Net")
    ]

    flows: pd.DataFrame = table.drop(columns=["fiscal_year", "quarter", "transaction"]).apply(
        pd.to_numeric, errors="coerce"
    )

    return flows.set_index(
        fiscal_quarter_to_period(fiscal_year=table["fiscal_year"], quarter=table["quarter"])
    ).sort_index()


def _current_account_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the current-account items of a parsed balance of payments sheet."""
    columns = {}
    for prefix, name in CURRENT_ACCOUNT_ITEMS.items():
        found = [column for column in frame.columns if column.startswith(prefix)]
        if not found:
            raise ValueError(f"No balance of payments column starts with {prefix!r}.")
        columns[found[0]] = name

    return frame[list(columns)].rename(columns=columns)


def transform_current_account(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Stack the net current-account flows of every sheet of the RBI quarterly balance of payments workbook.

    Parameters
    ----------
    sheets : dict of str to DataFrame
        Each sheet read with no header. The workbook splits its quarters across sheets by decade.

    Returns
    -------
    current_account : DataFrame
        Net flows in US$ millions of the current account and its goods, services, transfers and primary income items,
        indexed by calendar quarter.
    """
    current_account = pd.concat(
        [_current_account_items(_parse_balance_of_payments(sheet, sheet_name=name)) for name, sheet in sheets.items()]
    ).sort_index()
    if not current_account.index.is_unique:
        raise ValueError("The balance of payments sheets overlap in time.")

    return current_account


def _all_india_general_cpi(sheet: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Parse the all-India combined general index of one base-year sheet of the CPI workbook."""
    base = sheet_name.split()[-1].split("=")[0]
    header_row = _find_header_row(sheet, label="month", where=f"the CPI sheet {sheet_name!r}")
    labels = sheet.iloc[header_row : header_row + 3].astype(str).apply(lambda row: row.str.strip())
    group = labels.iloc[0].replace("nan", np.nan).ffill()
    # Each index column is followed by its inflation column.
    index_positions = [
        position
        for position in range(sheet.shape[1])
        if "general index" in str(group.iloc[position]).lower()
        and labels.iloc[1, position] == "Combined"
        and labels.iloc[2, position] == "Index"
    ]
    if not index_positions:
        raise ValueError(f"The CPI sheet {sheet_name!r} has no combined general index column.")
    index_position = index_positions[0]

    rows = sheet.iloc[header_row + 3 :]
    rows = rows[rows.iloc[:, 2].astype(str).str.strip() == "ALL INDIA"]

    return pd.DataFrame(
        {
            "base": base,
            "month": pd.to_datetime(rows.iloc[:, 1].astype(str), format="mixed").dt.to_period("M"),
            "index": pd.to_numeric(rows.iloc[:, index_position], errors="coerce"),
            "reported_inflation": pd.to_numeric(rows.iloc[:, index_position + 1], errors="coerce"),
        }
    )


def transform_all_india_cpi(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Read the all-India combined CPI general index from every base-year sheet of the RBI CPI workbook.

    Months listed in ``CPI_BAD_MONTHS`` are masked, and year-on-year inflation is recomputed from the index within each
    base.

    Parameters
    ----------
    sheets : dict of str to DataFrame
        Each sheet read with no header, keyed by its name, which ends in the base, as ``"CPI - 2012=100"``.

    Returns
    -------
    cpi : DataFrame
        The index, the RBI's reported inflation and the recomputed inflation in percent, indexed by base year and month.
    """
    cpi = (
        pd.concat([_all_india_general_cpi(sheet, sheet_name=name) for name, sheet in sheets.items()])
        .set_index(["base", "month"])
        .sort_index()
    )
    for base, months in CPI_BAD_MONTHS.items():
        bad = (cpi.index.get_level_values("base") == base) & cpi.index.get_level_values("month").isin(months)
        cpi.loc[bad, ["index", "reported_inflation"]] = np.nan
    cpi["computed_inflation"] = 100 * cpi["index"].groupby("base").pct_change(12)

    return cpi


def transform_monthly_trade(sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Read monthly merchandise trade from the RBI foreign trade workbook.

    Parameters
    ----------
    sheet : DataFrame
        The workbook's one sheet, read with no header. Rows carry a fiscal year and a month name.

    Returns
    -------
    trade : DataFrame
        Exports, imports and the trade balance in US$ millions, indexed by calendar month.
    """
    header_row = _find_header_row(sheet, label="month", where="the foreign trade sheet")
    trade = sheet.iloc[header_row + 1 :, 1:6].set_axis(
        ["fiscal_year", "month", "exports", "imports", "balance"], axis=1
    )
    trade = trade[trade["fiscal_year"].astype(str).str.strip().str.fullmatch(FISCAL_YEAR_PATTERN)]

    fiscal_start = trade["fiscal_year"].astype(str).str[:4].astype(int)
    month_number = pd.to_datetime(trade["month"].astype(str).str.strip(), format="%B").dt.month
    months = _periods(year=fiscal_start + (month_number <= 3), month=month_number, freq="M")

    return trade[["exports", "imports", "balance"]].astype(float).set_axis(months).sort_index()


def _collapse_spaces(label: object) -> str:
    """Return ``label`` as a string with line breaks and runs of spaces collapsed to single spaces."""
    return " ".join(str(label).split())


def _cell_position(row: pd.Series, label: str, where: str) -> int:
    """Return the position of the cell in ``row`` reading ``label``, case and spaces aside."""
    positions = np.flatnonzero(row.astype(str).str.strip().str.lower() == label)
    if positions.size == 0:
        raise ValueError(f"No cell in the header of {where} reads {label!r}.")

    return int(positions[0])


def _parse_district_statement(sheet: pd.DataFrame, items: Mapping[str, str], where: str) -> pd.DataFrame:
    """Parse a district statement whose quarters run across the sheet, each split into the items in ``items``."""
    header_row = _find_header_row(sheet, label="region", where=where)
    region = _cell_position(sheet.iloc[header_row], label="region", where=where)
    state, district = region + 1, region + 2
    body = sheet.iloc[header_row + 1 :]
    # The item labels sit in the last header row, directly above the first district.
    item_row = header_row + int(np.argmax(body.iloc[:, region].notna().to_numpy()))
    value_positions = list(range(district + 1, sheet.shape[1]))

    item_labels = [_collapse_spaces(label) for label in sheet.iloc[item_row, value_positions]]
    unknown = sorted(set(item_labels) - set(items))
    if unknown:
        raise ValueError(f"The columns {unknown} in {where} match none of {list(items)}.")
    quarter_labels = sheet.iloc[header_row, value_positions].ffill().astype(str).str.strip()
    fiscal_year, quarter = quarter_labels.str.split(":", expand=True).to_numpy().T
    quarters = fiscal_quarter_to_period(fiscal_year=pd.Series(fiscal_year), quarter=pd.Series(quarter))

    # State subtotals leave region and district blank, and the all-India totals leave state blank too.
    districts = body[body.iloc[:, region].notna() & body.iloc[:, district].notna()]
    names = districts.iloc[:, [region, state, district]].map(_collapse_spaces)
    values = districts.iloc[:, value_positions].astype(float)
    values.index = pd.MultiIndex.from_frame(names.set_axis(["region", "state", "district"], axis=1))
    values.columns = pd.MultiIndex.from_arrays(
        [quarters, [items[label] for label in item_labels]], names=["date", "item"]
    )
    statement = pd.DataFrame(values.stack("date", future_stack=True)).sort_index()
    statement.columns.name = None
    if not statement.index.is_unique:
        raise ValueError(f"A district appears twice in {where}.")

    return statement


def transform_district_deposits_and_credit(sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Read district deposits and credit of scheduled commercial banks from the RBI's Statement 4A.

    State subtotals and the all-India totals are dropped.

    Parameters
    ----------
    sheet : DataFrame
        The statement's one sheet, read with no header. Each fiscal quarter's offices, deposits and credit sit side by
        side.

    Returns
    -------
    statement : DataFrame
        The number of reporting offices, and deposits and credit in rupees crore, indexed by region, state, district and
        calendar quarter.
    """
    return _parse_district_statement(sheet, items=DISTRICT_STATEMENT_ITEMS, where="Statement 4A")


def transform_district_deposit_types(sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Read district deposits of scheduled commercial banks by account type from the RBI's Statement 4B.

    State subtotals and the all-India totals are dropped.

    Parameters
    ----------
    sheet : DataFrame
        The statement's one sheet, read with no header. Each fiscal quarter's current, savings, term and total deposits
        sit side by side.

    Returns
    -------
    deposits : DataFrame
        Deposits in rupees crore by account type and in total, indexed by region, state, district and calendar quarter.
    """
    return _parse_district_statement(sheet, items=DEPOSIT_TYPES, where="Statement 4B")


def transform_district_bank_credit(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Stack the BSR-1 credit records of scheduled commercial banks from every data sheet of the RBI workbook.

    Labels are upper-cased and the spellings in ``BANK_CREDIT_LABEL_SYNONYMS`` merged, since the workbook changes case
    and spelling from year to year. The few records with no district are dropped.

    Parameters
    ----------
    sheets : dict of str to DataFrame
        Each sheet read with no header, keyed by its name. The records run on from ``"Data"`` into ``"Data(1)"`` and
        onward with no header of their own, and the summary sheets are ignored.

    Returns
    -------
    credit : DataFrame
        One row per year, district, population group, bank group and occupation, with the number of accounts, and the
        credit limit and amount outstanding in rupees crore.
    """
    data_sheets = [sheet for name, sheet in sheets.items() if name.startswith("Data")]
    if not data_sheets:
        raise ValueError(f"The BSR-1 workbook has no data sheet among {list(sheets)}.")
    first = data_sheets[0]
    header_row = _find_header_row(first, label="year", where="the first BSR-1 data sheet")
    positions = np.flatnonzero(first.iloc[header_row].notna().to_numpy())
    header = [_collapse_spaces(label) for label in first.iloc[header_row, positions]]
    unknown = sorted(set(header) - set(BANK_CREDIT_COLUMNS))
    if unknown:
        raise ValueError(f"The BSR-1 columns {unknown} match none of {list(BANK_CREDIT_COLUMNS)}.")

    records = pd.concat(
        [first.iloc[header_row + 1 :, positions], *(sheet.iloc[:, positions] for sheet in data_sheets[1:])],
        ignore_index=True,
    ).dropna(how="all")
    records.columns = [BANK_CREDIT_COLUMNS[label] for label in header]
    records = records.dropna(subset=["district"])

    numeric = ["accounts", "credit_limit", "amount_outstanding"]
    labels = [column for column in records.columns if column not in ["year", *numeric]]

    return records.assign(
        year=records["year"].astype(int),
        **{column: records[column].astype(float) for column in numeric},
        **{
            column: records[column].str.strip().str.upper().replace(BANK_CREDIT_LABEL_SYNONYMS).astype("category")
            for column in labels
        },
    )


def transform_nbfc_sector_credit(sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Read credit extended by non-bank financial companies to each sector from the RBI's Table VI.5.

    Parameters
    ----------
    sheet : DataFrame
        The table's one sheet, read with no header. Items run down the sheet and half-year ends across it, with ``-``
        where a value is not available.

    Returns
    -------
    credit : DataFrame
        Credit outstanding in rupees crore, indexed by item, labeled as the table numbers it, and month. The table's
        totals and subtotals are kept as items, so summing over items counts the same credit more than once.
    """
    where = "the NBFC credit table"
    header_row = _find_header_row(sheet, label="item", where=where)
    item = _cell_position(sheet.iloc[header_row], label="item", where=where)
    date_positions = item + 1 + np.flatnonzero(sheet.iloc[header_row, item + 1 :].notna().to_numpy())
    dates = sheet.iloc[header_row, date_positions].astype(str).str.strip().str.removeprefix("End-")
    months = pd.PeriodIndex(pd.to_datetime(dates, format="%B %Y"), freq="M")

    # A row numbering the columns follows the header, and notes follow a blank row after the last item.
    body = sheet.iloc[header_row + 1 :]
    last_item = int(np.argmax(body.iloc[:, item].isna().to_numpy()))
    rows = body.iloc[:last_item]
    rows = rows[rows.iloc[:, item].map(lambda label: isinstance(label, str))]

    values = rows.iloc[:, date_positions]
    values = values.mask(values.map(lambda value: isinstance(value, str) and value.strip() == "-")).astype(float)
    values.index = pd.Index(rows.iloc[:, item].map(_collapse_spaces), name="item")
    values.columns = pd.Index(months, name="date")

    return pd.DataFrame({"credit": values.sort_index(axis=1).stack(future_stack=True)})


def rbi_fingerprints() -> dict[str, str]:
    """
    Fingerprint how each RBI frame is built and which release it reads, so editing a parser, a rule such as
    ``CPI_BAD_MONTHS``, or a workbook's pinned release turns its cache over.

    Returns
    -------
    fingerprints : dict of str to str
        One digest per frame, keyed by its cache name.
    """
    shared = (_find_header_row, fiscal_quarter_to_period, _periods, FISCAL_YEAR_PATTERN)

    return {
        "rbi_quarterly_gdp": builder_fingerprint(
            transform_quarterly_gdp, _parse_quarterly_gdp, _component_name, GDP_COMPONENTS, QUARTERLY_GDP, *shared
        ),
        "rbi_current_account": builder_fingerprint(
            transform_current_account,
            _parse_balance_of_payments,
            _current_account_items,
            CURRENT_ACCOUNT_ITEMS,
            BALANCE_OF_PAYMENTS,
            *shared,
        ),
        "rbi_cpi": builder_fingerprint(
            transform_all_india_cpi, _all_india_general_cpi, CPI_BAD_MONTHS, CONSUMER_PRICES, *shared
        ),
        "rbi_monthly_trade": builder_fingerprint(transform_monthly_trade, FOREIGN_TRADE, *shared),
        "rbi_district_deposits_and_credit": builder_fingerprint(
            transform_district_deposits_and_credit,
            _parse_district_statement,
            _cell_position,
            _collapse_spaces,
            DISTRICT_STATEMENT_ITEMS,
            DISTRICT_DEPOSITS_AND_CREDIT,
            *shared,
        ),
        "rbi_district_deposit_types": builder_fingerprint(
            transform_district_deposit_types,
            _parse_district_statement,
            _cell_position,
            _collapse_spaces,
            DEPOSIT_TYPES,
            DISTRICT_DEPOSIT_TYPES,
            *shared,
        ),
        "rbi_district_bank_credit": builder_fingerprint(
            transform_district_bank_credit,
            _collapse_spaces,
            BANK_CREDIT_COLUMNS,
            BANK_CREDIT_LABEL_SYNONYMS,
            DISTRICT_BANK_CREDIT,
            *shared,
        ),
        "rbi_nbfc_sector_credit": builder_fingerprint(
            transform_nbfc_sector_credit, _cell_position, _collapse_spaces, NBFC_SECTOR_CREDIT, *shared
        ),
    }


def _read_sheets(source: ManualSource, cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every sheet of an RBI workbook with no header row, keyed by sheet name."""
    path = source.require(rbi_directory(cache_dir))
    _log.info(f"Reading {path}")
    # DBIE exports carry no default cell style, which openpyxl warns about on every read.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        return pd.read_excel(path, sheet_name=None, header=None)


def _read_first_sheet(source: ManualSource, cache_dir: Path) -> pd.DataFrame:
    """Read the first sheet of an RBI workbook with no header row."""
    return next(iter(_read_sheets(source, cache_dir).values()))


def load_rbi_quarterly_gdp(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return India's quarterly GDP by expenditure at constant prices for every base year the RBI publishes.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the ``rbi`` workbook directory and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    gdp : DataFrame
        Components in rupees crore, indexed by base year and calendar quarter.

    Examples
    --------
    The workbook has to be placed in ``data/rbi`` by hand first:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.rbi import load_rbi_quarterly_gdp

        gdp = load_rbi_quarterly_gdp(Path("data"))
    """
    return cached(
        cache_dir,
        "rbi_quarterly_gdp",
        lambda: transform_quarterly_gdp(_read_sheets(QUARTERLY_GDP, cache_dir)),
        pandas_parquet(),
        params={"reading": rbi_fingerprints()["rbi_quarterly_gdp"]},
        force=force_reload,
    )


def load_rbi_current_account(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return India's quarterly net current-account flows from the RBI balance of payments.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the ``rbi`` workbook directory and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    current_account : DataFrame
        Net flows in US$ millions, indexed by calendar quarter.

    Examples
    --------
    The workbook has to be placed in ``data/rbi`` by hand first:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.rbi import load_rbi_current_account

        current_account = load_rbi_current_account(Path("data"))
    """
    return cached(
        cache_dir,
        "rbi_current_account",
        lambda: transform_current_account(_read_sheets(BALANCE_OF_PAYMENTS, cache_dir)),
        pandas_parquet(),
        params={"reading": rbi_fingerprints()["rbi_current_account"]},
        force=force_reload,
    )


def load_rbi_cpi(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return the all-India combined consumer price index for every base the RBI publishes.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the ``rbi`` workbook directory and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    cpi : DataFrame
        The index and inflation in percent, indexed by base year and month.

    Examples
    --------
    The workbook has to be placed in ``data/rbi`` by hand first:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.rbi import load_rbi_cpi

        cpi = load_rbi_cpi(Path("data"))
    """
    return cached(
        cache_dir,
        "rbi_cpi",
        lambda: transform_all_india_cpi(_read_sheets(CONSUMER_PRICES, cache_dir)),
        pandas_parquet(),
        params={"reading": rbi_fingerprints()["rbi_cpi"]},
        force=force_reload,
    )


def load_rbi_monthly_trade(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return India's monthly merchandise exports, imports and trade balance from the RBI.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the ``rbi`` workbook directory and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    trade : DataFrame
        Flows in US$ millions, indexed by calendar month.

    Examples
    --------
    The workbook has to be placed in ``data/rbi`` by hand first:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.rbi import load_rbi_monthly_trade

        trade = load_rbi_monthly_trade(Path("data"))
    """
    return cached(
        cache_dir,
        "rbi_monthly_trade",
        lambda: transform_monthly_trade(_read_first_sheet(FOREIGN_TRADE, cache_dir)),
        pandas_parquet(),
        params={"reading": rbi_fingerprints()["rbi_monthly_trade"]},
        force=force_reload,
    )


def load_rbi_district_deposits_and_credit(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return the reporting offices, deposits and credit of scheduled commercial banks in each Indian district.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the ``rbi`` workbook directory and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    statement : DataFrame
        The number of reporting offices, and deposits and credit in rupees crore, indexed by region, state, district and
        calendar quarter.

    Examples
    --------
    The workbook has to be placed in ``data/rbi`` by hand first:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.rbi import load_rbi_district_deposits_and_credit

        statement = load_rbi_district_deposits_and_credit(Path("data"))
    """
    return cached(
        cache_dir,
        "rbi_district_deposits_and_credit",
        lambda: transform_district_deposits_and_credit(_read_first_sheet(DISTRICT_DEPOSITS_AND_CREDIT, cache_dir)),
        pandas_parquet(),
        params={"reading": rbi_fingerprints()["rbi_district_deposits_and_credit"]},
        force=force_reload,
    )


def load_rbi_district_deposit_types(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return deposits with scheduled commercial banks in each Indian district by account type.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the ``rbi`` workbook directory and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    deposits : DataFrame
        Current, savings, term and total deposits in rupees crore, indexed by region, state, district and calendar
        quarter.

    Examples
    --------
    The workbook has to be placed in ``data/rbi`` by hand first:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.rbi import load_rbi_district_deposit_types

        deposits = load_rbi_district_deposit_types(Path("data"))
    """
    return cached(
        cache_dir,
        "rbi_district_deposit_types",
        lambda: transform_district_deposit_types(_read_first_sheet(DISTRICT_DEPOSIT_TYPES, cache_dir)),
        pandas_parquet(),
        params={"reading": rbi_fingerprints()["rbi_district_deposit_types"]},
        force=force_reload,
    )


def load_rbi_district_bank_credit(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return the BSR-1 credit records of scheduled commercial banks by district, bank group and occupation.

    The workbook holds about 2.2 million records, so the first read takes minutes.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the ``rbi`` workbook directory and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    credit : DataFrame
        One row per year, district, population group, bank group and occupation, with the number of accounts, and the
        credit limit and amount outstanding in rupees crore.

    Examples
    --------
    The workbook has to be placed in ``data/rbi`` by hand first:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.rbi import load_rbi_district_bank_credit

        credit = load_rbi_district_bank_credit(Path("data"))
    """
    return cached(
        cache_dir,
        "rbi_district_bank_credit",
        lambda: transform_district_bank_credit(_read_sheets(DISTRICT_BANK_CREDIT, cache_dir)),
        pandas_parquet(),
        params={"reading": rbi_fingerprints()["rbi_district_bank_credit"]},
        force=force_reload,
    )


def read_rbi_district_gadm_crosswalk(*, path: Path = DISTRICT_GADM_CROSSWALK) -> pd.DataFrame:
    """
    Read the links from RBI districts to the GADM 4.1 districts they lie in.

    Parameters
    ----------
    path : Path, optional
        The crosswalk table. Default the one shipped with the package.

    Returns
    -------
    crosswalk : DataFrame
        One row per RBI state, RBI district and GADM ``gid``, with the ``relation`` between them (``same``,
        ``renamed`` or ``carved_from``) and a ``note`` giving the evidence for every link other than ``same``. A
        district carved out after GADM's boundaries links to its parent, and a district spanning several GADM
        districts has one row for each.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data.rbi import read_rbi_district_gadm_crosswalk

        crosswalk = read_rbi_district_gadm_crosswalk()
    """
    return pd.read_csv(path)


def load_rbi_nbfc_sector_credit(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return credit extended by Indian non-bank financial companies to each sector, at half-year ends.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the ``rbi`` workbook directory and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    credit : DataFrame
        Credit outstanding in rupees crore, indexed by item and month.

    Examples
    --------
    The workbook has to be placed in ``data/rbi`` by hand first:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.rbi import load_rbi_nbfc_sector_credit

        credit = load_rbi_nbfc_sector_credit(Path("data"))
    """
    return cached(
        cache_dir,
        "rbi_nbfc_sector_credit",
        lambda: transform_nbfc_sector_credit(_read_first_sheet(NBFC_SECTOR_CREDIT, cache_dir)),
        pandas_parquet(),
        params={"reading": rbi_fingerprints()["rbi_nbfc_sector_credit"]},
        force=force_reload,
    )
