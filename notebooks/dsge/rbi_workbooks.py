from pathlib import Path

import numpy as np
import pandas as pd

GDP_COMPONENTS = {"PFCE": "consumption",
                  "GFCE": "government",
                  "GFCF": "investment",
                  "Change in Stock": "inventories",
                  "Valuables": "valuables",
                  "Export": "exports",
                  "Import": "imports",
                  "Discrepancies": "discrepancy",
                  "Gross Domestic Product": "gdp"}

CURRENT_ACCOUNT_ITEMS = {"1 CURRENT ACCOUNT": "current account",
                         "1.1 MERCHANDISE": "goods",
                         "1.2.1 Services": "services",
                         "1.2.2 Transfers": "transfers",
                         "1.2.3 Income": "primary income"}

FISCAL_YEAR_PATTERN = r"\d{4}-\d{2}"


def find_header_row(sheet: pd.DataFrame, label: str) -> int:
    matches = sheet.apply(lambda column: column.astype(str).str.strip().str.lower() == label)
    return sheet.index[matches.any(axis=1)][0]


def fiscal_quarter_to_period(fiscal_year: pd.Series, quarter: pd.Series) -> pd.PeriodIndex:
    """Indian fiscal quarters: Q1 is April-June of the fiscal year's first calendar year."""
    fiscal_start = fiscal_year.str[:4].astype(int)
    quarter_number = quarter.str[-1].astype(int)
    return pd.PeriodIndex.from_fields(year=fiscal_start + (quarter_number == 4),
                                      quarter=quarter_number % 4 + 1,
                                      freq="Q")


def fiscal_year_start(periods: pd.PeriodIndex) -> np.ndarray:
    """Calendar year in which the April-March fiscal year containing each period begins."""
    months = periods.to_timestamp().month
    return np.where(months <= 3, periods.year - 1, periods.year)


def sum_complete_years(frame: pd.DataFrame, year, periods_per_year: int) -> pd.DataFrame:
    grouped = frame.groupby(year)
    return grouped.sum().loc[grouped.size() == periods_per_year]


def _component_name(header: object) -> str:
    return next(name for key, name in GDP_COMPONENTS.items() if key in str(header))


def parse_quarterly_gdp(sheet: pd.DataFrame, base: str) -> pd.DataFrame:
    """Parse one base-year sheet of the quarterly GDP workbook, constant prices in rupees crore."""
    header_row = find_header_row(sheet, "quarter")
    table = sheet.iloc[header_row + 1:, 1:]
    table.columns = ["fiscal_year", "quarter", *[_component_name(header) for header in sheet.iloc[header_row, 3:]]]
    table = table.assign(fiscal_year=table["fiscal_year"].ffill().astype(str).str.strip(),
                         quarter=table["quarter"].astype(str).str.strip())
    table = table[table["quarter"].str.fullmatch(r"Q[1-4]") & table["fiscal_year"].str.fullmatch(FISCAL_YEAR_PATTERN)]

    return (table.drop(columns=["fiscal_year", "quarter"])
                 .astype(float)
                 .assign(base=base,
                         date=fiscal_quarter_to_period(table["fiscal_year"], table["quarter"]))
                 .set_index(["base", "date"])
                 .sort_index())


def load_quarterly_gdp(path: Path) -> pd.DataFrame:
    """Every base-year sheet of the quarterly GDP workbook, indexed by base and calendar quarter."""
    sheets = pd.read_excel(path,
                           sheet_name=None,
                           header=None)
    return pd.concat([parse_quarterly_gdp(sheet, base=sheet_name.split()[-1]) for sheet_name, sheet in sheets.items()])


def _parse_balance_of_payments(sheet: pd.DataFrame) -> pd.DataFrame:
    header_row = find_header_row(sheet, "quarter")
    table = sheet.iloc[header_row + 1:, 1:]
    item_names = [str(header).strip() for header in sheet.iloc[header_row, 4:]]
    table.columns = ["fiscal_year", "quarter", "transaction", *item_names]
    table = table.assign(fiscal_year=table["fiscal_year"].ffill().astype(str).str.strip(),
                         quarter=table["quarter"].ffill().astype(str).str.strip())
    table = table[table["quarter"].str.fullmatch(r"Q[1-4]")
                  & table["fiscal_year"].str.fullmatch(FISCAL_YEAR_PATTERN)
                  & (table["transaction"].astype(str).str.strip() == "Net")]

    return (table.drop(columns=["fiscal_year", "quarter", "transaction"])
                 .apply(pd.to_numeric, errors="coerce")
                 .set_index(fiscal_quarter_to_period(table["fiscal_year"], table["quarter"]))
                 .sort_index())


def _current_account_items(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {next(column for column in frame.columns if column.startswith(prefix)): name
               for prefix, name in CURRENT_ACCOUNT_ITEMS.items()}
    return frame[list(columns)].rename(columns=columns)


def load_current_account(path: Path) -> pd.DataFrame:
    """Quarterly net current-account flows in US$ millions, indexed by calendar quarter."""
    sheets = pd.read_excel(path,
                           sheet_name=None,
                           header=None)
    current_account = pd.concat([_current_account_items(_parse_balance_of_payments(sheet))
                                 for sheet in sheets.values()]).sort_index()
    if not current_account.index.is_unique:
        raise ValueError("balance of payments sheets overlap in time")
    return current_account


def _all_india_general_cpi(sheet: pd.DataFrame, base: str) -> pd.DataFrame:
    header_row = find_header_row(sheet, "month")
    labels = sheet.iloc[header_row:header_row + 3].astype(str).apply(lambda row: row.str.strip())
    group = labels.iloc[0].replace("nan", np.nan).ffill()
    index_column = next(column for column in sheet.columns
                        if "general index" in str(group[column]).lower()
                        and labels.iloc[1][column] == "Combined"
                        and labels.iloc[2][column] == "Index")

    rows = sheet.iloc[header_row + 3:]
    rows = rows[rows[2].astype(str).str.strip() == "ALL INDIA"]
    return pd.DataFrame({"base": base,
                         "month": pd.to_datetime(rows[1].astype(str), format="mixed").dt.to_period("M"),
                         "index": pd.to_numeric(rows[index_column], errors="coerce"),
                         "reported_inflation": pd.to_numeric(rows[index_column + 1], errors="coerce")})


# The 2024-base sheet reports June and July 2026 at roughly twice the level of May, with a matching
# 109% year-on-year figure in the RBI table itself.
CPI_BAD_MONTHS = {"2024": pd.period_range("2026-06", "2026-07", freq="M")}


def load_all_india_cpi(path: Path) -> pd.DataFrame:
    """All-India combined CPI general index for every base, with the known bad months masked."""
    sheets = pd.read_excel(path,
                           sheet_name=None,
                           header=None)
    cpi = (pd.concat([_all_india_general_cpi(sheet, base=sheet_name.split()[-1].split("=")[0])
                      for sheet_name, sheet in sheets.items()])
             .set_index(["base", "month"])
             .sort_index())
    for base, months in CPI_BAD_MONTHS.items():
        cpi.loc[[(base, month) for month in months], ["index", "reported_inflation"]] = np.nan
    cpi["computed_inflation"] = 100 * cpi["index"].groupby("base").pct_change(12, fill_method=None)
    return cpi


def load_monthly_trade(path: Path) -> pd.DataFrame:
    """Monthly merchandise exports, imports and balance in US$ millions, indexed by calendar month."""
    sheet = pd.read_excel(path, header=None)
    header_row = find_header_row(sheet, "month")
    trade = sheet.iloc[header_row + 1:, 1:6].set_axis(["fiscal_year", "month", "exports", "imports", "balance"], axis=1)
    trade = trade[trade["fiscal_year"].astype(str).str.strip().str.fullmatch(FISCAL_YEAR_PATTERN)]

    fiscal_start = trade["fiscal_year"].astype(str).str[:4].astype(int)
    month_number = pd.to_datetime(trade["month"].astype(str).str.strip(), format="%B").dt.month
    months = pd.PeriodIndex.from_fields(year=fiscal_start + (month_number <= 3),
                                        month=month_number,
                                        freq="M")
    return trade[["exports", "imports", "balance"]].astype(float).set_axis(months).sort_index()
