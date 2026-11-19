from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from statsmodels.tools.sm_exceptions import X13NotFoundError
from statsmodels.tsa.x13 import x13_arima_analysis

from climate_risk.data.bis import load_real_effective_exchange_rate
from climate_risk.data.cache import builder_fingerprint, cached, pandas_parquet
from climate_risk.data.fred import load_fred_data
from climate_risk.data.ilo import load_labour_income_share
from climate_risk.data.partner_activity import load_partner_activity
from climate_risk.data.pwt import load_pwt_data
from climate_risk.data.rbi import load_rbi_cpi, load_rbi_current_account, load_rbi_quarterly_gdp, rbi_fingerprints
from climate_risk.data.weo import load_government_debt
from climate_risk.data.world_bank import load_wb_data

QUARTERS_PER_YEAR = 4
NATIONAL_ACCOUNTS = ["gdp", "consumption", "investment", "government", "exports", "imports"]
# The base every national-accounts series is spliced onto; the others extend it on their growth rates.
GDP_ANCHOR_BASE = "2011-12"
CPI_BASES = ["2010", "2012", "2024"]
CPI_ANCHOR_BASE = "2024"
# FRED publishes world output quarterly. Its other series are daily or monthly and are averaged over complete quarters.
QUARTERLY_FRED_SERIES = ["world_output"]
PARTNER_ACTIVITY_LAST_YEAR = 2023

YearConvention = Literal["calendar", "fiscal"]
# The quarter an annual value is placed on, as (years after the year's label, calendar quarter). India's fiscal year,
# labeled by the calendar year it starts in, closes in the first quarter of the next.
CLOSING_QUARTER: dict[YearConvention, tuple[int, int]] = {"calendar": (0, 4), "fiscal": (1, 1)}


@dataclass(frozen=True)
class QuarterlySeries:
    """
    How one quarterly column of the panel is sourced and brought to quarters.

    Parameters
    ----------
    source : str
        Who publishes it.
    native_frequency : str
        How often the source observes it.
    units : str
        What one unit of the value is.
    seasonal_adjustment : str
        How it is seasonally adjusted, by the panel or by the source.
    aggregation : str
        How a quarter's value is made from the source's observations.
    """

    source: str
    native_frequency: str
    units: str
    seasonal_adjustment: str
    aggregation: str


@dataclass(frozen=True)
class AnnualSeries:
    """
    How one annual column of the panel is sourced and placed.

    Parameters
    ----------
    source : str
        Who publishes it.
    year_convention : {'calendar', 'fiscal'}
        Whether the source's years are calendar years or India's April to March fiscal years.
    aggregation : str
        How the annual value summarizes the year: a ratio, a mean or an end-of-period stock.
    units : str
        What one unit of the value is.
    """

    source: str
    year_convention: YearConvention
    aggregation: str
    units: str


def _fred(
    native_frequency: str, units: str, seasonal_adjustment: str = "none", aggregation: str = "mean"
) -> QuarterlySeries:
    return QuarterlySeries(
        source="FRED",
        native_frequency=native_frequency,
        units=units,
        seasonal_adjustment=seasonal_adjustment,
        aggregation=aggregation,
    )


QUARTERLY_SERIES = {
    **dict.fromkeys(
        NATIONAL_ACCOUNTS,
        QuarterlySeries(
            source="RBI",
            native_frequency="quarterly",
            units=f"rupees crore, {GDP_ANCHOR_BASE} prices",
            seasonal_adjustment="X-13, log",
            aggregation="sum",
        ),
    ),
    "cpi": QuarterlySeries(
        source="RBI",
        native_frequency="monthly",
        units=f"index, {CPI_ANCHOR_BASE} = 100",
        seasonal_adjustment="X-13, log",
        aggregation="mean",
    ),
    "current_account_usd": QuarterlySeries(
        source="RBI",
        native_frequency="quarterly",
        units="US$ millions",
        seasonal_adjustment="X-13, additive",
        aggregation="sum",
    ),
    "real_effective_exchange_rate": QuarterlySeries(
        source="BIS",
        native_frequency="monthly",
        units="index, 2020 = 100",
        seasonal_adjustment="none",
        aggregation="mean",
    ),
    "world_rate_3m": _fred(native_frequency="daily", units="percent per year"),
    "world_rate_10y": _fred(native_frequency="daily", units="percent per year"),
    "world_price_level": _fred(
        native_frequency="monthly", units="index, 1982-84 = 100", seasonal_adjustment="source-adjusted"
    ),
    "world_output": _fred(
        native_frequency="quarterly",
        units="billions of chained 2017 US$, annual rate",
        seasonal_adjustment="source-adjusted",
        aggregation="none",
    ),
    "commodity_all": _fred(native_frequency="monthly", units="index, 2016 = 100"),
    "commodity_energy": _fred(native_frequency="monthly", units="index, 2016 = 100"),
    "commodity_food": _fred(native_frequency="monthly", units="index, 2016 = 100"),
}
# The FRED series the panel carries, whatever else the FRED loader fetches.
FRED_SERIES = [name for name, series in QUARTERLY_SERIES.items() if series.source == "FRED"]


def _world_bank(year_convention: YearConvention, aggregation: str, units: str = "fraction") -> AnnualSeries:
    return AnnualSeries(source="World Bank", year_convention=year_convention, aggregation=aggregation, units=units)


ANNUAL_SERIES = {
    "nominal_consumption_share": _world_bank(year_convention="fiscal", aggregation="ratio"),
    "nominal_investment_share": _world_bank(year_convention="fiscal", aggregation="ratio"),
    "tax_revenue_gdp": _world_bank(year_convention="fiscal", aggregation="ratio"),
    "government_expense_gdp": _world_bank(year_convention="fiscal", aggregation="ratio"),
    "government_debt": AnnualSeries(
        source="IMF WEO", year_convention="fiscal", aggregation="end of period", units="fraction"
    ),
    "cpi_inflation_annual": _world_bank(year_convention="calendar", aggregation="mean"),
    "exchange_rate": _world_bank(year_convention="calendar", aggregation="mean", units="rupees per US$"),
    "lending_rate": _world_bank(year_convention="calendar", aggregation="mean"),
    "external_debt_usd": _world_bank(year_convention="calendar", aggregation="end of period", units="current US$"),
    "employment_rate": _world_bank(year_convention="calendar", aggregation="mean"),
    "labour_share_pwt": AnnualSeries(source="PWT", year_convention="calendar", aggregation="ratio", units="fraction"),
    "labour_share_ilo": AnnualSeries(source="ILO", year_convention="calendar", aggregation="ratio", units="fraction"),
    "partner_activity": AnnualSeries(
        source="partner activity", year_convention="calendar", aggregation="mean", units="index, 2015 base"
    ),
}


def load_india_quarterly_panel(cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return India's mixed-frequency macroeconomic panel, quarterly, for the CatDSGE model.

    Quarterly columns are seasonally adjusted national accounts, CPI and current account from the RBI, the BIS real
    effective exchange rate, and FRED's foreign block. Annual columns from the World Bank, the IMF, the Penn World
    Table, the ILO and partner activity sit on the last quarter of the year they cover, with every other quarter
    missing. :func:`india_panel_metadata` describes each column.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under, with the RBI workbooks in its ``rbi`` directory.
    force_reload : bool, optional
        Rebuild the panel and reload every source it reads. The FRED, BIS, World Bank, IMF and ILO series are then
        downloaded as their publishers serve them now, so revisions since the last build reach the panel. Default False.

    Returns
    -------
    panel : DataFrame
        One row per quarter, indexed by the quarter's first day under ``quarter_start``, from the first to the last
        quarter of the spliced national accounts.

    Examples
    --------
    The RBI workbooks have to be placed in ``data/rbi`` first, and seasonal adjustment needs the ``x13as`` binary:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel

        panel = load_india_quarterly_panel(Path("data"))
    """

    def build() -> pd.DataFrame:
        world_bank = _india_annual(load_wb_data(cache_dir, force_reload=force_reload))
        cpi_index = (
            load_rbi_cpi(cache_dir, force_reload=force_reload)["index"].unstack("base").reindex(columns=CPI_BASES)
        )
        fred = load_fred_data(cache_dir, force_reload=force_reload).to_pandas()
        exchange_rate = load_real_effective_exchange_rate(country="IN", cache_dir=cache_dir, force_reload=force_reload)

        national_accounts = seasonally_adjusted_national_accounts(
            load_rbi_quarterly_gdp(cache_dir, force_reload=force_reload)
        )
        cpi = seasonally_adjust(
            complete_quarter_mean(splice_on_growth(cpi_index, anchor=CPI_ANCHOR_BASE)).rename("cpi"), log=True
        )
        quarterly = {
            **{component: national_accounts[component] for component in NATIONAL_ACCOUNTS},
            "cpi": cpi,
            "current_account_usd": seasonally_adjust(
                load_rbi_current_account(cache_dir, force_reload=force_reload)["current account"], log=False
            ),
            "real_effective_exchange_rate": complete_quarter_mean(exchange_rate["real_effective_exchange_rate"]),
            **foreign_quarterly(fred),
        }

        annual = {
            **world_bank_annual_series(world_bank, cpi_first_year=cpi.index.min().year),
            "government_debt": _india_annual(
                load_government_debt(cache_dir, countries=["IND"], force_reload=force_reload)
            )["government_debt"],
            "labour_share_pwt": _india_annual(load_pwt_data(cache_dir, force_reload=force_reload))["labour_share"],
            "labour_share_ilo": _india_annual(
                load_labour_income_share(cache_dir, countries=["IND"], force_reload=force_reload)
            )["labour_income_share"],
            "partner_activity": _india_annual(
                load_partner_activity(
                    cache_dir, countries=["IND"], last_year=PARTNER_ACTIVITY_LAST_YEAR, force_reload=force_reload
                )
            )["partner_activity"],
        }

        return assemble_panel(quarterly, {name: annual[name] for name in ANNUAL_SERIES})

    return cached(
        cache_dir,
        "india_quarterly_panel",
        build,
        pandas_parquet(),
        params={"reading": builder_fingerprint(build, *PANEL_RULES, rbi_fingerprints())},
        force=force_reload,
    )


def india_panel_metadata(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Describe each column of the India panel: its source, native frequency, units, adjustment and coverage.

    Parameters
    ----------
    panel : DataFrame
        The panel :func:`load_india_quarterly_panel` returns.

    Returns
    -------
    metadata : DataFrame
        One row per column, with its first and last observed quarter and how many quarters it observes.

    Examples
    --------
    What each column of the panel is and how much of it is observed:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data_functions.india_panel import india_panel_metadata, load_india_quarterly_panel

        metadata = india_panel_metadata(load_india_quarterly_panel(Path("data")))
    """
    quarterly = pd.DataFrame(
        {
            name: {
                "source": series.source,
                "native frequency": series.native_frequency,
                "units": series.units,
                "seasonal adjustment": series.seasonal_adjustment,
                "aggregation": series.aggregation,
                "panel frequency": "quarterly",
                "year convention": "calendar",
            }
            for name, series in QUARTERLY_SERIES.items()
        }
    ).T
    annual = pd.DataFrame(
        {
            name: {
                "source": series.source,
                "native frequency": "annual",
                "units": series.units,
                "seasonal adjustment": "none",
                "aggregation": series.aggregation,
                "panel frequency": "annual, last quarter of year",
                "year convention": series.year_convention,
            }
            for name, series in ANNUAL_SERIES.items()
        }
    ).T

    return (
        pd.concat([quarterly, annual])
        .loc[list(panel.columns)]
        .assign(
            first=panel.apply(pd.Series.first_valid_index),
            last=panel.apply(pd.Series.last_valid_index),
            observations=panel.notna().sum(),
        )
        .rename_axis("column")
    )


def seasonally_adjusted_national_accounts(gdp: pd.DataFrame) -> pd.DataFrame:
    """
    Splice each national-accounts component across base years on its growth rates, then seasonally adjust it.

    Parameters
    ----------
    gdp : DataFrame
        The RBI quarterly GDP components, indexed by base year and calendar quarter.

    Returns
    -------
    national_accounts : DataFrame
        One column per component of ``NATIONAL_ACCOUNTS`` at ``GDP_ANCHOR_BASE`` prices, indexed by quarter.
    """
    spliced = pd.DataFrame(
        {
            component: splice_on_growth(gdp[component].unstack("base"), anchor=GDP_ANCHOR_BASE)
            for component in NATIONAL_ACCOUNTS
        }
    )
    spliced.index.name = "quarter"

    return spliced.apply(seasonally_adjust, log=True)


def splice_on_growth(levels: pd.DataFrame, anchor: str) -> pd.Series:
    """
    Extend one base's levels backward and forward on the growth rates of the other bases, as far as they reach.

    Bases break in level at each revision, so a long series cannot take levels from more than one of them. Each period
    outside the anchor's coverage takes its growth from the nearest base observing both it and its neighbor, the
    later bases first going backward and the earlier first going forward.

    Parameters
    ----------
    levels : DataFrame
        One column per base, ordered from oldest to newest, indexed by consecutive periods.
    anchor : str
        The column whose levels are kept.

    Returns
    -------
    spliced : Series
        The anchor's levels, extended, over the periods it reaches.
    """
    observed_bases = [column for column in levels.columns if levels[column].notna().any()]
    if anchor not in observed_bases:
        raise ValueError(
            f"The anchor base {anchor!r} has no observations; the bases observed are {observed_bases}. A renamed sheet "
            "changes the base it is read as."
        )

    spliced = levels[anchor].copy()
    others = [column for column in levels.columns if column != anchor]
    present = levels.notna()

    def donor(period: pd.Period, neighbor: pd.Period, candidates: list[str]) -> str | None:
        usable = (
            column for column in candidates if bool(present.at[period, column]) and bool(present.at[neighbor, column])
        )
        return next(usable, None)

    for period in reversed(levels.index[levels.index < spliced.first_valid_index()]):
        source = donor(period, period + 1, others[::-1])
        if source is None:
            break
        spliced[period] = spliced[period + 1] * levels.at[period, source] / levels.at[period + 1, source]

    for period in levels.index[levels.index > spliced.last_valid_index()]:
        source = donor(period, period - 1, others)
        if source is None:
            break
        spliced[period] = spliced[period - 1] * levels.at[period, source] / levels.at[period - 1, source]

    return spliced.dropna()


def seasonally_adjust(series: pd.Series, *, log: bool) -> pd.Series:
    """
    Seasonally adjust a quarterly series with X-13ARIMA-SEATS and automatic outlier detection.

    Parameters
    ----------
    series : Series
        Quarterly values indexed by quarterly period. Missing quarters at either end are dropped, and one inside the
        span raises, since X-13 needs consecutive quarters.
    log : bool
        Adjust multiplicatively on the log scale, or additively for a series that changes sign.

    Returns
    -------
    adjusted : Series
        The seasonally adjusted values, over the quarters ``series`` observes.
    """
    observed = series.dropna()
    quarterly = observed.set_axis(pd.PeriodIndex(observed.index).to_timestamp()).asfreq("QS")
    gaps = quarterly.index[quarterly.isna()]
    if not gaps.empty:
        missing = [str(quarter) for quarter in pd.DatetimeIndex(gaps).to_period("Q")]
        raise ValueError(
            f"X-13 needs consecutive quarters, and {series.name!r} is missing {missing} inside its span. Mask fewer "
            "months, fill the gap, or end the series before it."
        )

    try:
        result = x13_arima_analysis(quarterly, log=log, outlier=True)
    except X13NotFoundError as err:
        raise X13NotFoundError(
            "Seasonal adjustment runs the Census Bureau's X-13ARIMA-SEATS program, `x13as`. Put the binary on PATH or "
            "set X13PATH to its directory. Census publishes it at https://www.census.gov/data/software/x13as.html."
        ) from err

    adjusted: pd.Series = result.seasadj

    return adjusted.set_axis(observed.index)


def complete_quarter_mean(monthly: pd.Series) -> pd.Series:
    """
    Average a monthly series over each calendar quarter that observes all three months.

    Parameters
    ----------
    monthly : Series
        Values indexed by monthly period.

    Returns
    -------
    quarterly : Series
        Means indexed by quarterly period.
    """
    observed = monthly.dropna()
    grouped = observed.groupby(pd.PeriodIndex(observed.index).asfreq("Q"))

    return grouped.mean().loc[grouped.size() == 3]


def foreign_quarterly(fred: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Put FRED's foreign block on calendar quarters.

    Parameters
    ----------
    fred : DataFrame
        FRED's tidy panel of ``series``, ``date`` and ``value``.

    Returns
    -------
    foreign : dict of str to Series
        Each series of ``FRED_SERIES`` indexed by quarterly period: the means of complete quarters of monthly averages,
        and the quarterly series in ``QUARTERLY_FRED_SERIES`` as published.
    """
    missing = sorted(set(FRED_SERIES) - set(fred["series"]))
    if missing:
        raise ValueError(f"The FRED panel has no {missing}; add them to the FRED loader's SERIES_NAMES.")

    monthly = (
        fred[fred["series"].isin(FRED_SERIES)]
        .assign(month=lambda frame: pd.PeriodIndex(frame["date"], freq="M"))
        .groupby(["series", "month"])["value"]
        .mean()
        .unstack("series")
    )
    foreign = {
        name: complete_quarter_mean(monthly[name]) for name in monthly.columns if name not in QUARTERLY_FRED_SERIES
    }
    for name in QUARTERLY_FRED_SERIES:
        native = monthly[name].dropna()
        foreign[name] = native.set_axis(pd.PeriodIndex(native.index).asfreq("Q"))

    return foreign


def world_bank_annual_series(world_bank: pd.DataFrame, cpi_first_year: int) -> dict[str, pd.Series]:
    """
    Derive the panel's World Bank columns from India's annual World Bank rows.

    Parameters
    ----------
    world_bank : DataFrame
        India's World Bank indicators, indexed by year.
    cpi_first_year : int
        The first calendar year of the quarterly CPI. Annual CPI inflation stops the year before, where the quarterly
        series takes over.

    Returns
    -------
    series : dict of str to Series
        Each World Bank column of ``ANNUAL_SERIES``, indexed by year, with percentages as fractions.
    """
    nominal_gdp = world_bank["real_gdp_lcu"] * world_bank["gdp_deflator"] / 100

    return {
        "nominal_consumption_share": world_bank["nominal_consumption"] / nominal_gdp,
        "nominal_investment_share": world_bank["nominal_investment"] / nominal_gdp,
        "tax_revenue_gdp": world_bank["tax_revenue_gdp"] / 100,
        "government_expense_gdp": world_bank["government_expense_gdp"] / 100,
        "cpi_inflation_annual": world_bank["cpi"].pct_change().loc[: cpi_first_year - 1],
        "exchange_rate": world_bank["exchange_rate"],
        "lending_rate": world_bank["lending_rate"] / 100,
        "external_debt_usd": world_bank["external_debt"],
        "employment_rate": world_bank["employment_rate"] / 100,
    }


def place_annual(values: pd.Series, year_convention: YearConvention) -> pd.Series:
    """
    Move an annual series onto the last calendar quarter of the year each value covers.

    Parameters
    ----------
    values : Series
        Annual values indexed by year. Missing years are dropped.
    year_convention : {'calendar', 'fiscal'}
        Whether the years are calendar years or fiscal years labeled by the calendar year they start in.

    Returns
    -------
    placed : Series
        The values indexed by quarterly period.
    """
    observed = values.dropna()
    year_shift, quarter = CLOSING_QUARTER[year_convention]
    closing_quarters = pd.PeriodIndex([pd.Period(f"{int(year) + year_shift}Q{quarter}") for year in observed.index])

    return observed.set_axis(closing_quarters)


def assemble_panel(quarterly: Mapping[str, pd.Series], annual: Mapping[str, pd.Series]) -> pd.DataFrame:
    """
    Join quarterly and annual series on the quarters the national accounts span.

    An annual value enters only when the whole year it covers falls inside the sample.

    Parameters
    ----------
    quarterly : dict of str to Series
        Series indexed by quarterly period, ``NATIONAL_ACCOUNTS`` among them.
    annual : dict of str to Series
        Series indexed by year, each named in ``ANNUAL_SERIES``.

    Returns
    -------
    panel : DataFrame
        Quarterly columns, then annual ones, indexed by each quarter's first day under ``quarter_start``.
    """
    national_accounts = quarterly[NATIONAL_ACCOUNTS[0]]
    sample = pd.period_range(
        start=national_accounts.index.min(), end=national_accounts.index.max(), freq="Q", name="quarter"
    )

    placed = pd.DataFrame(
        {
            name: place_annual(values, year_convention=ANNUAL_SERIES[name].year_convention)
            for name, values in annual.items()
        }
    )
    year_starts_in_sample = (placed.index - (QUARTERS_PER_YEAR - 1)) >= sample.min()
    panel = pd.concat([pd.DataFrame(quarterly), placed.loc[year_starts_in_sample]], axis=1).reindex(sample)

    return panel.set_axis(pd.PeriodIndex(panel.index).to_timestamp(how="start")).rename_axis("quarter_start")


def _india_annual(frame: pl.DataFrame) -> pd.DataFrame:
    """India's rows of a country-year frame, indexed by year."""
    if "country_code" in frame.columns:
        frame = frame.filter(pl.col("country_code") == "IND")

    return frame.to_pandas().set_index("year")


# Every helper and rule the panel's build reads, so editing one turns the cached panel over.
PANEL_RULES = (
    seasonally_adjusted_national_accounts,
    splice_on_growth,
    seasonally_adjust,
    complete_quarter_mean,
    foreign_quarterly,
    world_bank_annual_series,
    place_annual,
    assemble_panel,
    _india_annual,
    ANNUAL_SERIES,
    CLOSING_QUARTER,
    QUARTERLY_SERIES,
    (NATIONAL_ACCOUNTS, GDP_ANCHOR_BASE, CPI_BASES, CPI_ANCHOR_BASE, QUARTERLY_FRED_SERIES, PARTNER_ACTIVITY_LAST_YEAR),
)
