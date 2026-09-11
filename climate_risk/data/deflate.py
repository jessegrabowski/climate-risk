from collections.abc import Sequence
from pathlib import Path

import polars as pl

from climate_risk.data.fred import load_fred_data

# The US consumer price index, which is what a constant-dollar amount in this project is constant
# against. It reaches the FRED panel under this name.
US_PRICE_LEVEL = "world_price_level"

MONTHS_IN_YEAR = 12

# A month an index skipped is a month it will not publish later: the BLS never issued an October 2025
# consumer price index, so demanding a full twelve would drop that year for good. An annual average
# short one or two months of a slow-moving index still lands on that year's level. Short a quarter of
# them it does not, and the series is more likely reported at some other frequency.
MIN_MONTHS = 10

# The year money is stated in. The World Bank constant-price series are 2015 US dollars and the
# partner-activity index is based there, so an amount restated anywhere else stops being comparable
# to the covariates it is regressed on.
BASE_YEAR = 2015


def _covered_years(price_level: pl.DataFrame) -> str:
    """The span an index reaches, for a message that has to say what was there instead."""
    years = price_level["year"].to_list()

    return f"{min(years)} to {max(years)}"


def annual_price_level(observations: pl.DataFrame, series: str = US_PRICE_LEVEL) -> pl.DataFrame:
    """
    Average a monthly price index onto calendar years.

    Parameters
    ----------
    observations : DataFrame
        The FRED panel, with ``series``, ``date`` and ``value`` columns, from
        :func:`~climate_risk.data.fred.load_fred_data`.
    series : str, optional
        Which series to read. Default ``"world_price_level"``, the US consumer price index.

    Returns
    -------
    price_level : DataFrame
        One row per closed calendar year the index covers, with ``year`` and ``price_level``, sorted.
    """
    monthly = observations.filter(pl.col("series") == series)
    if monthly.is_empty():
        raise ValueError(f"The panel carries no series named {series!r}.")

    annual = (
        monthly.group_by(pl.col("date").dt.year().alias("year"))
        .agg(pl.col("value").mean().alias("price_level"), pl.len().alias("months"))
        .sort("year")
    )

    # The newest year is still being published unless the index has reached its December, and the
    # months that happen to have arrived average to a level that is not that year's.
    open_year = monthly["date"].dt.year().max()
    closed = annual.filter((pl.col("year") < open_year) | (pl.col("months") == MONTHS_IN_YEAR))

    covered = closed.filter(pl.col("months") >= MIN_MONTHS)
    if covered.is_empty():
        raise ValueError(
            f"No closed year of {series!r} carries {MIN_MONTHS} months, reaching "
            f"{max(annual['months'].to_list())} at most, so it is not a monthly index."
        )

    return covered.drop("months")


def deflation_factors(price_level: pl.DataFrame, base_year: int = BASE_YEAR) -> pl.DataFrame:
    """
    Give each year the multiplier that carries an amount dated in it into base-year dollars.

    Parameters
    ----------
    price_level : DataFrame
        Annual index levels, with ``year`` and ``price_level``, from :func:`annual_price_level`.
    base_year : int, optional
        The year amounts are stated in. Default ``BASE_YEAR``.

    Returns
    -------
    factors : DataFrame
        ``year`` and ``deflator``, the latter one in ``base_year``, above one in the years before it
        and below one in the years after.
    """
    base = price_level.filter(pl.col("year") == base_year)
    if base.is_empty():
        raise ValueError(f"The price index runs {_covered_years(price_level)} and so cannot be based on {base_year}.")

    return price_level.select("year", (base["price_level"].item() / pl.col("price_level")).alias("deflator"))


def deflate(
    values: pl.DataFrame,
    price_level: pl.DataFrame,
    columns: Sequence[str],
    *,
    base_year: int = BASE_YEAR,
    year_column: str = "year",
) -> pl.DataFrame:
    """
    State nominal amounts in constant base-year dollars, each by the year it was measured in.

    Parameters
    ----------
    values : DataFrame
        Rows carrying money, dated by ``year_column``.
    price_level : DataFrame
        Annual index levels, from :func:`annual_price_level`.
    columns : sequence of str
        The money columns to restate. Every other column passes through.
    base_year : int, optional
        The year to state them in. Default ``BASE_YEAR``.
    year_column : str, optional
        Column holding the year each amount was measured in. Default ``"year"``.

    Returns
    -------
    deflated : DataFrame
        ``values`` with each named column multiplied by its row's factor.
    """
    years = values[year_column]
    if years.null_count():
        raise ValueError(f"{years.null_count()} rows carry no {year_column} and so cannot be dated to a price level.")

    factors = deflation_factors(price_level, base_year=base_year)
    uncovered = sorted(set(years) - set(factors["year"]))
    if uncovered:
        raise ValueError(f"The price index does not cover {uncovered}, which {len(uncovered)} amounts are dated to.")

    # Mapped rather than joined, so that a caller already carrying a column of this name keeps it and
    # the rows come back in the order they went in.
    factor = pl.col(year_column).replace_strict(dict(factors.iter_rows()), return_dtype=pl.Float64)

    return values.with_columns(*(pl.col(column) * factor for column in columns))


def rebase(
    values: pl.DataFrame,
    price_level: pl.DataFrame,
    columns: Sequence[str],
    *,
    source_year: int,
    base_year: int = BASE_YEAR,
) -> pl.DataFrame:
    """
    State amounts already constant in one year's dollars in another year's.

    EM-DAT publishes its adjusted damage columns in the dollars of a single year whatever year the
    event fell in, so they move by one factor rather than by their own dates.

    Parameters
    ----------
    values : DataFrame
        Rows carrying money, all of it constant in ``source_year`` dollars.
    price_level : DataFrame
        Annual index levels, from :func:`annual_price_level`.
    columns : sequence of str
        The money columns to restate. Every other column passes through.
    source_year : int
        The year the amounts are already stated in.
    base_year : int, optional
        The year to state them in. Default ``BASE_YEAR``.

    Returns
    -------
    rebased : DataFrame
        ``values`` with each named column multiplied by the one factor.
    """
    factors = deflation_factors(price_level, base_year=base_year)
    source = factors.filter(pl.col("year") == source_year)
    if source.is_empty():
        raise ValueError(
            f"The price index runs {_covered_years(factors)} and so does not reach {source_year}, which "
            f"the amounts are stated in."
        )

    return values.with_columns(*(pl.col(column) * source["deflator"].item() for column in columns))


def load_price_level(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Return the US consumer price index averaged onto calendar years.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    force_reload : bool, optional
        Download again and rebuild the cache rather than reading it. Default False.

    Returns
    -------
    price_level : DataFrame
        One row per closed calendar year the index covers, with ``year`` and ``price_level``, sorted.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.deflate import load_price_level

        price_level = load_price_level(Path("data"))
    """
    return annual_price_level(load_fred_data(cache_dir, force_reload=force_reload))
