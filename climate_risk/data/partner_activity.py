import logging

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from kuznets import imf

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.source import ApiSource
from climate_risk.data.world_bank import load_wb_data

_log = logging.getLogger(__name__)

IMTS = ApiSource(
    url="https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IMTS,1.0.0",
    license="\u00a9 International Monetary Fund. All rights reserved. https://www.imf.org/external/terms.htm",
    citation=(
        "International Monetary Fund, International Trade in Goods (by partner country), "
        "https://data.imf.org/en/datasets/IMF.STA:IMTS."
    ),
    retrieved="2026-09-05",
)

# IMTS reports goods exports FOB in US dollars. The counterpart dimension mixes real partners with
# the Fund's regional and income aggregates, which carry codes outside ISO 3166-1 alpha-3.
EXPORT_INDICATOR = "XG_FOB_USD"
ISO3_PATTERN = r"^[A-Z]{3}$"

# The year every partner's output is measured against. Partners without output in this year drop out
# of the index, so it wants to be a year with wide coverage rather than the newest one.
BASE_YEAR = 2015

# First year of the window the index is built over. Partner coverage thins out rapidly before this.
FIRST_YEAR = 2000

# A country whose retained partners take less than this share of its exports gets a warning: the
# index is then describing a minority of the markets it sells into.
MIN_COVERAGE = 0.8


def transform_partner_activity(
    exports: pl.DataFrame,
    gdp: pl.DataFrame,
    *,
    base_year: int,
    first_year: int,
    last_year: int,
) -> pl.DataFrame:
    r"""
    Weight partners' output by the share of a country's exports going to each.

    Partner output enters as a log index against ``base_year``, so the result is a unit-free measure
    of activity in the markets a country sells into. Levels in local currency are not comparable
    across countries. The ratio of each partner to its own base year is.

    Weights are export shares over the whole of ``exports``, so a caller chooses the weighting window
    by filtering before calling.

    The index is built from one fixed set of partners: those with output in the base year and in
    every year from ``first_year`` to ``last_year``. Weights are renormalized over that set once. A
    set that changed from year to year would shift the weighted mean by a discontinuous constant
    whenever a partner entered or left, which is indistinguishable in the series from a movement in
    foreign demand.

    Parameters
    ----------
    exports : DataFrame
        Bilateral exports as ``kuznets`` returns them, with ``country``, ``counterpart``, ``period``
        and ``value`` columns.
    gdp : DataFrame
        The World Bank panel, keyed by ``country_code`` and ``year``, carrying ``real_gdp_lcu``.
    base_year : int
        Year each partner's output is indexed to.
    first_year : int
        First year of the window the index covers.
    last_year : int
        Last year of the window the index covers.

    Returns
    -------
    activity : DataFrame
        One row per country and year, carrying ``partner_activity`` and the ``partner_coverage``
        share of exports the retained partners take. Coverage is constant within a country, since
        the partner set is.

    Notes
    -----
    The construction is the country-specific foreign variable of the global VAR literature, where
    :math:`x^*_{it} = \sum_j w_{ij} x_{jt}` with :math:`x` in logs and weights summing to one ([1]_,
    [2]_). Weighting logs and exponentiating is the geometric aggregation of the BIS effective
    exchange rate scheme ([3]_, [4]_). Foreign output serves as an observable in an estimated small
    open economy model in [5]_.

    Weights are single bilateral export shares. [3]_ weights a foreign demand variable by exports alone while using
    double weights for competitiveness. The asymmetry is the point: a competitiveness index has to price rivalry in
    third markets, whereas an activity index asks only whose spending buys a country's output. The global VAR papers
    weight by total trade because one matrix there serves prices and interest rates as well as output.

    The partner set is fixed across the window rather than renormalized year by year as in [1]_, so
    that a partner entering or leaving cannot move the level of the index.

    References
    ----------
    .. [1] Pesaran, M. H., T. Schuermann, and S. M. Weiner (2004). "Modeling Regional
           Interdependencies Using a Global Error-Correcting Macroeconometric Model." Journal of
           Business & Economic Statistics 22(2), 129-162.
    .. [2] Dees, S., F. di Mauro, M. H. Pesaran, and L. V. Smith (2007). "Exploring the
           International Linkages of the Euro Area: A Global VAR Analysis." Journal of Applied
           Econometrics 22(1), 1-38.
    .. [3] Turner, P., and J. Van 't dack (1993). "Measuring International Price and Cost
           Competitiveness." BIS Economic Papers 39, Bank for International Settlements.
    .. [4] Klau, M., and S. S. Fung (2006). "The New BIS Effective Exchange Rate Indices." BIS
           Quarterly Review, March 2006, 51-65.
    .. [5] Adolfson, M., S. Laseen, J. Linde, and M. Villani (2007). "Bayesian Estimation of an
           Open Economy DSGE Model with Incomplete Pass-Through." Journal of International
           Economics 72(2), 481-511.
    """
    if last_year < first_year:
        raise ValueError(f"the window {first_year}-{last_year} ends before it starts")

    required_years = sorted({*range(first_year, last_year + 1), base_year})
    complete_partners = (
        gdp.filter(pl.col("year").is_in(required_years))
        .drop_nulls("real_gdp_lcu")
        .group_by("country_code")
        .agg(pl.col("year").n_unique().alias("years"))
        .filter(pl.col("years") == len(required_years))
        .select(pl.col("country_code").alias("partner_code"))
    )
    if complete_partners.is_empty():
        raise ValueError(f"No country reports output in every year of {first_year}-{last_year} and {base_year}")

    weights = _export_weights(exports)
    retained = weights.join(complete_partners, on="partner_code", how="inner")
    coverage = retained.group_by("country_code").agg(pl.col("weight").sum().alias("partner_coverage"))
    _warn_on_thin_coverage(coverage)

    base_year_output = gdp.filter(pl.col("year") == base_year).select(
        pl.col("country_code").alias("partner_code"), pl.col("real_gdp_lcu").alias("base_output")
    )
    indexed = (
        gdp.filter(pl.col("year").is_between(first_year, last_year))
        .select(pl.col("country_code").alias("partner_code"), "year", "real_gdp_lcu")
        .join(base_year_output, on="partner_code")
        .select("partner_code", "year", (pl.col("real_gdp_lcu") / pl.col("base_output")).log().alias("log_output"))
    )

    return (
        retained.join(indexed, on="partner_code")
        .group_by("country_code", "year")
        .agg(((pl.col("weight") * pl.col("log_output")).sum() / pl.col("weight").sum()).exp().alias("partner_activity"))
        .join(coverage, on="country_code")
        .sort("country_code", "year")
    )


def _export_weights(exports: pl.DataFrame) -> pl.DataFrame:
    """Return each country's exports to each partner as a share of its exports to all of them."""
    partners = (
        exports.drop_nulls("value")
        .filter(pl.col("counterpart").str.contains(ISO3_PATTERN))
        # A country reports trade with itself in some years; it is not a foreign market.
        .filter(pl.col("counterpart") != pl.col("country"))
        .group_by("country", "counterpart")
        .agg(pl.col("value").sum().alias("exported"))
    )

    return (
        partners.with_columns((pl.col("exported") / pl.col("exported").sum().over("country")).alias("weight"))
        .select(
            pl.col("country").alias("country_code"),
            pl.col("counterpart").alias("partner_code"),
            "weight",
        )
        .sort("country_code", "partner_code")
    )


def _warn_on_thin_coverage(coverage: pl.DataFrame) -> None:
    """Name the countries whose retained partners take less than ``MIN_COVERAGE`` of their exports."""
    thin = coverage.filter(pl.col("partner_coverage") < MIN_COVERAGE).sort("country_code")
    for country, share in thin.iter_rows():
        _log.warning(f"{country}: partners with complete output cover only {share:.1%} of its exports")


def load_partner_activity(
    cache_dir: Path,
    countries: Iterable[str],
    *,
    base_year: int = BASE_YEAR,
    first_year: int = FIRST_YEAR,
    last_year: int,
    force_reload: bool = False,
) -> pl.DataFrame:
    """
    Return trade-weighted partner activity for ``countries``, one row per country and year.

    Both upstreams publish annually, so the result is annual.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the cached panels.
    countries : iterable of str
        ISO 3166-1 alpha-3 codes to build the index for.
    base_year : int, optional
        Year each partner's output is indexed to. Default 2015.
    first_year : int, optional
        First year the index covers. Default 2000.
    last_year : int
        Last year the index covers. It keys the cache, so a newer vintage of the output panel is a
        different entry rather than a silent extension of an existing one.
    force_reload : bool, optional
        Rebuild even when the index is already cached. Default False.

    Returns
    -------
    activity : DataFrame
        One row per country and year, carrying ``partner_activity`` and ``partner_coverage``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.partner_activity import load_partner_activity

        activity = load_partner_activity(Path("data"), ["LAO", "THA", "VNM"], last_year=2023)
    """
    codes = sorted(countries)

    def build() -> pl.DataFrame:
        # Both upstreams are reached inside the builder, so a warm cache touches neither.
        gdp = load_wb_data(cache_dir, force_reload=force_reload)
        _log.info(f"Downloading IMTS bilateral exports for {', '.join(codes)}")
        exports = imf.IMTSReader(codes, indicator=EXPORT_INDICATOR, freq="A", output_type="polars").read()
        if not isinstance(exports, pl.DataFrame):
            raise TypeError(f"kuznets returned a {type(exports).__name__} for output_type='polars'")

        return transform_partner_activity(exports, gdp, base_year=base_year, first_year=first_year, last_year=last_year)

    return cached(
        cache_dir,
        "partner_activity",
        build,
        polars_parquet(),
        params={
            "countries": "-".join(codes),
            "base_year": base_year,
            "first_year": first_year,
            "last_year": last_year,
            "reading": builder_fingerprint(build, EXPORT_INDICATOR, ISO3_PATTERN),
        },
        force=force_reload,
    )
