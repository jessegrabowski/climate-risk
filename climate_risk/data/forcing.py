from pathlib import Path

import polars as pl

from climate_risk.data.cache import cached, polars_parquet
from climate_risk.data.fetch import fetch
from climate_risk.data.source import DataSource
from climate_risk.exceptions import DataValidationError

FORCING_URL = "https://zenodo.org/records/5705391/files"
FORCING_SUBDIRECTORY = "forcing"
FORCING_LICENSE = "CC BY 4.0"
FORCING_CITATION = (
    "IPCC, 2021: Annex III: Tables of historical and projected well-mixed greenhouse gas mixing ratios and "
    "effective radiative forcing of all climate forcers [Dentener, F.J., B. Hall, C. Smith (eds.)]. In Climate "
    "Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the "
    "Intergovernmental Panel on Climate Change. Extended data, doi:10.5281/zenodo.5705391"
)
FORCING_RETRIEVED = "2026-10-25"

# Annex III publishes the history in table A3.3 and each Tier 1 scenario in a lettered part of A3.4,
# each from 1750, the scenarios running to 2500.
FORCING_TABLES = {
    "historical": "A3.3_historical_ERF_1750-2019",
    "ssp119": "A3.4a_ssp119_ERF_1750-2500",
    "ssp126": "A3.4b_ssp126_ERF_1750-2500",
    "ssp245": "A3.4c_ssp245_ERF_1750-2500",
    "ssp370": "A3.4d_ssp370_ERF_1750-2500",
    "ssp585": "A3.4e_ssp585_ERF_1750-2500",
}
FORCING_ESTIMATES = ("best_estimate", "5th_percentile", "95th_percentile")


def forcing_dir(cache_dir: Path) -> Path:
    """Directory the forcing tables are cached under."""
    return cache_dir / FORCING_SUBDIRECTORY


def forcing_source(scenario: str, estimate: str) -> DataSource:
    """
    Declare one Annex III table of effective radiative forcing.

    Parameters
    ----------
    scenario : str
        ``'historical'`` or a Tier 1 scenario, one of ``FORCING_TABLES``.
    estimate : str
        ``'best_estimate'`` or a percentile of the assessed range, one of ``FORCING_ESTIMATES``.

    Returns
    -------
    source : DataSource
        The table as Zenodo serves it.
    """
    if scenario not in FORCING_TABLES:
        raise DataValidationError(f"Annex III publishes forcing for {tuple(FORCING_TABLES)}, not {scenario!r}")
    if estimate not in FORCING_ESTIMATES:
        raise DataValidationError(f"Annex III publishes the estimates {FORCING_ESTIMATES}, not {estimate!r}")

    filename = f"table_{FORCING_TABLES[scenario]}_{estimate}.csv"

    return DataSource(
        url=f"{FORCING_URL}/{filename}?download=1",
        filename=filename,
        license=FORCING_LICENSE,
        citation=FORCING_CITATION,
        retrieved=FORCING_RETRIEVED,
    )


def load_effective_radiative_forcing(
    scenario: str, estimate: str, cache_dir: Path, *, force_reload: bool = False
) -> pl.DataFrame:
    """
    Load IPCC AR6 effective radiative forcing by forcing agent, one row per year from 1750.

    The scenario tables carry the years through 2019 as well, and some agents in them, black carbon
    on snow among them, differ from the historical table's, so a series is read from one table only.

    Parameters
    ----------
    scenario : str
        ``'historical'`` for 1750 to 2019, or a Tier 1 scenario such as ``'ssp245'`` for 1750 to 2500.
    estimate : str
        ``'best_estimate'``, ``'5th_percentile'`` or ``'95th_percentile'``.
    cache_dir : Path
        Directory the source caches live under.
    force_reload : bool, optional
        Download again and rebuild the cache rather than reading it. Default False.

    Returns
    -------
    forcing : DataFrame
        A ``year`` column and one column per forcing agent as Annex III names it, in W m-2, ending in
        ``total_anthropogenic``, ``total_natural`` and ``total``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.forcing import load_effective_radiative_forcing

        forcing = load_effective_radiative_forcing("ssp245", "best_estimate", Path("data"))
    """
    source = forcing_source(scenario, estimate)

    def build() -> pl.DataFrame:
        # Fetching inside the builder keeps a warm cache from reaching the network.
        raw = fetch(source, forcing_dir(cache_dir), force=force_reload)

        return pl.read_csv(raw).with_columns(pl.col("year").cast(pl.Int64)).sort("year")

    return cached(
        forcing_dir(cache_dir),
        "effective_radiative_forcing",
        build,
        polars_parquet(),
        params={"scenario": scenario, "estimate": estimate},
        force=force_reload,
    )
