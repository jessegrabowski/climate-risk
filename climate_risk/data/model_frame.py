import polars as pl

from climate_risk.data_functions.emdat_processing import GEOMETRY_SOURCES
from climate_risk.exceptions import DataValidationError

EVENT_WINDOW_COLUMNS = ("DisNo.", "ISO", "year", "gids", "n_units", "finest_level", "geometry_source")

# Ordered by declaration rather than alphabetically, so the best source of several is the min.
SOURCE_ORDER = pl.Enum(GEOMETRY_SOURCES)


def _finest_level(gids: pl.Expr) -> pl.Expr:
    """Depth of the deepest unit a window holds, counting dots in the GADM identifier."""
    return gids.list.eval(pl.element().str.split("_").list.first().str.count_matches(r"\.")).list.max()


def event_windows(events: pl.DataFrame, geography: pl.DataFrame) -> pl.DataFrame:
    """
    Collect each event into the one observation window its geography describes.

    An event is located to whatever units the sources reached. Those units are not all at one administrative level: a
    district for one event, a province for the next, nothing at all for a third. The window is the union of the units
    named, so the resolution an event was reported at is carried rather than normalized away, and an event naming
    nothing reduces to its whole country.

    Parameters
    ----------
    events : DataFrame
        The workbook filtered to a place's window, from
        :func:`~climate_risk.data_functions.emdat_processing.load_emdat_events`.
    geography : DataFrame
        One row per event-unit, from
        :func:`~climate_risk.data_functions.emdat_processing.event_geography`.

    Returns
    -------
    windows : DataFrame
        One row per event, ordered by identifier. ``gids`` holds the units the window covers, empty
        where the window is the whole country; ``n_units`` and ``finest_level`` describe its extent,
        the latter null on a whole-country window. ``geometry_source`` is the best source that
        reached the event, ranked by ``GEOMETRY_SOURCES``.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data.model_frame import event_windows
        from climate_risk.data_functions.emdat_processing import event_geography

        windows = event_windows(events, event_geography(events))
    """
    missing = set(events["DisNo."]) - set(geography["DisNo."])
    if missing:
        raise DataValidationError(
            f"{len(missing)} events carry no geography row, so their window is undefined: {sorted(missing)[:5]}"
        )

    windows = (
        geography.select("DisNo.", "gid", pl.col("geometry_source").cast(SOURCE_ORDER))
        .unique()
        .group_by("DisNo.")
        .agg(
            pl.col("gid").drop_nulls().sort().alias("gids"),
            pl.col("geometry_source").min(),
        )
    )

    return (
        events.select("DisNo.", "ISO", pl.col("Start_Year").dt.year().alias("year"))
        .join(windows, on="DisNo.", how="left")
        .with_columns(
            pl.col("gids").list.len().alias("n_units"),
            _finest_level(pl.col("gids")).alias("finest_level"),
            pl.col("geometry_source").cast(pl.String),
        )
        .select(EVENT_WINDOW_COLUMNS)
        .sort("DisNo.")
    )
