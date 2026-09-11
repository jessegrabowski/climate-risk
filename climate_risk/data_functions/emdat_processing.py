import datetime as dt
import json
import re

from pathlib import Path
from typing import NamedTuple

import polars as pl

from climate_risk.config.schema import EventFilters
from climate_risk.data.source import ManualSource
from climate_risk.exceptions import DataValidationError

# EM-DAT requires an account and forbids redistribution, so it is fetched by a person, not by code.
EMDAT = ManualSource(
    filename="emdat.xlsx",
    homepage="https://public.emdat.be/",
    license=(
        "Free for non-commercial use with attribution. Redistribution of the database is not "
        "permitted; users must download it themselves after registering."
    ),
    citation=(
        "EM-DAT, CRED / UCLouvain, Brussels, Belgium. Delforge, D. et al. (2025), EM-DAT: the "
        "Emergency Events Database. International Journal of Disaster Risk Reduction 124, 105509. "
        "https://doi.org/10.1016/j.ijdrr.2025.105509"
    ),
    retrieved="2026-08-03",
)

EM_DAT_COL_DICT = {
    "Start Year": "Start_Year",
    "Total Deaths": "Deaths",
    "No. Injured": "Injured",
    "No. Affected": "Numb_Affected",
    "No. Homeless": "Homeless",
    "Total Affected": "Total_Affected",
    "Reconstruction Costs ('000 US$)": "Reconstruction_Costs",
    "Reconstruction Costs, Adjusted ('000 US$)": "Reconstruction_Costs_Adjusted",
    "Insured Damage ('000 US$)": "Insured_Damage",
    "Insured Damage, Adjusted ('000 US$)": "Insured_Damage_Adjusted",
    "Total Damage ('000 US$)": "Total_Damage",
    "Total Damage, Adjusted ('000 US$)": "Total_Damage_Adjusted",
}

PROB_COLS = [
    "Country",
    "ISO",
    "Start_Year",
    "Drought",
    "Extreme temperature",
    "Flood",
    "Storm",
    "Wildfire",
    "Mass movement (dry)",
    "Mass movement (wet)",
    "Region",
    "Subregion",
]

INTENSITY_COLS = [
    "Country",
    "ISO",
    "Start_Year",
    "Region",
    "Deaths",
    "Injured",
    "Numb_Affected",
    "Homeless",
    "Total_Affected",
    "Total_Damage",
    "Total_Damage_Adjusted",
    "Disaster Type",
]

# The study window opens in 1969 and closes on the newest event in the workbook.
EMDAT_WINDOW_START = dt.date(1969, 1, 1)

# The types the panel counts. A country with no wildfires still needs a Wildfire column, so these
# are the columns the count frames carry whether or not the data contains them.
DISASTER_TYPES = tuple(c for c in PROB_COLS if c not in {"Country", "ISO", "Start_Year", "Region", "Subregion"})

# Columns read before any rename. Nothing detects upstream schema drift, so this check is the
# earliest point a changed export becomes a named error rather than a missing attribute.
REQUIRED_EMDAT_COLUMNS = {"ISO", "Region", "Subregion", "Disaster Type", "GADM Admin Units"} | set(EM_DAT_COL_DICT)

# EM-DAT writes empty strings, not blank cells: undeclared, a cost column reads as text and an
# all-empty text column warns.
EMDAT_DTYPES = {
    "AID Contribution ('000 US$)": "float",
    "Reconstruction Costs ('000 US$)": "float",
    "Reconstruction Costs, Adjusted ('000 US$)": "float",
    "Admin Units": "string",
    "GADM Admin Units": "string",
}

HYDROMETEOROLOGICAL = "Hydrometeorological"
CLIMATOLOGICAL = "Climatological"

DISASTER_CLASSES = {
    "Storm": HYDROMETEOROLOGICAL,
    "Flood": HYDROMETEOROLOGICAL,
    "Mass movement (wet)": HYDROMETEOROLOGICAL,
    "Wildfire": CLIMATOLOGICAL,
    "Extreme temperature": CLIMATOLOGICAL,
    "Drought": CLIMATOLOGICAL,
}

DAMAGE_VARS = [
    "Deaths",
    "Injured",
    "Numb_Affected",
    "Homeless",
    "Total_Affected",
    "Total_Damage",
    "Total_Damage_Adjusted",
]

# A country-year with no events reads as missing rather than zero, which is what the replication
# panel relies on. Floats keep that true whether the frame is read as polars or as pandas.
COUNT_DTYPE = pl.Float64


def _read_workbook(emdat_path: Path) -> pl.DataFrame:
    """Read the EM-DAT sheet, raising a named error rather than letting a changed export surface later."""
    workbook = pl.read_excel(
        emdat_path,
        sheet_name="EM-DAT Data",
        raise_if_empty=False,
        read_options={"dtypes": EMDAT_DTYPES},
    )

    if workbook.is_empty():
        raise ValueError(f"The `EM-DAT Data` sheet in `{emdat_path}` has no rows.")

    missing_columns = REQUIRED_EMDAT_COLUMNS - set(workbook.columns)
    if missing_columns:
        raise ValueError(
            f"The `EM-DAT Data` sheet in `{emdat_path}` is missing {sorted(missing_columns)}. "
            f"Re-download the database, or update EM_DAT_COL_DICT if the export has changed."
        )

    return workbook.rename(EM_DAT_COL_DICT).with_columns(
        pl.date(pl.col("Start_Year"), 1, 1).alias("Start_Year"),
        pl.col("Disaster Type").replace_strict(DISASTER_CLASSES, default=None).alias("disaster_class"),
    )


def country_year_grid(events: pl.DataFrame, *, window_start: dt.date = EMDAT_WINDOW_START) -> pl.DataFrame:
    """
    Cross every country with every year in the window, carrying each country's region.

    This is the grid the count and damage panels are laid over, so it is built from the *unfiltered*
    workbook: a country-year with no qualifying event still needs a row.

    Parameters
    ----------
    events : DataFrame
        The workbook, as :func:`load_emdat_events` returns it.
    window_start : datetime.date, optional
        First year of the panel. Default ``EMDAT_WINDOW_START``.

    Returns
    -------
    grid : DataFrame
        ``ISO``, ``Start_Year``, ``Region`` and ``Subregion``, sorted by country and year.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data_functions.emdat_processing import country_year_grid

        grid = country_year_grid(events)
    """
    newest_event = events["Start_Year"].max()
    if not isinstance(newest_event, dt.date):
        raise ValueError("Every Start_Year in the workbook is missing, so the window has no end.")

    if window_start > newest_event:
        raise ValueError(
            f"window_start={window_start} is after the newest event in the workbook "
            f"({newest_event}), so every output frame would be empty."
        )

    years: pl.Series = pl.date_range(window_start, newest_event, interval="1y", eager=True)

    regions = events.select("ISO", "Region", "Subregion").unique(subset="ISO", keep="first")

    return (
        events.select(pl.col("ISO").unique())
        .join(years.alias("Start_Year").to_frame(), how="cross")
        .join(regions, on="ISO", how="left")
        .sort("ISO", "Start_Year")
    )


def count_events_by_type(events: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """
    Count events per country, year and disaster type, over every row of ``grid``.

    Parameters
    ----------
    events : DataFrame
        Events to count, already narrowed to whichever ones should be counted.
    grid : DataFrame
        The country-year panel from :func:`country_year_grid`.

    Returns
    -------
    counts : DataFrame
        One row per country-year, one column per disaster type. A country-year with no events
        reads as null rather than zero.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data_functions.emdat_processing import count_events_by_type, country_year_grid

        counts = count_events_by_type(events, country_year_grid(events))
    """
    counted = (
        events.filter(pl.col("Disaster Type").is_in(DISASTER_TYPES))
        .group_by("ISO", "Start_Year", "Disaster Type")
        .len()
        .with_columns(pl.col("len").cast(COUNT_DTYPE))
        .pivot(on="Disaster Type", index=["ISO", "Start_Year"], values="len")
    )
    absent = [pl.lit(None, dtype=COUNT_DTYPE).alias(name) for name in DISASTER_TYPES if name not in counted.columns]

    return (
        grid.join(counted.with_columns(absent), on=["ISO", "Start_Year"], how="left")
        .select("ISO", "Start_Year", "Region", "Subregion", *DISASTER_TYPES)
        .sort("ISO", "Start_Year")
    )


def total_damage(events: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """
    Total each damage measure per country and year, over every row of ``grid``.

    Parameters
    ----------
    events : DataFrame
        Events to total, already narrowed to whichever ones should count.
    grid : DataFrame
        The country-year panel from :func:`country_year_grid`.

    Returns
    -------
    damage : DataFrame
        One row per country-year, one column per measure in ``DAMAGE_VARS``.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data_functions.emdat_processing import country_year_grid, total_damage

        damage = total_damage(events, country_year_grid(events))
    """
    totals = (
        events.filter(pl.col("Disaster Type").is_in(DISASTER_TYPES))
        .select(INTENSITY_COLS)
        .group_by("ISO", "Start_Year")
        .agg(pl.col(name).sum().cast(COUNT_DTYPE) for name in DAMAGE_VARS)
    )

    return (
        grid.join(totals, on=["ISO", "Start_Year"], how="left")
        .select("ISO", "Start_Year", *DAMAGE_VARS, "Region", "Subregion")
        .sort("ISO", "Start_Year")
    )


def load_emdat_events(cache_dir: Path) -> pl.DataFrame:
    """
    Read the EM-DAT workbook, renamed and classified, with nothing filtered out.

    Narrow it with :func:`event_filter`, or with any other polars predicate.

    Parameters
    ----------
    cache_dir : Path
        Directory holding ``emdat.xlsx``.

    Returns
    -------
    events : DataFrame
        One row per recorded event, keyed by ``DisNo.``, carrying ``disaster_class`` and the renamed
        damage columns.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_emdat_events

        events = load_emdat_events(Path("data"))
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    return _read_workbook(EMDAT.require(cache_dir))


LOCATION_SEPARATORS = ",;"

# `X, and Y` splits on the comma and strands the conjunction on the second span. No GADM unit is
# named with a leading `and`, so a span that starts with one is an artifact of the split.
STRANDED_CONJUNCTION = re.compile(r"^(?:and|&)\b\s*", re.IGNORECASE)

# A spreadsheet label EM-DAT leaks into the location field, as `Level 1 = Uttar Pradesh`.
FIELD_LABEL = re.compile(r"^\s*level\s*\d+\s*=\s*", re.IGNORECASE)

# `Montgomery County in (Tennessee state)` joins a place to its container with a preposition, which
# the split leaves dangling on the end of the name.
DANGLING_PREPOSITION = re.compile(r"\s+(?:in|of|at|on)$", re.IGNORECASE)


class NamedPlace(NamedTuple):
    """A place an event's location text names, and the place the text says contains it.

    Parameters
    ----------
    name : str
        The place as written, whitespace collapsed and nothing else changed.
    parent : str or None
        The containing place the prose gave in parentheses, or None where it gave none.
    """

    name: str
    parent: str | None


def named_places(location: str | None) -> list[NamedPlace]:
    """
    Explode an event's location text into the places it names.

    EM-DAT writes a list of places, optionally followed by a parenthesised container that applies
    to every place since the last one. ``Muang Nakhon Si Thammarat, Hua Sai, Pak Phanang, Ron
    Phibun districts (Nakhon Si Thammarat province)`` names four districts of one province. The
    parent is what tells a repeated district name which province it belongs to, so it is carried
    rather than flattened into a sibling.

    Separators are commas and semicolons outside parentheses. ``and`` is not one: 75 GADM units are
    named like ``Newfoundland and Labrador``, and splitting on it would destroy them. Where the prose writes ``X, and
    Y`` the comma splits and the conjunction is left stranded, so a leading one is dropped from each place. The same
    goes for a preposition left dangling on the end by ``X in (Y)``, and for the ``Level 1 =`` label the workbook leaks
    into the column.

    Parameters
    ----------
    location : str or None
        The ``Location`` column as EM-DAT writes it.

    Returns
    -------
    places : list of NamedPlace
        One entry per place named, in the order written, without repeats. Empty when the text names
        nothing.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data_functions.emdat_processing import named_places

        for place in named_places("Hua Sai, Pak Phanang districts (Nakhon Si Thammarat province)"):
            print(place)
    """
    spans: list[str] = []
    places: list[NamedPlace] = []
    written: list[str] = []
    container: list[str] = []
    depth = 0

    def close_span() -> None:
        collapsed = " ".join("".join(written).split())
        without_conjunction = STRANDED_CONJUNCTION.sub("", FIELD_LABEL.sub("", collapsed))
        if trimmed := DANGLING_PREPOSITION.sub("", without_conjunction).strip():
            spans.append(trimmed)
        written.clear()

    def close_group() -> None:
        # Nested parentheses gloss the container rather than nesting inside it — `Region I (Ilocos
        # region)` is one place under two names — so the inner text is dropped, not treated as a
        # further parent.
        parent = " ".join(re.sub(r"\([^()]*\)", " ", "".join(container)).split()) or None
        places.extend(NamedPlace(span, parent) for span in spans)
        spans.clear()
        container.clear()

    for character in str(location or ""):
        if character == "(":
            depth += 1
            if depth == 1:
                close_span()
                continue
        elif character == ")":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0:
                close_group()
                continue

        if depth:
            container.append(character)
        elif character in LOCATION_SEPARATORS:
            close_span()
        else:
            written.append(character)

    close_span()
    # An unclosed parenthesis still names a container; the text was truncated, not unstructured.
    close_group()

    return list(dict.fromkeys(places))


GADM_UNITS_COLUMN = "GADM Admin Units"


class UncodedEvent(NamedTuple):
    """An event EM-DAT recorded a location for without coding any administrative unit.

    Parameters
    ----------
    disno : str
        The EM-DAT event identifier.
    iso : str
        ISO 3166-1 alpha-3 code of the country the event is filed under.
    places : list of NamedPlace
        The places its location text names, never empty.
    """

    disno: str
    iso: str
    places: list[NamedPlace]


def events_missing_units(events: pl.DataFrame) -> list[UncodedEvent]:
    """
    Collect the events whose geography has to be recovered from their location text.

    Two thirds of the workbook carries no administrative units, and an event whose text names
    nothing is left out: there is nothing to resolve and no reason to fetch a gazetteer for it.

    Parameters
    ----------
    events : DataFrame
        The workbook, from :func:`load_emdat_events`.

    Returns
    -------
    uncoded : list of UncodedEvent
        One entry per event with places to place, in workbook order.
    """
    uncoded = []
    for disno, iso, location, units in zip(
        events["DisNo."], events["ISO"], events["Location"], events[GADM_UNITS_COLUMN], strict=True
    ):
        if str(units or "").strip() and json.loads(units):
            continue
        if places := named_places(location):
            uncoded.append(UncodedEvent(str(disno), str(iso), places))

    return uncoded


# One row per affected administrative unit. `name` and `migration_method` are absent on some units:
# EM-DAT omits the name where GADM has none, and the method where the unit was never migrated.
EVENT_UNIT_SCHEMA = {
    "DisNo.": pl.String,
    "gid": pl.String,
    "name": pl.String,
    "admin_level": pl.Int8,
    "migration_method": pl.String,
}


def event_units(events: pl.DataFrame) -> pl.DataFrame:
    """
    Explode each event's GADM administrative units into one row per event-unit.

    The level comes from the key EM-DAT files a unit under, ``gid_1`` or ``gid_2``, because a GADM
    identifier does not reliably state it: Ghana's are numbered ``GHA11_2`` and ``GHA7.13_2``.

    Parameters
    ----------
    events : DataFrame
        The workbook, as :func:`load_emdat_events` returns it.

    Returns
    -------
    units : DataFrame
        ``DisNo.``, ``gid``, ``name``, ``admin_level`` and ``migration_method``. An event carrying no
        units contributes no rows, so this is narrower than ``events``.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data_functions.emdat_processing import event_units

        units = event_units(events)
    """
    rows = []

    for disno, raw in zip(events["DisNo."], events[GADM_UNITS_COLUMN], strict=True):
        if not str(raw or "").strip():
            continue

        for unit in json.loads(raw):
            level = 2 if "gid_2" in unit else 1
            if f"gid_{level}" not in unit:
                raise ValueError(f"{disno}: an administrative unit carries no gid_1 or gid_2: {unit}")

            rows.append(
                {
                    "DisNo.": disno,
                    "gid": unit[f"gid_{level}"],
                    "name": unit.get(f"name_{level}"),
                    "admin_level": level,
                    "migration_method": unit.get("migration_method"),
                }
            )

    return pl.DataFrame(rows, schema=EVENT_UNIT_SCHEMA)


# Where an event's geometry comes from, best first. Nothing is filtered on this: the column records
# what the source holds, and a model chooses which tiers it will accept.
GEOMETRY_SOURCES = ("gadm", "geo_disasters", "location_text", "emdat_point", "country")

EVENT_GEOGRAPHY_COLUMNS = (
    "DisNo.",
    "ISO",
    "geometry_source",
    "gid",
    "name",
    "admin_level",
    "migration_method",
    "geocoding_q",
    "overlap",
    "names_written",
    "names_reached",
    "Latitude",
    "Longitude",
)

# What the `geo_disasters` tier adds, and what a caller has to supply for it to be emitted. Null on
# every other tier, as `migration_method` already is outside `gadm`.
RESOLVED_UNIT_SCHEMA = pl.Schema(
    {
        "DisNo.": pl.String,
        "gid": pl.String,
        "name": pl.String,
        "admin_level": pl.Int8,
        "geocoding_q": pl.Int8,
        "overlap": pl.Float64,
    }
)

# The units an event's own prose reaches, resolved elsewhere: this layer holds no gazetteer.
TEXT_UNIT_SCHEMA = pl.Schema({"DisNo.": pl.String, "gid": pl.String, "name": pl.String, "admin_level": pl.Int8})

# How complete that reading was, per event rather than per unit, so an event whose prose reached
# nothing still records that it wrote names. Null where the event carries no parseable text.
TEXT_RESOLUTION_SCHEMA = pl.Schema({"DisNo.": pl.String, "names_written": pl.Int16, "names_reached": pl.Int16})

# What a tier assembles on its own. The rest of the table is joined on afterwards, per event
# rather than per unit.
_TIER_COLUMNS = tuple(
    column for column in EVENT_GEOGRAPHY_COLUMNS if column == "DisNo." or column not in TEXT_RESOLUTION_SCHEMA
)

# The columns a tier need not carry, and the type each takes where it does not. A tier states only
# what it knows, so a column added to the table is declared here once instead of in every tier.
_ABSENT_COLUMN_TYPES = pl.Schema(
    {
        "gid": pl.String,
        "name": pl.String,
        "admin_level": pl.Int8,
        "migration_method": pl.String,
        "geocoding_q": pl.Int8,
        "overlap": pl.Float64,
    }
)


def _as_frame(given: pl.DataFrame | None, schema: pl.Schema) -> pl.DataFrame:
    """Take the caller's frame down to the columns and types the schema names, or an empty one."""
    return pl.DataFrame(schema=schema) if given is None else given.select(list(schema)).cast(schema)


def _one_row_per_event(counts: pl.DataFrame) -> pl.DataFrame:
    """Refuse a per-event lookup that names an event twice."""
    # Attached by a join on the event alone, so a repeated event fans out every geography row it
    # has, including rows of tiers the reading had nothing to do with.
    repeated = sorted(counts.filter(pl.col("DisNo.").is_duplicated())["DisNo."].unique().to_list())
    if repeated:
        raise DataValidationError(f"The resolution counts name an event more than once: {repeated[:5]}.")

    return counts


def _nesting_path(gid: pl.Expr) -> pl.Expr:
    """Strip a GADM identifier's version suffix, leaving the path it nests by: `LAO.1.2_1` -> `LAO.1.2`."""
    return gid.str.split("_").list.first()


def _naming_ground_the_units_do_not(overlay: pl.DataFrame, units: pl.DataFrame) -> pl.DataFrame:
    """
    Keep the overlay rows for ground an event's own units leave unnamed.

    GADM identifiers nest by dotted prefix, so an overlay unit repeats ground already named where
    the event carries that unit, something containing it, or something inside it. What survives is
    area no unit of the event covers, which an event with no units of its own is all of.

    Parameters
    ----------
    overlay : DataFrame
        Units another gazetteer places, with the columns of ``RESOLVED_UNIT_SCHEMA``.
    units : DataFrame
        The event's own units, as :func:`event_units` returns them.

    Returns
    -------
    extra : DataFrame
        The rows of ``overlay`` naming ground ``units`` does not.
    """
    already_named = (
        overlay.select("DisNo.", "gid", offered=_nesting_path(pl.col("gid")))
        .join(units.select("DisNo.", carried=_nesting_path(pl.col("gid"))), on="DisNo.", how="inner")
        .filter(
            (pl.col("offered") == pl.col("carried"))
            | pl.col("offered").str.starts_with(pl.col("carried") + ".")
            | pl.col("carried").str.starts_with(pl.col("offered") + ".")
        )
        .select("DisNo.", "gid")
        .unique()
    )

    return overlay.join(already_named, on=["DisNo.", "gid"], how="anti")


def _with_absent_columns(tier: pl.DataFrame) -> pl.DataFrame:
    """Fill every geography column the tier does not carry with a null of the right type."""
    return tier.with_columns(
        [
            pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in _ABSENT_COLUMN_TYPES.items()
            if name not in tier.columns
        ]
    )


def event_geography(
    events: pl.DataFrame,
    *,
    resolved: pl.DataFrame | None = None,
    from_text: pl.DataFrame | None = None,
    text_resolution: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Say where every event's geometry comes from, one row per event-unit and one per event otherwise.

    An event coded to administrative units contributes a row per unit. One with only a coordinate,
    or with nothing, contributes a single row. Every event in ``events`` appears, so an absence of
    geography is stated rather than implied by a missing row.

    ``Latitude`` and ``Longitude`` are carried wherever EM-DAT supplies them, including on ``gadm``
    rows, so the two claims stay visible side by side rather than one being discarded.

    Sources are taken in the order ``GEOMETRY_SOURCES`` declares. An event keeps every unit EM-DAT
    codes it to, and each later source contributes only units naming ground the ones before it leave
    uncovered. GADM ids nest, so a unit an earlier source already names is the same ground at a
    second resolution. Taking it again would put a province and the districts inside it into one
    observation.

    ``names_written`` and ``names_reached`` say how much of an event's prose resolved, on every row
    of an event whose text was read. They describe the prose rather than the tier, so an event whose
    names all failed carries them at ``country`` tier, which is where a partial reading is least
    visible and most worth recording.

    Parameters
    ----------
    events : DataFrame
        The workbook, as :func:`load_emdat_events` returns it.
    resolved : DataFrame, optional
        Units another gazetteer places, with the columns of ``RESOLVED_UNIT_SCHEMA``. Default None,
        which emits no ``geo_disasters`` rows. Reading them is the caller's job: they come from a
        polygon overlay, and this layer holds no geometry.
    from_text : DataFrame, optional
        Units an event's own location text reaches, with the columns of ``TEXT_UNIT_SCHEMA``.
        Default None, which emits no ``location_text`` rows. Resolving them needs a gazetteer, which
        this layer does not hold either.
    text_resolution : DataFrame, optional
        How many placeable names each event wrote and how many reached a unit, with the columns of
        ``TEXT_RESOLUTION_SCHEMA``. Default None, which leaves both columns null.

    Returns
    -------
    geography : DataFrame
        ``DisNo.``, ``ISO``, ``geometry_source`` from ``GEOMETRY_SOURCES``, the unit columns of
        :func:`event_units` where one applies, ``geocoding_q`` and ``overlap`` on the
        ``geo_disasters`` tier, ``names_written`` and ``names_reached`` wherever the event's text was
        read, and the event's coordinate where it has one.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data_functions.emdat_processing import event_geography

        geography = event_geography(events)
    """
    units = event_units(events)
    located = events.select("DisNo.", "ISO", "Latitude", "Longitude")
    overlay = _as_frame(resolved, RESOLVED_UNIT_SCHEMA)
    prose = _as_frame(from_text, TEXT_UNIT_SCHEMA)

    from_units = units.join(located, on="DisNo.", how="left").with_columns(pl.lit("gadm").alias("geometry_source"))

    unnamed = _naming_ground_the_units_do_not(overlay, units)
    from_overlay = located.join(unnamed, on="DisNo.", how="inner").with_columns(
        pl.lit("geo_disasters").alias("geometry_source")
    )

    # Each source answers only for ground the ones before it left unnamed, so the order in
    # `GEOMETRY_SOURCES` decides who wins a repeat rather than who is read last.
    already = pl.concat([units.select("DisNo.", "gid"), unnamed.select("DisNo.", "gid")])
    from_prose = located.join(_naming_ground_the_units_do_not(prose, already), on="DisNo.", how="inner").with_columns(
        pl.lit("location_text").alias("geometry_source")
    )

    placed = pl.concat([tier.select("DisNo.") for tier in (units, unnamed, from_prose)]).unique()
    rest = located.join(placed, on="DisNo.", how="anti").with_columns(
        pl.when(pl.col("Latitude").is_not_null())
        .then(pl.lit("emdat_point"))
        .otherwise(pl.lit("country"))
        .alias("geometry_source")
    )

    tiers = (from_units, from_overlay, from_prose, rest)
    every = pl.concat([_with_absent_columns(tier).select(_TIER_COLUMNS) for tier in tiers])

    return (
        every.join(_one_row_per_event(_as_frame(text_resolution, TEXT_RESOLUTION_SCHEMA)), on="DisNo.", how="left")
        .select(EVENT_GEOGRAPHY_COLUMNS)
        .sort("DisNo.", "gid", nulls_last=True)
    )


def event_filter(filters: EventFilters) -> pl.Expr:
    """
    Turn a place's thresholds into a predicate for :meth:`polars.DataFrame.filter`.

    Combine it with ``&`` to narrow further, such as to one country.

    Parameters
    ----------
    filters : EventFilters
        The window and severity thresholds an event has to clear.

    Returns
    -------
    predicate : Expr
        True for events that count. An event whose severity is unrecorded clears every threshold a
        place leaves unset, and fails any it sets.

    Examples
    --------
    Keep the events one place's configuration counts:

    .. code-block:: python

        from climate_risk.config.registry import load_place
        from climate_risk.data_functions.emdat_processing import event_filter

        laos = load_place("lao")
        counted = events.filter(event_filter(laos.event_filters))
    """
    counts = pl.col("Start_Year") >= pl.date(filters.start_year, 1, 1)

    if filters.end_year is not None:
        counts = counts & (pl.col("Start_Year") <= pl.date(filters.end_year, 12, 31))

    if filters.min_total_affected is not None:
        counts = counts & (pl.col("Total_Affected") > filters.min_total_affected)

    if filters.min_deaths is not None:
        counts = counts & (pl.col("Deaths") > filters.min_deaths)

    return counts
