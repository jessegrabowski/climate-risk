import sqlite3

from collections.abc import Collection, Iterator
from pathlib import Path

import geopandas as gpd
import pandas as pd

from climate_risk.data.cache import builder_fingerprint, cached, geo_parquet
from climate_risk.data.source import ManualSource
from climate_risk.exceptions import DataValidationError

# GADM's terms exclude redistribution and automated download, so it is declared as a manual source
# and never appears in the fetchable registry.
GADM = ManualSource(
    filename="gadm_410.gpkg",
    homepage="https://gadm.org/download_world.html",
    license=(
        "Academic use and other non-commercial use only. Redistribution, and use as part of a "
        "commercial product or service, require prior written permission."
    ),
    citation="GADM, Database of Global Administrative Areas, version 4.1. https://gadm.org",
    retrieved="2026-08-09",
)

# One table holds every country at its finest level, so a unit above that is the union of its rows.
GADM_LAYER = "gadm_410"

GID_COLUMNS = {level: (f"GID_{level}", f"NAME_{level}") for level in (1, 2, 3, 4)}

# A WHERE clause naming every id at once grows past what the driver will parse, and costs more to
# plan than the extra passes cost to run.
GID_CHUNK = 400


def gadm_dir(cache_dir: Path) -> Path:
    """Return the directory the GADM GeoPackage lives in, inside ``cache_dir``."""
    return cache_dir / "gadm"


def gadm_path(cache_dir: Path) -> Path:
    """
    Return the path to the GADM GeoPackage, raising if it has not been placed in the cache.

    Parameters
    ----------
    cache_dir : Path
        Directory the caches live under.

    Returns
    -------
    path : Path
        Location of ``gadm_410.gpkg``.
    """
    return GADM.require(gadm_dir(cache_dir))


def _chunks(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def administered_territories(iso: str, cache_dir: Path, *, layer: str = GADM_LAYER) -> tuple[str, ...]:
    """
    Name the disputed territories GADM files apart from the country administering them.

    Kashmir is not filed under ``IND`` or ``PAK`` but under codes of its own, so an event named
    there belongs to a country whose own code does not contain it. GADM records the administering
    country on each territory, and that is what this reads.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to read.
    cache_dir : Path
        Directory the caches live under.
    layer : str, optional
        Layer to read inside the GeoPackage. Default ``GADM_LAYER``.

    Returns
    -------
    territories : tuple of str
        The territory codes, empty for a country administering none.
    """
    with sqlite3.connect(gadm_path(cache_dir)) as connection:
        named = connection.execute(f'SELECT COUNTRY FROM "{layer}" WHERE GID_0 = ? LIMIT 1', (iso,)).fetchone()
        if named is None:
            return ()
        held = connection.execute(
            f'SELECT DISTINCT GID_0 FROM "{layer}" WHERE COUNTRY = ? AND GID_0 <> ? ORDER BY GID_0',
            (named[0], iso),
        )

        return tuple(row[0] for row in held)


def load_units_in_country(
    iso: str, level: int, cache_dir: Path, *, layer: str = GADM_LAYER, force_reload: bool = False
) -> gpd.GeoDataFrame:
    """
    Read every GADM unit one country holds at one administrative level.

    Where :func:`load_admin_units` answers "which polygon is this id", this answers "what units are
    there", which is what a search by geometry needs.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to read.
    level : int
        Administrative level, 1 through 4.
    cache_dir : Path
        Directory the caches live under.
    layer : str, optional
        Layer to read inside the GeoPackage. Default ``GADM_LAYER``.
    force_reload : bool, optional
        Rebuild even when the units are already cached. Default False.

    Returns
    -------
    units : GeoDataFrame
        Columns ``gid``, ``name``, ``admin_level`` and ``geometry``, one row per unit. Empty where
        the country is not divided at that level.

    Examples
    --------
    Read every district in Laos:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.gadm import load_units_in_country

        districts = load_units_in_country("LAO", 2, Path("data"))
    """
    if level not in GID_COLUMNS:
        raise DataValidationError(f"Units are read at levels {sorted(GID_COLUMNS)}, not {level}")

    def build() -> gpd.GeoDataFrame:
        gid_column, name_column = GID_COLUMNS[level]
        rows = gpd.read_file(
            gadm_path(cache_dir), layer=layer, columns=[gid_column, name_column], where=f"GID_0 = '{iso}'"
        )
        # A country not divided at this level stores a blank identifier on every row.
        rows = rows[rows[gid_column].astype(str) != ""]
        if rows.empty:
            return _no_units(gadm_path(cache_dir), layer)

        # The table stores the finest level, so a unit above it spans several rows.
        units = rows.dissolve(by=gid_column, aggfunc="first", as_index=False)

        return units.rename(columns={gid_column: "gid", name_column: "name"}).assign(admin_level=level)[
            ["gid", "name", "admin_level", "geometry"]
        ]

    return cached(
        gadm_dir(cache_dir),
        "units",
        build,
        geo_parquet(),
        # `layer` chooses what is read and the fingerprint covers how, neither of which the other
        # parameters record.
        params={"iso": iso, "layer": layer, "level": level, "reading": builder_fingerprint(build, GID_COLUMNS)},
        force=force_reload,
    )


def _units_at_level(gids: list[str], level: int, path: Path, layer: str) -> gpd.GeoDataFrame | None:
    gid_column, name_column = GID_COLUMNS[level]
    read = []

    for chunk in _chunks(gids, GID_CHUNK):
        quoted = ",".join(f"'{gid}'" for gid in chunk)
        rows = gpd.read_file(path, layer=layer, columns=[gid_column, name_column], where=f"{gid_column} IN ({quoted})")
        if not rows.empty:
            # The table stores the finest level, so a unit above it spans several rows.
            read.append(rows.dissolve(by=gid_column, aggfunc="first", as_index=False))

    if not read:
        return None

    found = pd.concat(read, ignore_index=True) if len(read) > 1 else read[0]

    return found.rename(columns={gid_column: "gid", name_column: "name"}).assign(admin_level=level)


def _no_units(path: Path, layer: str) -> gpd.GeoDataFrame:
    """
    An empty result carrying the layer's CRS, read without any of its rows.

    A CRS-less frame concatenates with projected geometry without complaint and reprojects to the
    wrong place, so the empty case has to match the populated one.
    """
    return gpd.GeoDataFrame(
        {"gid": [], "name": [], "admin_level": pd.Series([], dtype="int64")},
        geometry=[],
        crs=gpd.read_file(path, layer=layer, where="0=1").crs,
    )


def load_admin_units(
    units: Collection[tuple[str, int]], cache_dir: Path, *, layer: str = GADM_LAYER
) -> gpd.GeoDataFrame:
    """
    Read one polygon per GADM unit named.

    The level comes from the caller because a GADM id does not reliably state it: most countries are
    numbered ``LAO.11_1`` and ``LAO.11.3_1``, but Ghana's provinces are ``GHA11_2`` and its districts
    ``GHA7.13_2``. EM-DAT records the level in the key it files each unit under.

    Parameters
    ----------
    units : collection of tuple of (str, int)
        Each a GADM identifier and its administrative level, 1 through 4. Duplicates are read once.
    cache_dir : Path
        Directory the caches live under.
    layer : str, optional
        Layer to read inside the GeoPackage. Default ``GADM_LAYER``.

    Returns
    -------
    admin_units : GeoDataFrame
        Columns ``gid``, ``name``, ``admin_level`` and ``geometry``, one row per distinct id.

    Examples
    --------
    Read named units, each identified by its country and level:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.gadm import load_admin_units

        units = load_admin_units([("LAO", 1), ("THA", 1)], Path("data"))
    """
    path = gadm_path(cache_dir)
    requested = set(units)

    by_level: dict[int, list[str]] = {}
    for gid, level in sorted(requested):
        by_level.setdefault(level, []).append(gid)

    unsupported = sorted(set(by_level) - set(GID_COLUMNS))
    if unsupported:
        raise DataValidationError(
            f"Units are read at levels {sorted(GID_COLUMNS)}, but these were asked for at "
            f"{unsupported}: {[gid for level in unsupported for gid in by_level[level][:3]]}"
        )

    frames = [
        frame
        for level, gids_at_level in by_level.items()
        if (frame := _units_at_level(gids_at_level, level, path, layer)) is not None
    ]
    resolved = (
        gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs) if frames else _no_units(path, layer)
    )

    missing = sorted({gid for gid, _ in requested} - set(resolved["gid"]))
    if missing:
        raise DataValidationError(
            f"GADM holds no unit for {len(missing)} of {len(requested)} ids, starting {missing[:5]}. "
            f"The GeoPackage version may not match the one the ids were coded against."
        )

    return resolved[["gid", "name", "admin_level", "geometry"]]
