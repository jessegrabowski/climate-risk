import logging

from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd

from climate_risk.config.registry import resolve_isos
from climate_risk.config.schema import CountryConfig, Place
from climate_risk.data.fetch import fetch
from climate_risk.data.source import DataSource, ShapefileArchive
from climate_risk.exceptions import DataValidationError, ISOCodeValidationError

_log = logging.getLogger(__name__)

# The catalog's metadata API is rate-limited; its file host is not, and is where the published
# artifact actually lives.
WORLD = DataSource(
    url=("https://datacatalogfiles.worldbank.org/ddh-published/0038272/DR0046659/wb_countries_admin0_10m.zip"),
    filename="wb_countries_admin0_10m.zip",
    license="CC BY 4.0",
    citation=(
        "World Bank Official Boundaries, dataset 0038272 in the World Bank Data Catalog. "
        "https://datacatalog.worldbank.org/search/dataset/0038272"
    ),
    retrieved="2026-08-08",
)

COASTLINE = DataSource(
    url="https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip",
    filename="gshhg-shp-2.3.7.zip",
    license="LGPL",
    citation=(
        "Wessel, P., and Smith, W. H. F. (1996): A global, self-consistent, hierarchical, "
        "high-resolution shoreline database. Journal of Geophysical Research, 101(B4), 8741-8743. "
        "https://doi.org/10.1029/96JB00104"
    ),
    retrieved="2026-08-08",
)

SHAPEFILE_ARCHIVES = {
    "world": ShapefileArchive(WORLD, "WB_countries_Admin0_10m"),
    # The archive holds six layers and the driver does not enumerate them in order, so the
    # continental shoreline is named outright. L2-L6 are lakes, islands and Antarctic ice.
    "coastline": ShapefileArchive(COASTLINE, "GSHHS_shp/f/GSHHS_f_L1.shp"),
}

VALID_CHOICES = list(SHAPEFILE_ARCHIVES)

# The boundary file lists these separately but tags them with their owner's ISO code, or with no
# code at all, so counting them would double-count the owner or introduce a country that is not one.
DROPPED_TERRITORIES = (
    # Leased or far-flung dependencies carrying no code of their own.
    "Guantanamo Bay (US)",
    "Clipperton Island (Fr.)",
    "Cocos (Keeling) Islands (Aus.)",
    "Christmas Island (Aus.)",
    # Caribbean municipalities labeled as the Netherlands.
    "Bonaire (Neth.)",
    "Sint Eustatius (Neth.)",
    "Saba (Neth.)",
    # Labeled as New Zealand.
    "Tokelau (NZ)",
)

# The US minor outlying islands share one code, so they are matched by it rather than by name.
DROPPED_ISO_CODES = ("UMI",)

# Sovereign countries the boundary file leaves at -99. XKX is the World Bank's own code for Kosovo,
# which has none under ISO 3166-1, and is what the World Bank country table uses.
ISO_CODE_REPAIRS = {
    "France": "FRA",
    "Norway": "NOR",
    "Kosovo": "XKX",
}


def shapefile_dir(cache_dir: Path) -> Path:
    """Return the directory the downloaded shapefile archives live in, inside ``cache_dir``."""
    return cache_dir / "shapefiles"


def _archive_for(which: str) -> ShapefileArchive:
    try:
        return SHAPEFILE_ARCHIVES[which.lower()]
    except KeyError:
        raise ValueError(f"which should be one of {VALID_CHOICES}, got {which}") from None


def repair_iso_codes(world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop territories that would double-count their owner and supply the missing ISO codes.

    Parameters
    ----------
    world : GeoDataFrame
        The world boundary file, carrying ``WB_NAME`` and ``ISO_A3`` columns.

    Returns
    -------
    repaired : GeoDataFrame
        The repaired frame, reindexed from zero.
    """
    missing_columns = {"WB_NAME", "ISO_A3"} - set(world.columns)
    if missing_columns:
        raise DataValidationError(f"The boundary file is missing {sorted(missing_columns)}, so it cannot be repaired.")

    unmatched = sorted({*DROPPED_TERRITORIES, *ISO_CODE_REPAIRS} - set(world["WB_NAME"]))
    unmatched += sorted(set(DROPPED_ISO_CODES) - set(world["ISO_A3"]))
    if unmatched:
        raise DataValidationError(
            f"No row matches {unmatched}. The boundary file has changed, and applying the remaining "
            f"repairs would drop or relabel the wrong countries."
        )

    repaired = world[~world["WB_NAME"].isin(DROPPED_TERRITORIES) & ~world["ISO_A3"].isin(DROPPED_ISO_CODES)].copy()
    repaired["ISO_A3"] = repaired["WB_NAME"].map(ISO_CODE_REPAIRS).fillna(repaired["ISO_A3"])
    repaired = repaired.reset_index(drop=True)

    geometries_per_code = repaired["ISO_A3"].value_counts()
    duplicated = geometries_per_code[geometries_per_code > 1]
    if not duplicated.empty:
        raise ISOCodeValidationError(f"These ISO codes label more than one geometry: {dict(duplicated)}")

    return repaired


def download_shapefile(which: str, cache_dir: Path, *, force_reload: bool = False) -> Path:
    """Fetch the archive for ``which`` into the shapefile cache and return where it landed."""
    return fetch(_archive_for(which).source, shapefile_dir(cache_dir), force=force_reload)


def extract_shapefiles(which: str, cache_dir: Path, *, force_reload: bool = False) -> None:
    """Unpack the archive for ``which`` unless what it holds is already on disk."""
    _extract(_archive_for(which), shapefile_dir(cache_dir), force_reload=force_reload)


def _extract(archive: ShapefileArchive, directory: Path, *, force_reload: bool) -> None:
    if archive.extracted_path(directory).exists() and not force_reload:
        return

    _log.info(f"Extracting {archive.source.filename}")
    with ZipFile(archive.source.path(directory)) as zipped:
        zipped.extractall(path=directory)


def load_archive(archive: ShapefileArchive, cache_dir: Path, *, force_reload: bool = False) -> gpd.GeoDataFrame:
    """Fetch, unpack and read an archive, whether or not a registry entry names it."""
    directory = shapefile_dir(cache_dir)

    fetch(archive.source, directory, force=force_reload)
    _extract(archive, directory, force_reload=force_reload)

    return gpd.read_file(archive.extracted_path(directory), layer=0)


def load_shapefile(
    which: str, cache_dir: Path, *, force_reload: bool = False, repair_ISO_codes: bool = True
) -> gpd.GeoDataFrame:
    """
    Load one of the bundled boundary archives.

    Parameters
    ----------
    which : str
        Which archive to read: ``"world"`` for the World Bank country boundaries, ``"coastline"``
        for the GSHHG continental shoreline.
    cache_dir : Path
        Directory the source caches live under.
    force_reload : bool, optional
        Download again and re-extract rather than reading the cache. Default False.
    repair_ISO_codes : bool, optional
        Drop territories that would double-count their owner and supply the missing ISO codes.
        Applies to ``"world"`` only. Default True.

    Returns
    -------
    boundaries : GeoDataFrame
        The archive's geometries and attributes.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_shapefile

        world = load_shapefile("world", Path("data"))
        coastline = load_shapefile("coastline", Path("data"))
    """
    frame = load_archive(_archive_for(which), cache_dir, force_reload=force_reload)

    if which.lower() == "world" and repair_ISO_codes:
        return repair_iso_codes(frame)

    return frame


def load_place_boundary(place: Place, cache_dir: Path, *, force_reload: bool = False) -> gpd.GeoDataFrame:
    """
    Read the geometry a place covers.

    A place carrying its own boundary archive reads that. Everything else is sliced out of the world
    shapefile by the ISO codes the place resolves to.

    Parameters
    ----------
    place : CountryConfig or RegionConfig
        The place to read the geometry of.
    cache_dir : Path
        Directory the shapefile cache lives under.
    force_reload : bool, optional
        Re-download and re-unpack even when the cache is warm. Default False.

    Returns
    -------
    boundary : GeoDataFrame
        The place's geometry, in the boundary file's own CRS.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.config.registry import load_place
        from climate_risk.data_functions.shapefiles_data_loader import load_place_boundary

        boundary = load_place_boundary(load_place("lao"), Path("data"))
    """
    if isinstance(place, CountryConfig) and place.boundary is not None:
        return load_archive(place.boundary, cache_dir, force_reload=force_reload)

    codes = resolve_isos(place)
    world = load_shapefile("world", cache_dir, force_reload=force_reload)
    boundary = world[world["ISO_A3"].isin(codes)]

    if boundary.empty:
        raise DataValidationError(f"The boundary file holds no geometry for {list(codes)}, so the grid would be empty.")

    return boundary
