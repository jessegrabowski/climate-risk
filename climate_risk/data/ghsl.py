from pathlib import Path
from zipfile import ZipFile

import numpy as np

from climate_risk.data.fetch import fetch
from climate_risk.data.source import DataSource
from climate_risk.exceptions import DataValidationError
from climate_risk.geo.raster import CellGrid, sample_onto_cells

GHSL_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
GHSL_SUBDIRECTORY = "ghsl"
GHSL_RELEASE = "R2023A"
GHSL_LICENSE = "European Commission reuse notice: reuse authorised provided the source is acknowledged"
GHSL_CITATION = (
    "Schiavina, M., Freire, S., Carioli, A., MacManus, K. (2026): GHS-POP R2023A - GHS population "
    "grid multitemporal (1975-2030). European Commission, Joint Research Center. "
    "doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE"
)
GHSL_RETRIEVED = "2026-08-23"

# Epochs the release publishes. Five-yearly, so the panel's 1981 opening is covered from below.
POPULATION_EPOCHS = tuple(range(1975, 2031, 5))

# Arc-seconds per cell in the WGS84 grids. 30 is roughly a kilometer, ample under a 5 km analysis
# cell; the 3 arc-second grid is a hundred times the bytes for a mean that cannot move.
RESOLUTION = "30ss"


def ghsl_dir(cache_dir: Path) -> Path:
    """Return the directory the GHS-POP rasters live in, inside ``cache_dir``."""
    return cache_dir / GHSL_SUBDIRECTORY


def population_source(epoch: int) -> DataSource:
    """
    Declare the population raster for one epoch.

    Parameters
    ----------
    epoch : int
        Year the grid describes, one of ``POPULATION_EPOCHS``.

    Returns
    -------
    source : DataSource
        The zipped GeoTIFF as GHSL publishes it.
    """
    if epoch not in POPULATION_EPOCHS:
        raise DataValidationError(f"GHS-POP publishes {POPULATION_EPOCHS}, not {epoch}")

    stem = f"GHS_POP_E{epoch}_GLOBE_{GHSL_RELEASE}_4326_{RESOLUTION}"

    return DataSource(
        url=f"{GHSL_URL}/GHS_POP_GLOBE_{GHSL_RELEASE}/{stem}/V1-0/{stem}_V1_0.zip",
        filename=f"{stem}_V1_0.zip",
        license=GHSL_LICENSE,
        citation=GHSL_CITATION,
        retrieved=GHSL_RETRIEVED,
    )


def population_raster(epoch: int, cache_dir: Path) -> str:
    """
    Name the population raster for one epoch, fetching the archive if it is not there.

    The GeoTIFF is read where it lies, through GDAL's virtual filesystem, so the archive is the only
    copy on disk. The member is found by extension because GHSL ships metadata beside the raster.

    Parameters
    ----------
    epoch : int
        Year the grid describes, one of ``POPULATION_EPOCHS``.
    cache_dir : Path
        Directory the caches live under.

    Returns
    -------
    uri : str
        A URI :func:`rasterio.open` accepts, addressing the GeoTIFF inside its archive. The band
        holds people per cell.
    """
    source = population_source(epoch)
    archive = fetch(source, ghsl_dir(cache_dir))

    with ZipFile(archive) as unpacked:
        members = [name for name in unpacked.namelist() if name.lower().endswith(".tif")]

    if len(members) != 1:
        raise DataValidationError(f"{source.filename} holds {len(members)} rasters, expected exactly one")

    return f"zip://{archive}!/{members[0]}"


def population_on_cells(grid: CellGrid, epoch: int, cache_dir: Path) -> np.ndarray:
    """
    Count the people in each grid cell.

    Parameters
    ----------
    grid : CellGrid
        The lattice to count onto. Values come back in the order of ``grid.cells``.
    epoch : int
        Year the grid describes, one of ``POPULATION_EPOCHS``.
    cache_dir : Path
        Directory the caches live under.

    Returns
    -------
    population : ndarray
        People per cell, aligned with ``grid.cells``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.ghsl import population_on_cells

        people = population_on_cells(grid, 2020, Path("data"))
    """
    return sample_onto_cells(grid, population_raster(epoch, cache_dir), statistic="sum")
