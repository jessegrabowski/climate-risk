from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource, RasterioRasterSource
from pyproj import Geod
from shapely.geometry import box

from climate_risk.exceptions import DataValidationError
from climate_risk.geo.crs import GEOGRAPHIC_CRS

ISO_COLUMN = "ISO_A3"

# Spherical, and only ever used to turn a target resolution into a cell count. Cell areas, which
# are weights the model integrates with, are measured on the ellipsoid instead.
KM_PER_DEGREE_LATITUDE = 111.32

WGS84_ELLIPSOID = "WGS84"
SQUARE_METRES_PER_SQUARE_KM = 1e6

# One cell is a crude but well-defined quadrature over an extent smaller than the resolution. A
# place that small is screened out by the place configuration rather than here.
MIN_CELLS_PER_AXIS = 1


def dissolve_place_boundary(boundary: gpd.GeoDataFrame, *, iso3: str | None = None) -> gpd.GeoDataFrame:
    """
    Reduce a place's boundary to one polygon per country.

    A place either carries its own archive, which may hold many administrative features and no ISO
    column, or resolves to a slice of the world shapefile, which carries ``ISO_A3`` and one feature
    per country. Both grid to the same cells only once they have the same shape, so both are
    reduced here rather than at each call site.

    Parameters
    ----------
    boundary : GeoDataFrame
        The place's geometry, in any CRS.
    iso3 : str, optional
        Code to label the geometry with when it carries no ``ISO_A3`` column. A geometry that labels
        itself keeps its own codes, so passing one that disagrees is refused rather than ignored.
        Default None, which requires the geometry to label itself.

    Returns
    -------
    dissolved : GeoDataFrame
        One row per country, carrying ``ISO_A3`` and ``geometry``, in ``GEOGRAPHIC_CRS``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_shapefile
        from climate_risk.geo.raster import dissolve_place_boundary

        world = load_shapefile("world", Path("data"))
        laos = dissolve_place_boundary(world.query('ISO_A3 == "LAO"'))
    """
    if boundary.empty:
        raise DataValidationError("The boundary holds no geometry, so there would be nothing to grid.")

    if boundary.crs is None:
        raise DataValidationError("The boundary carries no CRS, so its coordinates cannot be placed on the globe.")

    labeled = ISO_COLUMN in boundary.columns
    if not labeled and iso3 is None:
        raise DataValidationError(
            f"The geometry carries no {ISO_COLUMN} column, so pass iso3 to say which country it covers."
        )

    if labeled and iso3 is not None and set(boundary[ISO_COLUMN]) != {iso3}:
        carried = sorted(set(boundary[ISO_COLUMN]))
        raise DataValidationError(
            f"The boundary carries {len(carried)} code(s), {carried[:3]}, so iso3={iso3!r} would not "
            f"label it and every country in it would be dissolved. Select the country first."
        )

    located = boundary.to_crs(GEOGRAPHIC_CRS)
    if not labeled:
        located = located.assign(**{ISO_COLUMN: iso3})

    dissolved = located.dissolve(by=ISO_COLUMN, as_index=False)

    return gpd.GeoDataFrame(dissolved[[ISO_COLUMN, "geometry"]], geometry="geometry", crs=GEOGRAPHIC_CRS)


def grid_axes(bounds: tuple[float, float, float, float], resolution_km: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cell center coordinates for a grid of approximately square cells tiling an extent.

    Resolution is stated in kilometers rather than points per axis, because points per axis means a
    different ground distance for every place and a different one along each axis of the same
    place. Longitude is converted at the extent's mean latitude, so cells are square in the middle
    of the place and drift from square towards its edges. The cell areas carry that drift rather
    than the spacing.

    Centers, not edges: the aggregation operator sums a field sampled once per cell, which is a
    midpoint rule, and a midpoint rule wants the sample in the middle of the ground it stands for.

    Parameters
    ----------
    bounds : tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` in degrees.
    resolution_km : float
        Target cell edge, in kilometers.

    Returns
    -------
    longitudes : ndarray
        Cell center longitudes, one per cell along the axis.
    latitudes : ndarray
        Cell center latitudes, one per cell along the axis.
    """
    if resolution_km <= 0.0:
        raise DataValidationError(f"The grid resolution must be a positive distance, not {resolution_km}.")

    lon_min, lat_min, lon_max, lat_max = bounds
    if lon_max < lon_min or lat_max < lat_min:
        raise DataValidationError(f"The bounds {bounds} are inverted, so they enclose nothing.")

    mean_latitude = (lat_min + lat_max) / 2.0
    km_per_degree_longitude = KM_PER_DEGREE_LATITUDE * np.cos(np.radians(mean_latitude))

    spans_km = (
        (lon_max - lon_min) * km_per_degree_longitude,
        (lat_max - lat_min) * KM_PER_DEGREE_LATITUDE,
    )
    counts = [max(MIN_CELLS_PER_AXIS, round(span / resolution_km)) for span in spans_km]

    longitude_edges = np.linspace(lon_min, lon_max, counts[0] + 1)
    latitude_edges = np.linspace(lat_min, lat_max, counts[1] + 1)

    return _midpoints(longitude_edges), _midpoints(latitude_edges)


def _midpoints(edges: np.ndarray) -> np.ndarray:
    centers: np.ndarray = (edges[:-1] + edges[1:]) / 2.0

    return centers


def cell_areas_km2(latitudes: np.ndarray, longitude_step: float, latitude_step: float) -> np.ndarray:
    """
    Ground area of each cell on a regular lon/lat grid, on the WGS84 ellipsoid.

    Cells of equal angular size cover less ground the further they sit from the equator.

    Parameters
    ----------
    latitudes : ndarray
        Cell center latitude, in degrees.
    longitude_step : float
        Angular cell width, in degrees.
    latitude_step : float
        Angular cell height, in degrees.

    Returns
    -------
    areas : ndarray
        Cell area in square kilometers, one per entry of ``latitudes``.
    """
    # A spherical cos(latitude) form is biased by a few parts in a thousand, and the bias runs with
    # latitude. Banding keeps the ellipsoidal figure cheap: one evaluation per row, not per cell.
    geodesic = Geod(ellps=WGS84_ELLIPSOID)
    bands, band_of_cell = np.unique(latitudes, return_inverse=True)

    half_width, half_height = longitude_step / 2.0, latitude_step / 2.0
    areas = np.array(
        [
            abs(geodesic.geometry_area_perimeter(box(-half_width, lat - half_height, half_width, lat + half_height))[0])
            for lat in bands
        ]
    )

    return areas[band_of_cell] / SQUARE_METRES_PER_SQUARE_KM


@dataclass(frozen=True, slots=True)
class CellGrid:
    """
    The quadrature lattice over a place, and the cells of it that fall inside.

    Both halves are needed. The model integrates over ``cells``, but assigning cells to administrative units treats the
    grid as a raster. A raster is defined by the whole lattice rather than by the part of it that survived clipping.
    ``cell_id`` on ``cells`` indexes the lattice in raster order, north row first, so it is the join key between the
    two.

    Parameters
    ----------
    longitudes : ndarray
        Cell center longitudes, ascending.
    latitudes : ndarray
        Cell center latitudes, descending, so row zero is the northernmost.
    longitude_step : float
        Angular cell width, in degrees.
    latitude_step : float
        Angular cell height, in degrees.
    cells : GeoDataFrame
        Surviving cells, with ``cell_id``, ``lon``, ``lat``, ``ISO_A3``, ``cell_area_km2`` for the whole
        cell and ``place_area_km2`` for the part of it inside the place.
    place : shapely geometry
        The ground the lattice was cut to, dissolved. Anything measured over the cells is measured
        over their intersection with this, so a raster sampled onto the grid cannot be clipped to
        different ground than the grid itself was.
    """

    longitudes: np.ndarray
    latitudes: np.ndarray
    longitude_step: float
    latitude_step: float
    cells: gpd.GeoDataFrame
    place: shapely.geometry.base.BaseGeometry

    @property
    def shape(self) -> tuple[int, int]:
        """Rows and columns of the full lattice, before clipping."""
        return len(self.latitudes), len(self.longitudes)

    @property
    def steps(self) -> tuple[float, float]:
        """Angular cell width and height, in degrees."""
        return self.longitude_step, self.latitude_step

    @property
    def footprints(self) -> gpd.GeoSeries:
        """
        The rectangle each cell stands for, rather than the center it is sampled at.

        ``cells.geometry`` is the center, where the model evaluates the field.
        """
        half_width, half_height = (step / 2.0 for step in self.steps)
        lon, lat = self.cells["lon"].to_numpy(), self.cells["lat"].to_numpy()

        return gpd.GeoSeries(
            shapely.box(lon - half_width, lat - half_height, lon + half_width, lat + half_height),
            index=self.cells.index,
            crs=GEOGRAPHIC_CRS,
        )

    def column_of_cell(self, cell_ids: np.ndarray) -> np.ndarray:
        """
        Operator column of each lattice cell.

        ``cell_id`` numbers the whole lattice while the operator's columns number the cells that
        survived clipping, so the two spaces diverge from the first dropped cell onward. An
        overlap table is keyed on the first and has to be translated to the second.

        Parameters
        ----------
        cell_ids : ndarray
            Lattice cell ids, as ``assign_cells_to_units`` reports them. Repeats are fine.

        Returns
        -------
        columns : ndarray
            Column index of each, positions into ``cells``.
        """
        wanted = np.asarray(cell_ids)
        column_of = pd.Series(np.arange(len(self.cells)), index=self.cells["cell_id"].to_numpy())

        missing = np.setdiff1d(wanted, column_of.index.to_numpy())
        if missing.size:
            raise DataValidationError(
                f"{missing.size} cells are not in the grid, starting {sorted(missing.tolist())[:5]}. They belong to "
                f"a different lattice, or to this one before it was clipped."
            )

        return np.asarray(column_of.loc[wanted].to_numpy())

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Outer edges of the lattice, half a cell beyond the outermost centers."""
        half_width, half_height = (step / 2.0 for step in self.steps)

        return (
            float(self.longitudes[0]) - half_width,
            float(self.latitudes[-1]) - half_height,
            float(self.longitudes[-1]) + half_width,
            float(self.latitudes[0]) + half_height,
        )


def cell_coverage(
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    features: gpd.GeoDataFrame,
    key_column: str,
) -> pd.DataFrame:
    """
    Fraction of each grid cell covered by each feature.

    Parameters
    ----------
    shape : tuple of int
        Rows and columns of the lattice.
    bounds : tuple of float
        Outer edges of the lattice, ``(lon_min, lat_min, lon_max, lat_max)``.
    features : GeoDataFrame
        Polygons to measure against, in any CRS.
    key_column : str
        Column naming each feature.

    Returns
    -------
    coverage : DataFrame
        One row per feature and overlapping cell, with the key, ``cell_id`` and ``coverage``. Cells
        a feature misses entirely produce no row.

    Examples
    --------
    .. code-block:: python

        from climate_risk.geo.raster import cell_coverage

        coverage = cell_coverage(grid.shape, grid.bounds, districts, "gid")
    """
    rows, columns = shape
    lon_min, lat_min, lon_max, lat_max = bounds
    lattice = NumPyRasterSource(np.zeros((rows, columns)), xmin=lon_min, ymin=lat_min, xmax=lon_max, ymax=lat_max)

    overlaps = exact_extract(
        lattice,
        features.to_crs(GEOGRAPHIC_CRS),
        ["cell_id", "coverage"],
        include_cols=[key_column],
        output="pandas",
    )
    widths = overlaps["cell_id"].map(len).to_numpy()

    return pd.DataFrame(
        {
            key_column: np.repeat(overlaps[key_column].to_numpy(), widths),
            "cell_id": np.concatenate([np.asarray(ids) for ids in overlaps["cell_id"]]).astype(int),
            "coverage": np.concatenate([np.asarray(share) for share in overlaps["coverage"]]),
        }
    )


def sample_onto_cells(grid: CellGrid, raster: str, *, statistic: str) -> np.ndarray:
    """
    Reduce a raster to one value per grid cell, over the part of each cell inside the place.

    Parameters
    ----------
    grid : CellGrid
        The lattice to reduce onto. Values come back in the order of ``grid.cells``.
    raster : str
        Path or URI :func:`rasterio.open` accepts.
    statistic : str
        An ``exactextract`` operation. It follows the quantity rather than the grid: an extensive
        one such as a headcount takes ``"sum"``, an intensive one such as an elevation ``"mean"``.

    Returns
    -------
    values : ndarray
        One value per surviving cell, aligned with ``grid.cells``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.ghsl import population_raster
        from climate_risk.geo.raster import sample_onto_cells

        raster = population_raster(2020, Path("data"))
        people = sample_onto_cells(grid, raster, statistic="sum")
    """
    clipped = grid.footprints.intersection(grid.place)
    missed = clipped.is_empty
    if missed.any():
        raise DataValidationError(
            f"{int(missed.sum())} cells were kept for overlapping the place but do not intersect it."
        )
    footprints = gpd.GeoDataFrame(geometry=clipped, crs=GEOGRAPHIC_CRS)

    with rasterio.open(raster) as source:
        reduced = exact_extract(RasterioRasterSource(source), footprints, [statistic, "count"], output="pandas")

    # A sum over no data is zero, which reads as an empty cell rather than an unmeasured one, so
    # coverage is what says whether the raster answered.
    uncovered = np.asarray(reduced["count"].to_numpy(), dtype=float) == 0.0
    if uncovered.any():
        raise DataValidationError(
            f"{int(uncovered.sum())} cells are not covered by {raster}, so their value is unknown rather than zero."
        )

    return np.asarray(reduced[statistic].to_numpy(), dtype=float)


def build_cell_grid(boundary: gpd.GeoDataFrame, *, resolution_km: float) -> CellGrid:
    """
    Lay a grid of cells over a dissolved boundary and keep every cell it touches.

    Membership is by coverage: a cell half inside the place is kept, credited with half its area.

    Parameters
    ----------
    boundary : GeoDataFrame
        One row per country, as :func:`dissolve_place_boundary` returns.
    resolution_km : float
        Target cell edge, in kilometers.

    Returns
    -------
    grid : CellGrid
        The lattice and its surviving cells, in ``GEOGRAPHIC_CRS``.

    Examples
    --------
    Lay a 25 km lattice over a boundary:

    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_shapefile
        from climate_risk.geo.raster import build_cell_grid, dissolve_place_boundary

        world = load_shapefile("world", Path("data"))
        boundary = dissolve_place_boundary(world.query('ISO_A3 == "LAO"'))
        grid = build_cell_grid(boundary, resolution_km=25.0)
    """
    # A country is mostly border at any resolution coarse enough to model, and a dropped border
    # cell takes its overlaps with the border units with it.
    if ISO_COLUMN not in boundary.columns:
        raise DataValidationError(f"Grid the output of dissolve_place_boundary, which carries {ISO_COLUMN}.")

    lon_min, lat_min, lon_max, lat_max = (float(edge) for edge in boundary.total_bounds)

    longitudes, ascending_latitudes = grid_axes((lon_min, lat_min, lon_max, lat_max), resolution_km)
    latitudes = ascending_latitudes[::-1]

    # Raster order: rows run north to south, columns west to east, so cell_id matches what a zonal
    # statistics pass over the same lattice reports.
    latitude_mesh, longitude_mesh = (axis.ravel() for axis in np.meshgrid(latitudes, longitudes, indexing="ij"))

    # A single-cell axis has no gap to measure, so the cell is as wide as the extent.
    steps = (
        longitudes[1] - longitudes[0] if len(longitudes) > 1 else lon_max - lon_min,
        ascending_latitudes[1] - ascending_latitudes[0] if len(latitudes) > 1 else lat_max - lat_min,
    )

    lattice = pd.DataFrame(
        {
            "cell_id": np.arange(longitude_mesh.size),
            "lon": longitude_mesh,
            "lat": latitude_mesh,
            "cell_area_km2": cell_areas_km2(latitude_mesh, *steps),
        }
    )

    shape = (len(latitudes), len(longitudes))
    edges = (
        float(longitudes[0]) - steps[0] / 2.0,
        float(ascending_latitudes[0]) - steps[1] / 2.0,
        float(longitudes[-1]) + steps[0] / 2.0,
        float(ascending_latitudes[-1]) + steps[1] / 2.0,
    )
    covered = cell_coverage(shape, edges, boundary, ISO_COLUMN)

    # A cell on an internal frontier touches two countries. It is one piece of ground, labeled by
    # whichever claims most of it and credited with everything the place covers.
    inside = covered.groupby("cell_id")[["coverage"]].sum()
    dominant = covered.sort_values("coverage", ascending=False).drop_duplicates("cell_id").set_index("cell_id")
    inside[ISO_COLUMN] = dominant[ISO_COLUMN]

    kept = lattice.join(inside, on="cell_id", how="inner").reset_index(drop=True)
    kept["place_area_km2"] = kept["cell_area_km2"] * kept["coverage"].clip(upper=1.0)

    return CellGrid(
        longitudes=longitudes,
        latitudes=latitudes,
        longitude_step=float(steps[0]),
        latitude_step=float(steps[1]),
        cells=gpd.GeoDataFrame(
            kept[["cell_id", "lon", "lat", ISO_COLUMN, "cell_area_km2", "place_area_km2"]],
            geometry=gpd.points_from_xy(kept["lon"], kept["lat"]),
            crs=GEOGRAPHIC_CRS,
        ),
        place=boundary.to_crs(GEOGRAPHIC_CRS).union_all(),
    )


def assign_cells_to_units(grid: CellGrid, units: gpd.GeoDataFrame, *, unit_column: str = "gid") -> pd.DataFrame:
    """
    Overlap between grid cells and administrative units, weighted by area.

    A cell on a border appears once per unit it touches, carrying the ground it contributes to
    each. Cells outside every unit produce no rows. A unit smaller than a cell still gets its share
    of the cell it sits in.

    Parameters
    ----------
    grid : CellGrid
        The lattice and its cells.
    units : GeoDataFrame
        Administrative units, in any CRS, carrying ``unit_column``.
    unit_column : str, optional
        Column naming each unit. Default 'gid', which is what ``event_geography`` keys on.

    Returns
    -------
    assignments : DataFrame
        One row per overlapping cell and unit, with the unit key, ``cell_id``, the fraction of the
        cell inside the unit, and the overlapping area in square kilometers.

    Examples
    --------
    Map each cell onto the district that holds it:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.gadm import load_units_in_country
        from climate_risk.geo.raster import assign_cells_to_units

        districts = load_units_in_country("LAO", 2, Path("data"))
        assignments = assign_cells_to_units(grid, districts)
    """
    if unit_column not in units.columns:
        raise DataValidationError(f"The units carry no {unit_column!r} column to key the operator's rows on.")

    if units.crs is None:
        raise DataValidationError("The units carry no CRS, so they cannot be placed against the grid.")

    overlaps = cell_coverage(grid.shape, grid.bounds, units, unit_column)

    areas = grid.cells.set_index("cell_id")["cell_area_km2"]
    within_place = overlaps[overlaps["cell_id"].isin(areas.index)].reset_index(drop=True)

    return within_place.assign(overlap_km2=lambda frame: frame["coverage"] * areas.loc[frame["cell_id"]].to_numpy())
