"""
Reduce gridded climate fields onto the model's cell grid, one value per cell and calendar year.

Every source lives on its own lattice (CHIRPS 0.05 degrees, ERA5-Land 0.1, ERA5 0.25, GloFAS 0.05).
A :class:`CellReducer` is built once per lattice: it measures how much of each lattice pixel lies
inside each model cell (clipped to the place), using ``exactextract``, and then turns a field into
cell values by an area-weighted mean or a maximum.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from scipy import sparse

from climate_risk.geo.crs import GEOGRAPHIC_CRS
from climate_risk.geo.raster import CellGrid

EULER_GAMMA = 0.5772156649015329
RETURN_PERIOD_YEARS = 5
BASELINE = (1991, 2020)


class PolygonReducer:
    """
    Reduce fields on one regular lattice onto a set of polygons (cells, countries, admin-1 units).

    Parameters
    ----------
    polygons : GeoSeries
        The shapes to reduce onto, in longitude/latitude.
    longitudes, latitudes : array
        Pixel centres of the source lattice, in either order.
    """

    def __init__(self, polygons: gpd.GeoSeries, longitudes: np.ndarray, latitudes: np.ndarray) -> None:
        self.n_cells = len(polygons)
        self.lat_order = np.argsort(-np.asarray(latitudes))  # raster order: north first
        self.lon_order = np.argsort(np.asarray(longitudes))
        lons = np.asarray(longitudes)[self.lon_order]
        lats = np.asarray(latitudes)[self.lat_order]
        dlon, dlat = float(np.median(np.diff(lons))), float(np.median(-np.diff(lats)))
        shape = (len(lats), len(lons))

        source = NumPyRasterSource(
            np.zeros(shape),
            xmin=lons[0] - dlon / 2,
            xmax=lons[-1] + dlon / 2,
            ymin=lats[-1] - dlat / 2,
            ymax=lats[0] + dlat / 2,
        )
        footprints = gpd.GeoDataFrame(geometry=list(polygons), crs=GEOGRAPHIC_CRS)
        overlaps = exact_extract(source, footprints, ["cell_id", "coverage"], output="pandas")

        rows, cols, weights = [], [], []
        for cell, (pixels, share) in enumerate(zip(overlaps["cell_id"], overlaps["coverage"], strict=True)):
            pixels = np.asarray(pixels, dtype=int)
            area = np.asarray(share) * np.cos(np.radians(lats[pixels // shape[1]]))  # pixel area shrinks with latitude
            rows.append(np.full(len(pixels), cell))
            cols.append(pixels)
            weights.append(area)
        self.weights = sparse.csr_matrix(
            (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_cells, shape[0] * shape[1]),
        )
        self.pixels_of_cell = [
            np.asarray(p, dtype=int)[np.asarray(s) > 0]
            for p, s in zip(overlaps["cell_id"], overlaps["coverage"], strict=True)
        ]

    def _flat(self, field: np.ndarray) -> np.ndarray:
        """Reorder the last two axes to raster order and flatten them."""
        field = np.asarray(field, dtype="float64")[..., self.lat_order, :][..., :, self.lon_order]
        return field.reshape(*field.shape[:-2], -1)

    def mean(self, field: np.ndarray) -> np.ndarray:
        """Area-weighted mean over the valid pixels of each cell. Leading axes are kept; NaN where none is valid."""
        flat = self._flat(field)
        valid = np.isfinite(flat)
        total = np.where(valid, flat, 0.0) @ self.weights.T
        weight = valid.astype("float64") @ self.weights.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight > 0, total / weight, np.nan)

    def maximum(self, field: np.ndarray) -> np.ndarray:
        """Maximum over the pixels touching each cell. Leading axes are kept; NaN where all are missing."""
        flat = self._flat(field)
        out = np.full((*flat.shape[:-1], self.n_cells), np.nan)
        for cell, pixels in enumerate(self.pixels_of_cell):
            if len(pixels):
                with np.errstate(all="ignore"):
                    out[..., cell] = (
                        np.nanmax(flat[..., pixels], axis=-1) if np.isfinite(flat[..., pixels]).any() else np.nan
                    )
        return out

    def weighted_mean(self, field: np.ndarray, pixel_weights: np.ndarray) -> np.ndarray:
        """
        Mean over the valid pixels of each polygon, weighted by area times ``pixel_weights`` (e.g. population).

        ``pixel_weights`` has the same two-dimensional shape as the field. NaN where the weights sum to zero.
        """
        flat = self._flat(field)
        extra = self._flat(pixel_weights)
        valid = np.isfinite(flat) & np.isfinite(extra)
        total = np.where(valid, flat * extra, 0.0) @ self.weights.T
        weight = np.where(valid, extra, 0.0) @ self.weights.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight > 0, total / weight, np.nan)


class CellReducer(PolygonReducer):
    """
    Reduce fields on one regular lattice onto the cells of ``grid``, each cell clipped to the place.

    Parameters
    ----------
    grid : CellGrid
        The model cells.
    longitudes, latitudes : array
        Pixel centres of the source lattice, in either order.
    """

    def __init__(self, grid: CellGrid, longitudes: np.ndarray, latitudes: np.ndarray) -> None:
        super().__init__(grid.footprints.intersection(grid.place), longitudes, latitudes)


def gumbel_return_level(annual_maxima: np.ndarray, period: float = RETURN_PERIOD_YEARS) -> np.ndarray:
    """
    Return level of a Gumbel distribution fitted by L-moments, column by column.

    Parameters
    ----------
    annual_maxima : array of shape (years, series)
        One annual maximum per year for each series.
    period : float
        Return period in years.

    Returns
    -------
    level : array of shape (series,)
        The value exceeded once every ``period`` years on average: mu - beta * log(-log(1 - 1/period)),
        with beta = L2 / log 2 and mu = L1 - gamma * beta from the first two sample L-moments.
    """
    sample = np.sort(np.asarray(annual_maxima, dtype="float64"), axis=0)
    n = sample.shape[0]
    ranks = np.arange(1, n + 1)[:, None]
    l1 = sample.mean(axis=0)
    b1 = ((ranks - 1) / (n - 1) * sample).mean(axis=0)
    l2 = 2 * b1 - l1
    beta = l2 / np.log(2)
    mu = l1 - EULER_GAMMA * beta
    return mu - beta * np.log(-np.log(1 - 1 / period))


def month_lengths(times: pd.DatetimeIndex) -> np.ndarray:
    """Days in each month of a monthly time axis."""
    return np.asarray(times.days_in_month, dtype="float64")


def long_frame(values: np.ndarray, years: np.ndarray, cell_ids: np.ndarray, name: str) -> pd.DataFrame:
    """Turn a (year, cell) array into one row per cell and year."""
    frame = pd.DataFrame(values, index=pd.Index(years, name="year"), columns=pd.Index(cell_ids, name="cell_id"))
    return frame.stack(future_stack=True).rename(name).reset_index()


def standardise_against_baseline(
    totals: np.ndarray, years: np.ndarray, baseline: tuple[int, int] = BASELINE
) -> np.ndarray:
    """Standardise a (year, cell) array by each cell's mean and standard deviation over the baseline years."""
    inside = (years >= baseline[0]) & (years <= baseline[1])
    return (totals - totals[inside].mean(axis=0)) / totals[inside].std(axis=0, ddof=1)


__all__ = [
    "CellReducer",
    "PolygonReducer",
    "gumbel_return_level",
    "long_frame",
    "month_lengths",
    "standardise_against_baseline",
]
