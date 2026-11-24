"""
GHS-POP people-per-pixel summed onto a coarser covariate lattice, for population-weighted country means.

The GHS-POP grid has 120 pixels per degree (30 arc-seconds). ERA5 (0.25 degrees), ERA5-Land (0.1) and
GloFAS (0.05) all have pixel edges on multiples of that, so each covariate pixel holds a whole number of
population pixels and the sum is exact.
"""

from pathlib import Path

import numpy as np
import rasterio

from climate_risk.data.ghsl import POPULATION_EPOCHS, population_raster

STRIP_ROWS = 1200
AVAILABLE_EPOCHS = tuple(range(1990, 2026, 5))  # the epochs fetched into the repo cache


def nearest_epoch(year: int) -> int:
    """The GHS-POP epoch closest to ``year`` among those on disk (years before 1990 use 1990)."""
    return min(AVAILABLE_EPOCHS, key=lambda epoch: abs(epoch - year))


def _groups(index: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Start positions and target indices of runs of equal ``index`` that fall inside ``0..size-1``."""
    starts = np.concatenate([[0], np.flatnonzero(np.diff(index)) + 1])
    targets = index[starts]
    keep = (targets >= 0) & (targets < size)
    return starts[keep], targets[keep]


def population_on_lattice(epoch: int, longitudes: np.ndarray, latitudes: np.ndarray, cache_dir: Path) -> np.ndarray:
    """
    People per covariate pixel, shape ``(len(latitudes), len(longitudes))`` in the order the coordinates are given.

    ``longitudes`` and ``latitudes`` are pixel centres of a regular lattice, in either order.
    """
    if epoch not in POPULATION_EPOCHS:
        raise ValueError(epoch)
    lons = np.asarray(longitudes, dtype="float64")
    lats = np.asarray(latitudes, dtype="float64")
    lon_order, lat_order = np.argsort(lons), np.argsort(-lats)
    sorted_lons, sorted_lats = lons[lon_order], lats[lat_order]
    dlon, dlat = float(np.median(np.diff(sorted_lons))), float(np.median(-np.diff(sorted_lats)))
    west, north = sorted_lons[0] - dlon / 2, sorted_lats[0] + dlat / 2
    out = np.zeros((len(sorted_lats), len(sorted_lons)))

    with rasterio.open(population_raster(epoch, cache_dir)) as raster:
        step = raster.transform.a
        raster_lons = raster.transform.c + (np.arange(raster.width) + 0.5) * step
        col_index = np.floor((raster_lons - west) / dlon + 1e-9).astype(int) % len(sorted_lons)
        col_starts, col_targets = _groups(col_index, len(sorted_lons))
        for top in range(0, raster.height, STRIP_ROWS):
            rows = min(STRIP_ROWS, raster.height - top)
            strip = raster.read(1, window=((top, top + rows), (0, raster.width))).astype("float64")
            strip = np.where(strip > 0, strip, 0.0)  # negative values are the no-data code
            raster_lats = raster.transform.f - (np.arange(top, top + rows) + 0.5) * step
            row_index = np.floor((north - raster_lats) / dlat + 1e-9).astype(int)
            row_starts, row_targets = _groups(row_index, len(sorted_lats))
            if len(row_starts) == 0:
                continue
            by_col = np.add.reduceat(strip, col_starts, axis=1)  # wrapped duplicates are added below
            ends = np.append(row_starts[1:], rows)
            for start, end, target in zip(row_starts, ends, row_targets, strict=True):
                np.add.at(out[target], col_targets, by_col[start:end].sum(axis=0))
    # back to the order the caller gave
    return out[np.argsort(lat_order)][:, np.argsort(lon_order)]
