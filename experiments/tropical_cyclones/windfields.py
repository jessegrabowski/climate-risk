"""
Step 2 of the cyclone build (CLIMADA environment, NOT pixi): the strongest modelled wind per cell and year.

IBTrACS v04r01 tracks are read with CLIMADA (winds rescaled to 1-minute sustained, missing pressure or wind filled
by CLIMADA's relations), interpolated to hourly steps, and turned into wind fields with the Holland 2008 model
(CLIMADA ``TropCyclone``, ``intensity_thres`` near zero so weak winds are kept, up to 300 km from the track, 61 degrees
of latitude at most). Each storm belongs to the calendar year its track starts in. The yearly value of a cell is the
largest wind any storm gave it, in m/s; cells no storm came near stay 0.

Usage: ``micromamba run -r ~/micromamba -n climada_env python experiments/tropical_cyclones/windfields.py [workers]``
Output: data/processed/covariates/tc_wind/<year>.npz (one value per cell of tc_cells.npz)
"""

import logging
import multiprocessing as mp
import sys

from pathlib import Path

import numpy as np

from climada.hazard import Centroids, TCTracks, TropCyclone

TRACKS = Path("data/raw/covariates/ibtracs")
CELLS = Path("data/processed/covariates/tc_cells.npz")
OUT = Path("data/processed/covariates/tc_wind")
YEARS = range(1980, 2026)
BATCH = 20  # storms per CLIMADA call, which keeps memory small
_log = logging.getLogger("tc_windfields")


def one_year(year: int) -> str:
    """Compute and save the annual maximum wind of one year."""
    target = OUT / f"{year}.npz"
    if target.exists():
        return f"{year} already done"
    cells = np.load(CELLS)
    centroids = Centroids(lat=cells["latitude"][cells["row"]], lon=cells["longitude"][cells["column"]])
    tracks = TCTracks.from_ibtracs_netcdf(
        year_range=(year - 1, year + 1),
        estimate_missing=True,
        file_name=str((TRACKS / "IBTrACS.ALL.v04r01.nc").resolve()),
    )
    mine = [track for track in tracks.data if int(track.time.dt.year.values[0]) == year]
    strongest = np.zeros(len(cells["row"]), dtype=np.float32)
    for start in range(0, len(mine), BATCH):
        batch = TCTracks(mine[start : start + BATCH])
        batch.equal_timestep(1.0)
        hazard = TropCyclone.from_tracks(
            batch,
            centroids=centroids,
            model="H08",
            intensity_thres=0.1,
            ignore_distance_to_coast=True,
            max_memory_gb=0.8,
        )
        peak = np.asarray(hazard.intensity.max(axis=0).todense()).ravel()
        strongest = np.maximum(strongest, peak.astype(np.float32))
    partial = target.with_name(f"{year}.part.npz")
    np.savez_compressed(partial, wind=strongest, storms=len(mine))
    partial.replace(target)
    return f"{year}: {len(mine)} storms, {int((strongest >= 17.5).sum())} cells reached 17.5 m/s"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("climada").setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for message in pool.imap_unordered(one_year, YEARS):
            _log.info(message)
