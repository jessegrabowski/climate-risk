"""
Build global yearly covariate statistics, one year at a time, deleting each raw file after it is reduced.

Steps (run each as a subcommand; every one skips years already done, so it can be stopped and restarted):

    era5       ERA5 daily precipitation  -> precip_total, rx5day            (1981-2025)
    era5land   ERA5-Land monthly means   -> soilm_*, t2m_mean, pet_annual   (1981-2025)
    glofas     GloFAS v5.0 daily flow    -> q_annual_max                    (1980-2025, raw files KEPT)
    glofas-days  fits the 5-year return level, counts days above it, then deletes the raw GloFAS files
    chirps     CHIRPS v2.0 (IRI mirror, 0.25-degree box-average, 50N-50S) -> precip_total, rx5day (1981-2025)
    anomalies  z-scores the ERA5 annual totals against 1991-2020 -> precip_anom_std

Example:  pixi run python experiments/covariates/global_annual_stats.py era5
Raw downloads go to data/raw/covariates/global/<source>/ (git-ignored); results to
data/processed/covariates/global/ (git-ignored).
"""

import argparse
import logging

from pathlib import Path

import xarray as xr

from climate_risk.data import chirps
from climate_risk.data import copernicus as cop
from climate_risk.data import global_stats as gs

RAW = Path("data/raw/covariates/global")
OUT = Path("data/processed/covariates/global")
YEARS = range(1981, 2026)
GLOFAS_YEARS = range(1980, 2026)

_log = logging.getLogger("global_annual_stats")


def save(dataset: xr.Dataset, target: Path, **attrs: str) -> None:
    """Write a small result file atomically (so a stopped run never leaves a half-written file)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.attrs.update(attrs)
    partial = target.with_suffix(".nc.part")
    dataset.to_netcdf(partial, encoding={name: {"zlib": True, "complevel": 4} for name in dataset.data_vars})
    partial.replace(target)


def run_source(name: str, years, request_for, dataset: str, url: str, reduce, raw_name: str, keep_raw: bool) -> None:
    """Download each missing year, reduce it, save the result, and delete the raw file unless told to keep it."""
    client = cop.make_client(url)
    for year in years:
        result = OUT / f"{name}_{year}.nc"
        if result.exists():
            continue
        raw = RAW / name / raw_name.format(year=year)
        cop.retrieve(client, dataset, request_for(year), raw)
        save(reduce(raw), result, source=name, year=str(year))
        _log.info("%s %d done", name, year)
        if not keep_raw:
            raw.unlink()


def do_chirps() -> None:
    """CHIRPS is not a Copernicus request, so it has its own small loop with the same skip/reduce/delete pattern."""
    for year in YEARS:
        result = OUT / f"chirps_v2_{year}.nc"
        if result.exists():
            continue
        raw = chirps.download_iri_global_year(year, RAW / "chirps" / f"chirps_v2_global_{year}.nc")
        save(
            gs.chirps_global_stats(raw),
            result,
            source="CHIRPS v2.0 daily improved, IRI mirror, box-averaged to 0.25 deg",
            year=str(year),
        )
        _log.info("chirps %d done", year)
        raw.unlink()


def do_era5() -> None:
    run_source(
        "era5_daily",
        YEARS,
        cop.era5_daily_request,
        cop.ERA5_DAILY,
        cop.CDS_URL,
        gs.era5_daily_stats,
        "era5_daily_{year}.nc",
        keep_raw=False,
    )


def do_era5land() -> None:
    run_source(
        "era5land",
        YEARS,
        lambda year: cop.global_request(cop.era5_land_request([year], (0, 0, 0, 0))),
        cop.ERA5_LAND,
        cop.CDS_URL,
        gs.era5_land_stats,
        "era5land_{year}.nc",
        keep_raw=False,
    )


def do_glofas() -> None:
    run_source(
        "glofas",
        GLOFAS_YEARS,
        lambda year: cop.global_request(cop.glofas_request([year], (0, 0, 0, 0))),
        cop.GLOFAS,
        cop.EWDS_URL,
        gs.glofas_annual_max,
        "glofas_{year}.nc",
        keep_raw=True,
    )


def do_glofas_days() -> None:
    maxima = xr.concat(
        [xr.open_dataset(OUT / f"glofas_{year}.nc")["q_annual_max"] for year in GLOFAS_YEARS],
        dim=xr.DataArray(list(GLOFAS_YEARS), dims="year", name="year"),
    ).load()
    level = gs.return_level_from_maxima(maxima)
    save(
        level.to_dataset(name="q_rp5_level"),
        OUT / "glofas_rp5_level.nc",
        method="Gumbel, L-moments, 1980-2025 annual maxima",
    )
    for year in GLOFAS_YEARS:
        target = OUT / f"glofas_days_{year}.nc"
        if not target.exists():
            save(gs.glofas_days_over(RAW / "glofas" / f"glofas_{year}.nc", level), target, year=str(year))
            (RAW / "glofas" / f"glofas_{year}.nc").unlink()
            _log.info("glofas days %d done", year)


def do_anomalies() -> None:
    totals = xr.concat(
        [xr.open_dataset(OUT / f"era5_daily_{year}.nc")["precip_total"].reset_coords(drop=True) for year in YEARS],
        dim=xr.DataArray(list(YEARS), dims="year", name="year"),
    ).load()
    z = gs.standardise_stack(totals).rename("precip_anom_std")
    for year in YEARS:
        save(
            z.sel(year=year, drop=True).to_dataset(), OUT / f"era5_anom_{year}.nc", year=str(year), baseline="1991-2020"
        )


STEPS = {
    "chirps": do_chirps,
    "era5": do_era5,
    "era5land": do_era5land,
    "glofas": do_glofas,
    "glofas-days": do_glofas_days,
    "anomalies": do_anomalies,
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS)
    STEPS[parser.parse_args().step]()
