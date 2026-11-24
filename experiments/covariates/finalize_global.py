"""
Add RONI and the panel's ``ISO`` key to the global country-year table, and write the global manifest.

Run last, after global_country_year.py (and again whenever GloFAS years are added):
    pixi run python experiments/covariates/finalize_global.py
Writes data/processed/covariates/global_country_year.parquet, global_admin1_year.parquet and manifest.json
(all git-ignored). The intermediate tables in .../global/ are left as they are.
"""

import json

from pathlib import Path

import pandas as pd

from climate_risk.data.roni import RONI_LICENCE, RONI_TABLE_URL, annual_roni, read_roni

PROCESSED = Path("data/processed/covariates")
GLOBAL = PROCESSED / "global"
RONI_FILE = sorted(Path("data/raw/covariates/NPL/roni").glob("RONI.ascii.*.txt"))[-1]
ACCESS = "2026-11-24"

MANIFEST = [
    {
        "source": "ERA5 daily total precipitation",
        "dataset": "derived-era5-single-levels-daily-statistics",
        "version": "ERA5 post-processed daily statistics (daily_sum, UTC)",
        "product": "total_precipitation, 0.25 degree, global",
        "url": "https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics",
        "licence": "Licence to use Copernicus Products (CC-BY 4.0 compatible)",
    },
    {
        "source": "ERA5-Land monthly means",
        "dataset": "reanalysis-era5-land-monthly-means",
        "version": "ERA5-Land",
        "product": "volumetric_soil_water_layer_1, 2m_temperature, total_precipitation, potential_evaporation; 0.1 degree, global",
        "url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means",
        "licence": "Licence to use Copernicus Products (CC-BY 4.0 compatible)",
    },
    {
        "source": "GloFAS historical",
        "dataset": "cems-glofas-historical",
        "version": "v5.0 (version_5_0, LISFLOOD, consolidated)",
        "product": "average river discharge in the last 24 hours, 0.05 degree, 60S-90N",
        "url": "https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical",
        "licence": "Licence to use Copernicus Products (CC-BY 4.0 compatible)",
    },
    {
        "source": "CHIRPS daily (stand-in for v3 rnl)",
        "dataset": "CHIRPS",
        "version": "v2.0 daily improved, IRI mirror",
        "product": "precipitation, box-averaged to 0.25 degree on the IRI server, 50S-50N; NOT v3 (v3 applies a gauge undercatch adjustment, daily = pentads split with ERA5)",
        "url": "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp",
        "licence": "Public domain, cite Funk et al. 2015",
    },
    {
        "source": "RONI",
        "dataset": "RONI.ascii.txt",
        "version": f"file {RONI_FILE.name}",
        "product": "Relative Oceanic Nino Index, 3-month seasons",
        "url": RONI_TABLE_URL,
        "licence": RONI_LICENCE,
    },
    {
        "source": "GHS-POP",
        "dataset": "GHS_POP_GLOBE_R2023A",
        "version": "R2023A, 30 arc-second, epochs 1990-2025",
        "product": "population weights for country means",
        "url": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL",
        "licence": "European Commission reuse notice",
    },
    {
        "source": "GADM",
        "dataset": "gadm_410",
        "version": "4.1",
        "product": "country and admin-1 outlines (simplified 0.002 deg)",
        "url": "https://gadm.org",
        "licence": "Non-commercial use only; not redistributed",
    },
]


def main() -> None:
    roni = annual_roni(read_roni(RONI_FILE))
    for level, name in (("country", "global_country_year"), ("admin1", "global_admin1_year")):
        table = pd.read_parquet(GLOBAL / f"{name}.parquet").rename(columns={"GID_0": "ISO"})
        table = table.merge(roni, on="year", how="left")
        table["date"] = pd.to_datetime(table["year"].astype(str) + "-01-01")  # the panel keys on ``date`` (1 January)
        table.to_parquet(PROCESSED / f"{name}.parquet")
        print(level, table.shape)
    for row in MANIFEST:
        row["access_date"] = ACCESS
    (PROCESSED / "manifest.json").write_text(json.dumps(MANIFEST, indent=2))


if __name__ == "__main__":
    main()
