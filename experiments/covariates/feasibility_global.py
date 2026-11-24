"""
Feasibility tests for global covariate downloads (step 1 of the 2026-11-23 global task).

Each test sends ONE small global request, reports whether the server accepts it, how long it
queued, and how big the file is. Nothing here loops over years. Run one test at a time:

    pixi run python experiments/covariates/feasibility_global.py glofas-year
    pixi run python experiments/covariates/feasibility_global.py glofas-month
    pixi run python experiments/covariates/feasibility_global.py era5land-year
"""

import sys
import time

from pathlib import Path

from climate_risk.data import copernicus as cop

OUT = Path("data/raw/covariates/global_tests")
GLOBAL = (-180.0, -90.0, 180.0, 90.0)


def glofas(months: list[str]) -> tuple[str, dict, str]:
    request = cop.glofas_request([1990], GLOBAL)
    request.pop("area")  # a whole-globe request has no area key
    request["month"] = months
    name = "glofas_global_1990_" + ("year" if len(months) == 12 else "month") + ".nc"
    return cop.GLOFAS, request, name


def era5_land_year() -> tuple[str, dict, str]:
    request = cop.era5_land_request([1990], GLOBAL)
    request.pop("area")
    return cop.ERA5_LAND, request, "era5land_global_1990.nc"


def era5_daily(months: list[str]) -> tuple[str, dict, str]:
    """ERA5 daily total precipitation (sum over the day) for the whole globe, 1990."""
    request = {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": "1990",
        "month": months,
        "day": cop.DAYS,
        "daily_statistic": "daily_sum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    return cop.ERA5_DAILY, request, "era5_daily_global_1990_" + ("year" if len(months) == 12 else "month") + ".nc"


TESTS = {
    "era5-daily-year": (cop.CDS_URL, lambda: era5_daily(cop.MONTHS)),
    "era5-daily-month": (cop.CDS_URL, lambda: era5_daily(["01"])),
    "glofas-year": (cop.EWDS_URL, lambda: glofas(cop.MONTHS)),
    "glofas-month": (cop.EWDS_URL, lambda: glofas(["01"])),
    "era5land-year": (cop.CDS_URL, era5_land_year),
}


def main(test: str) -> None:
    url, build = TESTS[test]
    dataset, request, name = build()
    if "month" in request and len(request["month"]) == 1:
        request["day"] = cop.DAYS
    started = time.time()
    try:
        target = cop.retrieve(cop.make_client(url), dataset, request, OUT / name)
    except cop.SourceFailed as error:
        print(f"{test}: FAILED after {time.time() - started:.0f} s: {error}")
        return
    print(f"{test}: OK in {(time.time() - started) / 60:.1f} min, {target.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
