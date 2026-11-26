"""
CMIP6 annual rx5day (ETCCDI "maximum_5_day_precipitation") from the Copernicus CDS.

Dataset ``sis-extreme-indices-cmip6``. The request keys and the valid combinations of model, member, experiment,
period and version come from the dataset's own constraint list (the "Show API request code" form reads the same list),
so nothing here is guessed: :func:`plan_requests` only emits combinations the catalogue says exist.
One member is used per model (``r1i1p1f1`` where it exists), and the newest dataset version available for it.
"""

import logging

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

DATASET = "sis-extreme-indices-cmip6"
CATALOGUE = f"https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{DATASET}"
VARIABLE = "maximum_5_day_precipitation"
EXPERIMENTS = ("historical", "ssp1_2_6", "ssp2_4_5", "ssp3_7_0", "ssp5_8_5")
PREFERRED_MEMBER = "r1i1p1f1"
_log = logging.getLogger(__name__)


def fetch_constraints() -> list[dict[str, Any]]:
    """The dataset's list of valid request combinations (public, no key needed)."""
    response = requests.get(f"{CATALOGUE}/constraints.json", timeout=60)
    response.raise_for_status()
    return list(response.json())


def plan_requests(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group the valid (model, experiment) pairs into as few requests as the constraints allow.

    Parameters
    ----------
    constraints : list of dict
        From :func:`fetch_constraints`.

    Returns
    -------
    requests : list of dict
        CDS request bodies. Each lists the models that share one experiment, member, period and version.
    """
    valid: dict[str, dict[str, dict[str, set[tuple[str, str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for block in constraints:
        if VARIABLE not in block["variable"] or "yearly" not in block["temporal_aggregation"]:
            continue
        for model in block["model"]:
            for member in block["ensemble_member"]:
                for experiment in block["experiment"]:
                    valid[model][member][experiment].update(
                        (version, period) for version in block["version"] for period in block["period"]
                    )

    groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for model in sorted(valid):
        members = valid[model]
        # the member that covers most experiments, preferring r1i1p1f1
        member = sorted(members, key=lambda m: (-len(set(members[m]) & set(EXPERIMENTS)), m != PREFERRED_MEMBER, m))[0]
        for experiment in EXPERIMENTS:
            options = members[member].get(experiment)
            if not options:
                continue
            version = max(v for v, _ in options)
            periods = sorted(p for v, p in options if v == version)
            period = "1850_2014" if "1850_2014" in periods else ("2015_2100" if "2015_2100" in periods else periods[0])
            groups[(experiment, member, version, period)].append(model)

    return [
        {
            "variable": [VARIABLE],
            "product_type": ["base_independent"],
            "model": models,
            "ensemble_member": [member],
            "experiment": [experiment],
            "temporal_aggregation": ["yearly"],
            "period": [period],
            "version": [version],
        }
        for (experiment, member, version, period), models in sorted(groups.items())
    ]


def target_name(request: Mapping[str, Any]) -> str:
    """A stable file name for one request, so a rerun skips what is already downloaded."""
    return "_".join([request["experiment"][0], request["ensemble_member"][0], request["period"][0]]) + ".bin"


def raw_directory(cache_dir: Path) -> Path:
    """Where the downloads go (git-ignored)."""
    return cache_dir / "raw" / "covariates" / "cmip6_rx5day"


_CALENDAR_YEAR_DAYS = {"360_day": 360.0, "365_day": 365.0, "noleap": 365.0, "366_day": 366.0, "all_leap": 366.0}


def _years_of(time: np.ndarray, units: str, calendar: str) -> np.ndarray:
    """Calendar year of each time stamp, read by hand because several CMIP6 calendars (360-day, noleap) break pandas."""
    reference = int(units.split("since")[1].strip().split("-")[0])
    if calendar in _CALENDAR_YEAR_DAYS:
        return reference + np.floor(np.asarray(time, dtype=float) / _CALENDAR_YEAR_DAYS[calendar]).astype(int)
    stamps = pd.Timestamp(f"{reference}-01-01") + pd.to_timedelta(np.asarray(time, dtype=float), unit="D")
    return np.asarray(stamps.year)


def read_rx5day(path: Path) -> xr.DataArray:
    """
    One CMIP6 file as annual rx5day (mm) with dims ``year``, ``latitude``, ``longitude`` (longitude -180..180, ascending).

    Parameters
    ----------
    path : Path
        A NetCDF file from the dataset (one model, one experiment).
    """
    with xr.open_dataset(path, decode_times=False) as opened:
        data = opened["rx5dayETCCDI"].load()
        years = _years_of(
            opened["time"].to_numpy(),
            str(opened["time"].attrs["units"]),
            str(opened["time"].attrs.get("calendar", "standard")),
        )
    out = data.rename({"lat": "latitude", "lon": "longitude", "time": "year"}).assign_coords(year=years)
    out = out.assign_coords(longitude=((out["longitude"] + 180) % 360) - 180).sortby("longitude").sortby("latitude")
    return out.drop_vars([name for name in out.coords if name not in out.dims]).rename("rx5day")


def split_name(filename: str) -> tuple[str, str]:
    """(model, experiment) from a file name such as ``rx5dayETCCDI_yr_FGOALS-g3_historical_r1i1p1f1_...nc``."""
    parts = filename.split("_")
    experiment_at = next(i for i, part in enumerate(parts) if part in EXPERIMENTS or part.startswith("ssp"))
    return "_".join(parts[2:experiment_at]), parts[experiment_at]
