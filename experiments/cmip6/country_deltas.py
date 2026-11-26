"""
Warming -> rx5day, from CMIP6 (task 2026-11-24b, part B).

For every model and scenario in ``data/raw/covariates/cmip6_rx5day`` (see download.py):
  1. put the model's annual rx5day on the ERA5 0.25-degree lattice (linear interpolation, longitude wrapped);
  2. population-weighted country mean per year (GHS-POP 2020 on that lattice, the same weights as the ERA5 country table);
  3. over 20-year windows, the relative change in the mean and in the standard deviation against 1995-2014
     (windows 2041-2060 and 2081-2100);
  4. the delta method on ERA5: dz = (ERA5 1991-2020 mean x relative change in the mean) / ERA5 1991-2020 SD, per country;
  5. ILLUSTRATIVE event-rate multiplier exp(SLOPE x dz), everything else held fixed.

Output: results/2026-11-24-cmip6-rx5day/country_deltas.parquet (one row per model, scenario, window, country),
        multipliers.csv (median and 10-90% range across models for the focus countries and a global population-weighted figure).
Usage: ``pixi run python experiments/cmip6/country_deltas.py``
"""

import logging
import tempfile
import zipfile

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from climate_risk.data import cmip6
from climate_risk.data.covariate_table import PolygonReducer
from climate_risk.replication_data import create_replication_data

CACHE = Path("data")
GLOBAL = CACHE / "processed" / "covariates" / "global"
COUNTRY_TABLE = CACHE / "processed" / "covariates" / "global_country_year.parquet"
RESULTS = Path("results/2026-11-24-cmip6-rx5day")
HISTORICAL = (1995, 2014)
WINDOWS = {"2041-2060": (2041, 2060), "2081-2100": (2081, 2100)}
ERA5_BASELINE = (1991, 2020)
SLOPE = 0.36  # rx5day slope of the frequency model (H1: 0.363, 89% interval 0.315 to 0.414)
FOCUS = ("LAO", "CRI", "BFA", "NPL", "ZMB", "MAR")
POPULATION_EPOCH = 2020

_log = logging.getLogger("cmip6_deltas")


def to_lattice(field: xr.DataArray, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Interpolate a (year, latitude, longitude) field onto the target lattice, treating longitude as a circle."""
    lon = field["longitude"].to_numpy()
    wrapped = xr.concat(
        [
            field.isel(longitude=[-1]).assign_coords(longitude=lon[-1:] - 360),
            field,
            field.isel(longitude=[0]).assign_coords(longitude=lon[:1] + 360),
        ],
        dim="longitude",
    )
    return np.asarray(wrapped.interp(longitude=longitude, latitude=latitude).to_numpy())


def country_series(
    directory: Path, reducer: PolygonReducer, weights: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> list[dict]:
    """Per file: the yearly population-weighted country series for the years the windows need."""
    needed = set(range(HISTORICAL[0], HISTORICAL[1] + 1))
    for first, last in WINDOWS.values():
        needed |= set(range(first, last + 1))
    out = []
    for archive in sorted(directory.glob("*.bin")):
        with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory() as scratch:
            for member in bundle.namelist():
                if not member.endswith(".nc"):
                    continue
                model, experiment = cmip6.split_name(Path(member).name)
                bundle.extract(member, scratch)
                field = cmip6.read_rx5day(Path(scratch) / member)
                years = [year for year in field["year"].to_numpy() if year in needed]
                if not years:
                    continue
                values = reducer.weighted_mean(to_lattice(field.sel(year=years), lon, lat), weights)
                out.append({"model": model, "experiment": experiment, "years": np.asarray(years), "values": values})
                _log.info("%s %s: %d years", model, experiment, len(years))
    return out


def deltas(series: list[dict], isos: list[str]) -> pd.DataFrame:
    """Relative change in mean and SD of annual rx5day, per model, scenario, window and country."""
    history = {s["model"]: s for s in series if s["experiment"] == "historical"}
    rows = []
    for s in series:
        if s["experiment"] == "historical" or s["model"] not in history:
            continue
        h = history[s["model"]]
        h_mask = (h["years"] >= HISTORICAL[0]) & (h["years"] <= HISTORICAL[1])
        h_values = h["values"][h_mask]
        h_mean, h_sd = np.nanmean(h_values, axis=0), np.nanstd(h_values, axis=0, ddof=1)
        for window, (first, last) in WINDOWS.items():
            mask = (s["years"] >= first) & (s["years"] <= last)
            if mask.sum() < 15:
                continue
            w_values = s["values"][mask]
            w_mean, w_sd = np.nanmean(w_values, axis=0), np.nanstd(w_values, axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel_mean, rel_sd = w_mean / h_mean - 1, w_sd / h_sd - 1
            rows.append(
                pd.DataFrame(
                    {
                        "model": s["model"],
                        "scenario": s["experiment"],
                        "window": window,
                        "ISO": isos,
                        "hist_mean_mm": h_mean,
                        "rel_change_mean": rel_mean,
                        "rel_change_sd": rel_sd,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    panel = create_replication_data(CACHE)
    isos_wanted = sorted(panel["ISO"].unique().to_list())
    boundaries = gpd.read_file(GLOBAL / "boundaries_country.gpkg").reset_index(drop=True)
    boundaries = boundaries[boundaries["GID_0"].isin(isos_wanted)].reset_index(drop=True)
    isos = boundaries["GID_0"].tolist()
    template = xr.open_dataset(GLOBAL / "era5_daily_2020.nc")
    lon, lat = template["longitude"].to_numpy(), template["latitude"].to_numpy()
    reducer = PolygonReducer(boundaries.geometry, lon, lat)
    weights = np.load(GLOBAL / f"population_era5_daily_{POPULATION_EPOCH}.npy")

    series = country_series(cmip6.raw_directory(CACHE), reducer, weights, lon, lat)
    table = deltas(series, isos)

    # ERA5 delta-method anchor: 1991-2020 mean and SD of the same population-weighted annual rx5day.
    era5 = (
        pl.read_parquet(COUNTRY_TABLE)
        .filter(pl.col("year").is_between(*ERA5_BASELINE))
        .group_by("ISO")
        .agg(pl.col("rx5day_popmean").mean().alias("era5_mean_mm"), pl.col("rx5day_popmean").std().alias("era5_sd_mm"))
        .to_pandas()
    )
    table = table.merge(era5, on="ISO", how="left")
    table["dz"] = table["era5_mean_mm"] * table["rel_change_mean"] / table["era5_sd_mm"]
    table["rate_multiplier"] = np.exp(SLOPE * table["dz"])  # ILLUSTRATIVE: everything else held fixed
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_parquet(RESULTS / "country_deltas.parquet")

    # Global population-weighted multiplier per model/scenario/window, then spread across models.
    population = panel.filter(pl.col("date").dt.year() == 2020).select("ISO", "population").to_pandas().dropna()
    merged = table.merge(population, on="ISO")
    merged["weighted"] = merged["rate_multiplier"] * merged["population"]
    keys = ["model", "scenario", "window"]
    valid = merged.dropna(subset=["rate_multiplier"])
    global_fig = (
        (valid.groupby(keys)["weighted"].sum() / valid.groupby(keys)["population"].sum())
        .rename("rate_multiplier")
        .reset_index()
    )
    global_fig["ISO"] = "GLOBAL"
    both = pd.concat(
        [
            table[table["ISO"].isin(FOCUS)][
                [*keys, "ISO", "rate_multiplier", "dz", "rel_change_mean", "rel_change_sd"]
            ],
            global_fig,
        ],
        ignore_index=True,
    )
    spread = (
        both.groupby(["ISO", "scenario", "window"])
        .agg(
            models=("model", "nunique"),
            multiplier_median=("rate_multiplier", "median"),
            multiplier_p10=("rate_multiplier", lambda v: v.quantile(0.1)),
            multiplier_p90=("rate_multiplier", lambda v: v.quantile(0.9)),
            dz_median=("dz", "median"),
            rel_mean_median=("rel_change_mean", "median"),
            rel_sd_median=("rel_change_sd", "median"),
        )
        .reset_index()
    )
    spread.to_csv(RESULTS / "multipliers.csv", index=False)
    pd.set_option("display.width", 220)
    print(spread.round(3).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
