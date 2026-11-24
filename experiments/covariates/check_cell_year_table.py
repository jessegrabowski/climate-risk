"""
Light checks on a country's covariate table: nulls, CHIRPS vs ERA5 correlation, two eye-check maps.

    pixi run python experiments/covariates/check_cell_year_table.py

Prints the numbers and writes two PNGs to ``results/2026-11-21-covariates/``.
"""

import matplotlib

matplotlib.use("Agg")

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COUNTRIES, PROCESSED, REPO

OUT = REPO / "results" / "2026-11-21-covariates"


def map_of(table: pd.DataFrame, column: str, title: str, path) -> None:
    mean = table.groupby(["cell_id", "lon", "lat"], as_index=False)[column].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    points = ax.scatter(mean["lon"], mean["lat"], c=mean[column], s=140, marker="s", cmap="viridis")
    fig.colorbar(points, ax=ax, label=column)
    ax.set(title=title, xlabel="longitude", ylabel="latitude", aspect="equal")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(iso: str) -> None:
    print(f"\n===== {iso} =====")
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(PROCESSED / f"{iso}_cell_year.parquet")
    value_columns = [c for c in table.columns if c not in ("cell_id", "year", "lon", "lat", "iso3")]

    print(
        "rows", len(table), "cells", table["cell_id"].nunique(), "years", table["year"].min(), "-", table["year"].max()
    )
    print("\nNulls per column:\n", table[value_columns].isna().sum().to_string())
    print(
        "\nNulls per year (all value columns):\n",
        table.groupby("year")[value_columns].apply(lambda g: g.isna().sum().sum()).to_string(),
    )
    empty = table.groupby("cell_id")[value_columns].apply(lambda g: g.isna().all())
    print("\nCells with an all-null column:\n", empty.sum()[empty.sum() > 0].to_string() or "none")

    pair = table[["precip_chirps_annual", "precip_era5"]].dropna()
    print(
        "\nCHIRPS vs ERA5 annual total, pooled cell-years: r =",
        round(float(np.corrcoef(pair.T)[0, 1]), 4),
        "n =",
        len(pair),
    )
    cell_means = table.groupby("cell_id")[["precip_chirps_annual", "precip_era5"]].mean().dropna()
    print(
        "Across cells (long-run means): r =", round(float(np.corrcoef(cell_means.T)[0, 1]), 4), "n =", len(cell_means)
    )

    map_of(table, "rx5day", f"{iso}: mean annual max 5-day rainfall (CHIRPS v2), mm", OUT / f"{iso}_map_rx5day.png")
    map_of(
        table,
        "q_annual_max",
        f"{iso}: mean annual max daily discharge (GloFAS v5), m3/s",
        OUT / f"{iso}_map_q_annual_max.png",
    )


if __name__ == "__main__":
    for code in sys.argv[1:] or COUNTRIES:
        main(code)
