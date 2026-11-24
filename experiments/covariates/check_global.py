"""
Light checks on the global country-year table: nulls, coverage gaps, CHIRPS vs ERA5 and one global map.

    pixi run python experiments/covariates/check_global.py
Writes results/2026-11-23-global-covariates/{checks.txt, map_rx5day_mean.png} (no data files).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

OUT = Path("data/processed/covariates/global")
RESULTS = Path("results/2026-11-23-global-covariates")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(OUT / "global_country_year.parquet")
    value_columns = [c for c in table.columns if c not in ("GID_0", "NAME_0", "year")]
    lines = [
        f"rows {len(table)}, countries {table['GID_0'].nunique()}, years {table['year'].min()}-{table['year'].max()}"
    ]

    lines.append("\nNull share by column (all country-years):")
    lines.append(table[value_columns].isna().mean().round(3).to_string())

    lines.append("\nNull share by year (mean over mean-columns):")
    mean_columns = [c for c in value_columns if c.endswith("_mean")]
    lines.append(table.groupby("year")[mean_columns].apply(lambda g: g.isna().mean().mean()).round(3).to_string())

    lines.append("\nCountries with NO value in a source (all years):")
    groups = {
        "era5": "rx5day_mean",
        "chirps": "chirps_rx5day_mean",
        "era5land": "soilm_mean_mean",
        "glofas": "q_annual_max_mean",
    }
    for name, column in groups.items():
        if column in table:
            empty = table.groupby("GID_0")[column].apply(lambda s: s.isna().all())
            lines.append(f"{name}: {int(empty.sum())} of {len(empty)}: {', '.join(sorted(empty[empty].index))}")

    both = table.dropna(subset=["precip_era5_mean", "chirps_precip_mean"])
    pooled = both["precip_era5_mean"].corr(both["chirps_precip_mean"])
    by_country = both.groupby("GID_0")[["precip_era5_mean", "chirps_precip_mean"]].mean()
    lines.append(
        f"\nCHIRPS v2 (0.25 deg box average) vs ERA5 annual totals, country means: pooled country-years r = {pooled:.4f} "
        f"(n = {len(both)}); across country averages r = {by_country.corr().iloc[0, 1]:.4f} (n = {len(by_country)})"
    )
    (RESULTS / "checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    stack = xr.concat(
        [xr.open_dataset(OUT / f"era5_daily_{y}.nc")["rx5day"].reset_coords(drop=True) for y in range(1981, 2026)],
        dim="year",
    )
    mean = stack.mean("year")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    image = ax.pcolormesh(
        mean["longitude"], mean["latitude"], mean, cmap="viridis", vmax=float(mean.quantile(0.99)), shading="auto"
    )
    fig.colorbar(image, ax=ax, label="mean annual rx5day, mm (ERA5, 1981-2025)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(RESULTS / "map_rx5day_mean.png", dpi=130)


if __name__ == "__main__":
    main()
