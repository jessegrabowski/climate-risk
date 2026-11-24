"""
Plot the year effect of fit 1a (hydro alone, with co2) against the shared year effect of fit 2b (joint model).

Usage (from the repo root): ``pixi run python experiments/identification/plot_year_effect.py``
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from common import CACHE, FITS, RESULTS_DIR

mpl.use("Agg")

SOURCES = {
    "1a  hydro alone, with co2": CACHE / "idata" / "frequency_stage2_yeareffect_long.nc",
    "2b  joint, one shared year effect, co2 on hydro only": FITS
    / "test2_geophysical_control_2b_joint_co2_hydro_only.nc",
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, path in SOURCES.items():
        effect = xr.open_datatree(path)["posterior"]["year_effect"]
        draws = effect.stack(sample=("chain", "draw")).squeeze().transpose(..., "sample").to_numpy()
        years = effect["year"].to_numpy() if "year" in effect.coords else np.arange(draws.shape[0])
        low, mid, high = np.percentile(draws, [5.5, 50, 94.5], axis=-1)
        ax.plot(years, mid, label=label)
        ax.fill_between(years, low, high, alpha=0.2)
    ax.set_xlabel("year")
    ax.set_ylabel("year effect (log scale, zero in first year)")
    ax.set_title("Year effect: median and 89% interval")
    ax.legend(frameon=False)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "year_effect_1a_vs_2b.png", dpi=150)


if __name__ == "__main__":
    main()
